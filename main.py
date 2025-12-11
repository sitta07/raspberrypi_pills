import os
import sys
import time
import threading
import numpy as np
import cv2
import torch
import pickle
from collections import Counter
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

# ================= FIX RASPBERRY PI ENVIRONMENT =================
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["OMP_NUM_THREADS"] = "3"    

try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ Warning: Picamera2 not found.")

# ================= CONFIGURATION =================
MODEL_PILL_PATH = 'models/pills_seg.pt'      
MODEL_PACK_PATH = 'models/seg_best_process.pt' 
DB_FILES = {
    'pills': {'vec': 'database/db_register/db_pills.pkl', 'col': 'database/db_register/colors_pills.pkl'},
    'packs': {'vec': 'database/db_register/db_packs.pkl', 'col': 'database/db_register/colors_packs.pkl'}
}
IMG_DB_FOLDER = 'database_images'
HIS_FILE_PATH = 'prescription.txt' 

DISPLAY_W, DISPLAY_H = 1280, 720
AI_IMG_SIZE = 416 
ZOOM_FACTOR = 1.0   

# --- 🔥 UI EXCLUSION ZONE (กัน AI ตรวจจับ Dashboard ตัวเอง) ---
# มุมขวาบน กว้าง 320px สูง 180px ห้ามตรวจจับ!
UI_ZONE_X = DISPLAY_W - 320 
UI_ZONE_Y = 180 

# --- TUNING WEIGHTS (เพิ่ม Color ให้ Pack) ---
# Pills: Vector 50%, Color 40%, SIFT 10%
W_PILL = {'vec': 0.5, 'col': 0.4, 'sift': 0.1}

# Packs: Vector 60%, Color 30%, SIFT 10% (เดิม Color 0%)
W_PACK = {'vec': 0.6, 'col': 0.3, 'sift': 0.1} 

COLOR_REJECT_THRESHOLD = 0.40
RANK_WEIGHTS = [1.0, 0.6, 0.2] 

CONF_PILL = 0.5   
CONF_PACK = 0.5     
SCORE_WIN_THRESHOLD = 2.0 
CONSISTENCY_THRESHOLD = 3 

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (UI FIX + PACK COLOR)")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}C"
    except: return "N/A"

def is_point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 < px < x2 and y1 < py < y2

def apply_digital_zoom(frame, zoom_factor):
    if zoom_factor <= 1.0: return frame
    h, w = frame.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    top, left = (h - new_h) // 2, (w - new_w) // 2
    crop = frame[top:top+new_h, left:left+new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

# ================= 1. WEBCAM STREAM =================
class WebcamStream:
    __slots__ = ('stopped', 'frame', 'grabbed', 'picam2', 'lock', 'cam')
    def __init__(self):
        self.stopped = False; self.frame = None; self.grabbed = False
        self.picam2 = None; self.cam = None; self.lock = threading.Lock()

    def start(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (DISPLAY_W, DISPLAY_H), "format": "RGB888"},
                controls={"FrameDurationLimits": (100000, 100000)})
            self.picam2.configure(config); self.picam2.start(); time.sleep(2.0)
        except:
            self.cam = cv2.VideoCapture(0)
            self.cam.set(3, DISPLAY_W); self.cam.set(4, DISPLAY_H)
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                if self.picam2:
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        with self.lock: self.frame = frame; self.grabbed = True
                else:
                    ret, frame = self.cam.read()
                    if ret:
                        with self.lock: self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); self.grabbed = True
            except: self.stopped = True

    def read(self):
        with self.lock: return self.frame if self.grabbed else None
    def stop(self):
        self.stopped = True
        if self.picam2: self.picam2.stop(); self.picam2.close()
        if self.cam: self.cam.release()

# ================= 2. RESOURCES & STRICT SCOPED DB =================
class HISLoader:
    @staticmethod
    def load_database(filename):
        if not os.path.exists(filename): return {}
        db = {}
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.split('|')
                    if len(parts) < 3: continue
                    hn, name = parts[0].strip(), parts[1].strip()
                    drugs = [d.strip().lower().replace('\ufeff', '') for d in parts[2].split(',') if d.strip()]
                    db[hn] = {'name': name, 'drugs': drugs}
            return db
        except: return {}

class PrescriptionState:
    def __init__(self):
        self.all_drugs = []  
        self.verified_drugs = set()
        self.session_pills_mat = None; self.session_pills_lbl = None
        self.session_packs_mat = None; self.session_packs_lbl = None
        self.lock = threading.Lock()
    
    def load_drugs(self, drug_list):
        with self.lock:
            self.all_drugs = drug_list.copy()
            self.verified_drugs.clear()
            self.create_session_db()
            
    def create_session_db(self):
        """ STRICT MODE: Filter Global DB based on Prescription """
        s_pills_vecs, s_pills_lbls = [], []
        s_packs_vecs, s_packs_lbls = [], []
        
        if matrix_pills is None or matrix_packs is None: return

        # Strict Filter for Pills
        for idx, label in enumerate(pills_lbls):
            if any(t.lower() in label.lower() for t in self.all_drugs):
                s_pills_vecs.append(matrix_pills[idx])
                s_pills_lbls.append(label)

        # Strict Filter for Packs
        for idx, label in enumerate(packs_lbls):
            if any(t.lower() in label.lower() for t in self.all_drugs):
                s_packs_vecs.append(matrix_packs[idx])
                s_packs_lbls.append(label)

        if s_pills_vecs:
            self.session_pills_mat = torch.stack(s_pills_vecs).to(device)
            self.session_pills_lbl = s_pills_lbls
        else: self.session_pills_mat = None; self.session_pills_lbl = []

        if s_packs_vecs:
            self.session_packs_mat = torch.stack(s_packs_vecs).to(device)
            self.session_packs_lbl = s_packs_lbls
        else: self.session_packs_mat = None; self.session_packs_lbl = []

    def verify_drug(self, drug_name):
        with self.lock:
            if drug_name in self.verified_drugs: return
            for target in self.all_drugs:
                if target.lower() in drug_name.lower() or drug_name.lower() in target.lower():
                    self.verified_drugs.add(target)
                    return
            self.verified_drugs.add(drug_name)
    
    def is_verified(self, drug_name):
        with self.lock:
            for v in self.verified_drugs:
                if v.lower() in drug_name.lower() or drug_name.lower() in v.lower(): return True
            return False

    def get_session_matrices(self):
        with self.lock:
            return (self.session_pills_mat, self.session_pills_lbl, 
                    self.session_packs_mat, self.session_packs_lbl)

prescription_state = PrescriptionState()

def load_pkl_to_list(filepath):
    if not os.path.exists(filepath): return [], []
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            items = [(v, name) for name, vec_list in data.items() for v in vec_list]
            if items:
                vecs, lbls = zip(*items)
                return list(vecs), list(lbls)
            return [], []
    except: return [], []

# Load Global Resources
pills_vecs, pills_lbls = load_pkl_to_list(DB_FILES['pills']['vec'])
packs_vecs, packs_lbls = load_pkl_to_list(DB_FILES['packs']['vec'])
matrix_pills = torch.tensor(np.array(pills_vecs), device=device, dtype=torch.float32) if pills_vecs else None
matrix_packs = torch.tensor(np.array(packs_vecs), device=device, dtype=torch.float32) if packs_vecs else None
if matrix_pills is not None: matrix_pills = matrix_pills / matrix_pills.norm(dim=1, keepdim=True)
if matrix_packs is not None: matrix_packs = matrix_packs / matrix_packs.norm(dim=1, keepdim=True)

color_db = {}
for db_type in ['pills', 'packs']:
    try:
        with open(DB_FILES[db_type]['col'], 'rb') as f: color_db.update(pickle.load(f))
    except: pass

sift = cv2.SIFT_create(nfeatures=100)
bf = cv2.BFMatcher(crossCheck=False)
sift_db = {}
if os.path.exists(IMG_DB_FOLDER):
    for folder in os.listdir(IMG_DB_FOLDER):
        path = os.path.join(IMG_DB_FOLDER, folder)
        if not os.path.isdir(path): continue
        des_list = []
        for img_file in os.listdir(path)[:3]:
            img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if max(img.shape) > 512: img = cv2.resize(img, None, fx=512/max(img.shape), fy=512/max(img.shape))
                _, des = sift.detectAndCompute(img, None)
                if des is not None: des_list.append(des)
        if des_list: sift_db[folder] = des_list

try:
    model_pill = YOLO(MODEL_PILL_PATH, task='detect')
    model_pack = YOLO(MODEL_PACK_PATH, task='detect')
    weights = models.ResNet50_Weights.DEFAULT
    base_model = models.resnet50(weights=weights)
    embedder = torch.nn.Sequential(*list(base_model.children())[:-1])
    embedder.eval().to(device)
    del base_model
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch.set_grad_enabled(False)
except Exception as e: sys.exit(1)

# ================= 3. TRINITY ENGINE (COLOR ACTIVATED FOR PACKS) =================
COLOR_NORM = np.array([90.0, 255.0, 255.0])
SIFT_RATIO = 0.75; SIFT_MAX_MATCHES = 15.0

def trinity_inference(img_crop, is_pill=True, session_mat=None, session_lbl=None):
    if session_mat is None or session_lbl is None or len(session_lbl) == 0: return []

    try:
        # 1. Vector
        pil_img = Image.fromarray(img_crop if is_pill else cv2.merge([cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)]*3))
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        live_vec = embedder(input_tensor).flatten()
        live_vec = live_vec / live_vec.norm()

        scores = torch.matmul(live_vec, session_mat.T).squeeze(0)
        
        k_val = min(10, len(session_lbl))
        if k_val == 0: return []
        top_k_val, top_k_idx = torch.topk(scores, k=k_val)
        
        # 2. Color (Now applied to both Pills AND Packs)
        live_color = None
        
        # Calculate dominant color (Center area)
        h, w = img_crop.shape[:2]
        center_crop = img_crop[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        
        if center_crop.size > 0:
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_RGB2HSV)
            live_color = np.mean(hsv, axis=(0,1))

        # 3. Filter
        gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        _, des_live = sift.detectAndCompute(gray, None)
        
        averaged_candidates = []
        raw_candidates = {}

        for idx, sc in zip(top_k_idx.detach().cpu().numpy(), top_k_val.detach().cpu().numpy()):
            name = session_lbl[idx]
            vec_score = float(sc)
            
            # --- COLOR CHECK ---
            color_score = 0.5 # Default
            if live_color is not None and name in color_db:
                diff = np.abs(live_color - color_db[name])
                diff[0] = min(diff[0], 180 - diff[0]) 
                norm_diff = diff / COLOR_NORM
                dist = np.linalg.norm(norm_diff)
                
                # REJECT if color is way off (Strict)
                if dist > COLOR_REJECT_THRESHOLD: continue 
                
                color_score = np.clip(np.exp(-3.0 * dist), 0, 1)

            clean_name = name.replace("_pill", "").replace("_pack", "")
            
            # --- WEIGHTING ---
            weights = W_PILL if is_pill else W_PACK
            final_score = (vec_score * weights['vec']) + (color_score * weights['col']) + (0.5 * weights['sift']) # Simplified SIFT for speed
            
            if clean_name not in raw_candidates: raw_candidates[clean_name] = []
            raw_candidates[clean_name].append(final_score)

        for name, score_list in raw_candidates.items():
            avg_score = sum(sorted(score_list, reverse=True)[:3]) / min(len(score_list), 3)
            averaged_candidates.append((name, avg_score))

        return sorted(averaged_candidates, key=lambda x: x[1], reverse=True)

    except Exception as e: return []

# ================= 4. AI WORKER (ROI FILTER ADDED) =================
class AIProcessor:
    __slots__ = ('latest_frame', 'results', 'top_candidates', 'stopped', 'lock', 'is_rx_mode', 
                 'current_patient_info', 'scale_x', 'scale_y', 'resize_interpolation', 'consistency_counter',
                 'final_winner', 'winner_verified')
    
    def __init__(self):
        self.latest_frame = None; self.results = []; self.top_candidates = []
        self.stopped = False; self.lock = threading.Lock()
        self.is_rx_mode = False; self.current_patient_info = None
        self.scale_x = DISPLAY_W / AI_IMG_SIZE; self.scale_y = DISPLAY_H / AI_IMG_SIZE
        self.resize_interpolation = cv2.INTER_LINEAR; self.consistency_counter = {}
        self.final_winner = "Analyzing..."
        self.winner_verified = False

    def load_patient(self, patient_data):
        with self.lock:
            if not patient_data:
                self.is_rx_mode = False; self.current_patient_info = None; prescription_state.load_drugs([])
            else:
                self.is_rx_mode = True; self.current_patient_info = patient_data
                prescription_state.load_drugs(patient_data['drugs'])
                print(f"🏥 Loaded: {patient_data['name']}")
            self.consistency_counter.clear()
            self.final_winner = "Ready"

    def start(self): threading.Thread(target=self.run, daemon=True).start(); return self
    def update_frame(self, frame): with self.lock: self.latest_frame = frame
    def get_results(self): with self.lock: return self.results, self.top_candidates, self.final_winner, self.winner_verified

    def run(self):
        print("[DEBUG] AI Loop Started - ROI & Color Check Active")
        while not self.stopped:
            with self.lock:
                frame_HD = self.latest_frame; self.latest_frame = None
                s_pill_mat, s_pill_lbl, s_pack_mat, s_pack_lbl = prescription_state.get_session_matrices()
            
            if frame_HD is None: time.sleep(0.005); continue

            frame_yolo = cv2.resize(frame_HD, (AI_IMG_SIZE, AI_IMG_SIZE), interpolation=self.resize_interpolation)
            global_votes = {} 
            detected_boxes = []

            # --- PROCESS PACKS ---
            pack_res = model_pack(frame_yolo, verbose=False, conf=CONF_PACK, imgsz=AI_IMG_SIZE, max_det=5, agnostic_nms=True)
            for box in pack_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                x1, y1 = int(box[0] * self.scale_x), int(box[1] * self.scale_y)
                x2, y2 = int(box[2] * self.scale_x), int(box[3] * self.scale_y)
                
                # 🛑 ROI CHECK: Skip if in Top-Right UI Zone
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if cx > UI_ZONE_X and cy < UI_ZONE_Y:
                    continue # SKIP THIS BOX!

                crop = frame_HD[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Now using Color for Packs too
                candidates = trinity_inference(crop, is_pill=False, session_mat=s_pack_mat, session_lbl=s_pack_lbl)
                
                for i, (name, score) in enumerate(candidates[:3]):
                    weight = RANK_WEIGHTS[i] * 1.0 # Pack Base Weight
                    if name not in global_votes: global_votes[name] = 0.0
                    global_votes[name] += (score ** 2) * weight
                
                detected_boxes.append({'box': (x1,y1,x2,y2), 'type': 'pack'})

            # --- PROCESS PILLS ---
            pill_res = model_pill(frame_yolo, verbose=False, conf=CONF_PILL, imgsz=AI_IMG_SIZE, max_det=20, agnostic_nms=True)
            for box in pill_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                x1, y1 = int(box[0] * self.scale_x), int(box[1] * self.scale_y)
                x2, y2 = int(box[2] * self.scale_x), int(box[3] * self.scale_y)
                
                # 🛑 ROI CHECK for Pills too
                cx, cy = (x1+x2)//2, (y1+y2)//2
                if cx > UI_ZONE_X and cy < UI_ZONE_Y: continue

                crop = frame_HD[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                candidates = trinity_inference(crop, is_pill=True, session_mat=s_pill_mat, session_lbl=s_pill_lbl)
                
                in_pack = False
                for p in detected_boxes:
                    if p['type'] == 'pack' and is_point_in_box((cx, cy), p['box']): in_pack = True; break
                
                if in_pack:
                    for i, (name, score) in enumerate(candidates[:3]):
                        clean_name = name.replace("_pill", "").lower()
                        weight = RANK_WEIGHTS[i] * 4.0 # Pill Base Weight (High)
                        if clean_name not in global_votes: global_votes[clean_name] = 0.0
                        global_votes[clean_name] += (score ** 2) * weight

                detected_boxes.append({'box': (x1,y1,x2,y2), 'type': 'pill'})

            # --- DECISION ---
            sorted_candidates = sorted(global_votes.items(), key=lambda x: x[1], reverse=True)
            current_top_list = sorted_candidates[:3]
            
            winner_name = "Unknown"
            winner_score = 0.0
            
            if sorted_candidates:
                winner_name, winner_score = sorted_candidates[0]
                second_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0
                margin = winner_score - second_score

                if winner_score > SCORE_WIN_THRESHOLD or (margin > 1.0 and winner_score > 1.0):
                    self.consistency_counter[winner_name] = self.consistency_counter.get(winner_name, 0) + 1
                    if self.consistency_counter[winner_name] >= CONSISTENCY_THRESHOLD:
                        prescription_state.verify_drug(winner_name)
                else:
                    winner_name = "Analyzing..."

            is_verified = prescription_state.is_verified(winner_name)
            for k in list(self.consistency_counter.keys()):
                if k != winner_name: self.consistency_counter[k] = 0

            with self.lock: 
                self.results = detected_boxes
                self.top_candidates = current_top_list
                self.final_winner = winner_name
                self.winner_verified = is_verified
    
    def stop(self): self.stopped = True

# ================= 5. UI DRAWING =================
FONT = cv2.FONT_HERSHEY_SIMPLEX
RGB_GREEN = (0, 255, 0); RGB_RED = (0, 0, 255); RGB_YELLOW = (0, 255, 255); 
RGB_WHITE = (255, 255, 255); RGB_BLACK = (0, 0, 0); RGB_GRAY = (50, 50, 50)

def draw_boxes(frame, results, winner):
    for r in results:
        x1, y1, x2, y2 = r['box']
        color = RGB_GREEN if r.get('type') == 'pack' else RGB_YELLOW
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def draw_ui_overlay(frame, candidates, winner, verified):
    h, w = frame.shape[:2]
    
    # 1. Dashboard (Top-Right)
    panel_w, panel_h = 320, 160
    x_start, y_start = w - panel_w - 10, 10
    
    # Draw Background (This corresponds to the Exclusion Zone)
    sub = frame[y_start:y_start+panel_h, x_start:x_start+panel_w]
    white = np.ones(sub.shape, dtype=np.uint8) * 255
    cv2.addWeighted(sub, 0.4, white, 0.6, 0, sub)
    cv2.rectangle(frame, (x_start, y_start), (x_start+panel_w, y_start+panel_h), RGB_BLACK, 2)
    
    # Debug Text for ROI
    cv2.putText(frame, "ZONE LOCKED (NO DETECT)", (x_start+10, y_start+150), FONT, 0.4, RGB_RED, 1)

    if candidates:
        cv2.putText(frame, "COLOR FUSION RANK", (x_start+10, y_start+25), FONT, 0.6, RGB_BLACK, 2)
        for i, (name, score) in enumerate(candidates):
            y_pos = y_start + 60 + (i * 25)
            bar_len = int(min(score, 15.0) / 15.0 * 150) 
            cv2.putText(frame, f"{i+1}. {name[:12]}", (x_start+10, y_pos), FONT, 0.55, RGB_BLACK, 1)
            cv2.rectangle(frame, (x_start+130, y_pos-10), (x_start+130+bar_len, y_pos), (0, 100, 255), -1)
            cv2.putText(frame, f"{score:.1f}", (x_start+135+bar_len, y_pos), FONT, 0.5, RGB_BLACK, 1)

    # 2. Status Bar
    bar_h = 60
    cv2.rectangle(frame, (0, h-bar_h), (w, h), RGB_BLACK, -1)
    
    status_text = f"CONCLUSION: {winner.upper()}" if winner != "Analyzing..." else "ANALYZING..."
    if verified: status_text = f"VERIFIED: {winner.upper()} (MATCHED)"
    
    if verified:
        cv2.rectangle(frame, (0, h-bar_h), (w, h), RGB_GREEN, -1)
        text_color = RGB_BLACK
    else:
        text_color = RGB_WHITE

    tsize = cv2.getTextSize(status_text, FONT, 1.2, 3)[0]
    tx = (w - tsize[0]) // 2
    ty = h - (bar_h - tsize[1]) // 2 - 5
    cv2.putText(frame, status_text, (tx, ty), FONT, 1.2, text_color, 3)

# ================= 6. MAIN =================
def main():
    TARGET_HN = "HN-101" 
    cam = WebcamStream().start()
    ai = AIProcessor().start()
    
    his_db = HISLoader.load_database(HIS_FILE_PATH)
    if TARGET_HN in his_db: 
        d = his_db[TARGET_HN]; d['hn'] = TARGET_HN; ai.load_patient(d)
    
    while cam.read() is None: time.sleep(0.1)
    window_name = "PillTrack: UI Fixed + Pack Color"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H) 

    print("🎥 SYSTEM READY. UI Zone Excluded. Pack Color Active.")
    
    try:
        while True:
            frame = cam.read()
            if frame is None: time.sleep(0.01); continue
            frame = apply_digital_zoom(frame, ZOOM_FACTOR)
            
            ai.update_frame(frame.copy()) 
            results, candidates, winner, verified = ai.get_results()
            draw_boxes(frame, results, winner)
            draw_ui_overlay(frame, candidates, winner, verified)
            
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db: d = his_db[TARGET_HN]; d['hn'] = TARGET_HN; ai.load_patient(d)

    except KeyboardInterrupt: pass
    finally: cam.stop(); ai.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()