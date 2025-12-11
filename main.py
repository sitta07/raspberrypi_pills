import os
import sys
import time
import threading
import numpy as np
import cv2
import torch
import pickle
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
# Paths
MODEL_PILL_PATH = 'models/pills_seg.pt'      #seg    
MODEL_PACK_PATH = 'models/seg_best_process.pt' #seg
DB_FILES = {
    'pills': {'vec': 'database/db_register/db_pills.pkl', 'col': 'database/db_register/colors_pills.pkl'},
    'packs': {'vec': 'database/db_register/db_packs.pkl', 'col': 'database/db_register/colors_packs.pkl'}
}

IMG_DB_FOLDER = 'database_images'
HIS_FILE_PATH = 'prescription.txt' 

# Display & AI Resolution
DISPLAY_W, DISPLAY_H = 1280, 720
AI_IMG_SIZE = 416 

# ZOOM CONFIGURATION
ZOOM_FACTOR = 1.0   

# Thresholds
CONF_PILL = 0.5   
CONF_PACK = 0.5     
SCORE_PASS_PILL = 0.2
SCORE_PASS_PACK = 0.2

# --- SENIOR UPGRADES ---
CONSISTENCY_THRESHOLD = 2   
MAX_OBJ_AREA_RATIO = 0.40   

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (RGB888 SCOPED MODE)")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}C"
    except: 
        return "N/A"

def is_point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 < px < x2 and y1 < py < y2

def apply_digital_zoom(frame, zoom_factor):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    crop = frame[top:top+new_h, left:left+new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

# ================= 1. WEBCAM STREAM (RGB888 ONLY) =================
class WebcamStream:
    __slots__ = ('stopped', 'frame', 'grabbed', 'picam2', 'lock', 'cam')
    
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.picam2 = None
        self.cam = None
        self.lock = threading.Lock()

    def start(self):
        print("[DEBUG] Initializing Camera (RGB888)...")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (DISPLAY_W, DISPLAY_H), "format": "RGB888"},
                controls={"FrameDurationLimits": (100000, 100000)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
            print("[DEBUG] Picamera2 Started in RGB888")
        except Exception as e:
            print(f"[ERROR] Picamera2 Failed, using OpenCV: {e}")
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
            
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                if self.picam2:
                    frame = self.picam2.capture_array() 
                    if frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.grabbed = True
                    else: self.stopped = True
                else:
                    ret, frame = self.cam.read() 
                    if ret:
                        with self.lock:
                            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            self.grabbed = True
            except: 
                self.stopped = True
                break

    def read(self):
        with self.lock:
            return self.frame if self.grabbed else None
    
    def stop(self):
        self.stopped = True
        if self.picam2: 
            self.picam2.stop()
            self.picam2.close()
        if self.cam:
            self.cam.release()

# ================= 2. RESOURCES & STATE (UPDATED FOR SCOPING) =================
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
        
        # 🆕 New: Session-Specific Databases (For Filtering)
        self.session_pills_mat = None
        self.session_pills_lbl = None
        self.session_packs_mat = None
        self.session_packs_lbl = None
        
        self.lock = threading.Lock()
    
    def load_drugs(self, drug_list):
        with self.lock:
            self.all_drugs = drug_list.copy()
            self.verified_drugs.clear()
            # 🚀 Create Optimized DB for this patient immediately
            self.create_session_db()
            
    def create_session_db(self):
        """ สร้าง Matrix เฉพาะยาที่มีในใบสั่งยาเท่านั้น """
        s_pills_vecs, s_pills_lbls = [], []
        s_packs_vecs, s_packs_lbls = [], []

        if matrix_pills is None or matrix_packs is None: return

        # Filter Pills
        for idx, label in enumerate(pills_lbls):
            clean_label = label.lower()
            if any(target_drug.lower() in clean_label for target_drug in self.all_drugs):
                s_pills_vecs.append(matrix_pills[idx]) 
                s_pills_lbls.append(label)

        # Filter Packs
        for idx, label in enumerate(packs_lbls):
            clean_label = label.lower()
            if any(target_drug.lower() in clean_label for target_drug in self.all_drugs):
                s_packs_vecs.append(matrix_packs[idx])
                s_packs_lbls.append(label)

        if s_pills_vecs:
            self.session_pills_mat = torch.stack(s_pills_vecs).to(device)
            self.session_pills_lbl = s_pills_lbls
            print(f"🎯 Session Scope (Pills): Reduced to {len(s_pills_lbls)} items")
        else:
            self.session_pills_mat = None
            self.session_pills_lbl = []

        if s_packs_vecs:
            self.session_packs_mat = torch.stack(s_packs_vecs).to(device)
            self.session_packs_lbl = s_packs_lbls
            print(f"🎯 Session Scope (Packs): Reduced to {len(s_packs_lbls)} items")
        else:
            self.session_packs_mat = None
            self.session_packs_lbl = []

    def get_remaining_drugs(self):
        with self.lock:
            return [d for d in self.all_drugs if d not in self.verified_drugs]
    
    def toggle_drug(self, drug_name):
        with self.lock:
            if drug_name in self.verified_drugs:
                self.verified_drugs.remove(drug_name)
            else:
                self.verified_drugs.add(drug_name)

    def verify_drug(self, drug_name):
        with self.lock:
            if drug_name in self.verified_drugs: return
            found = False
            for target in self.all_drugs:
                if target.lower() in drug_name.lower() or drug_name.lower() in target.lower():
                    self.verified_drugs.add(target)
                    print(f"✨ VERIFIED (Matched): {target}")
                    found = True
                    break
            if not found:
                self.verified_drugs.add(drug_name)
    
    def is_verified(self, drug_name):
        with self.lock:
            for v in self.verified_drugs:
                if v.lower() in drug_name.lower() or drug_name.lower() in v.lower():
                    return True
            return False
    
    def get_all_drugs(self):
        with self.lock: return self.all_drugs.copy()

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

# Load Global DB
pills_vecs, pills_lbls = load_pkl_to_list(DB_FILES['pills']['vec'])
packs_vecs, packs_lbls = load_pkl_to_list(DB_FILES['packs']['vec'])

matrix_pills = torch.tensor(np.array(pills_vecs), device=device, dtype=torch.float32) if pills_vecs else None
matrix_packs = torch.tensor(np.array(packs_vecs), device=device, dtype=torch.float32) if packs_vecs else None

if matrix_pills is not None: matrix_pills = matrix_pills / matrix_pills.norm(dim=1, keepdim=True)
if matrix_packs is not None: matrix_packs = matrix_packs / matrix_packs.norm(dim=1, keepdim=True)

color_db = {}
for db_type in ['pills', 'packs']:
    try:
        with open(DB_FILES[db_type]['col'], 'rb') as f: 
            color_db.update(pickle.load(f))
    except: pass

sift = cv2.SIFT_create(nfeatures=100)
bf = cv2.BFMatcher(crossCheck=False)
sift_db = {}

if os.path.exists(IMG_DB_FOLDER):
    for folder in os.listdir(IMG_DB_FOLDER):
        path = os.path.join(IMG_DB_FOLDER, folder)
        if not os.path.isdir(path): continue
        des_list = []
        image_files = [x for x in os.listdir(path) if x.lower().endswith(('jpg', 'png', 'jpeg'))][:3]
        for img_file in image_files:
            img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if max(img.shape) > 512:
                    scale = 512 / max(img.shape)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                _, des = sift.detectAndCompute(img, None)
                if des is not None: des_list.append(des)
        if des_list:
            sift_db[folder] = des_list

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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch.set_grad_enabled(False)
except Exception as e: 
    print(f"[CRITICAL] Model Error: {e}")
    sys.exit(1)

# ================= 3. TRINITY ENGINE (RGB LOGIC) =================
COLOR_NORM = np.array([90.0, 255.0, 255.0])
SIFT_RATIO = 0.75
SIFT_MAX_MATCHES = 15.0

def trinity_inference(img_crop, is_pill=True, 
                      session_pills=None, session_pills_lbl=None,
                      session_packs=None, session_packs_lbl=None):
    
    target_matrix = (session_pills if session_pills is not None else matrix_pills) if is_pill else \
                    (session_packs if session_packs is not None else matrix_packs)
    target_labels = (session_pills_lbl if session_pills_lbl is not None else pills_lbls) if is_pill else \
                    (session_packs_lbl if session_packs_lbl is not None else packs_lbls)

    if target_matrix is None: return "DB Error", 0.0

    try:
        if is_pill: 
            pil_img = Image.fromarray(img_crop) 
        else:
            gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
            crop_3ch_gray = cv2.merge([gray_crop, gray_crop, gray_crop])
            pil_img = Image.fromarray(crop_3ch_gray)

        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        live_vec = embedder(input_tensor).flatten()
        live_vec = live_vec / live_vec.norm()
        
        scores = torch.matmul(live_vec, target_matrix.T).squeeze(0)
        k_val = min(10, len(target_labels))
        if k_val == 0: return "Unknown", 0.0
        
        top_k_val, top_k_idx = torch.topk(scores, k=k_val)
        candidates = []
        seen = set()
        
        for idx, sc in zip(top_k_idx.detach().cpu().numpy(), top_k_val.detach().cpu().numpy()):
            name = target_labels[idx]
            if name not in seen:
                candidates.append((name, float(sc)))
                seen.add(name)
                if len(candidates) >= 3: break

        live_color = None
        gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        _, des_live = sift.detectAndCompute(gray, None)
        
        if is_pill: 
            h, w = img_crop.shape[:2]
            center = img_crop[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            if center.size > 0:
                hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
                live_color = np.mean(hsv, axis=(0,1))

        best_score = -1
        final_name = "Unknown"
        
        for name, vec_score in candidates:
            clean_name = name.replace("_pill", "").replace("_pack", "")
            
            sift_score = 0.0
            if des_live is not None and clean_name in sift_db:
                max_good = 0
                for ref_des in sift_db[clean_name]:
                    try:
                        matches = bf.knnMatch(des_live, ref_des, k=2)
                        good = sum(1 for m, n in matches if len([m, n]) == 2 and m.distance < SIFT_RATIO * n.distance)
                        max_good = max(max_good, good)
                    except: pass
                sift_score = min(max_good / SIFT_MAX_MATCHES, 1.0)
                
            color_score = 0.0
            if is_pill and live_color is not None and name in color_db:
                diff = np.abs(live_color - color_db[name])
                diff[0] = min(diff[0], 180 - diff[0]) 
                norm_diff = diff / COLOR_NORM
                dist = np.linalg.norm(norm_diff)
                color_score = np.clip(np.exp(-3.0 * dist), 0, 1)
                
            w_vec, w_sift, w_col = (0.5, 0.4, 0.1) if is_pill else (0, 1, 0.0)
            total = vec_score * w_vec + sift_score * w_sift + color_score * w_col
            
            if total > best_score: 
                best_score = total
                final_name = clean_name

        return final_name, best_score
    except Exception as e:
        print(f"[Trinity Error] {e}")
        return "Error", 0.0

# ================= 4. AI WORKER (SMART & SCOPED) =================
class AIProcessor:
    __slots__ = ('latest_frame', 'results', 'stopped', 'lock', 'is_rx_mode', 
                 'current_patient_info', 'scale_x', 'scale_y',
                 'resize_interpolation', 'consistency_counter')
    
    def __init__(self):
        self.latest_frame = None 
        self.results = [] 
        self.stopped = False
        self.lock = threading.Lock()
        self.is_rx_mode = False
        self.current_patient_info = None
        self.scale_x = DISPLAY_W / AI_IMG_SIZE
        self.scale_y = DISPLAY_H / AI_IMG_SIZE
        self.resize_interpolation = cv2.INTER_LINEAR
        self.consistency_counter = {}

    def load_patient(self, patient_data):
        with self.lock:
            if not patient_data:
                self.is_rx_mode = False
                self.current_patient_info = None
                prescription_state.load_drugs([])
                self.consistency_counter.clear()
            else:
                self.is_rx_mode = True
                self.current_patient_info = patient_data
                drugs = patient_data['drugs']
                prescription_state.load_drugs(drugs) # 🚀 Triggers session DB creation
                self.consistency_counter.clear()
                print(f"🏥 Loaded: {patient_data['name']}")
    
    def start(self): 
        threading.Thread(target=self.run, daemon=True).start()
        return self
    
    def update_frame(self, frame): 
        with self.lock: 
            self.latest_frame = frame
        
    def get_results(self): 
        with self.lock: 
            return self.results, self.current_patient_info

    def is_valid_detection(self, box, img_w, img_h):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        image_area = img_w * img_h
        if area / image_area > MAX_OBJ_AREA_RATIO:
            return False 
        return True

    def run(self):
        print("[DEBUG] AI Worker Loop Started - Scoped Mode")
        
        while not self.stopped:
            with self.lock:
                frame_HD = self.latest_frame 
                self.latest_frame = None
                # 🔥 Get Scoped Matrices
                s_pill_mat, s_pill_lbl, s_pack_mat, s_pack_lbl = prescription_state.get_session_matrices()
            
            if frame_HD is None: 
                time.sleep(0.005)
                continue

            # Use scoped DB if available, else fallback to global (e.g., if rx list is empty)
            use_pill_mat = s_pill_mat if s_pill_mat is not None else matrix_pills
            use_pill_lbl = s_pill_lbl if s_pill_lbl else pills_lbls
            use_pack_mat = s_pack_mat if s_pack_mat is not None else matrix_packs
            use_pack_lbl = s_pack_lbl if s_pack_lbl else packs_lbls

            frame_yolo = cv2.resize(frame_HD, (AI_IMG_SIZE, AI_IMG_SIZE), 
                                   interpolation=self.resize_interpolation)
            
            final_detections = []
            active_packs = [] 
            found_in_this_frame = set()

            try:
                # ================= PHASE 1: DETECT PACKS =================
                pack_res = model_pack(frame_yolo, verbose=False, conf=CONF_PACK, 
                                     imgsz=AI_IMG_SIZE, max_det=5, agnostic_nms=True)
                
                for box in pack_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    x1, y1 = int(x1_s * self.scale_x), int(y1_s * self.scale_y)
                    x2, y2 = int(x2_s * self.scale_x), int(y2_s * self.scale_y)
                    
                    if not self.is_valid_detection((x1, y1, x2, y2), DISPLAY_W, DISPLAY_H): continue
                    if (x2-x1) < 50 or (y2-y1) < 50: continue
                    
                    crop = frame_HD[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    # Pass scoped matrices
                    real_name, real_score = trinity_inference(crop, is_pill=False,
                                              session_pills=None, session_pills_lbl=None,
                                              session_packs=use_pack_mat,
                                              session_packs_lbl=use_pack_lbl)
                    
                    final_name = real_name
                    final_score = real_score
                    is_wrong_drug = False
                    
                    # If scoped DB returns Unknown, it means it's not in prescription
                    if "Unknown" in real_name or "?" in real_name:
                         final_name = "Wrong / Unknown"
                         is_wrong_drug = True
                         final_score = 0.0

                    clean_name = final_name.replace("_pack", "").lower()

                    if not is_wrong_drug and final_score >= SCORE_PASS_PACK:
                        self.consistency_counter[clean_name] = self.consistency_counter.get(clean_name, 0) + 1
                        found_in_this_frame.add(clean_name)
                        if self.consistency_counter[clean_name] >= CONSISTENCY_THRESHOLD:
                            prescription_state.verify_drug(clean_name)
                    
                    pack_data = {
                        'label': final_name, 'score': final_score, 'type': 'pack',
                        'verified': prescription_state.is_verified(clean_name), 
                        'box': (x1, y1, x2, y2), 'is_wrong': is_wrong_drug,
                        'clean_name': clean_name
                    }
                    active_packs.append(pack_data)
                    final_detections.append(pack_data)

                # ================= PHASE 2: DETECT PILLS =================
                pill_res = model_pill(frame_yolo, verbose=False, conf=CONF_PILL, 
                                     imgsz=AI_IMG_SIZE, max_det=20, agnostic_nms=True)
                
                for box in pill_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    x1, y1 = int(x1_s * self.scale_x), int(y1_s * self.scale_y)
                    x2, y2 = int(x2_s * self.scale_x), int(y2_s * self.scale_y)
                    
                    if not self.is_valid_detection((x1, y1, x2, y2), DISPLAY_W, DISPLAY_H): continue
                    if (x2-x1) < 30 or (y2-y1) < 30: continue
                    
                    cx, cy = (x1+x2)>>1, (y1+y2)>>1
                    parent_pack = None
                    for pack in active_packs:
                        if is_point_in_box((cx, cy), pack['box']):
                            parent_pack = pack
                            break
                    
                    final_name = "Unknown"
                    final_score = 0.0
                    is_wrong_drug = False
                    is_verified = False

                    if parent_pack:
                        final_name = parent_pack['label'] 
                        final_score = parent_pack['score'] 
                        is_wrong_drug = parent_pack['is_wrong']
                        is_verified = parent_pack['verified']
                        clean_name = parent_pack['clean_name']
                        if not is_wrong_drug:
                             self.consistency_counter[clean_name] = self.consistency_counter.get(clean_name, 0) + 1
                             found_in_this_frame.add(clean_name)
                    else:
                        crop = frame_HD[y1:y2, x1:x2]
                        if crop.size > 0:
                            # Pass scoped matrices
                            real_name, real_score = trinity_inference(crop, is_pill=True,
                                                      session_pills=use_pill_mat,       
                                                      session_pills_lbl=use_pill_lbl,
                                                      session_packs=use_pack_mat,
                                                      session_packs_lbl=use_pack_lbl)
                            final_name = real_name
                            final_score = real_score

                            if "Unknown" in real_name or "?" in real_name:
                                final_name = "Wrong / Unknown"
                                final_score = 0.0 
                                is_wrong_drug = True

                            clean_name = final_name.lower()
                            if not is_wrong_drug and final_score > SCORE_PASS_PILL:
                                self.consistency_counter[clean_name] = self.consistency_counter.get(clean_name, 0) + 1
                                found_in_this_frame.add(clean_name)
                                if self.consistency_counter[clean_name] >= CONSISTENCY_THRESHOLD:
                                    prescription_state.verify_drug(clean_name)
                            
                            is_verified = prescription_state.is_verified(clean_name)

                    final_detections.append({
                        'label': final_name, 'score': final_score, 'type': 'pill',
                        'verified': is_verified, 'box': (x1, y1, x2, y2), 'is_wrong': is_wrong_drug
                    })

                all_tracked = list(self.consistency_counter.keys())
                for k in all_tracked:
                    if k not in found_in_this_frame:
                        self.consistency_counter[k] = 0

                with self.lock: 
                    self.results = final_detections
            
            except Exception as e:
                print(f"[ERROR-AI-LOOP] {e}")
            
    def stop(self): 
        self.stopped = True

# ================= 5. UI DRAWING (RGB COLORS) =================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.6
THICKNESS = 2
THICKNESS_BOX = 3

RGB_GREEN = (0, 255, 0)
RGB_RED   = (255, 0, 0)
RGB_YELLOW = (255, 255, 0)

def draw_boxes_on_items(frame, results):
    for r in results:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        score = r['score']
        obj_type = r.get('type', 'pill')
        is_verified = r.get('verified', False)
        is_wrong = r.get('is_wrong', False)
        
        if is_wrong:
            color = RGB_RED 
            label_display = f"!! {label} !!"
        elif is_verified:
            color = RGB_GREEN
            label_display = f"OK {label}"
        elif obj_type == 'pack':
            color = RGB_GREEN if score >= SCORE_PASS_PACK else RGB_YELLOW
            label_display = label
        elif "?" in label or score < SCORE_PASS_PILL:
            color = RGB_RED
            label_display = label
        else:
            color = RGB_YELLOW
            label_display = label

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS_BOX)
        cv2.putText(frame, f"{label_display} {score:.0%}", (x1, y1-10), 
                   FONT, FONT_SCALE_SMALL, color, THICKNESS)

# ================= 6. MAIN =================
def main():
    TARGET_HN = "HN-101" 
    
    cam = WebcamStream().start()
    ai = AIProcessor().start()
    
    his_db = HISLoader.load_database(HIS_FILE_PATH)
    if TARGET_HN in his_db: 
        d = his_db[TARGET_HN]
        d['hn'] = TARGET_HN
        ai.load_patient(d)
    
    print("⏳ Waiting for camera feed...")
    while cam.read() is None: time.sleep(0.1)
    
    window_name = "PillTrack Senior Edition (Scoped)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H) 

    print(f"🎥 RUNNING... (RGB888 SCOPED) - ZOOM FACTOR: {ZOOM_FACTOR}x")
    
    fps = 0
    prev_time = time.perf_counter()
    TARGET_FPS = 15 
    FRAME_TIME = 1.0 / TARGET_FPS
    try:
        while True:
            start_loop = time.perf_counter()
            frame_rgb = cam.read()
            if frame_rgb is None: 
                time.sleep(0.01)
                continue
            
            frame_rgb = apply_digital_zoom(frame_rgb, ZOOM_FACTOR)
            
            ai.update_frame(frame_rgb.copy()) 
            results, cur_patient = ai.get_results()
            draw_boxes_on_items(frame_rgb, results)
            
            curr_time = time.perf_counter()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            temp = get_cpu_temperature()
            cv2.putText(frame_rgb, f"FPS: {fps:.1f} | {temp} | ZOOM: {ZOOM_FACTOR}x", (30, 50), 
                       FONT, 1.2, RGB_GREEN, THICKNESS_BOX)
            
            cv2.imshow(window_name, frame_rgb)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db: 
                    d = his_db[TARGET_HN]
                    d['hn'] = TARGET_HN
                    ai.load_patient(d)

            elapsed = time.perf_counter() - start_loop
            if elapsed < FRAME_TIME: 
                time.sleep(FRAME_TIME - elapsed)

    except KeyboardInterrupt: 
        print("\n⏹️ Stopping...")
    finally: 
        cam.stop()
        ai.stop()
        cv2.destroyAllWindows()
        print("👋 Bye Bye!")

if __name__ == "__main__":
    main()