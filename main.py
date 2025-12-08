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
# 1. Fix Display (X11 backend)
os.environ["QT_QPA_PLATFORM"] = "xcb"
# 2. Fix GPU Device Discovery Failed (Force Software Rendering)
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

# Import Picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ Warning: Picamera2 not found. Is this a Raspberry Pi?")
    sys.exit(1)

# ================= CONFIGURATION =================
MODEL_PILL_PATH = 'models/pills.onnx'          
MODEL_PACK_PATH = 'models/best_process_2.onnx'
DB_FILES = {
    'pills': {'vec': 'database/db_pills.pkl', 'col': 'database/colors_pills.pkl'},
    'packs': {'vec': 'database/db_packs.pkl', 'col': 'database/colors_packs.pkl'}
}
IMG_DB_FOLDER = 'database_images'
HIS_FILE_PATH = 'prescription.txt' 

CONF_PILL = 0.50    
CONF_PACK = 0.70    
SCORE_PASS_PILL = 0.75  
SCORE_PASS_PACK = 0.60  

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device}")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000.0
        return f"{temp:.1f}C"
    except:
        return "N/A"

# ================= 1. WEBCAM STREAM (RGB888) =================
class WebcamStream:
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.picam2 = None
        self.lock = threading.Lock()

    def start(self):
        print(" Initializing Picamera2 (HD Mode)...")
        try:
            self.picam2 = Picamera2()
            # 🔥 Config RGB888 DIRECTLY
            config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameDurationLimits": (16666, 16666)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
            print(" Camera Ready (1280x720 RGB888)!")
        except Exception as e:
            print(f" Camera Init Failed: {e}")
            self.stopped = True
            
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.grabbed = True
                else:
                    self.stopped = True
            except:
                self.stopped = True

    def read(self):
        with self.lock:
            if self.grabbed:
                return self.frame.copy()
            else:
                return None

    def stop(self):
        self.stopped = True
        if self.picam2:
            try: self.picam2.stop(); self.picam2.close()
            except: pass

# ================= 2. RESOURCES & LOGIC =================

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
                    drugs = [d.strip().lower() for d in parts[2].split(',') if d.strip()]
                    db[hn] = {'name': name, 'drugs': drugs}
            return db
        except: return {}

class PrescriptionManager:
    def __init__(self, global_vecs, global_lbls):
        self.g_vec = global_vecs; self.g_lbl = global_lbls
    def create_session_db(self, drug_names_list):
        if not drug_names_list: return None, None 
        s_vec, s_lbl = [], []
        for idx, label in enumerate(self.g_lbl):
            clean = label.replace("_pill", "").replace("_pack", "").lower()
            for target in drug_names_list:
                if target in clean:
                    s_vec.append(self.g_vec[idx]); s_lbl.append(label)
        if not s_vec: return None, None
        return torch.tensor(np.array(s_vec)).to(device), s_lbl

# --- Global Load ---
print("⏳ Loading Resources...")
vec_db, color_db = {}, {}
try:
    with open(DB_FILES['pills']['vec'], 'rb') as f: vec_db.update(pickle.load(f))
    with open(DB_FILES['pills']['col'], 'rb') as f: color_db.update(pickle.load(f))
    with open(DB_FILES['packs']['vec'], 'rb') as f: vec_db.update(pickle.load(f))
    with open(DB_FILES['packs']['col'], 'rb') as f: color_db.update(pickle.load(f))
except: pass

global_vectors, global_labels = [], []
for name, vec_list in vec_db.items():
    for vec in vec_list:
        global_vectors.append(vec); global_labels.append(name)
global_matrix = torch.tensor(np.array(global_vectors)).to(device) if global_vectors else None
rx_manager = PrescriptionManager(global_vectors, global_labels)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
sift_db = {}
if os.path.exists(IMG_DB_FOLDER):
    for folder in os.listdir(IMG_DB_FOLDER):
        path = os.path.join(IMG_DB_FOLDER, folder)
        if not os.path.isdir(path): continue
        des_list = []
        for f in [x for x in os.listdir(path) if x.endswith(('jpg','png'))][:3]:
            img = cv2.imread(os.path.join(path, f), 0)
            if img is not None:
                _, des = sift.detectAndCompute(img, None)
                if des is not None: des_list.append(des)
        sift_db[folder] = des_list

try:
    model_pill = YOLO(MODEL_PILL_PATH)
    model_pack = YOLO(MODEL_PACK_PATH)
    weights = models.ResNet50_Weights.DEFAULT
    embedder = torch.nn.Sequential(*list(models.resnet50(weights=weights).children())[:-1])
    embedder.eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except Exception as e: print(f"❌ Model Error: {e}"); sys.exit(1)

print("✅ AI Logic Ready!")

# --- Trinity Engine (RGB Version) ---
def trinity_inference(img_crop, is_pill=True, custom_matrix=None, custom_labels=None):
    target_matrix = custom_matrix if custom_matrix is not None else global_matrix
    target_labels = custom_labels if custom_labels is not None else global_labels
    if target_matrix is None: return "DB Error", 0.0

    # Assume img_crop is RGB
    if is_pill:
        pil_img = Image.fromarray(img_crop) 
    else:
        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        crop_3ch_gray = cv2.merge([gray_crop, gray_crop, gray_crop])
        pil_img = Image.fromarray(crop_3ch_gray)

    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        live_vec = embedder(input_tensor).flatten()
        live_vec = live_vec / live_vec.norm()
        
    scores = torch.matmul(live_vec, target_matrix.T).squeeze(0)
    k_val = min(10, len(target_labels))
    if k_val == 0: return "Unknown", 0.0
    
    top_k_val, top_k_idx = torch.topk(scores, k=k_val)
    candidates, seen = [], set()
    target_suffix = "_pill" if is_pill else "_pack"
    
    for idx, sc in zip(top_k_idx.cpu().numpy(), top_k_val.cpu().numpy()):
        name = target_labels[idx]
        if name.endswith(target_suffix) and name not in seen:
            candidates.append((name, float(sc))); seen.add(name)
            if len(candidates) >= 3: break

    live_color = None; des_live = None
    if is_pill: 
        h, w = img_crop.shape[:2]
        center = img_crop[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV) # RGB -> HSV
        live_color = np.mean(hsv, axis=(0,1))
    
    gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
    _, des_live = sift.detectAndCompute(gray, None)

    best_score = -1; final_name = "Unknown"
    for name, vec_score in candidates:
        clean_name = name.replace("_pill", "").replace("_pack", "")
        sift_score = 0.0
        if des_live is not None and clean_name in sift_db:
            max_good = 0
            for ref_des in sift_db[clean_name]:
                try:
                    matches = bf.knnMatch(des_live, ref_des, k=2)
                    good = [m for m,n in matches if m.distance < 0.75 * n.distance]
                    if len(good) > max_good: max_good = len(good)
                except: pass
            sift_score = min(max_good / 15.0, 1.0)
            
        color_score = 0.0
        if is_pill and name in color_db:
            diff = np.abs(live_color - color_db[name])
            diff[0] = min(diff[0], 180 - diff[0]) 
            norm_diff = diff / np.array([90.0, 255.0, 255.0])
            dist = np.linalg.norm(norm_diff)
            color_score = np.clip(np.exp(-3.0 * dist), 0, 1)
            
        w_vec, w_sift, w_col = (0.3, 0.1, 0.6) if is_pill else (0.3, 0.7, 0.0)
        total = (vec_score * w_vec) + (sift_score * w_sift) + (color_score * w_col)
        if total > best_score: best_score = total; final_name = clean_name

    return final_name, best_score

# ================= 3. AI WORKER (ASYNC) =================
class AIProcessor:
    def __init__(self):
        self.latest_frame = None 
        self.results = [] 
        self.stopped = False
        self.lock = threading.Lock()
        
        # State
        self.current_patient_info = None 
        self.session_matrix = None
        self.session_labels = None
        self.is_rx_mode = False

    def load_patient(self, patient_data):
        with self.lock:
            if not patient_data:
                self.is_rx_mode = False; self.current_patient_info = None
                self.session_matrix = None; self.session_labels = None
            else:
                self.is_rx_mode = True; self.current_patient_info = patient_data
                drugs = patient_data['drugs']
                self.session_matrix, self.session_labels = rx_manager.create_session_db(drugs)
                print(f"🏥 Loaded: {patient_data['name']}")

    def start(self): 
        threading.Thread(target=self.run, daemon=True).start()
        return self
    
    def update_frame(self, frame): 
        with self.lock: 
            self.latest_frame = frame.copy() 
        
    def get_results(self): 
        with self.lock: 
            return self.results, self.current_patient_info

    def run(self):
        print(" AI Worker Started...")
        while not self.stopped:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None 
            
            if frame_to_process is None: 
                time.sleep(0.01); continue

            # === PROCESSING RGB FRAME DIRECTLY ===
            detections = []
            pill_names_batch = [] 
            pill_coords = []      

            def process_crop(crop, is_pill_mode):
                name, score = trinity_inference(crop, is_pill=is_pill_mode,
                                                custom_matrix=self.session_matrix,
                                                custom_labels=self.session_labels)
                threshold = SCORE_PASS_PILL if is_pill_mode else SCORE_PASS_PACK
                label_prefix, color = "", (0,0,255)
                if score > threshold:
                    color = (0,255,0) if is_pill_mode else (255,0,255)
                    if self.is_rx_mode: label_prefix = "OK "; color = (0,255,0)
                else:
                    if self.is_rx_mode: name = "WRONG"; label_prefix = "!!! "
                return name, score, label_prefix, color

            # 1. Pills (YOLO takes RGB fine)
            pill_res = model_pill(frame_to_process, verbose=False, conf=CONF_PILL)
            if len(pill_res[0].boxes) > 0:
                for box in pill_res[0].boxes.xyxy.cpu().numpy().astype(int):
                    x1,y1,x2,y2 = box
                    crop = frame_to_process[y1:y2, x1:x2]
                    if crop.size > 0:
                        nm, sc, pf, clr = process_crop(crop, True)
                        if "WRONG" not in nm and "Unknown" not in nm:
                            pill_names_batch.append(nm); pill_coords.append((x1,y1,x2,y2))
                        detections.append({'box':box, 'label':nm, 'full':f"{pf}{nm} {sc:.0%}", 'color':clr, 'type':'pill'})

            # 2. Packs
            pack_res = model_pack(frame_to_process, verbose=False, conf=CONF_PACK, retina_masks=True)
            if pack_res[0].masks is not None:
                masks = pack_res[0].masks.data.cpu().numpy()
                boxes = pack_res[0].boxes.xyxy.cpu().numpy().astype(int)
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = box
                    raw_mask = masks[i]
                    mask_resized = cv2.resize(raw_mask, (frame_to_process.shape[1], frame_to_process.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    masked = frame_to_process.copy(); masked[mask_binary == 0] = [128,128,128]
                    crop = masked[y1:y2, x1:x2]
                    if crop.size > 0:
                        nm, sc, pf, clr = process_crop(crop, False)
                        detections.append({'box':box, 'label':nm, 'full':f"{pf}{nm} {sc:.0%}", 'color':clr, 'type':'pack'})

            # 3. Group Box
            if pill_coords:
                counts = Counter(pill_names_batch)
                if counts:
                    majority_name, count = counts.most_common(1)[0]
                    all_x1 = [c[0] for c in pill_coords]; all_y1 = [c[1] for c in pill_coords]
                    all_x2 = [c[2] for c in pill_coords]; all_y2 = [c[3] for c in pill_coords]
                    gx1, gy1 = max(0, min(all_x1) - 20), max(0, min(all_y1) - 40)
                    gx2, gy2 = max(all_x2) + 20, max(all_y2) + 20
                    detections.append({
                        'box': (gx1, gy1, gx2, gy2),
                        'label': majority_name,
                        'full': f"BATCH: {majority_name} (Total: {len(pill_coords)})",
                        'color': (0, 255, 255), 'type': 'group_box'
                    })

            with self.lock: 
                self.results = detections
            
    def stop(self): self.stopped = True

# ================= 4. UI DRAWING (USER LOGIC) =================
def draw_patient_info(frame, patient_data):
    if not patient_data: return
    H, W = frame.shape[:2]
    box_w = 300; start_x = W - box_w
    lines = [f"HN: {patient_data.get('hn', 'N/A')}",
             f"Name: {patient_data.get('name', 'N/A')}", "--- Rx List ---"]
    for d in patient_data.get('drugs', [])[:5]: lines.append(f"- {d}")
    box_h = (len(lines) * 25) + 15
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (50,50,50), -1)
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (0,255,255), 2)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (start_x+10, 25+(i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

def draw_summary_box(frame, results):
    H, W = frame.shape[:2]
    items = [r['label'] for r in results 
             if r['type'] in ['pill', 'pack'] 
             and "Unknown" not in r['label'] and "WRONG" not in r['label']]
    if not items: return
    counts = Counter(items)
    box_w = 300; line_h = 30; padding = 10
    total_lines = len(counts) + 1
    total_h = (total_lines * line_h) + (padding * 2)
    start_x = W - box_w - 20; start_y = H - total_h - 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (W - 20, H - 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (start_x, start_y), (W - 20, H - 20), (255, 255, 255), 2)
    cv2.putText(frame, "--- DETECTED ---", (start_x + 10, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for i, (name, count) in enumerate(counts.items()):
        y_pos = start_y + 25 + ((i + 1) * line_h)
        cv2.putText(frame, f"{name}: {count}", (start_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# ================= 5. MAIN (ASYNC PATTERN) =================
def main():
    TARGET_HN = "HN-101" 
    
    # 1. Start Camera
    cam = WebcamStream().start()
    
    # 2. Start AI Worker
    ai = AIProcessor().start()
    
    # 3. Load Patient Data
    his_db = HISLoader.load_database(HIS_FILE_PATH)
    if TARGET_HN in his_db: 
        d = his_db[TARGET_HN]; d['hn'] = TARGET_HN; ai.load_patient(d)
    
    print(" Waiting for camera feed...")
    while cam.read() is None: time.sleep(0.1)
    
    window_name = "PillTrack - Raspberry Pi"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("🎥 RUNNING... Press 'R' to Reload")
    prev_time = 0
    fps = 0

    try:
        while True:
            # Get latest frame (RGB888)
            frame = cam.read()
            if frame is None: 
                time.sleep(0.01); continue
            
            # Send to AI
            ai.update_frame(frame)
            
            # Use Frame for Display (Still RGB - Colors might be swapped on screen, as requested)
            display = frame.copy()
            
            # Get AI Results
            results, cur_patient = ai.get_results()
            
            # Draw UI
            for det in results:
                x1,y1,x2,y2 = det['box']
                color = det['color']
                if det.get('type') == 'group_box':
                    cv2.rectangle(display, (x1,y1), (x2,y2), color, 4) 
                    cv2.putText(display, det['full'], (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                else:
                    cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(display, det['label'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if cur_patient: draw_patient_info(display, cur_patient)
            draw_summary_box(display, results)
            
            # FPS & Temp
            curr_time = time.time()
            if (curr_time - prev_time) > 0: fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            temp = get_cpu_temperature()
            
            # 🔥 DISPLAY DIRECTLY (NO RGB->BGR CONVERSION)
            cv2.putText(display, f"FPS: {fps:.1f} | Temp: {temp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db: d = his_db[TARGET_HN]; d['hn'] = TARGET_HN; ai.load_patient(d)
                
    except KeyboardInterrupt:
        print("\n Stopping...")
    finally:
        cam.stop(); ai.stop(); cv2.destroyAllWindows()
        print(" Bye Bye!")

if __name__ == "__main__":
    main()