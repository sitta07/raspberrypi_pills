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
    sys.exit(1)

# ================= CONFIGURATION =================
MODEL_PILL_PATH = 'models/pills.pt'          
MODEL_PACK_PATH = 'models/best_process_2.pt'

DB_FILES = {
    'pills': {'vec': 'database/db_pills.pkl', 'col': 'database/colors_pills.pkl'},
    'packs': {'vec': 'database/db_packs.pkl', 'col': 'database/colors_packs.pkl'}
}
IMG_DB_FOLDER = 'database_images'
HIS_FILE_PATH = 'prescription.txt' 

# 📺 Display Resolution
DISPLAY_W, DISPLAY_H = 1280, 720

# 🚀 AI Resolution
AI_IMG_SIZE = 416 

# Thresholds
CONF_PILL = 0.5    
CONF_PACK = 0.75    

# Accuracy Thresholds (แยกกัน)
SCORE_PASS_PILL = 0.2  # ยาเม็ด: 10% เอาหมด
SCORE_PASS_PACK = 0.85  # กล่อง: ต้อง 85% (แต่ถ้ามีเม็ดอยู่ข้างใน จะถูก override)

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (10 FPS + Smart Box Logic)")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}C"
    except: return "N/A"

def is_point_in_box(point, box):
    """เช็คว่าจุดกึ่งกลาง (px, py) อยู่ในกล่อง (x1, y1, x2, y2) หรือไม่"""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 < px < x2 and y1 < py < y2

# ================= 1. WEBCAM STREAM (LOCKED @ 10 FPS) =================
class WebcamStream:
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.picam2 = None
        self.lock = threading.Lock()

    def start(self):
        print("[DEBUG] Initializing Picamera2 (10 FPS Mode)...")
        try:
            self.picam2 = Picamera2()
            # 🔥 Config 10 FPS (1,000,000 / 10 = 100,000 us)
            config = self.picam2.create_preview_configuration(
                main={"size": (DISPLAY_W, DISPLAY_H), "format": "RGB888"},
                controls={"FrameDurationLimits": (100000, 100000)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
            print("[DEBUG] Camera Started")
        except Exception as e:
            print(f"[ERROR] Camera Init Failed: {e}")
            self.stopped = True
            
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    with self.lock:
                        self.frame = frame.copy()
                        self.grabbed = True
                else: self.stopped = True
            except: self.stopped = True

    def read(self):
        with self.lock:
            if self.grabbed: return self.frame.copy()
            return None
    
    def stop(self):
        self.stopped = True
        if self.picam2: self.picam2.stop(); self.picam2.close()

# ================= 2. RESOURCES =================
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
            for target in drug_names_list:
                if target in label.lower():
                    s_vec.append(self.g_vec[idx]); s_lbl.append(label)
        if s_vec: return torch.tensor(np.array(s_vec)).to(device), s_lbl
        return None, None

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
    model_pill = YOLO(MODEL_PILL_PATH, task='detect')
    model_pack = YOLO(MODEL_PACK_PATH, task='detect')
    weights = models.ResNet50_Weights.DEFAULT
    embedder = torch.nn.Sequential(*list(models.resnet50(weights=weights).children())[:-1])
    embedder.eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except Exception as e: print(f"[CRITICAL] Model Error: {e}"); sys.exit(1)

# ================= 3. TRINITY ENGINE =================
def trinity_inference(img_crop, is_pill=True, custom_matrix=None, custom_labels=None):
    target_matrix = custom_matrix if custom_matrix is not None else global_matrix
    target_labels = custom_labels if custom_labels is not None else global_labels
    if target_matrix is None: return "DB Error", 0.0

    try:
        if is_pill: pil_img = Image.fromarray(img_crop) 
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
            hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
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
                
            w_vec, w_sift, w_col = (0.3, 0.1, 0.6) if is_pill else (0.8, 0.2, 0.0)
            total = (vec_score * w_vec) + (sift_score * w_sift) + (color_score * w_col)
            if total > best_score: best_score = total; final_name = clean_name

        return final_name, best_score
    except: return "Error", 0.0

# ================= 4. AI WORKER (SMART BOX LOGIC) =================
class AIProcessor:
    def __init__(self):
        self.latest_frame = None 
        self.results = [] 
        self.stopped = False
        self.lock = threading.Lock()
        self.is_rx_mode = False
        self.current_patient_info = None
        self.session_matrix = None; self.session_labels = None

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
        with self.lock: self.latest_frame = frame.copy() 
        
    def get_results(self): 
        with self.lock: return self.results, self.current_patient_info

    def run(self):
        print("[DEBUG] AI Worker Loop Started.")
        while not self.stopped:
            frame_HD = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_HD = self.latest_frame
                    self.latest_frame = None 
            
            if frame_HD is None: 
                time.sleep(0.005); continue

            # Resize for YOLO
            frame_yolo = cv2.resize(frame_HD, (AI_IMG_SIZE, AI_IMG_SIZE))
            frame_yolo_clean = np.ascontiguousarray(frame_yolo)
            
            # Scale Factors
            scale_x = DISPLAY_W / AI_IMG_SIZE
            scale_y = DISPLAY_H / AI_IMG_SIZE

            final_detections = []
            
            # Temporary storage for smart logic
            valid_pills = [] 

            def process_crop(crop, is_pill_mode):
                name, score = trinity_inference(crop, is_pill=is_pill_mode,
                                                custom_matrix=self.session_matrix,
                                                custom_labels=self.session_labels)
                threshold = SCORE_PASS_PILL if is_pill_mode else SCORE_PASS_PACK
                # ถ้าเป็น Pills คะแนนต่ำก็ส่งชื่อไปเลย แต่ถ้า Pack เดี๋ยวไปเช็คข้างในอีกที
                if score <= threshold: name = f"{name}?"
                return name, score

            try:
                # 1. DETECT PILLS FIRST
                pill_res = model_pill(frame_yolo_clean, verbose=False, conf=CONF_PILL, imgsz=AI_IMG_SIZE, max_det=10, agnostic_nms=True)
                for box in pill_res[0].boxes.xyxy.cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    # Scale to HD
                    x1 = int(x1_s * scale_x); y1 = int(y1_s * scale_y)
                    x2 = int(x2_s * scale_x); y2 = int(y2_s * scale_y)
                    
                    if (x2-x1) < 30 or (y2-y1) < 30: continue 
                    crop = frame_HD[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    nm, sc = process_crop(crop, True)
                    
                    # Store Valid Pills for Box Checking
                    # ถ้าชื่อไม่ Unknown และไม่ ? ให้ถือว่าเป็น Verified Pill
                    if "?" not in nm and "Unknown" not in nm:
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        valid_pills.append({'name': nm, 'center': (cx, cy)})

                    final_detections.append({'label':nm, 'score':sc, 'type':'pill', 'box':(x1,y1,x2,y2)})

                # 2. DETECT PACKS & CHECK INSIDE
                pack_res = model_pack(frame_yolo_clean, verbose=False, conf=CONF_PACK, imgsz=AI_IMG_SIZE, max_det=5, agnostic_nms=True)
                for box in pack_res[0].boxes.xyxy.cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    x1 = int(x1_s * scale_x); y1 = int(y1_s * scale_y)
                    x2 = int(x2_s * scale_x); y2 = int(y2_s * scale_y)
                    
                    if (x2-x1) < 50 or (y2-y1) < 50: continue
                    
                    # 🔥 SMART LOGIC: Check if any valid pill is inside this pack
                    found_inner_pill_name = None
                    for pill in valid_pills:
                        if is_point_in_box(pill['center'], (x1, y1, x2, y2)):
                            found_inner_pill_name = pill['name']
                            break # Found one! assume the pack is this drug
                    
                    if found_inner_pill_name:
                        # ✅ Override Pack Name with Pill Name!
                        nm = f"Box: {found_inner_pill_name}"
                        sc = 1.0 # Force High Score
                        final_detections.append({'label':nm, 'score':sc, 'type':'pack', 'box':(x1,y1,x2,y2)})
                    else:
                        # ❌ No pill inside, use Trinity
                        crop = frame_HD[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        nm, sc = process_crop(crop, False)
                        final_detections.append({'label':nm, 'score':sc, 'type':'pack', 'box':(x1,y1,x2,y2)})

                with self.lock: self.results = final_detections
            
            except Exception as e:
                print(f"[ERROR-AI-LOOP] {e}")
            
    def stop(self): self.stopped = True

# ================= 5. UI DRAWING =================
def draw_patient_info(frame, patient_data):
    if not patient_data: return
    H, W = frame.shape[:2]
    box_w = 400; start_x = W - box_w
    lines = [f"HN: {patient_data.get('hn', 'N/A')}",
             f"Name: {patient_data.get('name', 'N/A')}", "--- Rx List ---"]
    for d in patient_data.get('drugs', [])[:5]: lines.append(f"- {d}")
    line_h = 40
    box_h = (len(lines) * line_h) + 20
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (50,50,50), -1)
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (0,255,255), 2)
    for i, line in enumerate(lines):
        y = 35 + (i * line_h)
        cv2.putText(frame, line, (start_x+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

def draw_boxes_on_items(frame, results):
    for r in results:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        score = r['score']
        
        # Color Logic
        color = (0, 255, 0) # Green
        if "?" in label or score < SCORE_PASS_PILL: color = (0, 0, 255) # Red
        if "Unknown" in label: color = (255, 0, 0) # Blue
        if "Box:" in label: color = (0, 255, 255) # Yellow for Verified Box

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{label} {score:.0%}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ================= 6. MAIN (10 FPS) =================
def main():
    TARGET_HN = "HN-101" 
    cam = WebcamStream().start()
    ai = AIProcessor().start()
    his_db = HISLoader.load_database(HIS_FILE_PATH)
    if TARGET_HN in his_db: d = his_db[TARGET_HN]; d['hn'] = TARGET_HN; ai.load_patient(d)
    
    print(" Waiting for camera feed...")
    while cam.read() is None: time.sleep(0.1)
    
    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"🎥 RUNNING... (10 FPS Optimized)")
    fps = 0; prev_time = 0
    TARGET_FPS = 10 
    FRAME_TIME = 1.0 / TARGET_FPS

    try:
        while True:
            start_loop = time.time()
            frame_rgb = cam.read()
            if frame_rgb is None: time.sleep(0.01); continue
            
            ai.update_frame(frame_rgb)
            display = frame_rgb.copy()
            results, cur_patient = ai.get_results()
            
            draw_boxes_on_items(display, results)
            if cur_patient: draw_patient_info(display, cur_patient)
            
            curr_time = time.time()
            if (curr_time - prev_time) > 0: fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            temp = get_cpu_temperature()
            
            cv2.putText(display, f"FPS: {fps:.1f} | {temp}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.imshow(window_name, display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if cv2.waitKey(1) & 0xFF == ord('r'):
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db: d = his_db[TARGET_HN]; d['hn'] = TARGET_HN; ai.load_patient(d)

            elapsed = time.time() - start_loop
            if elapsed < FRAME_TIME: time.sleep(FRAME_TIME - elapsed)

    except KeyboardInterrupt: print("\n Stopping...")
    finally: cam.stop(); ai.stop(); cv2.destroyAllWindows(); print(" Bye Bye!")

if __name__ == "__main__":
    main()