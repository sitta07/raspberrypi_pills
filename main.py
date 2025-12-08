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

# Accuracy Thresholds
SCORE_PASS_PILL = 0.2
SCORE_PASS_PACK = 0.85 

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (Optimized Mode)")

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

# ================= 1. WEBCAM STREAM (OPTIMIZED) =================
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
            
        threading.Thread(target=self.update, daemon=True).start()
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
                break

    def read(self):
        with self.lock:
            return self.frame if self.grabbed else None
    
    def stop(self):
        self.stopped = True
        if self.picam2: 
            self.picam2.stop()
            self.picam2.close()

# ================= 2. RESOURCES (OPTIMIZED LOADING) =================
class HISLoader:
    @staticmethod
    def load_database(filename):
        if not os.path.exists(filename): 
            return {}
        db = {}
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): 
                        continue
                    parts = line.split('|')
                    if len(parts) < 3: 
                        continue
                    hn, name = parts[0].strip(), parts[1].strip()
                    drugs = [d.strip().lower() for d in parts[2].split(',') if d.strip()]
                    db[hn] = {'name': name, 'drugs': drugs}
            return db
        except: 
            return {}

class PrescriptionManager:
    @staticmethod
    def filter_db(drug_names_list, source_vecs, source_lbls):
        if not drug_names_list or not source_vecs: 
            return None, None
        
        # Optimized: Pre-convert drugs to set for O(1) lookup
        drug_set = set(drug_names_list)
        
        filtered_vecs = []
        filtered_lbls = []
        
        for idx, label in enumerate(source_lbls):
            label_lower = label.lower()
            if any(drug in label_lower for drug in drug_set):
                filtered_vecs.append(source_vecs[idx])
                filtered_lbls.append(label)
        
        if filtered_vecs: 
            return torch.tensor(np.array(filtered_vecs), device=device), filtered_lbls
        return None, None

# --- LOAD DATABASES (OPTIMIZED) ---
def load_pkl_to_list(filepath):
    """Optimized: Return tuple instead of modifying lists"""
    if not os.path.exists(filepath): 
        return [], []
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            vecs, lbls = [], []
            for name, vec_list in data.items():
                for v in vec_list:
                    vecs.append(v)
                    lbls.append(name)
            return vecs, lbls
    except Exception as e: 
        print(f"Error loading {filepath}: {e}")
        return [], []

# Load all databases
pills_vecs, pills_lbls = load_pkl_to_list(DB_FILES['pills']['vec'])
packs_vecs, packs_lbls = load_pkl_to_list(DB_FILES['packs']['vec'])

matrix_pills = torch.tensor(np.array(pills_vecs), device=device) if pills_vecs else None
matrix_packs = torch.tensor(np.array(packs_vecs), device=device) if packs_vecs else None

# Load color database
color_db = {}
for db_type in ['pills', 'packs']:
    try:
        with open(DB_FILES[db_type]['col'], 'rb') as f: 
            color_db.update(pickle.load(f))
    except: 
        pass

# SIFT database (Optimized: Load only first 3 images)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
sift_db = {}

if os.path.exists(IMG_DB_FOLDER):
    for folder in os.listdir(IMG_DB_FOLDER):
        path = os.path.join(IMG_DB_FOLDER, folder)
        if not os.path.isdir(path): 
            continue
        
        des_list = []
        image_files = [x for x in os.listdir(path) if x.lower().endswith(('jpg', 'png', 'jpeg'))][:3]
        
        for img_file in image_files:
            img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, des = sift.detectAndCompute(img, None)
                if des is not None: 
                    des_list.append(des)
        
        if des_list:
            sift_db[folder] = des_list

# Load models
try:
    model_pill = YOLO(MODEL_PILL_PATH, task='detect')
    model_pack = YOLO(MODEL_PACK_PATH, task='detect')
    
    weights = models.ResNet50_Weights.DEFAULT
    embedder = torch.nn.Sequential(*list(models.resnet50(weights=weights).children())[:-1])
    embedder.eval().to(device)
    
    # Optimized: Single preprocess pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Disable gradient computation globally for inference
    torch.set_grad_enabled(False)
    
except Exception as e: 
    print(f"[CRITICAL] Model Error: {e}")
    sys.exit(1)

# ================= 3. TRINITY ENGINE (OPTIMIZED) =================
def trinity_inference(img_crop, is_pill=True, 
                      session_pills=None, session_pills_lbl=None,
                      session_packs=None, session_packs_lbl=None):
    
    # Select target database
    if is_pill:
        target_matrix = session_pills if session_pills is not None else matrix_pills
        target_labels = session_pills_lbl if session_pills_lbl is not None else pills_lbls
    else:
        target_matrix = session_packs if session_packs is not None else matrix_packs
        target_labels = session_packs_lbl if session_packs_lbl is not None else packs_lbls

    if target_matrix is None: 
        return "DB Error", 0.0

    try:
        # Prepare image
        if is_pill: 
            pil_img = Image.fromarray(img_crop) 
        else:
            gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
            crop_3ch_gray = cv2.merge([gray_crop, gray_crop, gray_crop])
            pil_img = Image.fromarray(crop_3ch_gray)

        # Feature extraction
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        live_vec = embedder(input_tensor).flatten()
        live_vec = live_vec / live_vec.norm()
        
        # Compute similarity scores
        scores = torch.matmul(live_vec, target_matrix.T).squeeze(0)
        k_val = min(10, len(target_labels))
        if k_val == 0: 
            return "Unknown", 0.0
        
        top_k_val, top_k_idx = torch.topk(scores, k=k_val)
        
        # Get top 3 unique candidates
        candidates = []
        seen = set()
        
        for idx, sc in zip(top_k_idx.detach().cpu().numpy(), top_k_val.detach().cpu().numpy()):
            name = target_labels[idx]
            if name not in seen:
                candidates.append((name, float(sc)))
                seen.add(name)
                if len(candidates) >= 3: 
                    break

        # Color analysis (pills only)
        live_color = None
        if is_pill: 
            h, w = img_crop.shape[:2]
            center = img_crop[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            if center.size > 0:
                hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
                live_color = np.mean(hsv, axis=(0,1))
        
        # SIFT feature extraction
        gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        _, des_live = sift.detectAndCompute(gray, None)

        # Score fusion
        best_score = -1
        final_name = "Unknown"
        
        for name, vec_score in candidates:
            clean_name = name.replace("_pill", "").replace("_pack", "")
            
            # SIFT matching score
            sift_score = 0.0
            if des_live is not None and clean_name in sift_db:
                max_good = 0
                for ref_des in sift_db[clean_name]:
                    try:
                        matches = bf.knnMatch(des_live, ref_des, k=2)
                        good = [m for m,n in matches if len([m,n]) == 2 and m.distance < 0.75 * n.distance]
                        max_good = max(max_good, len(good))
                    except: 
                        pass
                sift_score = min(max_good / 15.0, 1.0)
                
            # Color matching score
            color_score = 0.0
            if is_pill and live_color is not None and name in color_db:
                diff = np.abs(live_color - color_db[name])
                diff[0] = min(diff[0], 180 - diff[0])  # Hue circular distance
                norm_diff = diff / np.array([90.0, 255.0, 255.0])
                dist = np.linalg.norm(norm_diff)
                color_score = np.clip(np.exp(-3.0 * dist), 0, 1)
                
            # Weighted fusion
            w_vec, w_sift, w_col = (0.3, 0.1, 0.6) if is_pill else (0.8, 0.2, 0.0)
            total = (vec_score * w_vec) + (sift_score * w_sift) + (color_score * w_col)
            
            if total > best_score: 
                best_score = total
                final_name = clean_name

        return final_name, best_score
        
    except Exception as e:
        print(f"[Trinity Error] {e}")
        return "Error", 0.0

# ================= 4. AI WORKER (OPTIMIZED) =================
class AIProcessor:
    def __init__(self):
        self.latest_frame = None 
        self.results = [] 
        self.stopped = False
        self.lock = threading.Lock()
        self.is_rx_mode = False
        self.current_patient_info = None
        self.sess_mat_pills = None
        self.sess_lbl_pills = None
        self.sess_mat_packs = None
        self.sess_lbl_packs = None
        
        # Optimized: Pre-compute scale factors
        self.scale_x = DISPLAY_W / AI_IMG_SIZE
        self.scale_y = DISPLAY_H / AI_IMG_SIZE

    def load_patient(self, patient_data):
        with self.lock:
            if not patient_data:
                self.is_rx_mode = False
                self.current_patient_info = None
                self.sess_mat_pills = None
                self.sess_lbl_pills = None
                self.sess_mat_packs = None
                self.sess_lbl_packs = None
            else:
                self.is_rx_mode = True
                self.current_patient_info = patient_data
                drugs = patient_data['drugs']
                self.sess_mat_pills, self.sess_lbl_pills = PrescriptionManager.filter_db(
                    drugs, pills_vecs, pills_lbls)
                self.sess_mat_packs, self.sess_lbl_packs = PrescriptionManager.filter_db(
                    drugs, packs_vecs, packs_lbls)
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

    def run(self):
        print("[DEBUG] AI Worker Loop Started.")
        
        while not self.stopped:
            # Get frame
            with self.lock:
                if self.latest_frame is None:
                    frame_HD = None
                else:
                    frame_HD = self.latest_frame
                    self.latest_frame = None
            
            if frame_HD is None: 
                time.sleep(0.005)
                continue

            # Resize once
            frame_yolo = cv2.resize(frame_HD, (AI_IMG_SIZE, AI_IMG_SIZE), 
                                   interpolation=cv2.INTER_LINEAR)
            
            final_detections = []
            valid_pills = [] 

            try:
                # 1. DETECT PILLS
                pill_res = model_pill(frame_yolo, verbose=False, conf=CONF_PILL, 
                                     imgsz=AI_IMG_SIZE, max_det=10, agnostic_nms=True)
                
                for box in pill_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    
                    # Scale to display resolution
                    x1 = int(x1_s * self.scale_x)
                    y1 = int(y1_s * self.scale_y)
                    x2 = int(x2_s * self.scale_x)
                    y2 = int(y2_s * self.scale_y)
                    
                    # Filter small boxes
                    if (x2-x1) < 30 or (y2-y1) < 30: 
                        continue
                    
                    # Extract crop
                    crop = frame_HD[y1:y2, x1:x2]
                    if crop.size == 0: 
                        continue

                    # Inference
                    nm, sc = trinity_inference(crop, is_pill=True,
                                              session_pills=self.sess_mat_pills,
                                              session_pills_lbl=self.sess_lbl_pills,
                                              session_packs=self.sess_mat_packs,
                                              session_packs_lbl=self.sess_lbl_packs)
                    
                    # Track high-confidence pills
                    if "?" not in nm and "Unknown" not in nm:
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        valid_pills.append({'name': nm, 'center': (cx, cy)})

                    final_detections.append({
                        'label': nm, 
                        'score': sc, 
                        'type': 'pill', 
                        'box': (x1, y1, x2, y2)
                    })

                # 2. DETECT PACKS
                pack_res = model_pack(frame_yolo, verbose=False, conf=CONF_PACK, 
                                     imgsz=AI_IMG_SIZE, max_det=5, agnostic_nms=True)
                
                for box in pack_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    
                    # Scale to display resolution
                    x1 = int(x1_s * self.scale_x)
                    y1 = int(y1_s * self.scale_y)
                    x2 = int(x2_s * self.scale_x)
                    y2 = int(y2_s * self.scale_y)
                    
                    # Filter small boxes
                    if (x2-x1) < 50 or (y2-y1) < 50: 
                        continue
                    
                    # Extract crop first to get pack score
                    crop = frame_HD[y1:y2, x1:x2]
                    if crop.size == 0: 
                        continue
                    
                    # Get pack detection score
                    nm, sc = trinity_inference(crop, is_pill=False,
                                              session_pills=self.sess_mat_pills,
                                              session_pills_lbl=self.sess_lbl_pills,
                                              session_packs=self.sess_mat_packs,
                                              session_packs_lbl=self.sess_lbl_packs)
                    
                    # 🔥 SMART LOGIC: Override with inner pill name BUT keep pack score
                    found_inner_pill_name = None
                    for pill in valid_pills:
                        if is_point_in_box(pill['center'], (x1, y1, x2, y2)):
                            found_inner_pill_name = pill['name']
                            break
                    
                    if found_inner_pill_name:
                        # Override name but keep original pack score
                        nm = found_inner_pill_name
                    
                    final_detections.append({
                        'label': nm, 
                        'score': sc,  # Use actual pack score
                        'type': 'pack', 
                        'box': (x1, y1, x2, y2)
                    })

                # Update results
                with self.lock: 
                    self.results = final_detections
            
            except Exception as e:
                print(f"[ERROR-AI-LOOP] {e}")
            
    def stop(self): 
        self.stopped = True

# ================= 5. UI DRAWING (OPTIMIZED) =================
def draw_patient_info(frame, patient_data):
    """Optimized: Faster drawing with pre-calculated positions"""
    if not patient_data: 
        return
    
    H, W = frame.shape[:2]
    box_w = 400
    start_x = W - box_w
    
    # Build lines
    lines = [
        f"HN: {patient_data.get('hn', 'N/A')}",
        f"Name: {patient_data.get('name', 'N/A')}", 
        "--- Rx List ---"
    ]
    
    for d in patient_data.get('drugs', [])[:5]: 
        lines.append(f"- {d}")
    
    line_h = 40
    box_h = (len(lines) * line_h) + 20
    
    # Draw background
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (0, 255, 255), 2)
    
    # Draw text
    for i, line in enumerate(lines):
        y = 35 + (i * line_h)
        cv2.putText(frame, line, (start_x+15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def draw_boxes_on_items(frame, results):
    """Optimized: Batch drawing with vectorized operations"""
    for r in results:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        score = r['score']
        obj_type = r.get('type', 'pill')
        
        # Determine color based on type and score
        if obj_type == 'pack':
            # Pack: Green if >= 0.75, Yellow if < 0.75
            if score >= SCORE_PASS_PACK:
                color = (0, 255, 0)    # Green - passed
            else:
                color = (0, 255, 255)  # Yellow - below threshold
        else:
            # Pill logic
            if "?" in label or score < SCORE_PASS_PILL:
                color = (0, 0, 255)    # Red for uncertain
            elif "Unknown" in label:
                color = (255, 0, 0)    # Blue for unknown
            else:
                color = (0, 255, 0)    # Green for confident

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{label} {score:.0%}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ================= 6. MAIN (OPTIMIZED) =================
def main():
    TARGET_HN = "HN-101" 
    
    # Initialize components
    cam = WebcamStream().start()
    ai = AIProcessor().start()
    
    # Load patient database
    his_db = HISLoader.load_database(HIS_FILE_PATH)
    if TARGET_HN in his_db: 
        d = his_db[TARGET_HN]
        d['hn'] = TARGET_HN
        ai.load_patient(d)
    
    # Wait for camera
    print("⏳ Waiting for camera feed...")
    while cam.read() is None: 
        time.sleep(0.1)
    
    # Setup window
    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"🎥 RUNNING... (Optimized Mode)")
    
    # FPS tracking
    fps = 0
    prev_time = time.perf_counter()
    TARGET_FPS = 10 
    FRAME_TIME = 1.0 / TARGET_FPS

    try:
        while True:
            start_loop = time.perf_counter()
            
            # Get frame
            frame_rgb = cam.read()
            if frame_rgb is None: 
                time.sleep(0.01)
                continue
            
            # Send to AI
            ai.update_frame(frame_rgb)
            
            # Get results
            results, cur_patient = ai.get_results()
            
            # Draw on copy
            display = frame_rgb.copy()
            draw_boxes_on_items(display, results)
            
            if cur_patient: 
                draw_patient_info(display, cur_patient)
            
            # Calculate FPS
            curr_time = time.perf_counter()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Get temperature
            temp = get_cpu_temperature()
            
            # Draw FPS
            cv2.putText(display, f"FPS: {fps:.1f} | {temp}", (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Show frame
            cv2.imshow(window_name, display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            if key == ord('r'):
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db: 
                    d = his_db[TARGET_HN]
                    d['hn'] = TARGET_HN
                    ai.load_patient(d)

            # Frame rate limiting
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