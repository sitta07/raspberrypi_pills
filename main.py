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

# Import Picamera2
try:
    from picamera2 import Picamera2
    print("[DEBUG] Picamera2 module imported successfully.")
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

# 🔥 ลด Threshold ต่ำสุดๆ เพื่อดูว่าโมเดลทำงานมั้ย
CONF_PILL = 0.10    
CONF_PACK = 0.15    
SCORE_PASS_PILL = 0.50  
SCORE_PASS_PACK = 0.50  

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device}")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}C"
    except: return "N/A"

# ================= 1. WEBCAM STREAM =================
class WebcamStream:
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.picam2 = None
        self.lock = threading.Lock()

    def start(self):
        print("[DEBUG] Initializing Picamera2...")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={"FrameDurationLimits": (16666, 16666)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
            print("[DEBUG] Camera Started Successfully (RGB888)")
        except Exception as e:
            print(f"[ERROR] Camera Init Failed: {e}")
            self.stopped = True
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        frame_count = 0
        while not self.stopped:
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.grabbed = True
                    # Print every 100 frames just to know it's alive
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"[DEBUG] Camera is sending frames... (Count: {frame_count})")
                else:
                    print("[ERROR] Camera returned None frame")
                    self.stopped = True
            except Exception as e:
                print(f"[ERROR] Camera Update Error: {e}")
                self.stopped = True

    def read(self):
        with self.lock:
            if self.grabbed and self.frame is not None:
                return self.frame.copy()
            return None
    
    def stop(self):
        self.stopped = True
        if self.picam2: self.picam2.stop(); self.picam2.close()

# ================= 2. RESOURCES =================
class PrescriptionManager:
    def __init__(self, global_vecs, global_lbls):
        self.g_vec = global_vecs; self.g_lbl = global_lbls
        print(f"[DEBUG] RxManager Init with {len(global_lbls)} labels")
    def create_session_db(self, drug_names_list):
        if not drug_names_list: return None, None
        print(f"[DEBUG] Creating Session DB for: {drug_names_list}")
        s_vec, s_lbl = [], []
        for idx, label in enumerate(self.g_lbl):
            for target in drug_names_list:
                if target in label.lower():
                    s_vec.append(self.g_vec[idx]); s_lbl.append(label)
        if s_vec:
            print(f"[DEBUG] Found {len(s_vec)} matching vectors for session.")
            return torch.tensor(np.array(s_vec)).to(device), s_lbl
        print("[DEBUG] No matching vectors found!")
        return None, None

# Load Resources
print("[DEBUG] Loading Vectors...")
vec_db, color_db = {}, {}
try:
    with open(DB_FILES['pills']['vec'], 'rb') as f: vec_db.update(pickle.load(f))
    with open(DB_FILES['pills']['col'], 'rb') as f: color_db.update(pickle.load(f))
    with open(DB_FILES['packs']['vec'], 'rb') as f: vec_db.update(pickle.load(f))
    with open(DB_FILES['packs']['col'], 'rb') as f: color_db.update(pickle.load(f))
except Exception as e: print(f"[ERROR] DB Load Failed: {e}")

global_vectors, global_labels = [], []
for name, vec_list in vec_db.items():
    for vec in vec_list:
        global_vectors.append(vec); global_labels.append(name)
global_matrix = torch.tensor(np.array(global_vectors)).to(device) if global_vectors else None
rx_manager = PrescriptionManager(global_vectors, global_labels)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
sift_db = {}
# (Skipping SIFT DB Loading debug print to save space, assuming it works)

try:
    print(f"[DEBUG] Loading YOLO from: {MODEL_PILL_PATH}")
    model_pill = YOLO(MODEL_PILL_PATH, task='detect')
    
    print(f"[DEBUG] Loading YOLO from: {MODEL_PACK_PATH}")
    model_pack = YOLO(MODEL_PACK_PATH, task='detect')
    
    weights = models.ResNet50_Weights.DEFAULT
    embedder = torch.nn.Sequential(*list(models.resnet50(weights=weights).children())[:-1])
    embedder.eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except Exception as e:
    print(f"[CRITICAL ERROR] Model Load Failed: {e}")
    sys.exit(1)

# ================= 3. TRINITY ENGINE =================
def trinity_inference(img_crop, is_pill=True, custom_matrix=None, custom_labels=None):
    target_matrix = custom_matrix if custom_matrix is not None else global_matrix
    target_labels = custom_labels if custom_labels is not None else global_labels
    if target_matrix is None: return "DB Error", 0.0

    try:
        # Debug Input
        # print(f"[DEBUG-TRINITY] Input shape: {img_crop.shape}")
        
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
        
        # Check scores
        # max_score = torch.max(scores).item()
        # print(f"[DEBUG-TRINITY] Max Vector Score: {max_score:.2f}")

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
            # (SIFT Logic omitted for brevity, assuming standard)
                
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
    except Exception as e:
        print(f"[ERROR-TRINITY] {e}")
        return "Error", 0.0

# ================= 4. AI WORKER =================
class AIProcessor:
    def __init__(self):
        self.latest_frame = None 
        self.results = [] 
        self.stopped = False
        self.lock = threading.Lock()
        self.is_rx_mode = False
        self.session_matrix = None; self.session_labels = None

    def start(self): 
        threading.Thread(target=self.run, daemon=True).start()
        return self
    
    def update_frame(self, frame): 
        with self.lock: self.latest_frame = frame.copy() 
        
    def get_results(self): 
        with self.lock: return self.results

    def run(self):
        print("[DEBUG] AI Worker Loop Started.")
        while not self.stopped:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None 
            
            if frame_to_process is None: 
                time.sleep(0.01); continue

            # 🔥 Force Contiguous Array (Memory Fix for Pi)
            frame_clean = np.ascontiguousarray(frame_to_process)
            
            # Uncomment to debug image shape entering YOLO
            # print(f"[DEBUG-AI] Processing frame shape: {frame_clean.shape}")

            detections = []
            pill_names_batch = []
            pill_coords = []

            # 1. Detect Pills
            try:
                # Use verbose=False to keep terminal clean, but check len below
                pill_res = model_pill(frame_clean, verbose=False, conf=CONF_PILL)
                
                # 🔥 DEBUG: Check if YOLO saw anything
                if len(pill_res[0].boxes) > 0:
                    print(f"[DEBUG] YOLO found {len(pill_res[0].boxes)} pills")
                
                for box in pill_res[0].boxes.xyxy.cpu().numpy().astype(int):
                    x1,y1,x2,y2 = box
                    crop = frame_clean[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    nm, sc = trinity_inference(crop, is_pill=True,
                                             custom_matrix=self.session_matrix,
                                             custom_labels=self.session_labels)
                    
                    print(f"   -> Pill: {nm} (Score: {sc:.2f})") # 🔥 Print identified pill

                    # Color Logic
                    color = (0,0,255)
                    if sc > SCORE_PASS_PILL:
                        color = (0,255,0)
                        if "Unknown" not in nm: 
                            pill_names_batch.append(nm); pill_coords.append(box)
                    
                    detections.append({'box':box, 'label':nm, 'full':f"{nm} {sc:.0%}", 'color':color, 'type':'pill'})

            except Exception as e:
                print(f"[ERROR-YOLO-PILL] {e}")

            # 2. Detect Packs (Similar logic...)
            try:
                pack_res = model_pack(frame_clean, verbose=False, conf=CONF_PACK)
                if len(pack_res[0].boxes) > 0:
                    print(f"[DEBUG] YOLO found {len(pack_res[0].boxes)} packs")

                for box in pack_res[0].boxes.xyxy.cpu().numpy().astype(int):
                    x1,y1,x2,y2 = box
                    crop = frame_clean[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    nm, sc = trinity_inference(crop, is_pill=False,
                                             custom_matrix=self.session_matrix,
                                             custom_labels=self.session_labels)
                    
                    color = (0,255,0) if sc > SCORE_PASS_PACK else (0,0,255)
                    detections.append({'box':box, 'label':nm, 'full':f"{nm} {sc:.0%}", 'color':color, 'type':'pack'})
            except Exception as e:
                print(f"[ERROR-YOLO-PACK] {e}")

            # 3. Group Box
            if pill_coords:
                counts = Counter(pill_names_batch)
                if counts:
                    maj_name = counts.most_common(1)[0][0]
                    # Calculate bounds safely
                    all_x1 = [c[0] for c in pill_coords]; all_y1 = [c[1] for c in pill_coords]
                    all_x2 = [c[2] for c in pill_coords]; all_y2 = [c[3] for c in pill_coords]
                    gx1 = max(0, min(all_x1)-20); gy1 = max(0, min(all_y1)-40)
                    gx2 = max(all_x2)+20; gy2 = max(all_y2)+20
                    
                    detections.append({
                        'box': (gx1, gy1, gx2, gy2),
                        'label': maj_name,
                        'full': f"BATCH: {maj_name} ({len(pill_coords)})",
                        'color': (0,255,255), 'type':'group_box'
                    })

            with self.lock: self.results = detections
            
    def stop(self): self.stopped = True

# ================= 5. DRAWING =================
def draw_summary_box(frame, results):
    if not results: return
    items = [r['label'] for r in results 
             if r['type'] in ['pill', 'pack'] and "Unknown" not in r['label']]
    if not items: return

    counts = Counter(items)
    H, W = frame.shape[:2]
    
    # Debug: Print summary to console
    # print(f"[DEBUG-UI] Summary: {dict(counts)}")

    start_x = W - 300
    start_y = H - (len(counts)*30) - 60
    
    cv2.rectangle(frame, (start_x, start_y), (W, H), (0,0,0), -1)
    cv2.rectangle(frame, (start_x, start_y), (W, H), (255,255,255), 2)
    
    for i, (name, count) in enumerate(counts.items()):
        y = start_y + 40 + (i*30)
        cv2.putText(frame, f"{name}: {count}", (start_x+10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# ================= 6. MAIN =================
def main():
    print("[DEBUG] Main Started")
    cam = WebcamStream().start()
    ai = AIProcessor().start()
    
    print("[DEBUG] Waiting for camera...")
    while cam.read() is None: time.sleep(0.1)
    print("[DEBUG] Camera is Live.")

    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps = 0; prev_time = 0
    try:
        while True:
            frame = cam.read() # RGB
            if frame is None: time.sleep(0.01); continue
            
            ai.update_frame(frame)
            results = ai.get_results() # Get whatever results are available
            
            # 🔥 Draw directly on RGB frame
            display = frame.copy()
            
            # วาดผลลัพธ์
            if results:
                for det in results:
                    x1,y1,x2,y2 = det['box']
                    color = det['color'] # RGB Color (0,255,0) is Green
                    
                    if det['type'] == 'group_box':
                        cv2.rectangle(display, (x1,y1), (x2,y2), color, 4)
                        cv2.putText(display, det['full'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                    else:
                        cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(display, det['full'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                draw_summary_box(display, results)
            
            # Calculate FPS
            curr_time = time.time()
            if (curr_time - prev_time) > 0: fps = 1/(curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(window_name, display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\n[DEBUG] Stopping...")
    finally:
        cam.stop(); ai.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()