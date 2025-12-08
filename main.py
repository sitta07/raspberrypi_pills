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

# ================= 1. FIX RASPBERRY PI ENVIRONMENT =================
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Import Picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ Warning: Picamera2 not found.")
    sys.exit(1)

# ================= CONFIGURATION =================
# ⚠️ เปลี่ยนเป็น .pt ตามที่คุณบอกว่าเวิร์ค (หรือจะใช้ .onnx ก็ได้ถ้ามีไฟล์)
MODEL_PILL_PATH = 'models/pills.pt'          
MODEL_PACK_PATH = 'models/best_process_2.pt'

DB_FILES = {
    'pills': {'vec': 'database/db_pills.pkl', 'col': 'database/colors_pills.pkl'},
    'packs': {'vec': 'database/db_packs.pkl', 'col': 'database/colors_packs.pkl'}
}
IMG_DB_FOLDER = 'database_images'
HIS_FILE_PATH = 'prescription.txt' 

# Thresholds (ตั้งไว้กลางๆ เพื่อกรอง Noise)
CONF_PILL = 0.40    
CONF_PACK = 0.50    
SCORE_PASS_PILL = 0.65  
SCORE_PASS_PACK = 0.60  

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device}")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}C"
    except: return "N/A"

# ================= 2. WEBCAM STREAM (RGB888) =================
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
            # RGB888 ตามที่คุณต้องการ
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 640), "format": "RGB888"}, # 640x640 ดีสุดสำหรับ YOLO
                controls={"FrameDurationLimits": (33333, 33333)} 
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
            print("[DEBUG] Camera Started (640x640 RGB888)")
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
                        # 🔥 CRITICAL FIX: ต้อง .copy() เพื่อแก้ Memory Layout ของ Picamera
                        self.frame = frame.copy()
                        self.grabbed = True
                else:
                    self.stopped = True
            except:
                self.stopped = True

    def read(self):
        with self.lock:
            if self.grabbed and self.frame is not None:
                return self.frame.copy()
            return None
    
    def stop(self):
        self.stopped = True
        if self.picam2: self.picam2.stop(); self.picam2.close()

# ================= 3. RESOURCES =================
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

print("[DEBUG] Loading Resources...")
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
    # โหลดโมเดล (ใช้ .pt ตามที่เทสผ่าน)
    print(f"[DEBUG] Loading Models...")
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
except Exception as e:
    print(f"[CRITICAL] Model Error: {e}"); sys.exit(1)

# ================= 4. TRINITY ENGINE =================
def trinity_inference(img_crop, is_pill=True, custom_matrix=None, custom_labels=None):
    target_matrix = custom_matrix if custom_matrix is not None else global_matrix
    target_labels = custom_labels if custom_labels is not None else global_labels
    if target_matrix is None: return "DB Error", 0.0

    try:
        # img_crop is BGR here (Converted in AI Loop)
        if is_pill:
            # ResNet likes RGB
            pil_img = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        else:
            gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
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
            hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            live_color = np.mean(hsv, axis=(0,1))
        
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
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
    except: return "Error", 0.0

# ================= 5. AI WORKER (THE FIX) =================
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
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None 
            
            if frame_to_process is None: 
                time.sleep(0.01); continue

            # 🔥 1. Force Contiguous (แก้ปัญหา Memory Picamera)
            frame_clean = np.ascontiguousarray(frame_to_process)
            
            # 🔥 2. COLOR FIX: กล้องส่ง RGB มา แต่ YOLO เทรนมาด้วย BGR
            # ต้องแปลงเป็น BGR ก่อนส่งเข้า YOLO ไม่งั้นสีเพี้ยน YOLO จะตาบอด
            frame_bgr = cv2.cvtColor(frame_clean, cv2.COLOR_RGB2BGR)

            detections = []

            def process_crop(crop, is_pill_mode):
                name, score = trinity_inference(crop, is_pill=is_pill_mode,
                                                custom_matrix=self.session_matrix,
                                                custom_labels=self.session_labels)
                threshold = SCORE_PASS_PILL if is_pill_mode else SCORE_PASS_PACK
                if score <= threshold: name = "Unknown"
                return name, score

            try:
                # 1. Pills (ใช้ BGR image)
                pill_res = model_pill(frame_bgr, verbose=False, conf=CONF_PILL, max_det=10)
                for box in pill_res[0].boxes.xyxy.cpu().numpy().astype(int):
                    x1,y1,x2,y2 = box
                    # Crop จากภาพ BGR
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    nm, sc = process_crop(crop, True)
                    
                    # Debug Print
                    if sc > 0.5: print(f"Found Pill: {nm} ({sc:.2f})")
                    
                    detections.append({'label':nm, 'type':'pill'})

                # 2. Packs (ใช้ BGR image)
                pack_res = model_pack(frame_bgr, verbose=False, conf=CONF_PACK, max_det=5)
                for box in pack_res[0].boxes.xyxy.cpu().numpy().astype(int):
                    x1,y1,x2,y2 = box
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    nm, sc = process_crop(crop, False)
                    detections.append({'label':nm, 'type':'pack'})

                with self.lock: self.results = detections
            
            except Exception as e:
                print(f"[ERROR-AI-LOOP] {e}")
            
    def stop(self): self.stopped = True

# ================= 6. UI DRAWING =================
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
    # Filter items
    items = [r['label'] for r in results 
             if "Unknown" not in r['label'] and "WRONG" not in r['label']]
    
    # ถ้าไม่เจอให้โชว์ว่า Scanning
    if not items: 
        cv2.putText(frame, "Scanning...", (W-200, H-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        return

    counts = Counter(items)
    box_w = 350; line_h = 40; padding = 15
    total_h = (len(counts) * line_h) + 50
    start_x = W - box_w - 10; start_y = H - total_h - 10
    
    # Draw Clean Box
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (W-10, H-10), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (start_x, start_y), (W-10, H-10), (255,255,255), 2)
    
    # Header
    cv2.putText(frame, "DETECTED ITEMS", (start_x+15, start_y+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.line(frame, (start_x+15, start_y+40), (W-25, start_y+40), (200,200,200), 1)

    # Items
    for i, (name, count) in enumerate(counts.items()):
        y = start_y + 70 + (i * line_h)
        cv2.putText(frame, f"{name} ({count})", (start_x+15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# ================= 7. MAIN =================
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
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("🎥 RUNNING... (Clean UI)")
    fps = 0; prev_time = 0

    try:
        while True:
            frame_rgb = cam.read()
            if frame_rgb is None: time.sleep(0.01); continue
            
            ai.update_frame(frame_rgb)
            display = frame_rgb.copy() # Display RGB directly (User preference)
            
            results, cur_patient = ai.get_results()
            
            # 🔥 No Boxes, Just Summary
            draw_summary_box(display, results)
            if cur_patient: draw_patient_info(display, cur_patient)
            
            # FPS
            curr_time = time.time()
            if (curr_time - prev_time) > 0: fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            temp = get_cpu_temperature()
            
            cv2.putText(display, f"FPS: {fps:.1f} | Temp: {temp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(window_name, display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if cv2.waitKey(1) & 0xFF == ord('r'):
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db: d = his_db[TARGET_HN]; d['hn'] = TARGET_HN; ai.load_patient(d)

    except KeyboardInterrupt: print("\n Stopping...")
    finally: cam.stop(); ai.stop(); cv2.destroyAllWindows(); print(" Bye Bye!")

if __name__ == "__main__":
    main()