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
os.environ["OMP_NUM_THREADS"] = "2"    

try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ Warning: Picamera2 not found.")

# ================= CONFIGURATION =================
# Paths
MODEL_PILL_PATH = 'models/pills_seg.pt'          
DB_FILES = {
    'pills': {'vec': 'database/model_register/db_pills.pkl', 'col': 'database/model_register/colors_pills.pkl'}
}

IMG_DB_FOLDER = 'database_images'
HIS_FILE_PATH = 'prescription.txt' 

# Display & AI Resolution
DISPLAY_W, DISPLAY_H = 1280, 720
AI_IMG_SIZE = 384  # ลดขนาดเพื่อประสิทธิภาพบน RPi

# Thresholds
CONF_PILL = 0.20    # เพิ่มเล็กน้อยเพื่อลด false positives
SCORE_PASS_PILL = 0.25

# --- SENIOR UPGRADES ---
CONSISTENCY_THRESHOLD = 2   
MAX_OBJ_AREA_RATIO = 0.40   

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (PILLS ONLY MODE)")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}C"
    except: 
        return "N/A"

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
                    else: 
                        self.stopped = True
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

# ================= 2. RESOURCES & STATE =================
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
        except: 
            return {}

class PrescriptionState:
    def __init__(self):
        self.all_drugs = []  
        self.verified_drugs = set()
        self.lock = threading.Lock()
    
    def load_drugs(self, drug_list):
        with self.lock:
            self.all_drugs = drug_list.copy()
            self.verified_drugs.clear()
    
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
            for verified in list(self.verified_drugs):
                if verified == drug_name: return

            if drug_name not in self.verified_drugs:
                found = False
                for target in self.all_drugs:
                    if target == drug_name:
                        self.verified_drugs.add(target)
                        print(f"✨ VERIFIED (Direct): {target}")
                        found = True
                        break
                
                if not found:
                    self.verified_drugs.add(drug_name)
                    print(f"✨ VERIFIED (New): {drug_name}")
    
    def is_verified(self, drug_name):
        with self.lock:
            return drug_name in self.verified_drugs
    
    def get_all_drugs(self):
        with self.lock:
            return self.all_drugs.copy()

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
    except: 
        return [], []

# Load Global DB
pills_vecs, pills_lbls = load_pkl_to_list(DB_FILES['pills']['vec'])

matrix_pills = torch.tensor(np.array(pills_vecs), device=device, dtype=torch.float32) if pills_vecs else None

if matrix_pills is not None: 
    matrix_pills = matrix_pills / matrix_pills.norm(dim=1, keepdim=True)

color_db = {}
try:
    with open(DB_FILES['pills']['col'], 'rb') as f: 
        color_db.update(pickle.load(f))
except: 
    pass

sift = cv2.SIFT_create(nfeatures=80)  # ลดจาก 100 เพื่อประสิทธิภาพ
bf = cv2.BFMatcher(crossCheck=False)
sift_db = {}

if os.path.exists(IMG_DB_FOLDER):
    for folder in os.listdir(IMG_DB_FOLDER):
        path = os.path.join(IMG_DB_FOLDER, folder)
        if not os.path.isdir(path): continue
        des_list = []
        image_files = [x for x in os.listdir(path) if x.lower().endswith(('jpg', 'png', 'jpeg'))][:2]  # ลดจาก 3
        for img_file in image_files:
            img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if max(img.shape) > 400:  # ลดขนาดมากขึ้น
                    scale = 400 / max(img.shape)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                _, des = sift.detectAndCompute(img, None)
                if des is not None: 
                    des_list.append(des)
        if des_list:
            sift_db[folder] = des_list

try:
    model_pill = YOLO(MODEL_PILL_PATH, task='segment')  # เปลี่ยนเป็น segment
    model_pill.overrides['imgsz'] = AI_IMG_SIZE
    
    # ใช้ MobileNetV2 แทน ResNet50 เพื่อประสิทธิภาพบน RPi
    weights = models.MobileNet_V2_Weights.DEFAULT
    base_model = models.mobilenet_v2(weights=weights)
    embedder = torch.nn.Sequential(*list(base_model.children())[:-1])
    embedder.eval().to(device)
    del base_model
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch.set_grad_enabled(False)
    print("✅ Models loaded successfully (Pills Segmentation Mode)")
except Exception as e: 
    print(f"[CRITICAL] Model Error: {e}")
    sys.exit(1)

# ================= 3. TRINITY ENGINE (RGB LOGIC) =================
COLOR_NORM = np.array([90.0, 255.0, 255.0])
SIFT_RATIO = 0.75
SIFT_MAX_MATCHES = 12.0  # ลดลง

def trinity_inference(img_crop):
    if matrix_pills is None: 
        return "DB Error", 0.0

    try:
        pil_img = Image.fromarray(img_crop) 
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        live_vec = embedder(input_tensor).flatten()
        live_vec = live_vec / live_vec.norm()
        
        scores = torch.matmul(live_vec, matrix_pills.T).squeeze(0)
        k_val = min(8, len(pills_lbls))  # ลดจาก 10
        if k_val == 0: return "Unknown", 0.0
        
        top_k_val, top_k_idx = torch.topk(scores, k=k_val)
        candidates = []
        seen = set()
        
        for idx, sc in zip(top_k_idx.detach().cpu().numpy(), top_k_val.detach().cpu().numpy()):
            name = pills_lbls[idx]
            if name not in seen:
                candidates.append((name, float(sc)))
                seen.add(name)
                if len(candidates) >= 3: 
                    break

        live_color = None
        gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        _, des_live = sift.detectAndCompute(gray, None)
        
        h, w = img_crop.shape[:2]
        center = img_crop[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        if center.size > 0:
            hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
            live_color = np.mean(hsv, axis=(0,1))

        best_score = -1
        final_name = "Unknown"
        
        for name, vec_score in candidates:
            clean_name = name.replace("_pill", "")
            
            sift_score = 0.0
            if des_live is not None and clean_name in sift_db:
                max_good = 0
                for ref_des in sift_db[clean_name]:
                    try:
                        matches = bf.knnMatch(des_live, ref_des, k=2)
                        good = sum(1 for m, n in matches if len([m, n]) == 2 and m.distance < SIFT_RATIO * n.distance)
                        max_good = max(max_good, good)
                    except: 
                        pass
                sift_score = min(max_good / SIFT_MAX_MATCHES, 1.0)
                
            color_score = 0.0
            if live_color is not None and name in color_db:
                diff = np.abs(live_color - color_db[name])
                diff[0] = min(diff[0], 180 - diff[0]) 
                norm_diff = diff / COLOR_NORM
                dist = np.linalg.norm(norm_diff)
                color_score = np.clip(np.exp(-3.0 * dist), 0, 1)
                
            w_vec, w_sift, w_col = 0.5, 0.4, 0.1
            total = vec_score * w_vec + sift_score * w_sift + color_score * w_col
            
            if total > best_score: 
                best_score = total
                final_name = clean_name

        return final_name, best_score
    except Exception as e:
        print(f"[Trinity Error] {e}")
        return "Error", 0.0

# ================= 4. AI WORKER (PILLS ONLY) =================
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
                prescription_state.load_drugs(drugs)
                self.consistency_counter.clear()
                print(f"🏥 Loaded: {patient_data['name']}")
                print(f"📋 Prescription: {', '.join(drugs)}")
    
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
        print("[DEBUG] AI Worker Loop Started (PILLS ONLY MODE)")
        
        while not self.stopped:
            with self.lock:
                frame_HD = self.latest_frame
                self.latest_frame = None
            
            if frame_HD is None: 
                time.sleep(0.005)
                continue

            frame_yolo = cv2.resize(frame_HD, (AI_IMG_SIZE, AI_IMG_SIZE), 
                                   interpolation=self.resize_interpolation)
            
            final_detections = []
            found_in_this_frame = set()

            try:
                # ==========================================
                # DETECT PILLS ONLY (SEGMENTATION)
                # ==========================================
                pill_res = model_pill(frame_yolo, verbose=False, conf=CONF_PILL, 
                                     imgsz=AI_IMG_SIZE, max_det=30, agnostic_nms=True)
                
                # แสดงจำนวนที่เจอ
                num_detected = len(pill_res[0].boxes) if pill_res[0].boxes is not None else 0
                if num_detected > 0:
                    print(f"🔍 Detected {num_detected} pills in frame")
                
                for i, box in enumerate(pill_res[0].boxes.xyxy.detach().cpu().numpy().astype(int)):
                    x1_s, y1_s, x2_s, y2_s = box
                    x1, y1 = int(x1_s * self.scale_x), int(y1_s * self.scale_y)
                    x2, y2 = int(x2_s * self.scale_x), int(y2_s * self.scale_y)
                    
                    if not self.is_valid_detection((x1, y1, x2, y2), DISPLAY_W, DISPLAY_H):
                        continue
                    
                    if (x2-x1) < 30 or (y2-y1) < 30: 
                        continue
                    
                    crop = frame_HD[y1:y2, x1:x2]
                    if crop.size == 0: 
                        continue
                    
                    # Trinity Inference
                    real_name, real_score = trinity_inference(crop)
                    
                    final_name = real_name
                    final_score = real_score
                    is_wrong_drug = False
                    
                    # RX Check (ถ้าอยู่ใน prescription mode)
                    if self.is_rx_mode:
                        clean_real = real_name.lower().strip()
                        allowed_drugs = [d.lower() for d in prescription_state.get_all_drugs()]
                        match_found = False
                        
                        for allowed in allowed_drugs:
                            if allowed in clean_real or clean_real in allowed:
                                match_found = True
                                final_name = allowed
                                break
                        
                        if not match_found and "?" not in real_name and "Unknown" not in real_name:
                            final_name = f"WRONG: {real_name}"
                            final_score = 0.0
                            is_wrong_drug = True
                    
                    clean_name = final_name.lower()
                    
                    # Register Valid Pills
                    if not is_wrong_drug and "?" not in final_name and "Unknown" not in final_name and final_score >= SCORE_PASS_PILL:
                        self.consistency_counter[clean_name] = self.consistency_counter.get(clean_name, 0) + 1
                        found_in_this_frame.add(clean_name)
                        
                        if self.consistency_counter[clean_name] >= CONSISTENCY_THRESHOLD:
                            prescription_state.verify_drug(clean_name)
                    
                    is_verified = prescription_state.is_verified(clean_name)
                    
                    final_detections.append({
                        'label': final_name, 
                        'score': final_score, 
                        'type': 'pill',
                        'verified': is_verified, 
                        'box': (x1, y1, x2, y2), 
                        'is_wrong': is_wrong_drug
                    })

                # Reset consistency counter สำหรับ items ที่หายไป
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
FONT_SCALE = 0.7
FONT_SCALE_SMALL = 0.5
THICKNESS = 2
THICKNESS_BOX = 2
CHECKBOX_SIZE = 25

RGB_GREEN = (0, 255, 0)
RGB_RED   = (255, 0, 0)
RGB_BLUE  = (0, 0, 255)
RGB_YELLOW = (255, 255, 0)
RGB_WHITE = (255, 255, 255)
RGB_GRAY  = (50, 50, 50)
RGB_BLACK = (0, 0, 0)

def draw_patient_info(frame, patient_data):
    if not patient_data: return []
    
    H, W = frame.shape[:2]
    box_w = 450
    start_x = W - box_w
    
    header_lines = [
        f"HN: {patient_data.get('hn', 'N/A')}",
        f"Name: {patient_data.get('name', 'N/A')}", 
        "--- Prescription List ---"
    ]
    
    all_drugs = prescription_state.get_all_drugs()
    line_h = 45
    box_h = (len(header_lines) + len(all_drugs)) * line_h + 20
    
    cv2.rectangle(frame, (start_x, 0), (W, box_h), RGB_GRAY, -1)
    cv2.rectangle(frame, (start_x, 0), (W, box_h), RGB_YELLOW, 2)
    
    for i, line in enumerate(header_lines):
        y = 35 + i * line_h
        cv2.putText(frame, line, (start_x+15, y), FONT, FONT_SCALE, RGB_WHITE, THICKNESS)
    
    clickable_areas = []
    for i, drug in enumerate(all_drugs):
        y_base = 35 + (len(header_lines) + i) * line_h
        checkbox_x = start_x + 15
        checkbox_y = y_base - 20
        
        is_checked = prescription_state.is_verified(drug.lower())
        
        cv2.rectangle(frame, (checkbox_x, checkbox_y), 
                     (checkbox_x + CHECKBOX_SIZE, checkbox_y + CHECKBOX_SIZE), 
                     RGB_WHITE, 3)
        
        if is_checked:
            cv2.rectangle(frame, (checkbox_x + 3, checkbox_y + 3), 
                         (checkbox_x + CHECKBOX_SIZE - 3, checkbox_y + CHECKBOX_SIZE - 3), 
                         RGB_GREEN, -1)
            cv2.line(frame, (checkbox_x + 6, checkbox_y + 14), 
                    (checkbox_x + 11, checkbox_y + 20), RGB_WHITE, 4)
            cv2.line(frame, (checkbox_x + 11, checkbox_y + 20), 
                    (checkbox_x + 20, checkbox_y + 8), RGB_WHITE, 4)
        else:
            cv2.rectangle(frame, (checkbox_x + 3, checkbox_y + 3), 
                         (checkbox_x + CHECKBOX_SIZE - 3, checkbox_y + CHECKBOX_SIZE - 3), 
                         (60, 60, 60), -1)
        
        text_x = checkbox_x + CHECKBOX_SIZE + 10
        text_y = y_base
        drug_text = drug
        text_color = (100, 100, 100) if is_checked else RGB_WHITE
        cv2.putText(frame, drug_text, (text_x, text_y), FONT, 0.7, text_color, THICKNESS)
        
        if is_checked:
            text_size = cv2.getTextSize(drug_text, FONT, 0.7, THICKNESS)[0]
            cv2.line(frame, (text_x, text_y - 10), (text_x + text_size[0], text_y - 10), RGB_RED, 3)
        
        click_box = (checkbox_x - 5, checkbox_y - 5, checkbox_x + 300, checkbox_y + CHECKBOX_SIZE + 10)
        clickable_areas.append({'drug': drug, 'box': click_box})
    
    return clickable_areas

def draw_boxes_on_items(frame, results):
    """วาดกรอบทุก object ที่เจอ"""
    for r in results:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        score = r['score']
        is_verified = r.get('verified', False)
        is_wrong = r.get('is_wrong', False)
        
        # เลือกสีและข้อความ
        if is_wrong:
            color = RGB_RED 
            label_display = f"!! {label} !!"
        elif is_verified:
            color = RGB_GREEN
            label_display = f"✓ {label}"
        elif "?" in label or score < SCORE_PASS_PILL:
            color = RGB_RED
            label_display = label
        else:
            color = RGB_YELLOW
            label_display = label

        # วาดกรอบ
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS_BOX)
        
        # วาดพื้นหลังสำหรับ label
        label_text = f"{label_display} {score:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE_SMALL, THICKNESS)
        cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w+10, y1), color, -1)
        cv2.putText(frame, label_text, (x1+5, y1-5), 
                   FONT, FONT_SCALE_SMALL, RGB_BLACK, THICKNESS)

# ================= 6. MAIN =================
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clickable_areas, ai_processor = param
        for area in clickable_areas:
            x1, y1, x2, y2 = area['box']
            if x1 <= x <= x2 and y1 <= y <= y2:
                drug = area['drug']
                prescription_state.toggle_drug(drug.lower())
                return

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
    while cam.read() is None: 
        time.sleep(0.1)
    
    window_name = "PillTrack - Pills Only Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H) 

    print(f"🎥 RUNNING... (PILLS ONLY - OPTIMIZED FOR RPi)")
    
    fps = 0
    prev_time = time.perf_counter()
    TARGET_FPS = 12  # ลด FPS เป้าหมายสำหรับ RPi
    FRAME_TIME = 1.0 / TARGET_FPS
    
    clickable_areas = []

    try:
        while True:
            start_loop = time.perf_counter()
            frame_rgb = cam.read()
            if frame_rgb is None: 
                time.sleep(0.01)
                continue
            
            ai.update_frame(frame_rgb.copy()) 
            results, cur_patient = ai.get_results()
            
            # วาดกรอบทุก object
            draw_boxes_on_items(frame_rgb, results)
            
            # แสดง count
            num_items = len(results)
            cv2.putText(frame_rgb, f"Detected: {num_items} pills", (30, 100), 
                       FONT, 1.0, RGB_YELLOW, THICKNESS)
            
            if cur_patient: 
                clickable_areas = draw_patient_info(frame_rgb, cur_patient)
                cv2.setMouseCallback(window_name, mouse_callback, (clickable_areas, ai))
            
            curr_time = time.perf_counter()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            temp = get_cpu_temperature()
            cv2.putText(frame_rgb, f"FPS: {fps:.1f} | {temp}", (30, 50), 
                       FONT, 1.2, RGB_GREEN, THICKNESS)
            
            cv2.imshow(window_name, frame_rgb)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            if key == ord('r'):
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db: 
                    d = his_db[TARGET_HN]
                    d['hn'] = TARGET_HN
                    ai.load_patient(d)
                    print("🔄 Prescription reloaded")

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