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

# Display & AI Resolution
DISPLAY_W, DISPLAY_H = 1280, 720
AI_IMG_SIZE = 416 

# Thresholds
CONF_PILL = 0.3    
CONF_PACK = 0.7     # ลดลงนิดหน่อยเพื่อให้ Detect เจอกล่องง่ายขึ้น แล้วไปคัดที่ Logic แทน
SCORE_PASS_PILL = 0.2
SCORE_PASS_PACK = 0.7

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (Ultra-Optimized Mode)")

# ================= UTILS =================
def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}C"
    except: 
        return "N/A"

def is_point_in_box(point, box):
    """Optimized: inline unpacking"""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 < px < x2 and y1 < py < y2

# ================= 1. WEBCAM STREAM =================
class WebcamStream:
    __slots__ = ('stopped', 'frame', 'grabbed', 'picam2', 'lock')
    
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

# ================= 2. RESOURCES & STATE =================
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
        
        drug_set = set(drug_names_list)
        indices = [i for i, label in enumerate(source_lbls) 
                   if any(drug in label.lower() for drug in drug_set)]
        
        if indices:
            filtered_vecs = [source_vecs[i] for i in indices]
            filtered_lbls = [source_lbls[i] for i in indices]
            return torch.tensor(np.array(filtered_vecs), device=device), filtered_lbls
        return None, None

class PrescriptionState:
    """Manages verified/remaining drugs with checkbox state"""
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
        """Force mark a drug as verified (Automatic)"""
        with self.lock:
            if drug_name not in self.verified_drugs:
                print(f"✨ AUTO-VERIFIED: {drug_name}")
                self.verified_drugs.add(drug_name)
    
    def is_verified(self, drug_name):
        with self.lock:
            return drug_name in self.verified_drugs
    
    def get_all_drugs(self):
        with self.lock:
            return self.all_drugs.copy()

prescription_state = PrescriptionState()

# --- LOAD DATABASES ---
def load_pkl_to_list(filepath):
    if not os.path.exists(filepath): 
        return [], []
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            items = [(v, name) for name, vec_list in data.items() for v in vec_list]
            if items:
                vecs, lbls = zip(*items)
                return list(vecs), list(lbls)
            return [], []
    except Exception as e: 
        print(f"Error loading {filepath}: {e}")
        return [], []

pills_vecs, pills_lbls = load_pkl_to_list(DB_FILES['pills']['vec'])
packs_vecs, packs_lbls = load_pkl_to_list(DB_FILES['packs']['vec'])

matrix_pills = torch.tensor(np.array(pills_vecs), device=device, dtype=torch.float32) if pills_vecs else None
matrix_packs = torch.tensor(np.array(packs_vecs), device=device, dtype=torch.float32) if packs_vecs else None

if matrix_pills is not None:
    matrix_pills = matrix_pills / matrix_pills.norm(dim=1, keepdim=True)
if matrix_packs is not None:
    matrix_packs = matrix_packs / matrix_packs.norm(dim=1, keepdim=True)

color_db = {}
for db_type in ['pills', 'packs']:
    try:
        with open(DB_FILES[db_type]['col'], 'rb') as f: 
            color_db.update(pickle.load(f))
    except: 
        pass

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
                if des is not None: 
                    des_list.append(des)
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

# ================= 3. TRINITY ENGINE =================
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

    if target_matrix is None: 
        return "DB Error", 0.0

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
        if k_val == 0: 
            return "Unknown", 0.0
        
        top_k_val, top_k_idx = torch.topk(scores, k=k_val)
        candidates = []
        seen = set()
        
        for idx, sc in zip(top_k_idx.detach().cpu().numpy(), top_k_val.detach().cpu().numpy()):
            name = target_labels[idx]
            if name not in seen:
                candidates.append((name, float(sc)))
                seen.add(name)
                if len(candidates) >= 3: 
                    break

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
                    except: 
                        pass
                sift_score = min(max_good / SIFT_MAX_MATCHES, 1.0)
                
            color_score = 0.0
            if is_pill and live_color is not None and name in color_db:
                diff = np.abs(live_color - color_db[name])
                diff[0] = min(diff[0], 180 - diff[0]) 
                norm_diff = diff / COLOR_NORM
                dist = np.linalg.norm(norm_diff)
                color_score = np.clip(np.exp(-3.0 * dist), 0, 1)
                
            w_vec, w_sift, w_col = (0.3, 0.1, 0.6) if is_pill else (0.8, 0.2, 0.0)
            total = vec_score * w_vec + sift_score * w_sift + color_score * w_col
            
            if total > best_score: 
                best_score = total
                final_name = clean_name

        return final_name, best_score
    except Exception as e:
        print(f"[Trinity Error] {e}")
        return "Error", 0.0

# ================= 4. AI WORKER (AUTO-CHECK LOGIC) =================
class AIProcessor:
    __slots__ = ('latest_frame', 'results', 'stopped', 'lock', 'is_rx_mode', 
                 'current_patient_info', 'sess_mat_pills', 'sess_lbl_pills',
                 'sess_mat_packs', 'sess_lbl_packs', 'scale_x', 'scale_y',
                 'resize_interpolation')
    
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
        self.scale_x = DISPLAY_W / AI_IMG_SIZE
        self.scale_y = DISPLAY_H / AI_IMG_SIZE
        self.resize_interpolation = cv2.INTER_LINEAR

    def load_patient(self, patient_data):
        with self.lock:
            if not patient_data:
                self.is_rx_mode = False
                self.current_patient_info = None
                self.sess_mat_pills = None
                self.sess_lbl_pills = None
                self.sess_mat_packs = None
                self.sess_lbl_packs = None
                prescription_state.load_drugs([])
            else:
                self.is_rx_mode = True
                self.current_patient_info = patient_data
                drugs = patient_data['drugs']
                prescription_state.load_drugs(drugs)
                self.sess_mat_pills, self.sess_lbl_pills = PrescriptionManager.filter_db(
                    drugs, pills_vecs, pills_lbls)
                self.sess_mat_packs, self.sess_lbl_packs = PrescriptionManager.filter_db(
                    drugs, packs_vecs, packs_lbls)
                print(f"🏥 Loaded: {patient_data['name']}")
    
    def update_remaining_drugs(self):
        with self.lock:
            if self.current_patient_info:
                remaining = prescription_state.get_remaining_drugs()
                if remaining:
                    self.sess_mat_pills, self.sess_lbl_pills = PrescriptionManager.filter_db(
                        remaining, pills_vecs, pills_lbls)
                    self.sess_mat_packs, self.sess_lbl_packs = PrescriptionManager.filter_db(
                        remaining, packs_vecs, packs_lbls)
                else:
                    self.sess_mat_pills = matrix_pills
                    self.sess_lbl_pills = pills_lbls
                    self.sess_mat_packs = matrix_packs
                    self.sess_lbl_packs = packs_lbls

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
            with self.lock:
                frame_HD = self.latest_frame
                self.latest_frame = None
            
            if frame_HD is None: 
                time.sleep(0.005)
                continue

            frame_yolo = cv2.resize(frame_HD, (AI_IMG_SIZE, AI_IMG_SIZE), 
                                   interpolation=self.resize_interpolation)
            
            final_detections = []
            detected_pills_raw = [] 
            valid_pills = [] # Pills that are good enough to verify packs

            try:
                # --- 1. DETECT PILLS ---
                pill_res = model_pill(frame_yolo, verbose=False, conf=CONF_PILL, 
                                     imgsz=AI_IMG_SIZE, max_det=10, agnostic_nms=True)
                
                # Assumption Logic Helper
                best_pill_score = -1
                best_pill_name = None

                for box in pill_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    x1, y1 = int(x1_s * self.scale_x), int(y1_s * self.scale_y)
                    x2, y2 = int(x2_s * self.scale_x), int(y2_s * self.scale_y)
                    
                    if (x2-x1) < 30 or (y2-y1) < 30: continue
                    crop = frame_HD[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    nm, sc = trinity_inference(crop, is_pill=True,
                                              session_pills=self.sess_mat_pills,
                                              session_pills_lbl=self.sess_lbl_pills,
                                              session_packs=self.sess_mat_packs,
                                              session_packs_lbl=self.sess_lbl_packs)
                    
                    cx, cy = (x1+x2)>>1, (y1+y2)>>1
                    
                    # Store for assumption
                    if "?" not in nm and "Unknown" not in nm and sc > best_pill_score:
                        best_pill_score = sc
                        best_pill_name = nm
                        
                    detected_pills_raw.append({
                        'name': nm, 'score': sc, 'center': (cx, cy), 'box': (x1, y1, x2, y2)
                    })

                # Process Pills & Auto-Tick
                for pill in detected_pills_raw:
                    nm, sc = pill['name'], pill['score']
                    
                    # Apply Assumption
                    if best_pill_name and ("?" in nm or "Unknown" in nm or sc < SCORE_PASS_PILL):
                        nm, sc = best_pill_name, best_pill_score
                    
                    # --- LOGIC #1: Auto-Tick Pill ---
                    if "?" not in nm and "Unknown" not in nm and sc >= SCORE_PASS_PILL:
                        prescription_state.verify_drug(nm.lower())
                        self.update_remaining_drugs()

                    # Check verification status
                    is_verified = prescription_state.is_verified(nm.lower())
                    
                    if "?" not in nm and "Unknown" not in nm:
                        # Add to valid list for Pack checking
                        valid_pills.append({
                            'name': nm, 'center': pill['center'], 'verified': is_verified, 'score': sc
                        })

                    final_detections.append({
                        'label': nm, 'score': sc, 'type': 'pill',
                        'verified': is_verified, 'box': pill['box']
                    })

                # --- 2. DETECT PACKS ---
                pack_res = model_pack(frame_yolo, verbose=False, conf=CONF_PACK, 
                                     imgsz=AI_IMG_SIZE, max_det=5, agnostic_nms=True)
                
                for box in pack_res[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                    x1_s, y1_s, x2_s, y2_s = box
                    x1, y1 = int(x1_s * self.scale_x), int(y1_s * self.scale_y)
                    x2, y2 = int(x2_s * self.scale_x), int(y2_s * self.scale_y)
                    
                    if (x2-x1) < 50 or (y2-y1) < 50: continue
                    crop = frame_HD[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    nm, sc = trinity_inference(crop, is_pill=False,
                                              session_pills=self.sess_mat_pills,
                                              session_pills_lbl=self.sess_lbl_pills,
                                              session_packs=self.sess_mat_packs,
                                              session_packs_lbl=self.sess_lbl_packs)
                    
                    # --- LOGIC #2: Auto-Tick Pack (Native) ---
                    clean_name = nm.replace("_pack", "").lower()
                    if "?" not in nm and "Unknown" not in nm and sc >= SCORE_PASS_PACK:
                        prescription_state.verify_drug(clean_name)
                        self.update_remaining_drugs()
                    
                    # --- LOGIC #3: Check for Inner Pill (Inheritance) ---
                    pack_verified = prescription_state.is_verified(clean_name)
                    
                    for pill in valid_pills:
                        if is_point_in_box(pill['center'], (x1, y1, x2, y2)):
                            # Found a pill inside this pack
                            if pill['score'] >= SCORE_PASS_PILL or pill['verified']:
                                # TRUST THE PILL -> Force Verify Pack
                                pack_verified = True
                                nm = pill['name'] # Override Name
                                sc = max(sc, pill['score']) # Boost Score
                                
                                # Auto-tick based on inner pill
                                prescription_state.verify_drug(nm.lower())
                                self.update_remaining_drugs()
                                break
                    
                    final_detections.append({
                        'label': nm, 'score': sc, 'type': 'pack',
                        'verified': pack_verified,
                        'box': (x1, y1, x2, y2)
                    })

                with self.lock: 
                    self.results = final_detections
            
            except Exception as e:
                print(f"[ERROR-AI-LOOP] {e}")
            
    def stop(self): 
        self.stopped = True

# ================= 5. UI DRAWING =================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_SCALE_SMALL = 0.6
THICKNESS = 2
THICKNESS_BOX = 3
CHECKBOX_SIZE = 25

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
    
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (start_x, 0), (W, box_h), (0, 255, 255), 2)
    
    for i, line in enumerate(header_lines):
        y = 35 + i * line_h
        cv2.putText(frame, line, (start_x+15, y), FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
    
    clickable_areas = []
    for i, drug in enumerate(all_drugs):
        y_base = 35 + (len(header_lines) + i) * line_h
        checkbox_x = start_x + 15
        checkbox_y = y_base - 20
        
        is_checked = prescription_state.is_verified(drug.lower())
        
        cv2.rectangle(frame, (checkbox_x, checkbox_y), 
                     (checkbox_x + CHECKBOX_SIZE, checkbox_y + CHECKBOX_SIZE), 
                     (255, 255, 255), 3)
        
        if is_checked:
            cv2.rectangle(frame, (checkbox_x + 3, checkbox_y + 3), 
                         (checkbox_x + CHECKBOX_SIZE - 3, checkbox_y + CHECKBOX_SIZE - 3), 
                         (0, 255, 0), -1)
            cv2.line(frame, (checkbox_x + 6, checkbox_y + 14), 
                    (checkbox_x + 11, checkbox_y + 20), (255, 255, 255), 4)
            cv2.line(frame, (checkbox_x + 11, checkbox_y + 20), 
                    (checkbox_x + 20, checkbox_y + 8), (255, 255, 255), 4)
        else:
            cv2.rectangle(frame, (checkbox_x + 3, checkbox_y + 3), 
                         (checkbox_x + CHECKBOX_SIZE - 3, checkbox_y + CHECKBOX_SIZE - 3), 
                         (60, 60, 60), -1)
        
        text_x = checkbox_x + CHECKBOX_SIZE + 10
        text_y = y_base
        drug_text = drug
        text_color = (100, 100, 100) if is_checked else (255, 255, 255)
        cv2.putText(frame, drug_text, (text_x, text_y), FONT, 0.75, text_color, THICKNESS)
        
        if is_checked:
            text_size = cv2.getTextSize(drug_text, FONT, 0.75, THICKNESS)[0]
            cv2.line(frame, (text_x, text_y - 10), (text_x + text_size[0], text_y - 10), (255, 0, 0), 3)
        
        click_box = (checkbox_x - 5, checkbox_y - 5, checkbox_x + 300, checkbox_y + CHECKBOX_SIZE + 10)
        clickable_areas.append({'drug': drug, 'box': click_box})
    
    return clickable_areas

def draw_boxes_on_items(frame, results):
    for r in results:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        score = r['score']
        obj_type = r.get('type', 'pill')
        is_verified = r.get('verified', False)
        
        if is_verified:
            color = (0, 255, 0) # Green for verified
            label_display = f"OK {label}"
        elif obj_type == 'pack':
            if score >= SCORE_PASS_PACK:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255) # Yellow for unsure pack
            label_display = label
        elif "?" in label or score < SCORE_PASS_PILL:
            color = (0, 0, 255) # Red for bad pill
            label_display = label
        else:
            color = (0, 255, 0)
            label_display = label

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS_BOX)
        cv2.putText(frame, f"{label_display} {score:.0%}", (x1, y1-10), 
                   FONT, FONT_SCALE_SMALL, color, THICKNESS)

# ================= 6. MAIN =================
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clickable_areas, ai_processor = param
        for area in clickable_areas:
            x1, y1, x2, y2 = area['box']
            if x1 <= x <= x2 and y1 <= y <= y2:
                drug = area['drug']
                prescription_state.toggle_drug(drug.lower())
                ai_processor.update_remaining_drugs()
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
    while cam.read() is None: time.sleep(0.1)
    
    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H) 
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"🎥 RUNNING... (Auto-Check & Mouse Mode)")
    
    fps = 0
    prev_time = time.perf_counter()
    TARGET_FPS = 10 
    FRAME_TIME = 1.0 / TARGET_FPS
    
    clickable_areas = []

    try:
        while True:
            start_loop = time.perf_counter()
            frame_rgb = cam.read()
            if frame_rgb is None: 
                time.sleep(0.01)
                continue
            
            ai.update_frame(frame_rgb)
            results, cur_patient = ai.get_results()
            
            draw_boxes_on_items(frame_rgb, results)
            
            if cur_patient: 
                clickable_areas = draw_patient_info(frame_rgb, cur_patient)
                cv2.setMouseCallback(window_name, mouse_callback, (clickable_areas, ai))
            
            curr_time = time.perf_counter()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            temp = get_cpu_temperature()
            cv2.putText(frame_rgb, f"FPS: {fps:.1f} | {temp}", (30, 50), 
                       FONT, 1.2, (0, 255, 0), THICKNESS_BOX)
            
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