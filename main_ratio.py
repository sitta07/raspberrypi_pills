#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║ PILLTRACK: MASTER EDITION (DINOv2 + OCR + MASKING)           ║
║ - Brain: DINOv2 (ViT-S/14) for Pills                         ║
║ - Eyes: EasyOCR for Blister Packs                            ║
║ - Vision: YOLOv8 Segmentation + Smart Background Removal     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import pickle
from torchvision import transforms
from ultralytics import YOLO

# --- 🚀 RASPBERRY PI OPTIMIZATIONS ---
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["QT_QPA_PLATFORM"] = "xcb" # แก้ปัญหา OpenCV บน Pi

# --- 📦 IMPORT OCR ---
try:
    import easyocr
except ImportError:
    print("⚠️ EasyOCR not found. Install with: pip install easyocr")
    sys.exit(1)

# ================= ⚙️ CONFIGURATION =================
@dataclass
class Config:
    # --- PATHS ---
    MODEL_PACK: str = 'models/seg_best_process.pt'
    MODEL_PILL: str = 'models/pills_seg.pt'
    
    # DINOv2 Databases (ต้องเป็นไฟล์ที่สร้างจาก DINOv2 เท่านั้น)
    DB_PILLS_VEC: str = 'database/db_register/db_pills_dino.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs_dino.pkl'
    
    # Generic Databases
    DB_PILLS_COL: str = 'database/db_register/colors_pills.pkl'
    DB_PACKS_COL: str = 'database/db_register/colors_packs.pkl'
    PRESCRIPTION_FILE: str = 'prescription.txt'
    ALIAS_FILE: str = 'drug_aliases.json' # ไฟล์จับคู่ชื่อยา (Mapping)

    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416 # 416 เร็ว, 640 แม่น
    
    # 🚫 EXCLUSION ZONE (Dashboard Area - Top Right)
    UI_ZONE_X_START: int = 900
    UI_ZONE_Y_END: int = 220
    
    # 🎚️ TUNING THRESHOLDS
    CONF_THRESHOLD: float = 0.65       # DINOv2 ต้องมั่นใจ 65% ขึ้นไป
    VERIFY_FRAMES: int = 5             # ต้องเห็นต่อเนื่อง 5 เฟรมถึงจะติ๊กถูก
    
    # WEIGHTS FUSION
    # DINOv2 (Vec) 70% | Color 20% | SIFT 10%
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {'vec': 0.7, 'col': 0.2, 'sift': 0.1})
    
    SIFT_RATIO_TEST: float = 0.7

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 SYSTEM STARTING ON: {device}")

# ================= 🧠 PRESCRIPTION & ALIAS MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.patient_name = "Unknown"
        self.allowed_drugs_rx = [] # ชื่อยาตามใบสั่งแพทย์
        self.verified_drugs = set()
        self.aliases = self.load_aliases()
        self.load_prescription()

    def load_aliases(self):
        """ โหลดพจนานุกรมแปลชื่อยา (Generic Substitution) """
        if not os.path.exists(CFG.ALIAS_FILE):
            # สร้างไฟล์เปล่าถ้าไม่มี
            default_alias = {"paracetamol_500mg": ["sara", "tylenol", "gpo_para"]}
            try:
                with open(CFG.ALIAS_FILE, 'w') as f: json.dump(default_alias, f)
            except: pass
            return default_alias
        try:
            with open(CFG.ALIAS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except: return {}

    def load_prescription(self):
        if not os.path.exists(CFG.PRESCRIPTION_FILE):
            print("⚠️ Prescription file not found.")
            return
        try:
            with open(CFG.PRESCRIPTION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split('|')
                    if len(parts) >= 3:
                        self.patient_name = parts[1].strip()
                        raw = parts[2].split(',')
                        self.allowed_drugs_rx = [d.strip().lower() for d in raw if d.strip()]
                        print(f"📋 Rx Target: {self.allowed_drugs_rx}")
                        break
        except Exception as e: print(f"Rx Error: {e}")

    def verify(self, ai_detected_name):
        """ ตรวจสอบว่าชื่อที่ AI เห็น ตรงกับยาที่หมอสั่งไหม (รองรับ Alias) """
        clean_ai = ai_detected_name.lower().replace('_pack', '').replace('_pill', '')
        
        for rx_drug in self.allowed_drugs_rx:
            # 1. เช็คตรงตัว
            if rx_drug in clean_ai or clean_ai in rx_drug:
                self.verified_drugs.add(rx_drug)
                return True

            # 2. เช็คผ่าน Alias (JSON)
            if rx_drug in self.aliases:
                valid_brands = [b.lower() for b in self.aliases[rx_drug]]
                for brand in valid_brands:
                    if brand in clean_ai:
                        self.verified_drugs.add(rx_drug)
                        return True
        return False

# ================= 🎨 FEATURE ENGINE (DINOv2 + SIFT) =================
class FeatureEngine:
    def __init__(self):
        print("🦕 Loading DINOv2 (ViT-S/14)...")
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.eval().to(device)
            
            # Preprocessing ของ DINOv2
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ DINOv2 Ready (Vec: 384)")
        except Exception as e:
            sys.exit(f"❌ DINOv2 Error: {e}")

        # SIFT Setup
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    @torch.no_grad()
    def get_vector(self, img_rgb):
        t = self.preprocess(img_rgb).unsqueeze(0).to(device)
        vec = self.model(t).flatten().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def get_sift_features(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return des

# ================= 🤖 AI PROCESSOR (THE BRAIN) =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        
        # Load OCR
        print("📖 Loading OCR Engine...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False) # GPU False บน Pi เพื่อความเสถียร
        print("✅ OCR Ready")

        self.session_db_vec = {}
        self.session_db_col = {}
        self.session_db_sift = {}
        self.load_databases()
        
        try:
            self.yolo_pack = YOLO(CFG.MODEL_PACK)
            print("✅ YOLO Ready")
        except: sys.exit("❌ YOLO Error")

        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False
        
        # Stability Counters
        self.verify_counter = 0
        self.last_seen_drug = None

    def load_databases(self):
        def load_pkl(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: return pickle.load(f)
            return {}

        # 1. Load Vectors
        all_vecs = {**load_pkl(CFG.DB_PILLS_VEC), **load_pkl(CFG.DB_PACKS_VEC)}
        count = 0
        for name, vecs in all_vecs.items():
            # โหลดเฉพาะยาที่เกี่ยวข้อง (Performance Optimization)
            # แต่ถ้าอยากโหลดหมดก็เอาเงื่อนไข if ออกได้
            for v in vecs:
                if len(v) == 384: # เช็คว่าเป็น DINOv2 vector จริงๆ
                    self.session_db_vec[f"{name}_{count}"] = (name, np.array(v))
                    count += 1
        
        # 2. Load Colors
        self.session_db_col = {**load_pkl(CFG.DB_PILLS_COL), **load_pkl(CFG.DB_PACKS_COL)}

        # 3. Load SIFT (Optional - ข้ามได้ถ้าเครื่องช้า)
        pass 

    def match(self, vec, img_crop):
        candidates = []
        if not self.session_db_vec: return []

        # Color Sampling
        h, w = img_crop.shape[:2]
        center = img_crop[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        live_col = np.mean(cv2.cvtColor(center, cv2.COLOR_RGB2HSV), axis=(0,1)) if center.size > 0 else None

        for key, (real_name, db_v) in self.session_db_vec.items():
            # 1. Vector Score (Cosine)
            vec_score = np.dot(vec, db_v)
            
            # 2. Color Score
            col_score = 0.5
            if live_col is not None and real_name in self.session_db_col:
                db_col = self.session_db_col[real_name]
                dist = np.linalg.norm(live_col - db_col)
                col_score = np.clip(1.0 - (dist / 100.0), 0, 1)

            # 3. SIFT Score (ลดบทบาทลงเพื่อความเร็ว)
            sift_score = 0.0
            
            final_score = (vec_score * CFG.WEIGHTS['vec']) + \
                          (col_score * CFG.WEIGHTS['col']) + \
                          (sift_score * CFG.WEIGHTS['sift'])
            
            candidates.append((real_name, final_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate
        unique = []
        seen = set()
        for n, s in candidates:
            if n not in seen:
                unique.append((n, s))
                seen.add(n)
            if len(unique) >= 3: break
        return unique

    def process_frame(self, frame):
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        results = self.yolo_pack(img_ai, verbose=False, conf=0.5, imgsz=CFG.AI_SIZE, task='segment')
        
        detections = []
        res = results[0]
        
        if res.masks is None:
            with self.lock: self.results = []
            return

        for box, mask in zip(res.boxes, res.masks):
            # --- Coordinate Scaling ---
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
            scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
            rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
            rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
            
            # Limit to screen
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(CFG.DISPLAY_SIZE[0], rx2), min(CFG.DISPLAY_SIZE[1], ry2)
            
            # ROI Check
            cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
            if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END: continue

            # Contour
            contour = mask.xyn[0].copy()
            contour[:, 0] *= CFG.DISPLAY_SIZE[0]
            contour[:, 1] *= CFG.DISPLAY_SIZE[1]
            contour = contour.astype(np.int32)

            # --- 🛠️ STEP 1: SMART MASKING (แก้คะแนนต่ำ) ---
            crop_rgb = frame[ry1:ry2, rx1:rx2]
            if crop_rgb.size == 0: continue
            
            # สร้าง Mask สีดำเท่าเฟรม
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(full_mask, [contour], 255)
            crop_mask = full_mask[ry1:ry2, rx1:rx2]
            
            # สร้างพื้นหลังสีเทา (Gray 128)
            masked_input = np.full_like(crop_rgb, 128)
            mask_bool = crop_mask > 0
            try:
                masked_input[mask_bool] = crop_rgb[mask_bool]
            except: masked_input = crop_rgb # Fallback

            # --- 🛠️ STEP 2: DINOv2 INFERENCE ---
            vec = self.engine.get_vector(masked_input)
            candidates = self.match(vec, masked_input)
            
            label = "Unknown"
            score = 0.0
            ocr_text = ""
            
            if candidates:
                top_name, top_score = candidates[0]
                
                # --- 🛠️ STEP 3: OCR CHECK (เฉพาะ Packs) ---
                is_pack = "pack" in top_name
                # รัน OCR เฉพาะถ้ามั่นใจว่าเป็น Pack และคะแนน DINOv2 ยังปริ่มๆ หรือต้องการ Confirm
                if is_pack and top_score > 0.4:
                    try:
                        # OCR อ่านจากภาพดิบ (crop_rgb) จะดีกว่าภาพที่ mask แล้ว
                        ocr_res = self.ocr_reader.readtext(crop_rgb, detail=0)
                        ocr_text = " ".join(ocr_res).upper()
                        
                        # เช็คว่า OCR เจอชื่อยาที่หมอสั่งไหม
                        for rx in self.rx_manager.allowed_drugs_rx:
                            # ตัดคำเช่น _500mg ออก
                            keyword = rx.split('_')[0].upper()
                            if keyword in ocr_text:
                                top_score = 1.0 # Boost คะแนนเต็มเลยถ้าอ่านชื่อเจอ
                                label = rx # เปลี่ยนชื่อเป็นชื่อยาตามใบสั่ง
                                print(f"📖 OCR MATCH: {keyword} found!")
                                break
                    except: pass

                # --- 🛠️ STEP 4: VERIFICATION LOGIC ---
                if top_score > CFG.CONF_THRESHOLD:
                    # Counter Check (กันภาพกระพริบ)
                    if self.last_seen_drug == top_name:
                        self.verify_counter += 1
                    else:
                        self.last_seen_drug = top_name
                        self.verify_counter = 0
                    
                    if self.verify_counter >= CFG.VERIFY_FRAMES or top_score == 1.0:
                        label = top_name
                        score = top_score
                        self.rx_manager.verify(label)
                else:
                    self.verify_counter = 0

            detections.append({
                'box': (rx1, ry1, rx2, ry2),
                'contour': contour,
                'label': label,
                'score': score,
                'candidates': candidates,
                'ocr_text': ocr_text
            })
            
        with self.lock: self.results = detections

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self
    def _run(self):
        while not self.stopped:
            with self.lock: frame = self.latest_frame
            if frame is not None:
                try: self.process_frame(frame)
                except Exception as e: print(f"Err: {e}")
            time.sleep(0.01)

# ================= 📷 CAMERA =================
class Camera:
    def __init__(self):
        self.cap = None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            cfg = self.picam.create_preview_configuration(main={"size": CFG.DISPLAY_SIZE, "format": "RGB888"})
            self.picam.configure(cfg)
            self.picam.start()
            self.use_pi = True
            print("📷 PiCamera2: Active")
        except:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, CFG.DISPLAY_SIZE[0])
            self.cap.set(4, CFG.DISPLAY_SIZE[1])
            self.use_pi = False
            print("📷 USB Camera: Active")
    def get(self):
        if self.use_pi: return self.picam.capture_array()
        else:
            ret, frame = self.cap.read()
            if ret: return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
    def stop(self):
        if self.use_pi: self.picam.stop()
        else: self.cap.release()

# ================= 🖥️ UI RENDERER =================
def draw_ui(frame, results, rx_manager):
    h, w = frame.shape[:2]
    
    # 1. Draw Detections
    for det in results:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        score = det['score']
        ocr_txt = det['ocr_text']
        
        # Color: Green=Verified, Yellow=Detected, Red=Unknown
        color = (255, 0, 0) # Red
        if label != "Unknown":
            if label in rx_manager.verified_drugs: color = (0, 255, 0) # Green
            else: color = (255, 255, 0) # Yellow

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label Tag
        tag = f"{label} {score:.0%}"
        t_size = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1-25), (x1+t_size[0]+10, y1), color, -1)
        cv2.putText(frame, tag, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # Show OCR Text if available
        if ocr_txt:
            cv2.rectangle(frame, (x1, y2), (x2, y2+25), (0,0,0), -1)
            cv2.putText(frame, f"OCR: {ocr_txt[:15]}", (x1+5, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    # 2. Dashboard
    db_x, db_y = CFG.UI_ZONE_X_START, 10
    db_w, db_h = w - db_x - 10, CFG.UI_ZONE_Y_END
    
    # Semi-transparent background
    sub = frame[db_y:db_y+db_h, db_x:db_x+db_w]
    white = np.ones(sub.shape, dtype=np.uint8) * 30
    cv2.addWeighted(sub, 0.3, white, 0.7, 0, sub)
    cv2.rectangle(frame, (db_x, db_y), (db_x+db_w, db_y+db_h), (0, 255, 0), 2)
    
    # Patient Info
    cv2.putText(frame, f"RX: {rx_manager.patient_name}", (db_x+10, db_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # Drug List
    y_off = 60
    for drug in rx_manager.allowed_drugs_rx:
        is_done = drug in rx_manager.verified_drugs
        status = " [OK]" if is_done else " [...]"
        col = (0, 255, 0) if is_done else (200, 200, 200)
        cv2.putText(frame, f"- {drug[:15]}{status}", (db_x+10, db_y+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y_off += 25

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    print("✨ PILLTRACK MASTER EDITION: RUNNING...")
    print("   - DINOv2: Enabled (Masked Input)")
    print("   - OCR: Enabled (On Packs)")
    
    while cam.get() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack Master", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack Master", *CFG.DISPLAY_SIZE)
    
    try:
        while True:
            frame = cam.get()
            if frame is None: continue
            
            ai.latest_frame = frame.copy()
            draw_ui(frame, ai.results, ai.rx_manager)
            
            cv2.imshow("PillTrack Master", frame)
            if cv2.waitKey(1) == ord('q'): break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()