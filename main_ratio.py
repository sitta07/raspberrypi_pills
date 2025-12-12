#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  PILLTRACK: STRICT MODE (DUAL PIPELINE)                      ║
║  - Logic: Separate Pack vs Pill Processing                   ║
║  - Pack Strict Rule: Min SIFT Matches > 10                   ║
║  - Pill Strict Rule: Color Correlation > 0.90                ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import pickle
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# ================= ⚙️ CONFIGURATION =================
@dataclass
class Config:
    # --- PATHS ---
    MODEL_PACK: str = 'models/seg_best_process.pt' 
    MODEL_PILL: str = 'models/pills_seg.pt'
    
    # Databases
    DB_PILLS_VEC: str = 'database/db_register/db_pills.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs.pkl'
    DB_PILLS_COL: str = 'database/db_register/colors_pills.pkl'
    DB_PACKS_COL: str = 'database/db_register/colors_packs.pkl'
    
    IMG_DB_FOLDER: str = 'database_images' # สำหรับ SIFT ของ Pack
    PRESCRIPTION_FILE: str = 'prescription.txt'
    
    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416 
    
    # Dashboard Zone
    UI_ZONE_X_START: int = 900 
    UI_ZONE_Y_END: int = 220
    
    # 🎚️ STRICT TUNING THRESHOLDS (ปรับให้เข้มขึ้นที่นี่)
    CONF_THRESHOLD: float = 0.60         # มั่นใจอย่างน้อย 60%
    
    # Hard Rules
    MIN_SIFT_MATCHES: int = 10           # Pack: ต้องเจอจุดเหมือนอย่างน้อย 10 จุด
    MIN_COLOR_SCORE: float = 0.85        # Pill: สีต้องเหมือนต้นฉบับเกิน 85%
    
    # WEIGHTS (แยกประเภท)
    # Pack เน้น SIFT + Vector
    WEIGHTS_PACK: Dict[str, float] = field(default_factory=lambda: {'vec': 0.4, 'sift': 0.6, 'col': 0.0})
    # Pill เน้น Color + Vector (ไม่สน SIFT)
    WEIGHTS_PILL: Dict[str, float] = field(default_factory=lambda: {'vec': 0.5, 'col': 0.5, 'sift': 0.0})

    SIFT_RATIO_TEST: float = 0.70

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (STRICT MODE)")

# ================= 🧠 PRESCRIPTION STATE MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.patient_name = "Unknown"
        self.allowed_drugs = []
        self.verified_drugs = set()
        self.load_prescription()

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
                        raw_drugs = parts[2].split(',')
                        self.allowed_drugs = [d.strip().lower() for d in raw_drugs if d.strip()]
                        print(f"📋 Rx for {self.patient_name}: {self.allowed_drugs}")
                        break 
        except Exception as e: print(f"Rx Error: {e}")

    def is_allowed(self, db_name):
        db_clean = db_name.lower().replace('_pack', '').replace('_pill', '')
        for allowed in self.allowed_drugs:
            if allowed in db_clean or db_clean in allowed: return True
        return False

    def verify(self, name):
        clean = name.lower().replace('_pack', '').replace('_pill', '')
        for allowed in self.allowed_drugs:
            if allowed in clean or clean in allowed:
                self.verified_drugs.add(allowed)

# ================= 🎨 FEATURE ENGINE (Vec + Color + SIFT) =================
class FeatureEngine:
    def __init__(self):
        # 1. ResNet50 for Vectors
        try:
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
            self.model = torch.nn.Sequential(*list(base.children())[:-1])
            self.model.eval().to(device)
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e: sys.exit(f"❌ ResNet Error: {e}")

        # 2. SIFT Engine
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

    def get_color_hist(self, img_rgb):
        # ใช้ Hue-Saturation Histogram (ทนต่อแสงสว่างเปลี่ยน)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # H: 0-179, S: 0-255. ใช้ 8x8 bins
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def compare_color(self, hist1, hist2):
        # Correlation method: 1.0 = เหมือนเป๊ะ, < 0.8 = เริ่มไม่เหมือน
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# ================= 🤖 AI PROCESSOR (DUAL PIPELINE) =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        
        # แยก Session DB ชัดเจน
        self.db_packs = {'vec': {}, 'sift': {}, 'col': {}}
        self.db_pills = {'vec': {}, 'col': {}} # Pills ไม่ใช้ SIFT
        
        self.load_and_filter_db()
        
        try:
            print("🔄 Loading Models...")
            self.model_pack = YOLO(CFG.MODEL_PACK)
            self.model_pill = YOLO(CFG.MODEL_PILL)
            print("✅ Dual Models Loaded (Pack & Pill)")
        except Exception as e: sys.exit(f"❌ YOLO Error: {e}")

        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False

    def load_and_filter_db(self):
        print("🔍 Building Segmented Session Database...")
        def load_pkl(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: return pickle.load(f)
            return {}

        # 1. Load Packs Data
        raw_vecs_pack = load_pkl(CFG.DB_PACKS_VEC)
        raw_cols_pack = load_pkl(CFG.DB_PACKS_COL)
        
        for name, vecs in raw_vecs_pack.items():
            if self.rx_manager.is_allowed(name):
                # Flatten list of vecs
                for i, v in enumerate(vecs):
                    self.db_packs['vec'][f"{name}#{i}"] = (name, np.array(v))
        
        # Load SIFT for Packs only
        if os.path.exists(CFG.IMG_DB_FOLDER):
            for drug_name in os.listdir(CFG.IMG_DB_FOLDER):
                if not self.rx_manager.is_allowed(drug_name): continue
                if '_pack' not in drug_name.lower() and 'pack' not in drug_name.lower(): continue # กรองเอาแต่ Pack

                drug_path = os.path.join(CFG.IMG_DB_FOLDER, drug_name)
                if os.path.isdir(drug_path):
                    des_list = []
                    for img_file in sorted(os.listdir(drug_path))[:3]:
                        if img_file.lower().endswith(('jpg', 'png', 'jpeg')):
                            img = cv2.imread(os.path.join(drug_path, img_file))
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                des = self.engine.get_sift_features(img)
                                if des is not None: des_list.append(des)
                    if des_list:
                        self.db_packs['sift'][drug_name] = des_list
                        print(f"   + Pack SIFT: {drug_name}")

        # 2. Load Pills Data
        raw_vecs_pill = load_pkl(CFG.DB_PILLS_VEC)
        raw_cols_pill = load_pkl(CFG.DB_PILLS_COL)

        for name, vecs in raw_vecs_pill.items():
            if self.rx_manager.is_allowed(name):
                for i, v in enumerate(vecs):
                    self.db_pills['vec'][f"{name}#{i}"] = (name, np.array(v))
                
                # Assume Color DB store hist or compatible data
                if name in raw_cols_pill:
                    self.db_pills['col'][name] = raw_cols_pill[name]

    # --- PACK MATCHING LOGIC (เน้น SIFT) ---
    def match_pack(self, vec, img_crop):
        candidates = []
        if not self.db_packs['vec']: return []
        
        query_sift = self.engine.get_sift_features(img_crop)

        for key, (real_name, db_v) in self.db_packs['vec'].items():
            # 1. SIFT Check (HARD RULE)
            sift_score = 0.0
            matches_count = 0
            
            if real_name in self.db_packs['sift'] and query_sift is not None:
                max_matches = 0
                for ref_des in self.db_packs['sift'][real_name]:
                    try:
                        matches = self.engine.bf.knnMatch(query_sift, ref_des, k=2)
                        good = [m for m, n in matches if m.distance < CFG.SIFT_RATIO_TEST * n.distance]
                        if len(good) > max_matches: max_matches = len(good)
                    except: pass
                matches_count = max_matches
                sift_score = min(max_matches / 20.0, 1.0) # Normalize 20 matches = 100%

            # ⛔ STRICT FILTER: ถ้าเจอน้อยกว่า 10 จุด ตัดทิ้งทันที
            if matches_count < CFG.MIN_SIFT_MATCHES:
                continue 

            # 2. Vector Check
            vec_score = np.dot(vec, db_v)
            
            # Combine Score (No Color for Pack)
            final_score = (vec_score * CFG.WEIGHTS_PACK['vec']) + \
                          (sift_score * CFG.WEIGHTS_PACK['sift'])
            
            candidates.append((real_name, final_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    # --- PILL MATCHING LOGIC (เน้น Color + Vec) ---
    def match_pill(self, vec, img_crop):
        candidates = []
        if not self.db_pills['vec']: return []
        
        query_hist = self.engine.get_color_hist(img_crop)

        seen_drugs = set()
        
        for key, (real_name, db_v) in self.db_pills['vec'].items():
            if real_name in seen_drugs: continue # Optimization: One check per drug type

            # 1. Color Check (HARD RULE)
            col_score = 0.0
            if real_name in self.db_pills['col']:
                # *ต้องแน่ใจว่า DB เก็บ format เดียวกัน*
                # ถ้า DB เก็บแค่ RGB tuple โค้ดนี้ต้องแก้ให้ extract hist จาก DB ภาพต้นฉบับแทน
                # ในที่นี้สมมติว่า DB มีข้อมูลที่ compare ได้ หรือใช้ placeholder ถ้าไม่มี
                try:
                    # สมมติ db_col เก็บ histogram
                    col_score = self.engine.compare_color(query_hist, self.db_pills['col'][real_name])
                except:
                    col_score = 0.5 # Fallback ถ้าข้อมูลไม่ตรง

            # ⛔ STRICT FILTER: ถ้าสีไม่เหมือน (Correlation < 0.85) ตัดทิ้ง
            if col_score < CFG.MIN_COLOR_SCORE:
                continue

            # 2. Vector Check
            vec_score = np.dot(vec, db_v)
            
            final_score = (vec_score * CFG.WEIGHTS_PILL['vec']) + \
                          (col_score * CFG.WEIGHTS_PILL['col'])
            
            candidates.append((real_name, final_score))
            seen_drugs.add(real_name)

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def process_frame(self, frame):
        # Resize for Inference
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        
        all_detections = []

        # ================= LOOP 1: PACK DETECTION =================
        res_pack = self.model_pack(img_ai, verbose=False, conf=0.5, imgsz=CFG.AI_SIZE, task='segment')[0]
        if res_pack.masks:
            for box, mask in zip(res_pack.boxes, res_pack.masks):
                det = self._extract_and_id(frame, box, mask, type='pack')
                if det: all_detections.append(det)

        # ================= LOOP 2: PILL DETECTION =================
        res_pill = self.model_pill(img_ai, verbose=False, conf=0.5, imgsz=CFG.AI_SIZE, task='segment')[0]
        if res_pill.masks:
            for box, mask in zip(res_pill.boxes, res_pill.masks):
                # TODO: เพิ่ม Logic เช็ค Overlap ว่าถ้ายาเม็ดอยู่ในพื้นที่ Pack ให้ ignore (ทำทีหลังได้)
                det = self._extract_and_id(frame, box, mask, type='pill')
                if det: all_detections.append(det)
            
        with self.lock: self.results = all_detections

    def _extract_and_id(self, frame, box, mask, type='pack'):
        # Helper function to crop and identify
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
        scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
        rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
        
        # Dashboard Zone Filter
        cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
        if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END: return None

        # Crop
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10: return None

        # Identification
        vec = self.engine.get_vector(crop)
        
        if type == 'pack':
            candidates = self.match_pack(vec, crop)
        else:
            candidates = self.match_pill(vec, crop)

        # Final Decision
        if candidates and candidates[0][1] > CFG.CONF_THRESHOLD:
            label = candidates[0][0]
            score = candidates[0][1]
            self.rx_manager.verify(label)
        else:
            label = "Unknown"
            score = 0.0

        # Contour for Display
        contour = mask.xyn[0]
        contour[:, 0] *= CFG.DISPLAY_SIZE[0]
        contour[:, 1] *= CFG.DISPLAY_SIZE[1]
        
        return {
            'contour': contour.astype(np.int32),
            'label': label,
            'score': score,
            'type': type
        }

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

# ================= 📷 CAMERA (RGB888) =================
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
            print("📷 PiCamera2: RGB888 Source Locked")
        except:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, CFG.DISPLAY_SIZE[0])
            self.cap.set(4, CFG.DISPLAY_SIZE[1])
            self.use_pi = False
            print("📷 USB Camera: Converting BGR to RGB888")
            
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
    overlay = frame.copy()
    
    for det in results:
        contour = det['contour']
        label = det['label']
        dtype = det['type']
        
        # Color Coding:
        # Green = Verified
        # Yellow = Detected but Unknown
        # Cyan = Pack, Magenta = Pill (Debug)
        
        if label != "Unknown":
            color = (0, 255, 0) # Green
        else:
            color = (255, 0, 0) if dtype == 'pack' else (255, 100, 100) # Red-ish

        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(overlay, [contour], True, (255, 255, 255), 2)
        
        # Label
        if label != "Unknown":
            top_point = tuple(contour[contour[:, 1].argmin()])
            tx, ty = top_point
            cv2.putText(frame, f"{label} ({det['score']:.2f})", (tx, ty-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Dashboard
    db_x = CFG.UI_ZONE_X_START
    cv2.rectangle(frame, (db_x, 0), (w, CFG.UI_ZONE_Y_END), (0,0,0), -1)
    cv2.putText(frame, f"Patient: {rx_manager.patient_name}", (db_x+10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    y_off = 70
    for drug in rx_manager.allowed_drugs:
        status = "[OK]" if drug in rx_manager.verified_drugs else "[...]"
        col = (0, 255, 0) if drug in rx_manager.verified_drugs else (100, 100, 100)
        cv2.putText(frame, f"{drug} {status}", (db_x+10, y_off), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        y_off += 30

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    print("✨ System Ready. Waiting for video...")
    while cam.get() is None: time.sleep(0.1)
    
    try:
        while True:
            frame = cam.get()
            if frame is None: continue
            
            ai.latest_frame = frame.copy()
            draw_ui(frame, ai.results, ai.rx_manager)
            
            cv2.imshow("PillTrack: Dual Pipeline Strict Mode", frame) # ถ้าจอไม่รองรับ RGB อาจสีเพี้ยนในหน้าจอ debug แต่ logic ถูก
            
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt: pass
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()