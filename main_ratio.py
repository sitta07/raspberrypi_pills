#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  PILLTRACK: DUAL PIPELINE ARCHITECTURE (HIGH ACCURACY)       ║
║  - Pipeline 1: Pack Detection (Model A -> DB Packs)          ║
║  - Pipeline 2: Pill Detection (Model B -> DB Pills)          ║
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
    # แยกโมเดลกันทำงานชัดเจน
    MODEL_PACK_PATH: str = 'models/seg_best_process.pt' 
    MODEL_PILL_PATH: str = 'models/pills_seg.pt'
    
    # แยก Database ออกจากกัน
    DB_PILLS_VEC: str = 'database/db_register/db_pills.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs.pkl'
    DB_PILLS_COL: str = 'database/db_register/colors_pills.pkl'
    DB_PACKS_COL: str = 'database/db_register/colors_packs.pkl'
    
    IMG_DB_FOLDER: str = 'database_images' # SIFT Images
    PRESCRIPTION_FILE: str = 'prescription.txt'
    
    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 640 # เพิ่มความละเอียดเพื่อให้ Model 2 ตัวทำงานแม่นขึ้น (เดิม 416)
    
    # 🚫 EXCLUSION ZONE (Dashboard Area)
    UI_ZONE_X_START: int = 900 
    UI_ZONE_Y_END: int = 220
    
    # 🎚️ TUNING (แยก Threshold ได้อิสระ)
    CONF_PACK: float = 0.40
    CONF_PILL: float = 0.50
    
    # SIFT Tuning
    SIFT_RATIO_TEST: float = 0.75

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (DUAL PIPELINE MODE)")

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

# ================= 🎨 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        # 1. ResNet50
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

# ================= 🤖 AI PROCESSOR (DUAL CORE) =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        
        # --- SEPARATED DATABASES ---
        # เลน Pack
        self.db_packs_vec = {} 
        self.db_packs_sift = {}
        self.db_packs_col = {}
        
        # เลน Pill
        self.db_pills_vec = {}
        self.db_pills_col = {} # Pill ไม่เน้น SIFT
        
        self.load_separate_dbs()
        
        try:
            print("⏳ Loading Models...")
            # Load 2 Separate Models
            self.model_pack = YOLO(CFG.MODEL_PACK_PATH)
            self.model_pill = YOLO(CFG.MODEL_PILL_PATH)
            print("✅ DUAL Models Loaded (Pack & Pill)")
        except Exception as e: sys.exit(f"❌ Model Error: {e}")

        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False

    def load_separate_dbs(self):
        print("🔍 Building Dual Database...")
        def load_pkl(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: return pickle.load(f)
            return {}

        # 1. Load PACKS
        packs_raw = load_pkl(CFG.DB_PACKS_VEC)
        count_pack = 0
        for name, vecs in packs_raw.items():
            if self.rx_manager.is_allowed(name):
                for v in vecs:
                    self.db_packs_vec[f"{name}_{count_pack}"] = (name, np.array(v))
                    count_pack += 1
        
        # Load SIFT for PACKS ONLY
        if os.path.exists(CFG.IMG_DB_FOLDER):
            for drug_name in os.listdir(CFG.IMG_DB_FOLDER):
                # โหลดเฉพาะตัวที่เป็น Pack (สมมติชื่อไฟล์หรือ Folder บ่งบอก หรือ Rx check)
                if not self.rx_manager.is_allowed(drug_name): continue
                
                drug_path = os.path.join(CFG.IMG_DB_FOLDER, drug_name)
                if os.path.isdir(drug_path):
                    descriptors_list = []
                    for img_file in sorted(os.listdir(drug_path))[:2]: # จำกัดจำนวนรูป Reference
                        if img_file.lower().endswith(('jpg', 'png')):
                            img = cv2.imread(os.path.join(drug_path, img_file))
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                des = self.engine.get_sift_features(img)
                                if des is not None: descriptors_list.append(des)
                    if descriptors_list:
                        self.db_packs_sift[drug_name] = descriptors_list
        
        # 2. Load PILLS
        pills_raw = load_pkl(CFG.DB_PILLS_VEC)
        count_pill = 0
        for name, vecs in pills_raw.items():
            if self.rx_manager.is_allowed(name):
                for v in vecs:
                    self.db_pills_vec[f"{name}_{count_pill}"] = (name, np.array(v))
                    count_pill += 1

        print(f"   + Packs Loaded: {count_pack} vectors")
        print(f"   + Pills Loaded: {count_pill} vectors")

    def compute_sift_score(self, query_des, target_name, db_sift):
        if query_des is None or target_name not in db_sift: return 0.0
        max_matches = 0
        for ref_des in db_sift[target_name]:
            try:
                matches = self.engine.bf.knnMatch(query_des, ref_des, k=2)
                good = [m for m, n in matches if m.distance < CFG.SIFT_RATIO_TEST * n.distance]
                if len(good) > max_matches: max_matches = len(good)
            except: pass
        
        # Normalize score (สมมติว่าเจอ 10 จุดคือเต็ม 100%)
        return min(max_matches / 10.0, 1.0)

    # --- PIPELINE 1: PACK MATCHING ---
    def match_pack(self, vec, img_crop):
        candidates = []
        if not self.db_packs_vec: return []

        query_sift = self.engine.get_sift_features(img_crop)

        for key, (real_name, db_v) in self.db_packs_vec.items():
            # Vector Similarity
            vec_score = np.dot(vec, db_v)
            
            # SIFT Similarity (เฉพาะ Pack ที่เน้น)
            sift_score = self.compute_sift_score(query_sift, real_name, self.db_packs_sift)
            
            # Pack Weight: ให้ค่า SIFT สูงหน่อย
            final_score = (vec_score * 0.4) + (sift_score * 0.6) 
            
            candidates.append((real_name, final_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:1] # เอาตัวที่ดีที่สุด

    # --- PIPELINE 2: PILL MATCHING ---
    def match_pill(self, vec, img_crop):
        candidates = []
        if not self.db_pills_vec: return []

        # Pill ไม่ทำ SIFT เพื่อประหยัดแรงและลด Noise
        for key, (real_name, db_v) in self.db_pills_vec.items():
            vec_score = np.dot(vec, db_v)
            
            # Color Check (ใส่ Logic ง่ายๆ ไว้ก่อน หรือรอฟังก์ชันเทียบสี)
            # Pill Weight: Vector 100% (เดี๋ยวเพิ่ม Color ใน step ถัดไป)
            final_score = vec_score 
            
            candidates.append((real_name, final_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:1]

    def process_frame(self, frame):
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        combined_detections = []

        # ---------------------------------------------------------
        # PIPELINE 1: Detect PACKS
        # ---------------------------------------------------------
        results_pack = self.model_pack(img_ai, verbose=False, conf=CFG.CONF_PACK, imgsz=CFG.AI_SIZE, task='segment')
        if results_pack[0].masks is not None:
            combined_detections.extend(self._extract_detections(frame, results_pack[0], mode='pack'))

        # ---------------------------------------------------------
        # PIPELINE 2: Detect PILLS
        # ---------------------------------------------------------
        results_pill = self.model_pill(img_ai, verbose=False, conf=CFG.CONF_PILL, imgsz=CFG.AI_SIZE, task='segment')
        if results_pill[0].masks is not None:
            combined_detections.extend(self._extract_detections(frame, results_pill[0], mode='pill'))
            
        with self.lock: self.results = combined_detections

    def _extract_detections(self, frame, res, mode):
        detections = []
        # Loop ผ่าน Results
        for box, mask in zip(res.boxes, res.masks):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Scale กลับมาขนาดจอจริง
            scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
            scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
            rx1, ry1 = int(x1*scale_x), int(y1*scale_y)
            rx2, ry2 = int(x2*scale_x), int(y2*scale_y)
            
            # Dashboard Check
            cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
            if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END: continue 
            
            # Contour
            contour = mask.xyn[0]
            contour[:, 0] *= CFG.DISPLAY_SIZE[0]
            contour[:, 1] *= CFG.DISPLAY_SIZE[1]
            contour = contour.astype(np.int32)
            
            # Crop & Recognize
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0: continue
            
            vec = self.engine.get_vector(crop)
            
            # *** KEY: แยก Logic การ Match ***
            if mode == 'pack':
                candidates = self.match_pack(vec, crop)
                display_color = (0, 255, 255) # สีเหลืองสำหรับ Pack
            else:
                candidates = self.match_pill(vec, crop)
                display_color = (255, 0, 255) # สีม่วงสำหรับ Pill
            
            # Threshold Check
            label = "Unknown"
            score = 0.0
            if candidates:
                top_name, top_score = candidates[0]
                # ใช้ Threshold แยกกันได้
                thresh = CFG.CONF_PACK if mode == 'pack' else CFG.CONF_PILL
                if top_score > thresh:
                    label = top_name
                    score = top_score
                    self.rx_manager.verify(label)

            detections.append({
                'type': mode, # เก็บ Type ไว้ใช้แสดงผล
                'box': (rx1, ry1, rx2, ry2),
                'contour': contour,
                'label': label,
                'score': score,
                'color': display_color
            })
        return detections

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
    
    # 1. Draw Detections
    for det in results:
        contour = det['contour']
        label = det['label']
        score = det['score']
        dtype = det['type']
        
        # Color Logic: Known vs Unknown
        base_color = det['color'] if label != "Unknown" else (255, 0, 0)
        
        # Draw Mask
        cv2.fillPoly(overlay, [contour], base_color)
        cv2.polylines(overlay, [contour], True, (255,255,255), 2)
        
        # Draw Label (บนสุดของ Contour)
        top_point = tuple(contour[contour[:, 1].argmin()])
        tx, ty = top_point
        
        text = f"[{dtype.upper()}] {label} ({score:.2f})"
        cv2.putText(frame, text, (tx, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # 2. Dashboard
    db_x, db_y = CFG.UI_ZONE_X_START, 10
    db_w, db_h = w - db_x - 10, CFG.UI_ZONE_Y_END
    
    # Dashboard Background
    cv2.rectangle(frame, (db_x, db_y), (db_x+db_w, db_y+db_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (db_x, db_y), (db_x+db_w, db_y+db_h), (0, 255, 0), 2)
    
    cv2.putText(frame, f"RX: {rx_manager.patient_name}", (db_x+10, db_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    y_off = 70
    for drug in rx_manager.allowed_drugs:
        is_verified = drug in rx_manager.verified_drugs
        status_icon = " [OK]" if is_verified else " [...]"
        text_col = (0, 255, 0) if is_verified else (150, 150, 150)
        
        cv2.putText(frame, f"- {drug}{status_icon}", (db_x+10, db_y+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 1)
        y_off += 30

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    print("✨ System Ready. Waiting for video...")
    while cam.get() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack: Dual Pipeline", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack: Dual Pipeline", *CFG.DISPLAY_SIZE)
    
    try:
        while True:
            frame = cam.get()
            if frame is None: continue
            
            ai.latest_frame = frame.copy()
            draw_ui(frame, ai.results, ai.rx_manager)
            
            cv2.imshow("PillTrack: Dual Pipeline", frame)
            
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()