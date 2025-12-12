#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  PILLTRACK: SEGMENTATION MASTER (RGB STRICT)                 ║
║  - Model Type: YOLOv8 Segmentation (Masks/Contours)          ║
║  - Feature: Prescription Lock + SIFT + Vector + Color        ║
║  - Display: Real-time Mask Overlay (RGB888)                  ║
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
    # --- PATHS (แก้ให้ตรงกับเครื่องคุณ) ---
    # โมเดลต้องเป็น Type Segmentation
    MODEL_PACK: str = 'models/seg_best_process.pt' 
    MODEL_PILL: str = 'models/pills_seg.pt'
    
    # Databases
    DB_PILLS_VEC: str = 'database/db_register/db_pills.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs.pkl'
    DB_PILLS_COL: str = 'database/db_register/colors_pills.pkl'
    DB_PACKS_COL: str = 'database/db_register/colors_packs.pkl'
    
    IMG_DB_FOLDER: str = 'database_images' # For SIFT
    PRESCRIPTION_FILE: str = 'prescription.txt'
    
    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416 # Resize ก่อนเข้าโมเดล
    
    # 🚫 EXCLUSION ZONE (Dashboard Area)
    UI_ZONE_X_START: int = 900 
    UI_ZONE_Y_END: int = 220
    
    # 🎚️ TUNING THRESHOLDS
    CONF_THRESHOLD: float = 0.45
    
    # WEIGHTS FUSION: Vector 50%, Color 30%, SIFT 20%
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {'vec': 0.5, 'col': 0.3, 'sift': 0.2}) 
    
    # SIFT Tuning
    SIFT_RATIO_TEST: float = 0.85

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (SEGMENTATION MODE)")

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

# ================= 🤖 AI PROCESSOR (SEGMENTATION LOGIC) =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        
        # Session Databases
        self.session_db_vec = {} 
        self.session_db_col = {}
        self.session_db_sift = {}
        
        self.load_and_filter_db()
        
        try:
            # Load as Segmentation Models
            self.yolo_pack = YOLO(CFG.MODEL_PACK) if os.path.exists(CFG.MODEL_PACK) else YOLO('yolov8n-seg.pt')
            print("✅ YOLO Segmentation Models Loaded")
        except: sys.exit("❌ YOLO Error")

        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False

    def load_and_filter_db(self):
        print("🔍 Building Session Database...")
        def load_pkl(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: return pickle.load(f)
            return {}

        # 1. Load Vectors
        all_vecs = {**load_pkl(CFG.DB_PILLS_VEC), **load_pkl(CFG.DB_PACKS_VEC)}
        count = 0
        for name, vecs in all_vecs.items():
            if self.rx_manager.is_allowed(name):
                for v in vecs:
                    self.session_db_vec[f"{name}_{count}"] = (name, np.array(v)) 
                    count += 1
        
        # 2. Load Colors
        all_cols = {**load_pkl(CFG.DB_PILLS_COL), **load_pkl(CFG.DB_PACKS_COL)}
        for name, col in all_cols.items():
            if self.rx_manager.is_allowed(name):
                self.session_db_col[name] = col

        # 3. Load SIFT
        if os.path.exists(CFG.IMG_DB_FOLDER):
            for drug_name in os.listdir(CFG.IMG_DB_FOLDER):
                if not self.rx_manager.is_allowed(drug_name): continue
                drug_path = os.path.join(CFG.IMG_DB_FOLDER, drug_name)
                if os.path.isdir(drug_path):
                    descriptors_list = []
                    for img_file in sorted(os.listdir(drug_path))[:3]:
                        if img_file.lower().endswith(('jpg', 'png', 'jpeg')):
                            img = cv2.imread(os.path.join(drug_path, img_file))
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                des = self.engine.get_sift_features(img)
                                if des is not None: descriptors_list.append(des)
                    if descriptors_list:
                        self.session_db_sift[drug_name] = descriptors_list
                        print(f"   + SIFT: {drug_name}")

    def compute_sift_score(self, query_des, target_name):
        if query_des is None or target_name not in self.session_db_sift: return 0.0
        max_matches = 0
        for ref_des in self.session_db_sift[target_name]:
            try:
                matches = self.engine.bf.knnMatch(query_des, ref_des, k=2)
                good = [m for m, n in matches if m.distance < CFG.SIFT_RATIO_TEST * n.distance]
                if len(good) > max_matches: max_matches = len(good)
            except: pass
        return min(max_matches / 15.0, 1.0)

    def match(self, vec, img_crop):
        candidates = []
        if not self.session_db_vec: return []

        query_sift_des = self.engine.get_sift_features(img_crop)

        for key, (real_name, db_v) in self.session_db_vec.items():
            vec_score = np.dot(vec, db_v)
            col_score = 0.5 # Placeholder (If you have real color hist, compare here)
            sift_score = self.compute_sift_score(query_sift_des, real_name)
            
            final_score = (vec_score * CFG.WEIGHTS['vec']) + \
                          (col_score * CFG.WEIGHTS['col']) + \
                          (sift_score * CFG.WEIGHTS['sift'])
                          
            candidates.append((real_name, final_score, vec_score, sift_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        unique = []
        seen = set()
        for n, fs, vs, ss in candidates:
            if n not in seen:
                unique.append((n, fs, vs, ss))
                seen.add(n)
            if len(unique) >= 5: break
        return unique

    def process_frame(self, frame):
        # Resize for Inference
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        
        # Inference (Segmentation Task)
        # Note: We use the Pack model to detect both or separate if preferred.
        # Assuming model handles segmentation logic.
        results = self.yolo_pack(img_ai, verbose=False, conf=0.4, imgsz=CFG.AI_SIZE, task='segment')
        
        detections = []
        res = results[0]
        
        if res.masks is None:
            with self.lock: self.results = []
            return

        # Iterate over Masks and Boxes
        for box, mask in zip(res.boxes, res.masks):
            # 1. Bounding Box (For Cropping Logic)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Scale Box to Display Size
            scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
            scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
            
            rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
            rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
            
            # ROI Check
            cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
            if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END: continue 
            
            # 2. Extract Polygon (For Drawing)
            # mask.xyn gives normalized coordinates (0-1). We scale them up.
            contour = mask.xyn[0] # Get first contour
            contour[:, 0] *= CFG.DISPLAY_SIZE[0]
            contour[:, 1] *= CFG.DISPLAY_SIZE[1]
            contour = contour.astype(np.int32)
            
            # 3. Crop for Recognition (Using Box)
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0: continue
            
            # 4. Recognize
            vec = self.engine.get_vector(crop)
            candidates = self.match(vec, crop)
            
            if candidates:
                top_name, top_score, _, _ = candidates[0]
                label = top_name if top_score > CFG.CONF_THRESHOLD else "Unknown"
                if label != "Unknown": self.rx_manager.verify(label)
            else:
                label = "Unknown"
                candidates = []

            detections.append({
                'box': (rx1, ry1, rx2, ry2),
                'contour': contour, # Store the shape!
                'label': label,
                'score': candidates[0][1] if candidates else 0.0,
                'candidates': candidates
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

# ================= 🖥️ UI RENDERER (MASKS + OVERLAYS) =================
def draw_ui(frame, results, rx_manager):
    h, w = frame.shape[:2]
    
    # 1. Draw Masks (Segmentation Overlay)
    overlay = frame.copy()
    for det in results:
        contour = det['contour']
        label = det['label']
        
        # Color: Green if Known/Verified, Red if Unknown
        color = (0, 255, 0) if label != "Unknown" else (255, 0, 0)
        
        # Draw Filled Polygon on Overlay
        cv2.fillPoly(overlay, [contour], color)
        # Draw Contour Line on Overlay
        cv2.polylines(overlay, [contour], True, color, 2)
    
    # Apply Transparency (Alpha Blend)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # 2. Draw Labels & Boxes
    # for det in results:
    #     x1, y1, x2, y2 = det['box']
    #     label = det['label']
    #     score = det['score']
    #     candidates = det['candidates']
    #     contour = det['contour']
        
    #     # Get topmost point of contour for label placement
    #     top_point = tuple(contour[contour[:, 1].argmin()])
    #     tx, ty = top_point
        
    #     # Label Background
    #     color = (0, 255, 0) if label != "Unknown" else (255, 0, 0)
    #     cv2.rectangle(frame, (tx, ty-25), (tx + len(label)*15, ty), color, -1)
    #     cv2.putText(frame, f"{label} {score:.0%}", (tx+5, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
    #     # Candidate Panel
    #     panel_x = x2 + 5 if x2 + 180 < w else x1 - 185
    #     panel_y = y1
    #     cv2.rectangle(frame, (panel_x, panel_y), (panel_x+180, panel_y+60), (0,0,0), -1)
    #     cv2.putText(frame, "AI CANDIDATES:", (panel_x+5, panel_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
    #     for i, (c_name, c_score, c_vec, c_sift) in enumerate(candidates[:3]):
    #         d_name = (c_name[:9] + '.') if len(c_name) > 9 else c_name
    #         c_col = (0, 255, 0) if c_score > CFG.CONF_THRESHOLD else (255, 100, 0)
    #         line = f"{i+1}.{d_name} {c_score:.2f} (S:{c_sift:.1f})"
    #         cv2.putText(frame, line, (panel_x+5, panel_y+30+(i*15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_col, 1)

    # 3. Dashboard
    db_x, db_y = CFG.UI_ZONE_X_START, 10
    db_w, db_h = w - db_x - 10, CFG.UI_ZONE_Y_END
    sub = frame[db_y:db_y+db_h, db_x:db_x+db_w]
    white = np.ones(sub.shape, dtype=np.uint8) * 30
    cv2.addWeighted(sub, 0.3, white, 0.7, 0, sub)
    cv2.rectangle(frame, (db_x, db_y), (db_x+db_w, db_y+db_h), (0, 255, 0), 2)
    cv2.putText(frame, f"RX: {rx_manager.patient_name}", (db_x+10, db_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    y_off = 60
    for drug in rx_manager.allowed_drugs:
        status = " [OK]" if drug in rx_manager.verified_drugs else " [...]"
        col = (0, 255, 0) if drug in rx_manager.verified_drugs else (200, 200, 200)
        cv2.putText(frame, f"- {drug}{status}", (db_x+10, db_y+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y_off += 25

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    print("✨ Waiting for RGB888 feed (Segmentation Mode)...")
    while cam.get() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack Segmentation", *CFG.DISPLAY_SIZE)
    
    try:
        while True:
            frame = cam.get()
            if frame is None: continue
            
            ai.latest_frame = frame.copy()
            draw_ui(frame, ai.results, ai.rx_manager)
            
            # Display STRICT RGB (Requires display to support it, usually works fine)
            cv2.imshow("PillTrack Segmentation", frame)
            
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()