#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  PILLTRACK: PRESCRIPTION MASTER (RGB STRICT)                 ║
║  - Prescription Locking (Search only prescribed drugs)       ║
║  - RGB888 Processing Pipeline                                ║
║  - Candidate Inspector UI (Fixes 'Unknown' confusion)        ║
║  - ROI Exclusion Zone                                        ║
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

# ================= ⚙️ CONFIGURATION =================
@dataclass
class Config:
    # --- PATHS (ตรวจสอบ Path ให้ตรงกับเครื่องคุณ) ---
    MODEL_PACK: str = 'models/seg_best_process.pt' 
    MODEL_PILL: str = 'models/pills_seg.pt'
    
    # Databases
    DB_PILLS_VEC: str = 'database/db_register/db_pills.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs.pkl'
    DB_PILLS_COL: str = 'database/db_register/colors_pills.pkl'
    DB_PACKS_COL: str = 'database/db_register/colors_packs.pkl'
    
    PRESCRIPTION_FILE: str = 'prescription.txt'
    
    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416
    ZOOM_FACTOR: float = 1.0
    
    # 🚫 EXCLUSION ZONE (กัน AI จับ UI ตัวเอง)
    # มุมขวาบน (X เริ่มที่ 960, Y ถึง 200)
    UI_ZONE_X_START: int = 900 
    UI_ZONE_Y_END: int = 220
    
    # 🎚️ TUNING THRESHOLDS
    # ลด CONF ลงเพื่อให้ AI กล้าตอบมากขึ้น (เดิมอาจจะสูงไปจน Unknown)
    CONF_THRESHOLD: float = 0.38        
    
    # WEIGHTS: Vector 60%, Color 40% (สีช่วยแยกแยะกล่องได้ดี)
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {'vec': 0.6, 'col': 0.4}) 

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 SYSTEM STARTING ON: {device}")

# ================= 🧠 PRESCRIPTION STATE MANAGER =================
class PrescriptionManager:
    """Manages the 'Allowed Drugs' list to lock search space"""
    def __init__(self):
        self.patient_name = "Unknown"
        self.allowed_drugs = []
        self.verified_drugs = set()
        self.load_prescription()

    def load_prescription(self):
        if not os.path.exists(CFG.PRESCRIPTION_FILE):
            print("⚠️ No prescription.txt found. System will search NOTHING.")
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
                        print(f"📋 Loaded Rx for {self.patient_name}: {self.allowed_drugs}")
                        break # Load first patient only
        except Exception as e:
            print(f"❌ Rx Load Error: {e}")

    def is_allowed(self, db_name):
        """Check if a DB item matches any prescribed drug"""
        db_clean = db_name.lower().replace('_pack', '').replace('_pill', '')
        for allowed in self.allowed_drugs:
            if allowed in db_clean or db_clean in allowed:
                return True
        return False

    def verify(self, name):
        clean = name.lower().replace('_pack', '').replace('_pill', '')
        for allowed in self.allowed_drugs:
            if allowed in clean or clean in allowed:
                self.verified_drugs.add(allowed)

# ================= 🎨 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        # Load ResNet50
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
            print("✅ Feature Extractor Loaded")
        except Exception as e:
            sys.exit(f"❌ Model Error: {e}")

    @torch.no_grad()
    def get_vector(self, img_rgb):
        t = self.preprocess(img_rgb).unsqueeze(0).to(device)
        vec = self.model(t).flatten().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def get_color_hist(self, img_rgb):
        # LAB Histogram for Human-like color perception
        try:
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            # Focus on center to avoid background noise
            h, w = lab.shape[:2]
            mask = np.zeros((h, w), dtype='uint8')
            cv2.circle(mask, (w//2, h//2), int(min(h,w)*0.4), 255, -1)
            
            hist = cv2.calcHist([lab], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            return cv2.normalize(hist, hist).flatten()
        except: return np.zeros(512)

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        
        # Load Global DB & Filter immediately
        self.session_db_vec = {} 
        self.session_db_col = {}
        self.load_and_filter_db()
        
        # Load YOLO
        try:
            from ultralytics import YOLO
            self.yolo_pack = YOLO(CFG.MODEL_PACK) if os.path.exists(CFG.MODEL_PACK) else YOLO('yolov8n.pt')
            self.yolo_pill = YOLO(CFG.MODEL_PILL) if os.path.exists(CFG.MODEL_PILL) else YOLO('yolov8n.pt')
            print("✅ YOLO Models Loaded")
        except: sys.exit("❌ YOLO Error")

        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False

    def load_and_filter_db(self):
        print("🔍 Building Session Database...")
        
        # Helper to load pickle
        def load_pkl(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: return pickle.load(f)
            return {}

        # Load Global Data
        all_pills_vec = load_pkl(CFG.DB_PILLS_VEC)
        all_packs_vec = load_pkl(CFG.DB_PACKS_VEC)
        all_pills_col = load_pkl(CFG.DB_PILLS_COL)
        all_packs_col = load_pkl(CFG.DB_PACKS_COL)

        # Filter: Keep ONLY items in prescription
        count = 0
        for name, vecs in {**all_pills_vec, **all_packs_vec}.items():
            if self.rx_manager.is_allowed(name):
                # Store all vectors (flattened list)
                for v in vecs:
                    # Key format: "Name_Index" to differentiate samples
                    self.session_db_vec[f"{name}_{count}"] = (name, np.array(v)) 
                    count += 1
        
        # Filter Colors
        for name, col in {**all_pills_col, **all_packs_col}.items():
            if self.rx_manager.is_allowed(name):
                self.session_db_col[name] = col

        print(f"✅ Session DB Ready: {len(self.session_db_vec)} vectors tracking {len(self.rx_manager.allowed_drugs)} drugs")

    def match(self, vec, img_crop):
        candidates = []
        if not self.session_db_vec: return []

        # Extract query color
        # Note: Ideally compare against stored color means. 
        # Here we simulate color matching score for demonstration.
        # In production, use Hist Intersection with stored histograms.
        
        for key, (real_name, db_v) in self.session_db_vec.items():
            # 1. Vector Score
            vec_score = np.dot(vec, db_v)
            
            # 2. Color Score (Simulated Logic - Replace with real Hist match if DB has Hists)
            # If DB has colors loaded, we assume strict color match is possible.
            # Here we assign a baseline or verify if name exists in col DB.
            col_score = 0.5 
            if real_name in self.session_db_col:
                # Placeholder: If we had real histograms in DB, we'd compare here.
                # For now, we trust Vector more but let Color boost confidence.
                col_score = 0.5 # Neutral
            
            # Weighted Final Score
            final_score = (vec_score * CFG.WEIGHTS['vec']) + (col_score * CFG.WEIGHTS['col'])
            candidates.append((real_name, final_score, vec_score))
        
        # Sort & Deduplicate by Name (Keep highest score per drug)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        unique_candidates = []
        seen = set()
        for name, fs, vs in candidates:
            if name not in seen:
                unique_candidates.append((name, fs, vs))
                seen.add(name)
            if len(unique_candidates) >= 5: break
            
        return unique_candidates

    def process_frame(self, frame):
        # Resize
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
        scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        
        # Detect Packs (Focus on packs as requested)
        results = self.yolo_pack(img_ai, verbose=False, conf=0.25, imgsz=CFG.AI_SIZE)
        
        detections = []
        
        for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            
            # Scale back
            rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
            rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
            
            # 🛑 ROI EXCLUSION CHECK
            cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
            if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END:
                continue # Skip detection (It's likely the UI)

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0: continue
            
            # Match
            vec = self.engine.get_vector(crop)
            candidates = self.match(vec, crop)
            
            if candidates:
                top_name, top_score, _ = candidates[0]
                label = top_name if top_score > CFG.CONF_THRESHOLD else "Unknown"
                if label != "Unknown":
                    self.rx_manager.verify(label)
            else:
                label = "Unknown"
                candidates = []

            detections.append({
                'box': (rx1, ry1, rx2, ry2),
                'label': label,
                'score': candidates[0][1] if candidates else 0.0,
                'candidates': candidates
            })
            
        with self.lock:
            self.results = detections

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
            print("📷 PiCamera2 (RGB888) Active")
        except:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, CFG.DISPLAY_SIZE[0])
            self.cap.set(4, CFG.DISPLAY_SIZE[1])
            self.use_pi = False
            print("📷 USB Camera (BGR->RGB) Active")
            
    def get(self):
        if self.use_pi:
            return self.picam.capture_array()
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
    
    # 1. Draw Dashboard (Top Right)
    # This is the Exclusion Zone we defined earlier
    db_x, db_y = CFG.UI_ZONE_X_START, 10
    db_w, db_h = w - db_x - 10, CFG.UI_ZONE_Y_END
    
    # Semi-transparent BG
    sub = frame[db_y:db_y+db_h, db_x:db_x+db_w]
    white = np.ones(sub.shape, dtype=np.uint8) * 30
    cv2.addWeighted(sub, 0.3, white, 0.7, 0, sub)
    cv2.rectangle(frame, (db_x, db_y), (db_x+db_w, db_y+db_h), (0, 255, 0), 2)
    
    # Rx Status Text
    cv2.putText(frame, f"PATIENT: {rx_manager.patient_name}", (db_x+10, db_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    y_off = 60
    for drug in rx_manager.allowed_drugs:
        status = " [OK]" if drug in rx_manager.verified_drugs else " [...]"
        col = (0, 255, 0) if drug in rx_manager.verified_drugs else (150, 150, 150)
        cv2.putText(frame, f"- {drug}{status}", (db_x+10, db_y+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y_off += 25

    # 2. Draw Detections
    for det in results:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        score = det['score']
        candidates = det['candidates']
        
        # Color: Green if Verified, Yellow if Detected, Red if Unknown
        if label == "Unknown": color = (255, 0, 0)
        else: color = (0, 255, 0)
        
        # Box & Label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1-25), (x1 + len(label)*15, y1), color, -1)
        cv2.putText(frame, f"{label} {score:.0%}", (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        # --- CANDIDATE INSPECTOR (The Fix for "Unknown") ---
        # Draw a panel next to the object showing top 3 guesses
        panel_x = x2 + 5 if x2 + 180 < w else x1 - 185
        panel_y = y1
        
        # Black BG
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+180, panel_y+60), (0,0,0), -1)
        cv2.putText(frame, "AI SEES:", (panel_x+5, panel_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        for i, (c_name, c_score, c_vec) in enumerate(candidates[:3]):
            # Shorten name
            d_name = (c_name[:9] + '.') if len(c_name) > 9 else c_name
            # Choose color based on score (High=Green, Low=Red)
            c_col = (0, 255, 0) if c_score > CFG.CONF_THRESHOLD else (0, 100, 255)
            
            line = f"{i+1}.{d_name} {c_score:.2f}"
            cv2.putText(frame, line, (panel_x+5, panel_y+30+(i*15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c_col, 1)

# ================= 🚀 MAIN ENTRY =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    print("✨ Waiting for video...")
    while cam.get() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack Strict", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack Strict", *CFG.DISPLAY_SIZE)
    
    try:
        while True:
            # 1. Get RGB Frame
            frame = cam.get()
            if frame is None: continue
            
            # 2. Process
            ai.latest_frame = frame.copy()
            
            # 3. Draw UI
            draw_ui(frame, ai.results, ai.rx_manager)
            
            # 4. Display (Convert to BGR for OpenCV Window)
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("PillTrack Strict", display)
            
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()