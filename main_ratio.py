#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  PILLTRACK: DUAL-PIPELINE MASTER v2.1 (STABLE)               ║
║  - Architecture: Dual Model (Pack Seg + Pill Seg)            ║
║  - Logic: Separate Databases & Strict Feature Matching       ║
║  - Safety: Overlap Removal (Pill inside Pack = Ignore)       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import logging
import threading
import collections
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import cv2
import torch
import pickle
from torchvision import models, transforms
from ultralytics import YOLO

# ================= 📝 LOGGING SETUP =================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ================= ⚙️ CONFIGURATION =================
@dataclass
class Config:
    # --- PATHS ---
    MODEL_PACK: str = 'models/seg_best_process.pt' 
    MODEL_PILL: str = 'models/pills_seg.pt'
    
    # Databases
    DB_PILLS_VEC: str = 'database/db_register/db_pills.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs.pkl'
    IMG_DB_FOLDER: str = 'database_images'
    PRESCRIPTION_FILE: str = 'prescription.txt'
    
    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416
    
    # UI Zone (Dashboard)
    UI_ZONE_X_START: int = 900
    UI_ZONE_Y_END: int = 220
    
    # 🎚️ THRESHOLDS (STRICT MODE)
    CONF_THRESHOLD_PACK: float = 0.60  # แผงต้องมั่นใจสูง
    CONF_THRESHOLD_PILL: float = 0.65  # เม็ดยาต้องมั่นใจสูงมาก
    
    # Feature Weights (Separated Logic)
    # Pack: เน้น SIFT (ลาย) + Vector (ทรง)
    WEIGHTS_PACK: Dict[str, float] = field(default_factory=lambda: {'vec': 0.4, 'col': 0.1, 'sift': 0.5})
    # Pill: เน้น Color (สี) + Vector (ทรง)
    WEIGHTS_PILL: Dict[str, float] = field(default_factory=lambda: {'vec': 0.5, 'col': 0.5, 'sift': 0.0})
    
    # SIFT Tuning
    SIFT_RATIO: float = 0.70
    SIFT_MIN_MATCHES: int = 8  # ต้องเจอจุดเหมือนอย่างน้อย 8 จุดสำหรับแผงยา
    
    # Stability
    STABILITY_HISTORY: int = 8
    STABILITY_REQ: int = 5      # ต้องเจอซ้ำ 5 เฟรม

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 🧠 PRESCRIPTION MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.patient_name = "Unknown"
        self.allowed_drugs = []
        self.verified_drugs = set()
        self.load_prescription()

    def load_prescription(self):
        if not os.path.exists(CFG.PRESCRIPTION_FILE):
            return
        try:
            with open(CFG.PRESCRIPTION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split('|')
                    if len(parts) >= 3:
                        self.patient_name = parts[1].strip()
                        self.allowed_drugs = [d.strip().lower() for d in parts[2].split(',') if d.strip()]
                        logger.info(f"📋 Rx Loaded: {self.patient_name} -> {self.allowed_drugs}")
                        break 
        except Exception as e: logger.error(f"Rx Error: {e}")

    def is_allowed(self, name):
        clean = name.lower().replace('_pack', '').replace('_pill', '')
        for allowed in self.allowed_drugs:
            if allowed in clean or clean in allowed: return True
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
                transforms.ToPILImage(), transforms.Resize((224, 224)),
                transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except: logger.error("ResNet Load Failed")

        # 2. SIFT
        self.sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03)
        self.bf = cv2.BFMatcher()

    @torch.no_grad()
    def get_vector(self, img_rgb):
        try:
            t = self.preprocess(img_rgb).unsqueeze(0).to(device)
            vec = self.model(t).flatten().cpu().numpy()
            return vec / (np.linalg.norm(vec) + 1e-8)
        except: return None

    def get_sift_features(self, img_rgb):
        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            kp, des = self.sift.detectAndCompute(gray, None)
            return des
        except: return None

    def get_color_hist(self, img_rgb):
        try:
            img = cv2.resize(img_rgb, (64, 64))
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist.flatten()
        except: return None

# ================= 🛡️ STABILIZER =================
class ObjectStabilizer:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def get_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(areaA + areaB - inter + 1e-5)

    def update(self, detections):
        # detections: list of {'box', 'label', 'contour', 'type'}
        updated_tracks = {}
        used_dets = set()

        # Match existing
        for tid, track in self.tracks.items():
            best_iou, best_idx = 0, -1
            for i, det in enumerate(detections):
                if i in used_dets: continue
                iou = self.get_iou(track['box'], det['box'])
                if iou > best_iou: best_iou, best_idx = iou, i
            
            if best_iou > 0.3 and best_idx != -1:
                det = detections[best_idx]
                track['history'].append(det['label'])
                if len(track['history']) > CFG.STABILITY_HISTORY: track['history'].pop(0)
                track.update({'box': det['box'], 'contour': det['contour'], 'missing': 0})
                updated_tracks[tid] = track
                used_dets.add(best_idx)
            else:
                track['missing'] += 1
                if track['missing'] < 3: updated_tracks[tid] = track

        # Add new
        for i, det in enumerate(detections):
            if i not in used_dets:
                updated_tracks[self.next_id] = {
                    'history': [det['label']], 
                    'box': det['box'], 
                    'contour': det['contour'], 
                    'type': det['type'],
                    'missing': 0
                }
                self.next_id += 1

        self.tracks = updated_tracks
        
        # Output Generation
        results = []
        for tid, track in self.tracks.items():
            if track['missing'] > 0: continue
            
            counts = collections.Counter(track['history'])
            top_label, count = counts.most_common(1)[0]
            
            status = "pending"
            final_label = "Verifying..."
            
            if count >= CFG.STABILITY_REQ:
                final_label = top_label
                status = "confirmed" if top_label != "Unknown" else "unknown"
            
            results.append({
                'box': track['box'],
                'contour': track['contour'],
                'label': final_label,
                'status': status,
                'type': track['type']
            })
        return results

# ================= 🤖 AI PROCESSOR (DUAL PIPELINE) =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx = PrescriptionManager()
        self.stabilizer = ObjectStabilizer()
        
        # Separate Databases
        self.db_packs = {'vec': {}, 'sift': {}, 'col': {}}
        self.db_pills = {'vec': {}, 'col': {}} # No SIFT for pills
        
        self.load_databases()
        
        # Dual Models
        logger.info("🤖 Loading Models...")
        self.model_pack = self._load_yolo(CFG.MODEL_PACK)
        self.model_pill = self._load_yolo(CFG.MODEL_PILL)
        
        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False

    def _load_yolo(self, path):
        if os.path.exists(path): return YOLO(path)
        logger.warning(f"⚠️ Model not found: {path}, using default.")
        return YOLO('yolov8n-seg.pt')

    def load_databases(self):
        logger.info("📚 Building Databases...")
        def load_pkl(p): return pickle.load(open(p, 'rb')) if os.path.exists(p) else {}
        
        # 1. Load Packs (Strict SIFT + Vec)
        vecs = load_pkl(CFG.DB_PACKS_VEC)
        for name, vs in vecs.items():
            if self.rx.is_allowed(name):
                for i, v in enumerate(vs):
                    self.db_packs['vec'][f"{name}_{i}"] = (name, np.array(v))
                    
        # Load Pack Images for SIFT
        if os.path.exists(CFG.IMG_DB_FOLDER):
            for dname in os.listdir(CFG.IMG_DB_FOLDER):
                if not self.rx.is_allowed(dname): continue
                # Assume if folder name contains 'pack' -> Pack DB
                # But for safety, we check prescription logic or naming convention
                # For demo, load SIFT for all allowed drugs into Packs DB
                dpath = os.path.join(CFG.IMG_DB_FOLDER, dname)
                if os.path.isdir(dpath):
                    des_list = []
                    for f in sorted(os.listdir(dpath))[:3]:
                        img = cv2.imread(os.path.join(dpath, f))
                        if img is not None:
                            des = self.engine.get_sift_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                            if des is not None: des_list.append(des)
                    if des_list: self.db_packs['sift'][dname] = des_list

        # 2. Load Pills (Vec + Color)
        vecs = load_pkl(CFG.DB_PILLS_VEC)
        cols = load_pkl(CFG.DB_PILLS_COL)
        for name, vs in vecs.items():
            if self.rx.is_allowed(name):
                for i, v in enumerate(vs):
                    self.db_pills['vec'][f"{name}_{i}"] = (name, np.array(v))
        for name, c in cols.items():
             if self.rx.is_allowed(name):
                 self.db_pills['col'][name] = np.array(c)

    def compute_sift_score(self, query_des, target_name):
        if query_des is None or target_name not in self.db_packs['sift']: return 0.0
        max_matches = 0
        for ref_des in self.db_packs['sift'][target_name]:
            matches = self.engine.bf.knnMatch(query_des, ref_des, k=2)
            good = [m for m, n in matches if m.distance < CFG.SIFT_RATIO * n.distance]
            if len(good) > max_matches: max_matches = len(good)
        
        # Strict Rule: Must have minimum matches
        if max_matches < CFG.SIFT_MIN_MATCHES: return 0.0
        return min(max_matches / 20.0, 1.0)

    def match_object(self, vec, img_crop, obj_type):
        """Generic matcher based on object type"""
        candidates = []
        
        if obj_type == 'PACK':
            db = self.db_packs
            weights = CFG.WEIGHTS_PACK
            thresh = CFG.CONF_THRESHOLD_PACK
            query_sift = self.engine.get_sift_features(img_crop)
            
            for k, (name, db_v) in db['vec'].items():
                s_sift = self.compute_sift_score(query_sift, name)
                if s_sift == 0: continue # SIFT Failed -> Skip
                
                s_vec = np.dot(vec, db_v)
                score = (s_vec * weights['vec']) + (s_sift * weights['sift']) + (0.5 * weights['col'])
                candidates.append((name, score))
                
        else: # PILL
            db = self.db_pills
            weights = CFG.WEIGHTS_PILL
            thresh = CFG.CONF_THRESHOLD_PILL
            query_col = self.engine.get_color_hist(img_crop)
            
            for k, (name, db_v) in db['vec'].items():
                # Color Check
                s_col = 0.0
                if name in db['col'] and query_col is not None:
                    s_col = cv2.compareHist(query_col, db['col'][name], cv2.HISTCMP_CORREL)
                    s_col = max(0, s_col) # Clip negative
                
                if s_col < 0.8: continue # Color Mismatch -> Skip
                
                s_vec = np.dot(vec, db_v)
                score = (s_vec * weights['vec']) + (s_col * weights['col'])
                candidates.append((name, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates and candidates[0][1] >= thresh:
            return candidates[0][0]
        return "Unknown"

    def process_frame(self, frame):
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        detections = []

        # 1. PACK DETECTION
        res_pack = self.model_pack(img_ai, verbose=False, conf=0.4, imgsz=CFG.AI_SIZE, task='segment')[0]
        pack_boxes = [] # Keep track of pack areas
        
        if res_pack.masks:
            for box, mask in zip(res_pack.boxes, res_pack.masks):
                bx = box.xyxy[0].cpu().numpy().astype(int)
                contour = self._scale_contour(mask.xyn[0])
                crop, disp_box = self._get_crop(frame, bx)
                
                if crop.size == 0 or self._is_ui_zone(disp_box): continue
                
                vec = self.engine.get_vector(crop)
                if vec is not None:
                    label = self.match_object(vec, crop, 'PACK')
                    detections.append({'box': disp_box, 'contour': contour, 'label': label, 'type': 'PACK'})
                    pack_boxes.append(disp_box)

        # 2. PILL DETECTION
        res_pill = self.model_pill(img_ai, verbose=False, conf=0.4, imgsz=CFG.AI_SIZE, task='segment')[0]
        
        if res_pill.masks:
            for box, mask in zip(res_pill.boxes, res_pill.masks):
                bx = box.xyxy[0].cpu().numpy().astype(int)
                contour = self._scale_contour(mask.xyn[0])
                crop, disp_box = self._get_crop(frame, bx)
                
                if crop.size == 0 or self._is_ui_zone(disp_box): continue
                
                # Check Overlap: Is this pill inside any detected pack?
                cx, cy = (disp_box[0]+disp_box[2])//2, (disp_box[1]+disp_box[3])//2
                is_inside = False
                for pb in pack_boxes:
                    if pb[0] < cx < pb[2] and pb[1] < cy < pb[3]:
                        is_inside = True; break
                if is_inside: continue # Skip pills inside packs
                
                vec = self.engine.get_vector(crop)
                if vec is not None:
                    label = self.match_object(vec, crop, 'PILL')
                    detections.append({'box': disp_box, 'contour': contour, 'label': label, 'type': 'PILL'})

        # Stabilize & Verify
        final_dets = self.stabilizer.update(detections)
        for d in final_dets:
            if d['status'] == 'confirmed': self.rx.verify(d['label'])
            
        with self.lock: self.results = final_dets

    def _scale_contour(self, cnt_norm):
        cnt = cnt_norm.copy()
        cnt[:, 0] *= CFG.DISPLAY_SIZE[0]
        cnt[:, 1] *= CFG.DISPLAY_SIZE[1]
        return cnt.astype(np.int32)

    def _get_crop(self, frame, box_ai):
        sx = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
        sy = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        x1, y1, x2, y2 = box_ai
        rx1, ry1 = int(x1*sx), int(y1*sy)
        rx2, ry2 = int(x2*sx), int(y2*sy)
        return frame[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2)

    def _is_ui_zone(self, box):
        cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
        return cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        while not self.stopped:
            with self.lock: frame = self.latest_frame
            if frame is not None:
                try: self.process_frame(frame)
                except Exception as e: logger.error(f"Frame Err: {e}")
            time.sleep(0.01)

# ================= 🚀 MAIN UI =================
class Camera:
    def __init__(self):
        self.cap = None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            c = self.picam.create_preview_configuration(main={"size": CFG.DISPLAY_SIZE, "format": "RGB888"})
            self.picam.configure(c)
            self.picam.start()
            self.pi = True
            logger.info("📷 PiCamera2: RGB888")
        except:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, CFG.DISPLAY_SIZE[0])
            self.cap.set(4, CFG.DISPLAY_SIZE[1])
            self.pi = False
            logger.info("📷 USB Camera")

    def get(self):
        if self.pi: return self.picam.capture_array()
        ret, f = self.cap.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ret else None
    
    def stop(self):
        if self.pi: self.picam.stop()
        else: self.cap.release()

def draw_ui(frame, results, rx):
    overlay = frame.copy()
    for det in results:
        cnt, lbl, st, typ = det['contour'], det['label'], det['status'], det['type']
        
        # Color: Pack=Cyan, Pill=Green (if confirmed)
        if st == 'confirmed': col = (0, 255, 0) if typ == 'PILL' else (0, 255, 255)
        elif st == 'unknown': col = (255, 0, 0)
        else: col = (100, 100, 100) # Pending
        
        cv2.fillPoly(overlay, [cnt], col)
        cv2.polylines(overlay, [cnt], True, (255,255,255), 2)
        
        # Label
        tx, ty = tuple(cnt[cnt[:, 1].argmin()])
        cv2.putText(frame, lbl, (tx, ty-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Dashboard
    dbx, dby = CFG.UI_ZONE_X_START, 10
    sub = frame[dby:CFG.UI_ZONE_Y_END, dbx:1270]
    white = np.ones(sub.shape, dtype=np.uint8)*50
    cv2.addWeighted(sub, 0.5, white, 0.5, 0, sub)
    frame[dby:CFG.UI_ZONE_Y_END, dbx:1270] = sub
    
    cv2.putText(frame, f"Patient: {rx.patient_name}", (dbx+10, dby+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y = 70
    for d in rx.allowed_drugs:
        ok = d in rx.verified_drugs
        col = (0, 255, 0) if ok else (200, 200, 200)
        icon = "[OK]" if ok else "[ ]"
        cv2.putText(frame, f"{d} {icon}", (dbx+10, dby+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y += 25

if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    logger.info("✨ Waiting for video...")
    while cam.get() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack v2.1", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack v2.1", *CFG.DISPLAY_SIZE)
    
    try:
        while True:
            frame = cam.get()
            if frame is None: continue
            
            ai.latest_frame = frame.copy()
            draw_ui(frame, ai.results, ai.rx)
            
            cv2.imshow("PillTrack v2.1", frame)
            if cv2.waitKey(1) == ord('q'): break
    except KeyboardInterrupt: pass
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()