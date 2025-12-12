#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  PILLTRACK: STABILIZED VERSION (Stability & Filtering)       ║
║  - Feature: Object Tracking + Label Voting (ลดการกระพริบ)    ║
║  - Feature: Confidence Buffering (รอชัวร์ค่อยโชว์)           ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import threading
import collections
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
    
    IMG_DB_FOLDER: str = 'database_images'
    PRESCRIPTION_FILE: str = 'prescription.txt'
    
    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416
    
    # Dashboard Zone
    UI_ZONE_X_START: int = 900 
    UI_ZONE_Y_END: int = 220
    
    # 🎚️ TUNING THRESHOLDS
    CONF_THRESHOLD: float = 0.40  # เพิ่มความเข้มขึ้นเล็กน้อย
    
    # WEIGHTS
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {'vec': 0.5, 'col': 0.3, 'sift': 0.2}) 
    
    # SIFT Tuning
    SIFT_RATIO_TEST: float = 0.75

    # 🛡️ STABILITY SETTINGS (NEW!)
    STABILITY_HISTORY_LEN: int = 5   # จำประวัติ 10 เฟรมล่าสุด
    STABILITY_CONFIRM_REQ: int = 3    # ต้องเจอชื่อซ้ำกัน 6 ใน 10 เฟรม ถึงจะฟันธง
    TRACKING_IOU_THRESH: float = 0.3  # ถ้าย้ายที่เกินนี้ถือว่าเป็นวัตถุใหม่
    MAX_MISSING_FRAMES: int = 1       # ถ้าหายไป 5 เฟรม ให้ลืมวัตถุนั้น

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (STABILIZED MODE)")

# ================= 🧠 PRESCRIPTION STATE MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.patient_name = "Unknown"
        self.allowed_drugs = []
        self.verified_drugs = set()
        self.load_prescription()

    def load_prescription(self):
        if not os.path.exists(CFG.PRESCRIPTION_FILE): return
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

# ================= 🛡️ STABILIZER SYSTEM (NEW CLASS) =================
class ObjectStabilizer:
    """
    คลาสนี้ทำหน้าที่จำวัตถุข้ามเฟรม (Tracking) และโหวตชื่อ (Voting)
    เพื่อแก้ปัญหาชื่อกระพริบไปมา
    """
    def __init__(self):
        # tracks เก็บข้อมูล: { track_id: {'history': deque, 'missing': int, 'box': tuple, 'contour': np.array} }
        self.tracks = {} 
        self.next_id = 0

    def calculate_iou(self, boxA, boxB):
        # คำนวณ Intersection over Union เพื่อดูว่ากล่องซ้อนกันแค่ไหน
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def update(self, raw_detections):
        # raw_detections = list of dict {'box', 'contour', 'label', ...}
        
        updated_tracks = {}
        used_indices = set()
        
        # 1. MATCHING: จับคู่ของใหม่กับของเก่าที่ใกล้กันที่สุด
        for track_id, track_data in self.tracks.items():
            best_iou = 0
            best_idx = -1
            
            last_box = track_data['box']
            
            for i, det in enumerate(raw_detections):
                if i in used_indices: continue
                iou = self.calculate_iou(last_box, det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # ถ้าเจอคู่ที่ IOU สูงพอ แสดงว่าเป็นของเดิม
            if best_iou > CFG.TRACKING_IOU_THRESH and best_idx != -1:
                det = raw_detections[best_idx]
                
                # Update History
                track_data['history'].append(det['label'])
                if len(track_data['history']) > CFG.STABILITY_HISTORY_LEN:
                    track_data['history'].popleft()
                
                # Update Info
                track_data['box'] = det['box']
                track_data['contour'] = det['contour']
                track_data['missing'] = 0
                track_data['candidates'] = det['candidates']
                
                updated_tracks[track_id] = track_data
                used_indices.add(best_idx)
            else:
                # ถ้าหาไม่เจอ ให้โอกาสหายไปได้ไม่เกิน MAX_MISSING_FRAMES
                track_data['missing'] += 1
                if track_data['missing'] < CFG.MAX_MISSING_FRAMES:
                    updated_tracks[track_id] = track_data

        # 2. NEW OBJECTS: ของที่เหลือที่จับคู่ไม่ได้ คือของใหม่
        for i, det in enumerate(raw_detections):
            if i not in used_indices:
                new_id = self.next_id
                self.next_id += 1
                history = collections.deque([det['label']])
                updated_tracks[new_id] = {
                    'history': history,
                    'missing': 0,
                    'box': det['box'],
                    'contour': det['contour'],
                    'candidates': det['candidates']
                }
        
        self.tracks = updated_tracks
        
        # 3. VOTING: แปลงข้อมูล Track เป็น Output ที่นิ่งแล้ว
        stable_results = []
        for tid, data in self.tracks.items():
            if data['missing'] > 0: continue # ไม่แสดงถ้ากำลังหายไป
            
            # นับคะแนนโหวตใน History
            counter = collections.Counter(data['history'])
            most_common = counter.most_common(1)
            
            final_label = "Verifying..." # ค่า Default ถ้ายังไม่มั่นใจ
            color_status = "pending"
            
            if most_common:
                top_label, count = most_common[0]
                # กฎเหล็ก: ต้องชนะโหวตเกินเกณฑ์ที่ตั้งไว้
                if count >= CFG.STABILITY_CONFIRM_REQ:
                    final_label = top_label
                    color_status = "confirmed" if top_label != "Unknown" else "unknown"
                else:
                    final_label = f"Wait... ({count}/{CFG.STABILITY_CONFIRM_REQ})"
            
            stable_results.append({
                'box': data['box'],
                'contour': data['contour'],
                'label': final_label,
                'status': color_status,
                'candidates': data.get('candidates', [])
            })
            
        return stable_results

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        self.stabilizer = ObjectStabilizer()  # ✅ เรียกใช้ Stabilizer
        
        self.session_db_vec = {} 
        self.session_db_sift = {}
        self.load_and_filter_db()
        
        try:
            # ตอนนี้ใช้ 1 โมเดลไปก่อนตาม Demo เดิม
            self.yolo_pack = YOLO(CFG.MODEL_PACK) if os.path.exists(CFG.MODEL_PACK) else YOLO('yolov8n-seg.pt')
            print("✅ YOLO Segmentation Model Loaded")
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

        # 1. Load Vectors (รวมกันไปก่อนตาม Demo เดิม)
        all_vecs = {**load_pkl(CFG.DB_PILLS_VEC), **load_pkl(CFG.DB_PACKS_VEC)}
        count = 0
        for name, vecs in all_vecs.items():
            if self.rx_manager.is_allowed(name):
                for v in vecs:
                    self.session_db_vec[f"{name}_{count}"] = (name, np.array(v)) 
                    count += 1

        # 2. Load SIFT
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
            sift_score = self.compute_sift_score(query_sift_des, real_name)
            
            final_score = (vec_score * CFG.WEIGHTS['vec']) + \
                          (sift_score * CFG.WEIGHTS['sift']) + \
                          (0.5 * CFG.WEIGHTS['col']) # Dummy color score for now
                          
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
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        results = self.yolo_pack(img_ai, verbose=False, conf=0.35, imgsz=CFG.AI_SIZE, task='segment')
        
        raw_detections = [] # เก็บผลดิบเฟรมนี้
        res = results[0]
        
        if res.masks is not None:
            for box, mask in zip(res.boxes, res.masks):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
                scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
                rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
                rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
                
                cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
                if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END: continue 
                
                contour = mask.xyn[0]
                contour[:, 0] *= CFG.DISPLAY_SIZE[0]
                contour[:, 1] *= CFG.DISPLAY_SIZE[1]
                contour = contour.astype(np.int32)
                
                crop = frame[ry1:ry2, rx1:rx2]
                if crop.size == 0: continue
                
                vec = self.engine.get_vector(crop)
                candidates = self.match(vec, crop)
                
                if candidates:
                    top_name, top_score, _, _ = candidates[0]
                    # กรองคะแนนดิบที่นี่ก่อนเข้า Stabilizer
                    label = top_name if top_score > CFG.CONF_THRESHOLD else "Unknown"
                else:
                    label = "Unknown"

                raw_detections.append({
                    'box': (rx1, ry1, rx2, ry2),
                    'contour': contour,
                    'label': label,
                    'candidates': candidates
                })

        # ✅ ส่งผลดิบเข้า Stabilizer เพื่อให้ช่วยกรองและจำ
        final_detections = self.stabilizer.update(raw_detections)
        
        # ✅ Verify เฉพาะยาที่ผ่านการ Stabilize มาแล้วว่าเป็นของจริง
        for det in final_detections:
            lbl = det['label']
            if lbl not in ["Unknown", "Verifying..."] and "Wait" not in lbl:
                self.rx_manager.verify(lbl)

        with self.lock: self.results = final_detections

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

# ================= 🖥️ UI RENDERER (UPDATED) =================
def draw_ui(frame, results, rx_manager):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    for det in results:
        contour = det['contour']
        label = det['label']
        status = det['status'] # confirmed, unknown, pending
        box = det['box']
        
        # Color Coding
        if status == "confirmed":
            color = (0, 255, 0) # Green
            border = 2
        elif status == "unknown":
            color = (255, 0, 0) # Red
            border = 2
        else: # pending / verifying
            color = (255, 255, 0) # Yellow
            border = 1
            
        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(overlay, [contour], True, color, border)
        
        # Label Drawing
        top_point = tuple(contour[contour[:, 1].argmin()])
        tx, ty = top_point
        
        # Background box for text
        cv2.rectangle(frame, (tx, ty-25), (tx + len(label)*14, ty), color, -1)
        
        # Text color (Black text on Yellow/Green, White on Red)
        txt_col = (0,0,0) if status != "unknown" else (255,255,255)
        cv2.putText(frame, label, (tx+5, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_col, 2)

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Dashboard
    db_x, db_y = CFG.UI_ZONE_X_START, 10
    db_w, db_h = w - db_x - 10, CFG.UI_ZONE_Y_END
    
    # Semi-transparent BG for dashboard
    sub = frame[db_y:db_y+db_h, db_x:db_x+db_w]
    white = np.ones(sub.shape, dtype=np.uint8) * 40
    cv2.addWeighted(sub, 0.4, white, 0.6, 0, sub)
    frame[db_y:db_y+db_h, db_x:db_x+db_w] = sub
    
    cv2.rectangle(frame, (db_x, db_y), (db_x+db_w, db_y+db_h), (0, 255, 0), 2)
    cv2.putText(frame, f"RX: {rx_manager.patient_name}", (db_x+10, db_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    y_off = 60
    for drug in rx_manager.allowed_drugs:
        is_ver = drug in rx_manager.verified_drugs
        icon = " [OK]" if is_ver else " [...]"
        col = (0, 255, 0) if is_ver else (180, 180, 180)
        cv2.putText(frame, f"- {drug}{icon}", (db_x+10, db_y+y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        y_off += 25

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    print("✨ Waiting for RGB888 feed (Stabilized Mode)...")
    while cam.get() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack Stabilized", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack Stabilized", *CFG.DISPLAY_SIZE)
    
    try:
        while True:
            frame = cam.get()
            if frame is None: continue
            
            ai.latest_frame = frame.copy()
            draw_ui(frame, ai.results, ai.rx_manager)
            
            cv2.imshow("PillTrack Stabilized", frame)
            if cv2.waitKey(1) == ord('q'): break
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()