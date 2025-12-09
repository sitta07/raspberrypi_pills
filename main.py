import os
import sys
import time
import threading
import pickle
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any, Union

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# SAHI IMPORTS
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.prediction import ObjectPrediction
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False
    print("⚠️ Warning: SAHI not installed. Running in standard YOLO mode.")

# ================= ENVIRONMENT SETUP =================
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["OMP_NUM_THREADS"] = "3"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================
@dataclass
class AppConfig:
    # Hardware / Display
    DISPLAY_W: int = 1280
    DISPLAY_H: int = 720
    AI_IMG_SIZE: int = 416 # Size for Standard YOLO
    DEVICE: str = "cpu"
    
    # SAHI Configuration (Optimized for Pi 5)
    USE_SAHI: bool = True     # Toggle SAHI on/off
    SAHI_SLICE_H: int = 320   # เล็กลงเพื่อความเร็ว (Standard 512)
    SAHI_SLICE_W: int = 320
    SAHI_OVERLAP_H: float = 0.1 # ลด Overlap เพื่อลดจำนวน Inference
    SAHI_OVERLAP_W: float = 0.1
    
    # Thresholds
    CONF_PILL: float = 0.25   # SAHI แม่นขึ้น อาจต้องเพิ่ม Conf นิดหน่อย
    CONF_PACK: float = 0.85
    SCORE_PASS_PILL: float = 0.2
    SCORE_PASS_PACK: float = 0.2
    MAX_OBJ_AREA_RATIO: float = 0.40
    CONSISTENCY_THRESHOLD: int = 2
    
    # Paths
    MODEL_PILL_PATH: str = 'models/pills_seg.pt'
    MODEL_PACK_PATH: str = 'models/seg_best_process.pt'
    HIS_FILE_PATH: str = 'prescription.txt'
    IMG_DB_FOLDER: str = 'database_images'
    
    DB_FILES: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'pills': {'vec': 'database/model_register/db_pills.pkl', 'col': 'database/model_register/colors_pills.pkl'},
        'packs': {'vec': 'database/model_register/db_packs.pkl', 'col': 'database/model_register/colors_packs.pkl'}
    })

    WEIGHTS_PILL: Tuple[float, float, float] = (0.5, 0.4, 0.1)
    WEIGHTS_PACK: Tuple[float, float, float] = (0.2, 0.8, 0.0)

# ================= DATA STRUCTURES =================
@dataclass
class DetectionResult:
    label: str
    score: float
    type: str
    box: Tuple[int, int, int, int]
    verified: bool = False
    is_wrong: bool = False
    clean_name: str = ""

# ================= CAMERA & UTILS (Same as before) =================
class SystemMonitor:
    @staticmethod
    def get_cpu_temperature() -> str:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return f"{float(f.read()) / 1000.0:.1f}C"
        except FileNotFoundError:
            return "N/A"

class CameraStream:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.lock = threading.Lock()
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cam.read()
            if ret:
                with self.lock:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.grabbed = True
            else:
                self.stopped = True

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.cam.release()

# ================= LOGIC MODULES (Patient & Trinity) =================
class PatientManager:
    def __init__(self, his_path: str):
        self.his_path = his_path
        self.current_patient: Optional[Dict] = None
        self.all_drugs: List[str] = []
        self.verified_drugs: Set[str] = set()
        self.lock = threading.Lock()

    def load_his_database(self) -> Dict[str, Dict]:
        if not os.path.exists(self.his_path): return {}
        db = {}
        try:
            with open(self.his_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"): continue
                    parts = line.split('|')
                    if len(parts) < 3: continue
                    hn, name = parts[0].strip(), parts[1].strip()
                    drugs = [d.strip().lower().replace('\ufeff', '') for d in parts[2].split(',') if d.strip()]
                    db[hn] = {'name': name, 'drugs': drugs}
            return db
        except Exception: return {}

    def set_patient(self, patient_data: Optional[Dict]):
        with self.lock:
            self.current_patient = patient_data
            if patient_data:
                self.all_drugs = patient_data['drugs'].copy()
                self.verified_drugs.clear()
            else:
                self.all_drugs = []
                self.verified_drugs.clear()

    def check_rx_compliance(self, detected_name: str) -> Tuple[str, bool]:
        with self.lock:
            if not self.current_patient: return detected_name, False
            clean_real = detected_name.replace("_pack", "").replace("_pill", "").lower().strip()
            if "?" in detected_name or "Unknown" in detected_name: return detected_name, False
            for allowed in self.all_drugs:
                if allowed in clean_real or clean_real in allowed: return allowed, False
            return f"WRONG: {detected_name}", True

    def verify_drug(self, drug_name: str):
        with self.lock:
            if drug_name in self.verified_drugs: return
            for target in self.all_drugs:
                if target == drug_name:
                    self.verified_drugs.add(target)
                    return
            self.verified_drugs.add(drug_name)

    def is_verified(self, drug_name: str) -> bool:
        with self.lock: return drug_name in self.verified_drugs

class TrinityEngine:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.device = torch.device(config.DEVICE)
        self.pills_vecs, self.pills_lbls = self._load_pkl_db(config.DB_FILES['pills']['vec'])
        self.packs_vecs, self.packs_lbls = self._load_pkl_db(config.DB_FILES['packs']['vec'])
        self.color_db = self._load_color_db()
        self.matrix_pills = self._prep_tensor(self.pills_vecs)
        self.matrix_packs = self._prep_tensor(self.packs_vecs)
        self.sift = cv2.SIFT_create(nfeatures=100)
        self.bf = cv2.BFMatcher(crossCheck=False)
        self.sift_db = self._load_sift_db()
        self._init_models()

    def _init_models(self):
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        self.embedder = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.embedder.eval().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        torch.set_grad_enabled(False)

    def _load_pkl_db(self, filepath):
        if not os.path.exists(filepath): return [], []
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                items = [(v, name) for name, vec_list in data.items() for v in vec_list]
                if items:
                    vecs, lbls = zip(*items)
                    return list(vecs), list(lbls)
        except: pass
        return [], []

    def _load_color_db(self):
        db = {}
        for k in ['pills', 'packs']:
            try: 
                with open(self.cfg.DB_FILES[k]['col'], 'rb') as f: db.update(pickle.load(f))
            except: pass
        return db

    def _load_sift_db(self):
        s_db = {}
        if os.path.exists(self.cfg.IMG_DB_FOLDER):
            for folder in os.listdir(self.cfg.IMG_DB_FOLDER):
                path = os.path.join(self.cfg.IMG_DB_FOLDER, folder)
                if not os.path.isdir(path): continue
                des_list = []
                for img_file in [x for x in os.listdir(path) if x.lower().endswith(('jpg', 'png'))][:3]:
                    img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    if max(img.shape) > 512:
                        scale = 512 / max(img.shape)
                        img = cv2.resize(img, None, fx=scale, fy=scale)
                    _, des = self.sift.detectAndCompute(img, None)
                    if des is not None: des_list.append(des)
                if des_list: s_db[folder] = des_list
        return s_db

    def _prep_tensor(self, vecs):
        if not vecs: return None
        t = torch.tensor(np.array(vecs), device=self.device, dtype=torch.float32)
        return t / t.norm(dim=1, keepdim=True)

    def identify(self, img_crop: np.ndarray, is_pill: bool) -> Tuple[str, float]:
        target_matrix = self.matrix_pills if is_pill else self.matrix_packs
        target_labels = self.pills_lbls if is_pill else self.packs_lbls
        if target_matrix is None: return "DB Error", 0.0

        try:
            if is_pill: pil_img = Image.fromarray(img_crop)
            else:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
                pil_img = Image.fromarray(cv2.merge([gray, gray, gray]))

            input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            live_vec = self.embedder(input_tensor).flatten()
            live_vec = live_vec / live_vec.norm()

            scores = torch.matmul(live_vec, target_matrix.T).squeeze(0)
            k_val = min(10, len(target_labels))
            if k_val == 0: return "Unknown", 0.0

            top_k_val, top_k_idx = torch.topk(scores, k=k_val)
            candidates = []
            seen = set()
            for idx, sc in zip(top_k_idx.detach().cpu().numpy(), top_k_val.detach().cpu().numpy()):
                name = target_labels[idx]
                if name not in seen:
                    candidates.append((name, float(sc)))
                    seen.add(name)
                    if len(candidates) >= 3: break

            live_color = None
            gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
            _, des_live = self.sift.detectAndCompute(gray, None)
            if is_pill:
                h, w = img_crop.shape[:2]
                center = img_crop[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
                if center.size > 0: live_color = np.mean(cv2.cvtColor(center, cv2.COLOR_RGB2HSV), axis=(0,1))

            best_score = -1.0
            final_name = "Unknown"
            COLOR_NORM = np.array([90.0, 255.0, 255.0])

            for name, vec_score in candidates:
                clean = name.replace("_pill", "").replace("_pack", "")
                sift_score = 0.0
                if des_live is not None and clean in self.sift_db:
                    max_good = 0
                    for ref_des in self.sift_db[clean]:
                        try:
                            matches = self.bf.knnMatch(des_live, ref_des, k=2)
                            good = sum(1 for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance)
                            max_good = max(max_good, good)
                        except: pass
                    sift_score = min(max_good / 15.0, 1.0)

                col_score = 0.0
                if is_pill and live_color is not None and name in self.color_db:
                    diff = np.abs(live_color - self.color_db[name])
                    diff[0] = min(diff[0], 180 - diff[0])
                    dist = np.linalg.norm(diff / COLOR_NORM)
                    col_score = np.clip(np.exp(-3.0 * dist), 0, 1)

                w_vec, w_sift, w_col = self.cfg.WEIGHTS_PILL if is_pill else self.cfg.WEIGHTS_PACK
                total = vec_score * w_vec + sift_score * w_sift + col_score * w_col
                if total > best_score: best_score, final_name = total, clean

            return final_name, best_score
        except: return "Error", 0.0

# ================= 4. AI WORKER (UPDATED FOR SAHI) =================
class AIProcessor:
    def __init__(self, config: AppConfig, patient_mgr: PatientManager, trinity: TrinityEngine):
        self.cfg = config
        self.patient_mgr = patient_mgr
        self.trinity = trinity
        
        logger.info("Loading Models...")
        # 1. Standard YOLO for Packs (Always fast)
        self.model_pack = YOLO(config.MODEL_PACK_PATH, task='detect')
        
        # 2. Pill Model: Choose between SAHI or Standard
        self.sahi_model = None
        self.model_pill = None
        
        if HAS_SAHI and config.USE_SAHI:
            logger.info("⚡ SAHI Enabled for Pills (Slicing Mode)")
            # Load into SAHI Wrapper
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=config.MODEL_PILL_PATH,
                confidence_threshold=config.CONF_PILL,
                device=config.DEVICE # Important for Pi
            )
        else:
            logger.info("⚡ Standard YOLO Enabled for Pills")
            self.model_pill = YOLO(config.MODEL_PILL_PATH, task='detect')
        
        self.latest_frame = None
        self.results: List[DetectionResult] = []
        self.stopped = False
        self.lock = threading.Lock()
        self.scale_x = config.DISPLAY_W / config.AI_IMG_SIZE
        self.scale_y = config.DISPLAY_H / config.AI_IMG_SIZE
        self.consistency_counter: Dict[str, int] = {}

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def update_frame(self, frame):
        with self.lock: self.latest_frame = frame

    def get_results(self):
        with self.lock: return self.results, self.patient_mgr.current_patient

    def _is_valid_box(self, box):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        total_area = self.cfg.DISPLAY_W * self.cfg.DISPLAY_H
        return (area / total_area) <= self.cfg.MAX_OBJ_AREA_RATIO

    def _process_item(self, box, crop, is_pill: bool) -> DetectionResult:
        raw_name, score = self.trinity.identify(crop, is_pill=is_pill)
        final_name, is_wrong = self.patient_mgr.check_rx_compliance(raw_name)
        clean_name = final_name.replace("WRONG: ", "").lower()
        verified = self.patient_mgr.is_verified(clean_name)
        return DetectionResult(label=final_name, score=score, type='pill' if is_pill else 'pack',
                               box=box, verified=verified, is_wrong=is_wrong, clean_name=clean_name)

    def _run(self):
        logger.info(f"AI Loop Started on {self.cfg.DEVICE}")
        
        while not self.stopped:
            with self.lock:
                frame_hd = self.latest_frame
                self.latest_frame = None
            
            if frame_hd is None:
                time.sleep(0.005)
                continue

            active_packs: List[DetectionResult] = []
            final_detections: List[DetectionResult] = []
            frame_found_names = set()

            try:
                # --- PHASE 1: PACKS (Standard YOLO) ---
                # Packs are large, SAHI is unnecessary overkill
                frame_ai_pack = cv2.resize(frame_hd, (self.cfg.AI_IMG_SIZE, self.cfg.AI_IMG_SIZE))
                res_pack = self.model_pack(frame_ai_pack, verbose=False, conf=self.cfg.CONF_PACK, 
                                           imgsz=self.cfg.AI_IMG_SIZE, agnostic_nms=True)[0]
                
                for box_data in res_pack.boxes.xyxy.detach().cpu().numpy().astype(int):
                    x1, y1 = int(box_data[0] * self.scale_x), int(box_data[1] * self.scale_y)
                    x2, y2 = int(box_data[2] * self.scale_x), int(box_data[3] * self.scale_y)
                    
                    if not self._is_valid_box((x1, y1, x2, y2)): continue
                    crop = frame_hd[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    result = self._process_item((x1, y1, x2, y2), crop, is_pill=False)
                    if not result.is_wrong and result.score >= self.cfg.SCORE_PASS_PACK:
                        frame_found_names.add(result.clean_name)
                        self.consistency_counter[result.clean_name] = self.consistency_counter.get(result.clean_name, 0) + 1
                        if self.consistency_counter[result.clean_name] >= self.cfg.CONSISTENCY_THRESHOLD:
                             self.patient_mgr.verify_drug(result.clean_name)
                             result.verified = True
                    active_packs.append(result)
                    final_detections.append(result)

                # --- PHASE 2: PILLS (SAHI vs Standard) ---
                pill_boxes = []
                
                if HAS_SAHI and self.cfg.USE_SAHI:
                    # SAHI Inference on Full HD Frame (No Resize Needed)
                    sahi_result = get_sliced_prediction(
                        frame_hd, # Send original high-res frame
                        self.sahi_model,
                        slice_height=self.cfg.SAHI_SLICE_H,
                        slice_width=self.cfg.SAHI_SLICE_W,
                        overlap_height_ratio=self.cfg.SAHI_OVERLAP_H,
                        overlap_width_ratio=self.cfg.SAHI_OVERLAP_W,
                        verbose=0
                    )
                    
                    # Convert SAHI results to list of [x1, y1, x2, y2]
                    for prediction in sahi_result.object_prediction_list:
                        # prediction.bbox is a BoundingBox object
                        bbox = prediction.bbox.to_xyxy() # returns [x1, y1, x2, y2]
                        x1, y1, x2, y2 = map(int, bbox)
                        pill_boxes.append((x1, y1, x2, y2))
                else:
                    # Standard YOLO
                    frame_ai_pill = cv2.resize(frame_hd, (self.cfg.AI_IMG_SIZE, self.cfg.AI_IMG_SIZE))
                    res_pill = self.model_pill(frame_ai_pill, verbose=False, conf=self.cfg.CONF_PILL,
                                               imgsz=self.cfg.AI_IMG_SIZE, max_det=30, agnostic_nms=True)[0]
                    for box_data in res_pill.boxes.xyxy.detach().cpu().numpy().astype(int):
                        x1, y1 = int(box_data[0] * self.scale_x), int(box_data[1] * self.scale_y)
                        x2, y2 = int(box_data[2] * self.scale_x), int(box_data[3] * self.scale_y)
                        pill_boxes.append((x1, y1, x2, y2))

                # Process Pill Boxes
                for (x1, y1, x2, y2) in pill_boxes:
                    if not self._is_valid_box((x1, y1, x2, y2)): continue
                    if (x2-x1) < 20 or (y2-y1) < 20: continue # Skip noise
                    
                    cx, cy = (x1+x2)>>1, (y1+y2)>>1
                    parent_pack = next((p for p in active_packs if p.box[0] < cx < p.box[2] and p.box[1] < cy < p.box[3]), None)
                    
                    if parent_pack:
                        result = DetectionResult(
                            label=parent_pack.label, score=parent_pack.score, type='pill',
                            box=(x1, y1, x2, y2), verified=parent_pack.verified, 
                            is_wrong=parent_pack.is_wrong, clean_name=parent_pack.clean_name
                        )
                        if not result.is_wrong:
                             frame_found_names.add(result.clean_name)
                             self.consistency_counter[result.clean_name] = self.consistency_counter.get(result.clean_name, 0) + 1
                    else:
                        crop = frame_hd[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        result = self._process_item((x1, y1, x2, y2), crop, is_pill=True)
                        if not result.is_wrong and result.score > self.cfg.SCORE_PASS_PILL and "Unknown" not in result.label:
                            frame_found_names.add(result.clean_name)
                            self.consistency_counter[result.clean_name] = self.consistency_counter.get(result.clean_name, 0) + 1
                            if self.consistency_counter[result.clean_name] >= self.cfg.CONSISTENCY_THRESHOLD:
                                self.patient_mgr.verify_drug(result.clean_name)
                                result.verified = True
                    final_detections.append(result)

                # Reset consistency for missing items
                for k in list(self.consistency_counter.keys()):
                    if k not in frame_found_names: self.consistency_counter[k] = 0

                with self.lock: self.results = final_detections

            except Exception as e:
                logger.error(f"AI Loop Error: {e}")

    def stop(self): self.stopped = True

# ================= 5. MAIN APP =================
class PillTrackApp:
    def __init__(self):
        self.cfg = AppConfig()
        self.camera = CameraStream(self.cfg.DISPLAY_W, self.cfg.DISPLAY_H)
        self.patient_mgr = PatientManager(self.cfg.HIS_FILE_PATH)
        self.trinity = TrinityEngine(self.cfg)
        self.ai = AIProcessor(self.cfg, self.patient_mgr, self.trinity)
        self.window_name = "PillTrack Senior + SAHI"
        self.target_hn = "HN-101"

    def draw_ui(self, frame, results):
        for r in results:
            color = (255, 255, 0)
            text = r.label
            if r.is_wrong: color, text = (255, 0, 0), f"!! {r.label} !!"
            elif r.verified: color, text = (0, 255, 0), f"OK {r.label}"
            elif r.type == 'pack' and r.score >= self.cfg.SCORE_PASS_PACK: color = (0, 255, 0)
            elif "Unknown" in r.label: color = (50, 50, 50)
            cv2.rectangle(frame, (r.box[0], r.box[1]), (r.box[2], r.box[3]), color, 2)
            cv2.putText(frame, f"{text} {r.score:.0%}", (r.box[0], r.box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Temp: {SystemMonitor.get_cpu_temperature()}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    def run(self):
        self.camera.start()
        self.ai.start()
        db = self.patient_mgr.load_his_database()
        if self.target_hn in db: self.patient_mgr.set_patient(db[self.target_hn])
        
        while self.camera.read() is None: time.sleep(0.1)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.cfg.DISPLAY_W, self.cfg.DISPLAY_H)
        
        try:
            prev_time = time.time()
            while True:
                frame = self.camera.read()
                if frame is None: time.sleep(0.01); continue
                self.ai.update_frame(frame.copy())
                results, _ = self.ai.get_results()
                self.draw_ui(frame, results)
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {fps:.1f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                if key == ord('r'):
                    db = self.patient_mgr.load_his_database()
                    self.patient_mgr.set_patient(db.get(self.target_hn))
        finally:
            self.camera.stop()
            self.ai.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    PillTrackApp().run()