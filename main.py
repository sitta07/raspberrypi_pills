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
from datetime import datetime
import json
import logging
from collections import defaultdict, deque

# ================= FIX RASPBERRY PI ENVIRONMENT =================
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["OMP_NUM_THREADS"] = "3"

try:
    from picamera2 import Picamera2
except ImportError:
    print("⚠️ Warning: Picamera2 not found.")

# ================= ENHANCED LOGGING CONFIGURATION =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pilltrack_senior.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================
# Paths
MODEL_PILL_PATH = 'models/pills_seg.pt'      # seg    
MODEL_PACK_PATH = 'models/seg_best_process.pt' # seg
DB_FILES = {
    'pills': {'vec': 'database/db_register/db_pills.pkl', 'col': 'database/db_register/colors_pills.pkl'},
    'packs': {'vec': 'database/db_register/db_packs.pkl', 'col': 'database/db_register/colors_packs.pkl'}
}

IMG_DB_FOLDER = 'database_images'
HIS_FILE_PATH = 'prescription.txt'
CONFIG_FILE = 'config.json'

# Display & AI Resolution
DISPLAY_W, DISPLAY_H = 1280, 720
AI_IMG_SIZE = 416

# 🆕 NEW: ZOOM CONFIGURATION
ZOOM_FACTOR = 1.15

# Thresholds
CONF_PILL = 0.5
CONF_PACK = 0.25
SCORE_PASS_PILL = 0.2
SCORE_PASS_PACK = 0.2

# --- ENHANCED CONFIGURATION ---
CONSISTENCY_THRESHOLD = 3  # Increased for better stability
MAX_OBJ_AREA_RATIO = 0.40
MIN_PILL_SIZE = 30
MIN_PACK_SIZE = 50
MAX_DETECTIONS_PER_FRAME = 25
HEALTH_CHECK_INTERVAL = 30  # seconds
FRAME_BUFFER_SIZE = 5

# Enhanced validation
MIN_COLOR_SIMILARITY = 0.6
MIN_SIFT_MATCHES = 5
MAX_ASPECT_RATIO = 3.0
MIN_ASPECT_RATIO = 0.33

device = torch.device("cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (RGB888 STRICT MODE)")

# ================= ENHANCED UTILS =================
def get_system_info():
    """Get comprehensive system information"""
    info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'cpu_temp': get_cpu_temperature(),
        'memory': get_memory_usage(),
        'cpu_usage': get_cpu_usage(),
        'device': str(device)
    }
    return info

def get_cpu_temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return f"{float(f.read()) / 1000.0:.1f}°C"
    except:
        return "N/A"

def get_memory_usage():
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1])) for i in f.readlines())
            total = meminfo['MemTotal']
            free = meminfo['MemFree']
            used = total - free
            return f"{used/1024:.1f}MB/{total/1024:.1f}MB ({used/total*100:.1f}%)"
    except:
        return "N/A"

def get_cpu_usage():
    try:
        with open('/proc/stat', 'r') as f:
            fields = [float(column) for column in f.readline().strip().split()[1:]]
        idle, total = fields[3], sum(fields)
        return f"{(1.0 - idle / total) * 100:.1f}%"
    except:
        return "N/A"

def is_point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def validate_box(box, img_w, img_h):
    """Validate bounding box coordinates and dimensions"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    # Check coordinates
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        logger.warning(f"Invalid box coordinates: {box} for image size {img_w}x{img_h}")
        return False
    
    # Check size
    if w < 5 or h < 5:
        logger.warning(f"Box too small: {w}x{h}")
        return False
    
    # Check aspect ratio
    aspect_ratio = w / h if h > 0 else 0
    if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
        logger.warning(f"Invalid aspect ratio: {aspect_ratio:.2f}")
        return False
    
    # Check area ratio
    area = w * h
    image_area = img_w * img_h
    if area / image_area > MAX_OBJ_AREA_RATIO:
        logger.warning(f"Box too large: {area/image_area:.2%}")
        return False
    
    return True

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def apply_digital_zoom(frame, zoom_factor):
    """Apply digital zoom with edge handling"""
    if zoom_factor <= 1.0:
        return frame
    
    h, w = frame.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    
    # Ensure crop dimensions are valid
    new_h = max(1, min(new_h, h))
    new_w = max(1, min(new_w, w))
    
    top = max(0, (h - new_h) // 2)
    left = max(0, (w - new_w) // 2)
    
    # Ensure crop is within bounds
    top = min(top, h - new_h)
    left = min(left, w - new_w)
    
    crop = frame[top:top+new_h, left:left+new_w]
    
    if crop.size == 0:
        logger.error("Empty crop after zoom")
        return frame
    
    zoomed_frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed_frame

# ================= ENHANCED WEBCAM STREAM =================
class WebcamStream:
    __slots__ = ('stopped', 'frame', 'grabbed', 'picam2', 'lock', 'cam', 
                 'frame_counter', 'last_valid_frame', 'fps', 'last_fps_time')
    
    def __init__(self):
        self.stopped = False
        self.frame = None
        self.grabbed = False
        self.picam2 = None
        self.cam = None
        self.lock = threading.Lock()
        self.frame_counter = 0
        self.last_valid_frame = None
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_buffer = deque(maxlen=10)

    def start(self):
        logger.info("Initializing Camera (RGB888)...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (DISPLAY_W, DISPLAY_H), "format": "RGB888"},
                    controls={
                        "FrameDurationLimits": (100000, 100000),
                        "AwbEnable": True,
                        "AeEnable": True
                    }
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(2.0)
                logger.info("Picamera2 Started in RGB888")
                break
            except Exception as e:
                logger.error(f"Picamera2 initialization attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.info("Falling back to OpenCV")
                    self.cam = cv2.VideoCapture(0)
                    self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
                    self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
                    self.cam.set(cv2.CAP_PROP_FPS, 15)
                    if not self.cam.isOpened():
                        logger.error("OpenCV camera also failed")
                        return None
        
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while not self.stopped:
            try:
                current_time = time.time()
                
                if self.picam2:
                    frame = self.picam2.capture_array()
                    if frame is not None and frame.size > 0:
                        if frame.shape != (DISPLAY_H, DISPLAY_W, 3):
                            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                        
                        with self.lock:
                            self.frame = frame
                            self.grabbed = True
                            self.last_valid_frame = frame.copy()
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        logger.warning(f"Empty frame from picam2 (error {consecutive_errors})")
                else:
                    ret, frame = self.cam.read()
                    if ret and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if frame.shape != (DISPLAY_H, DISPLAY_W, 3):
                            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                        
                        with self.lock:
                            self.frame = frame
                            self.grabbed = True
                            self.last_valid_frame = frame.copy()
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        logger.warning(f"Failed to read from OpenCV (error {consecutive_errors})")
                
                # Calculate FPS
                self.frame_counter += 1
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_counter
                    self.fps_buffer.append(self.fps)
                    avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
                    logger.debug(f"Camera FPS: {self.fps} (avg: {avg_fps:.1f})")
                    self.frame_counter = 0
                    self.last_fps_time = current_time
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive camera errors")
                    self.stopped = True
                    break
                    
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Camera update error: {e}")
                consecutive_errors += 1
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            elif self.last_valid_frame is not None:
                return self.last_valid_frame.copy()
            return None
    
    def get_fps(self):
        with self.lock:
            return self.fps if hasattr(self, 'fps') else 0
    
    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
        if self.cam:
            try:
                self.cam.release()
            except:
                pass
        logger.info("Camera stopped")

# ================= ENHANCED RESOURCES & STATE =================
class HISLoader:
    @staticmethod
    def load_database(filename):
        if not os.path.exists(filename):
            logger.warning(f"Database file not found: {filename}")
            return {}
        db = {}
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split('|')
                    if len(parts) < 3:
                        logger.warning(f"Invalid line {line_num}: {line}")
                        continue
                    hn = parts[0].strip()
                    name = parts[1].strip()
                    drugs = []
                    for d in parts[2].split(','):
                        drug = d.strip().lower().replace('\ufeff', '')
                        if drug:
                            drugs.append(drug)
                    if hn and name and drugs:
                        db[hn] = {
                            'name': name,
                            'drugs': drugs,
                            'loaded_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        logger.info(f"Loaded patient: {name} ({hn}) with {len(drugs)} drugs")
                    else:
                        logger.warning(f"Incomplete data on line {line_num}")
            logger.info(f"Loaded {len(db)} patients from database")
            return db
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return {}

class PrescriptionState:
    def __init__(self):
        self.all_drugs = []
        self.verified_drugs = set()
        self.verification_history = []  # Track verification history
        self.detection_stats = defaultdict(lambda: {'count': 0, 'last_seen': None})
        self.lock = threading.Lock()
    
    def load_drugs(self, drug_list):
        with self.lock:
            self.all_drugs = drug_list.copy()
            self.verified_drugs.clear()
            self.verification_history.clear()
            self.detection_stats.clear()
            logger.info(f"Loaded {len(drug_list)} drugs into state")
    
    def get_remaining_drugs(self):
        with self.lock:
            remaining = [d for d in self.all_drugs if d not in self.verified_drugs]
            logger.debug(f"{len(remaining)} drugs remaining to verify")
            return remaining
    
    def toggle_drug(self, drug_name):
        with self.lock:
            drug_lower = drug_name.lower()
            if drug_lower in self.verified_drugs:
                self.verified_drugs.remove(drug_lower)
                self.verification_history.append({
                    'action': 'unverified',
                    'drug': drug_name,
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                logger.info(f"Manually unverified: {drug_name}")
            else:
                self.verified_drugs.add(drug_lower)
                self.verification_history.append({
                    'action': 'verified',
                    'drug': drug_name,
                    'time': datetime.now().strftime("%H:%M:%S")
                })
                logger.info(f"Manually verified: {drug_name}")

    def verify_drug(self, drug_name):
        with self.lock:
            drug_lower = drug_name.lower()
            
            # Check if already verified
            if drug_lower in self.verified_drugs:
                return
            
            # Log verification attempt
            verification_record = {
                'drug': drug_name,
                'time': datetime.now().strftime("%H:%M:%S"),
                'type': 'auto'
            }
            
            # Check against prescribed drugs
            matched = False
            for prescribed in self.all_drugs:
                if prescribed.lower() == drug_lower:
                    self.verified_drugs.add(drug_lower)
                    verification_record['action'] = 'verified'
                    verification_record['match_type'] = 'exact'
                    matched = True
                    logger.info(f"✨ AUTO VERIFIED (Exact match): {drug_name}")
                    break
            
            # If no exact match, try partial match
            if not matched:
                for prescribed in self.all_drugs:
                    if drug_lower in prescribed.lower() or prescribed.lower() in drug_lower:
                        self.verified_drugs.add(prescribed.lower())
                        verification_record['action'] = 'verified'
                        verification_record['match_type'] = 'partial'
                        verification_record['prescribed_drug'] = prescribed
                        logger.info(f"✨ AUTO VERIFIED (Partial match): {drug_name} -> {prescribed}")
                        matched = True
                        break
            
            if not matched:
                verification_record['action'] = 'rejected'
                logger.warning(f"Auto verification rejected: {drug_name} not in prescription")
            
            self.verification_history.append(verification_record)
    
    def record_detection(self, drug_name, confidence):
        with self.lock:
            drug_lower = drug_name.lower()
            self.detection_stats[drug_lower]['count'] += 1
            self.detection_stats[drug_lower]['last_seen'] = datetime.now().strftime("%H:%M:%S")
            if 'confidence' not in self.detection_stats[drug_lower]:
                self.detection_stats[drug_lower]['confidence'] = []
            self.detection_stats[drug_lower]['confidence'].append(confidence)
    
    def get_detection_stats(self, drug_name):
        with self.lock:
            stats = self.detection_stats.get(drug_name.lower(), {})
            if 'confidence' in stats and stats['confidence']:
                avg_conf = sum(stats['confidence']) / len(stats['confidence'])
                stats['avg_confidence'] = avg_conf
            return stats
    
    def is_verified(self, drug_name):
        with self.lock:
            return drug_name.lower() in self.verified_drugs
    
    def get_all_drugs(self):
        with self.lock:
            return self.all_drugs.copy()
    
    def get_verification_history(self):
        with self.lock:
            return self.verification_history.copy()
    
    def get_summary(self):
        with self.lock:
            total = len(self.all_drugs)
            verified = len(self.verified_drugs)
            remaining = total - verified
            return {
                'total': total,
                'verified': verified,
                'remaining': remaining,
                'progress': verified/total*100 if total > 0 else 0
            }

prescription_state = PrescriptionState()

def load_pkl_to_list(filepath):
    if not os.path.exists(filepath):
        logger.warning(f"PKL file not found: {filepath}")
        return [], []
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            items = [(v, name) for name, vec_list in data.items() for v in vec_list]
            if items:
                vecs, lbls = zip(*items)
                logger.info(f"Loaded {len(vecs)} vectors from {filepath}")
                return list(vecs), list(lbls)
            else:
                logger.warning(f"No data in {filepath}")
                return [], []
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return [], []

# Load Global DB with enhanced error handling
logger.info("Loading databases...")
pills_vecs, pills_lbls = load_pkl_to_list(DB_FILES['pills']['vec'])
packs_vecs, packs_lbls = load_pkl_to_list(DB_FILES['packs']['vec'])

if pills_vecs:
    matrix_pills = torch.tensor(np.array(pills_vecs), device=device, dtype=torch.float32)
    matrix_pills = matrix_pills / matrix_pills.norm(dim=1, keepdim=True)
    logger.info(f"Pills matrix: {matrix_pills.shape}")
else:
    matrix_pills = None
    logger.warning("No pills vectors loaded")

if packs_vecs:
    matrix_packs = torch.tensor(np.array(packs_vecs), device=device, dtype=torch.float32)
    matrix_packs = matrix_packs / matrix_packs.norm(dim=1, keepdim=True)
    logger.info(f"Packs matrix: {matrix_packs.shape}")
else:
    matrix_packs = None
    logger.warning("No packs vectors loaded")

# Load color database
color_db = {}
for db_type in ['pills', 'packs']:
    try:
        if os.path.exists(DB_FILES[db_type]['col']):
            with open(DB_FILES[db_type]['col'], 'rb') as f:
                db_data = pickle.load(f)
                color_db.update(db_data)
                logger.info(f"Loaded {len(db_data)} colors from {db_type}")
    except Exception as e:
        logger.error(f"Error loading color DB {db_type}: {e}")

# Initialize SIFT with enhanced parameters
sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=0.01, edgeThreshold=10)
bf = cv2.BFMatcher(crossCheck=False)
sift_db = {}

# Load SIFT database with progress tracking
if os.path.exists(IMG_DB_FOLDER):
    logger.info(f"Loading SIFT database from {IMG_DB_FOLDER}...")
    folders = [f for f in os.listdir(IMG_DB_FOLDER) if os.path.isdir(os.path.join(IMG_DB_FOLDER, f))]
    for idx, folder in enumerate(folders, 1):
        path = os.path.join(IMG_DB_FOLDER, folder)
        des_list = []
        image_files = [x for x in os.listdir(path) if x.lower().endswith(('jpg', 'png', 'jpeg'))][:5]
        
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if max(img.shape) > 512:
                    scale = 512 / max(img.shape)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                _, des = sift.detectAndCompute(img, None)
                if des is not None:
                    des_list.append(des)
        if des_list:
            sift_db[folder] = des_list
        
        if idx % 10 == 0 or idx == len(folders):
            logger.info(f"SIFT loading: {idx}/{len(folders)} folders")
    
    logger.info(f"Loaded SIFT features for {len(sift_db)} drugs")
else:
    logger.warning(f"Image database folder not found: {IMG_DB_FOLDER}")

# Load models with enhanced error handling
logger.info("Loading AI models...")
try:
    model_pill = YOLO(MODEL_PILL_PATH, task='detect')
    model_pack = YOLO(MODEL_PACK_PATH, task='detect')
    
    # Test models
    test_input = torch.randn(1, 3, AI_IMG_SIZE, AI_IMG_SIZE).to(device)
    logger.info("YOLO models loaded successfully")
    
    # Load feature extractor
    weights = models.ResNet50_Weights.DEFAULT
    base_model = models.resnet50(weights=weights)
    embedder = torch.nn.Sequential(*list(base_model.children())[:-1])
    embedder.eval().to(device)
    del base_model
    
    # Test feature extractor
    with torch.no_grad():
        test_features = embedder(test_input)
    logger.info(f"Feature extractor output shape: {test_features.shape}")
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    torch.set_grad_enabled(False)
    logger.info("All models loaded and tested successfully")
    
except Exception as e:
    logger.error(f"Model loading error: {e}")
    sys.exit(1)

# ================= ENHANCED TRINITY ENGINE =================
COLOR_NORM = np.array([90.0, 255.0, 255.0])
SIFT_RATIO = 0.75
SIFT_MAX_MATCHES = 20.0

def enhanced_color_analysis(img_crop):
    """Enhanced color analysis with multiple sampling points"""
    h, w = img_crop.shape[:2]
    color_samples = []
    
    # Sample from multiple regions
    regions = [
        (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)),  # Center
        (int(w*0.1), int(h*0.1), int(w*0.4), int(h*0.4)),      # Top-left
        (int(w*0.6), int(h*0.6), int(w*0.9), int(h*0.9)),      # Bottom-right
    ]
    
    for x1, y1, x2, y2 in regions:
        region = img_crop[y1:y2, x1:x2]
        if region.size > 0:
            hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            avg_color = np.mean(hsv, axis=(0,1))
            color_samples.append(avg_color)
    
    if color_samples:
        return np.mean(color_samples, axis=0)
    return None

def enhanced_sift_matching(des_live, drug_name):
    """Enhanced SIFT matching with multiple strategies"""
    if des_live is None or drug_name not in sift_db:
        return 0.0
    
    max_good = 0
    total_matches = 0
    total_keypoints = len(des_live)
    
    for ref_des in sift_db[drug_name]:
        if ref_des is None or len(ref_des) < 2:
            continue
            
        try:
            # KNN matching
            matches = bf.knnMatch(des_live, ref_des, k=2)
            good_matches = []
            
            for m, n in matches:
                if len([m, n]) == 2 and m.distance < SIFT_RATIO * n.distance:
                    good_matches.append(m)
            
            # Ratio test
            if len(matches) > 0:
                good_ratio = len(good_matches) / len(matches)
                max_good = max(max_good, len(good_matches))
                total_matches += len(good_matches)
                
        except Exception as e:
            logger.debug(f"SIFT matching error: {e}")
            continue
    
    # Calculate score based on multiple factors
    if total_keypoints == 0:
        return 0.0
    
    # Score 1: Best match ratio
    score1 = min(max_good / SIFT_MAX_MATCHES, 1.0)
    
    # Score 2: Average matches per descriptor
    score2 = min(total_matches / max(len(sift_db[drug_name]), 1) / total_keypoints * 10, 1.0)
    
    # Combined score
    return 0.7 * score1 + 0.3 * score2

def trinity_inference(img_crop, is_pill=True, 
                      session_pills=None, session_pills_lbl=None,
                      session_packs=None, session_packs_lbl=None):
    
    target_matrix = (session_pills if session_pills is not None else matrix_pills) if is_pill else \
                    (session_packs if session_packs is not None else matrix_packs)
    target_labels = (session_pills_lbl if session_pills_lbl is not None else pills_lbls) if is_pill else \
                    (session_packs_lbl if session_packs_lbl is not None else packs_lbls)
    
    if target_matrix is None or len(target_labels) == 0:
        logger.warning("Database empty in trinity_inference")
        return "DB Error", 0.0
    
    try:
        # Preprocessing
        if is_pill:
            pil_img = Image.fromarray(img_crop)
        else:
            gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
            # Enhance contrast for pack detection
            gray_crop = cv2.equalizeHist(gray_crop)
            crop_3ch_gray = cv2.merge([gray_crop, gray_crop, gray_crop])
            pil_img = Image.fromarray(crop_3ch_gray)
        
        # Feature extraction
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        live_vec = embedder(input_tensor).flatten()
        live_vec = live_vec / live_vec.norm()
        
        # Similarity calculation
        scores = torch.matmul(live_vec, target_matrix.T).squeeze(0)
        k_val = min(15, len(target_labels))
        
        if k_val == 0:
            return "Unknown", 0.0
        
        # Get top candidates
        top_k_val, top_k_idx = torch.topk(scores, k=k_val)
        candidates = []
        seen = set()
        
        for idx, sc in zip(top_k_idx.detach().cpu().numpy(), top_k_val.detach().cpu().numpy()):
            name = target_labels[idx]
            if name not in seen and sc > 0.1:  # Minimum similarity threshold
                candidates.append((name, float(sc)))
                seen.add(name)
                if len(candidates) >= 5:  # Consider more candidates
                    break
        
        if not candidates:
            return "Unknown", 0.0
        
        # Enhanced feature extraction for matching
        live_color = None
        gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        
        # Enhance image for better feature detection
        gray = cv2.equalizeHist(gray)
        _, des_live = sift.detectAndCompute(gray, None)
        
        if is_pill:
            live_color = enhanced_color_analysis(img_crop)
        
        # Evaluate candidates
        best_score = -1
        final_name = "Unknown"
        confidence_breakdown = {}
        
        for name, vec_score in candidates:
            clean_name = name.replace("_pill", "").replace("_pack", "")
            
            # SIFT matching
            sift_score = enhanced_sift_matching(des_live, clean_name)
            
            # Color matching
            color_score = 0.0
            if is_pill and live_color is not None and name in color_db:
                ref_color = color_db[name]
                # Handle hue circularity
                hue_diff = min(abs(live_color[0] - ref_color[0]), 
                             180 - abs(live_color[0] - ref_color[0]))
                sat_diff = abs(live_color[1] - ref_color[1])
                val_diff = abs(live_color[2] - ref_color[2])
                
                # Normalized differences
                norm_diff = np.array([hue_diff/180.0, sat_diff/255.0, val_diff/255.0])
                dist = np.linalg.norm(norm_diff)
                color_score = max(0, 1.0 - dist)
            
            # Weighted scoring
            if is_pill:
                w_vec, w_sift, w_col = (0.4, 0.4, 0.2)  # Balanced weights for pills
            else:
                w_vec, w_sift, w_col = (0.3, 0.7, 0.0)  # Emphasize SIFT for packs
            
            total_score = vec_score * w_vec + sift_score * w_sift + color_score * w_col
            
            # Store breakdown for debugging
            confidence_breakdown[clean_name] = {
                'total': total_score,
                'vector': vec_score,
                'sift': sift_score,
                'color': color_score
            }
            
            if total_score > best_score:
                best_score = total_score
                final_name = clean_name
        
        # Log detailed results for top candidate
        if final_name != "Unknown" and best_score > 0.3:
            logger.debug(f"Trinity inference: {final_name} (score: {best_score:.3f})")
            if final_name in confidence_breakdown:
                logger.debug(f"  Breakdown: {confidence_breakdown[final_name]}")
        
        return final_name, best_score
        
    except Exception as e:
        logger.error(f"Trinity inference error: {e}")
        return "Error", 0.0

# ================= ENHANCED AI WORKER =================
class AIProcessor:
    __slots__ = ('latest_frame', 'results', 'stopped', 'lock', 'is_rx_mode', 
                 'current_patient_info', 'scale_x', 'scale_y',
                 'resize_interpolation', 'consistency_counter',
                 'frame_buffer', 'last_health_check', 'detection_log',
                 'performance_stats')
    
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
        self.consistency_counter = defaultdict(int)
        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.last_health_check = time.time()
        self.detection_log = deque(maxlen=100)
        self.performance_stats = {
            'total_frames': 0,
            'avg_processing_time': 0,
            'detection_counts': defaultdict(int)
        }
    
    def load_patient(self, patient_data):
        with self.lock:
            if not patient_data:
                self.is_rx_mode = False
                self.current_patient_info = None
                prescription_state.load_drugs([])
                self.consistency_counter.clear()
                self.detection_log.clear()
                logger.info("Cleared patient data")
            else:
                self.is_rx_mode = True
                self.current_patient_info = patient_data
                drugs = patient_data['drugs']
                prescription_state.load_drugs(drugs)
                self.consistency_counter.clear()
                self.detection_log.clear()
                logger.info(f"🏥 Loaded patient: {patient_data['name']} ({len(drugs)} drugs)")
    
    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self
    
    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame
    
    def get_results(self):
        with self.lock:
            return self.results.copy(), self.current_patient_info
    
    def health_check(self):
        """Perform system health checks"""
        current_time = time.time()
        if current_time - self.last_health_check >= HEALTH_CHECK_INTERVAL:
            # Check memory usage
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent}%")
            except:
                pass
            
            # Log performance stats
            logger.info(f"Performance: {self.performance_stats['total_frames']} frames processed")
            
            self.last_health_check = current_time
    
    def run(self):
        logger.info("AI Worker Loop Started (Enhanced Mode)")
        
        frame_count = 0
        processing_times = deque(maxlen=30)
        
        while not self.stopped:
            start_time = time.time()
            
            # Perform health check
            self.health_check()
            
            # Get frame
            with self.lock:
                frame_HD = self.latest_frame
                self.latest_frame = None
            
            if frame_HD is None:
                time.sleep(0.005)
                continue
            
            # Add to frame buffer
            self.frame_buffer.append(frame_HD.copy())
            
            # Resize for YOLO
            frame_yolo = cv2.resize(frame_HD, (AI_IMG_SIZE, AI_IMG_SIZE),
                                   interpolation=self.resize_interpolation)
            
            final_detections = []
            active_packs = []
            found_in_this_frame = set()
            
            try:
                # ==========================================
                # PHASE 1: DETECT PACKS
                # ==========================================
                pack_start = time.time()
                pack_res = model_pack(frame_yolo, verbose=False, conf=CONF_PACK,
                                     imgsz=AI_IMG_SIZE, max_det=5, agnostic_nms=True)
                pack_time = time.time() - pack_start
                
                if pack_res and len(pack_res) > 0 and hasattr(pack_res[0], 'boxes'):
                    boxes = pack_res[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        boxes_array = boxes.xyxy.detach().cpu().numpy().astype(int)
                        
                        for box in boxes_array:
                            x1_s, y1_s, x2_s, y2_s = box
                            x1, y1 = int(x1_s * self.scale_x), int(y1_s * self.scale_y)
                            x2, y2 = int(x2_s * self.scale_x), int(y2_s * self.scale_y)
                            
                            # Enhanced validation
                            if not validate_box((x1, y1, x2, y2), DISPLAY_W, DISPLAY_H):
                                continue
                            
                            if (x2 - x1) < MIN_PACK_SIZE or (y2 - y1) < MIN_PACK_SIZE:
                                continue
                            
                            crop = frame_HD[y1:y2, x1:x2]
                            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                                continue
                            
                            # Trinity inference
                            real_name, real_score = trinity_inference(
                                crop, is_pill=False,
                                session_pills=matrix_pills,
                                session_pills_lbl=pills_lbls,
                                session_packs=matrix_packs,
                                session_packs_lbl=packs_lbls
                            )
                            
                            final_name = real_name
                            final_score = real_score
                            is_wrong_drug = False
                            
                            # Prescription validation
                            if self.is_rx_mode and real_score >= SCORE_PASS_PACK:
                                clean_real = real_name.replace("_pack", "").lower().strip()
                                allowed_drugs = [d.lower() for d in prescription_state.get_all_drugs()]
                                match_found = False
                                
                                for allowed in allowed_drugs:
                                    # Enhanced matching logic
                                    if (allowed == clean_real or
                                        allowed in clean_real or
                                        clean_real in allowed or
                                        allowed.split()[0] == clean_real.split()[0]):
                                        match_found = True
                                        final_name = allowed
                                        break
                                
                                if not match_found and "?" not in real_name and "Unknown" not in real_name:
                                    final_name = f"WRONG: {real_name}"
                                    final_score = 0.0
                                    is_wrong_drug = True
                            
                            clean_name = final_name.replace("_pack", "").lower()
                            
                            # Update consistency counter
                            if (not is_wrong_drug and
                                "?" not in final_name and
                                "Unknown" not in final_name and
                                final_score >= SCORE_PASS_PACK):
                                
                                self.consistency_counter[clean_name] += 1
                                found_in_this_frame.add(clean_name)
                                
                                # Record detection for statistics
                                prescription_state.record_detection(clean_name, final_score)
                                
                                # Auto-verification
                                if self.consistency_counter[clean_name] >= CONSISTENCY_THRESHOLD:
                                    prescription_state.verify_drug(clean_name)
                            
                            # Check verification status
                            pack_verified = prescription_state.is_verified(clean_name)
                            
                            pack_data = {
                                'label': final_name,
                                'score': final_score,
                                'type': 'pack',
                                'verified': pack_verified,
                                'box': (x1, y1, x2, y2),
                                'is_wrong': is_wrong_drug,
                                'clean_name': clean_name,
                                'detection_time': time.time()
                            }
                            
                            active_packs.append(pack_data)
                            final_detections.append(pack_data)
                            
                            # Log detection
                            self.detection_log.append({
                                'type': 'pack',
                                'name': final_name,
                                'score': final_score,
                                'verified': pack_verified,
                                'wrong': is_wrong_drug,
                                'time': time.time()
                            })
                
                # ==========================================
                # PHASE 2: DETECT PILLS
                # ==========================================
                pill_start = time.time()
                pill_res = model_pill(frame_yolo, verbose=False, conf=CONF_PILL,
                                     imgsz=AI_IMG_SIZE, max_det=MAX_DETECTIONS_PER_FRAME,
                                     agnostic_nms=True)
                pill_time = time.time() - pill_start
                
                if pill_res and len(pill_res) > 0 and hasattr(pill_res[0], 'boxes'):
                    boxes = pill_res[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        boxes_array = boxes.xyxy.detach().cpu().numpy().astype(int)
                        
                        for box in boxes_array:
                            x1_s, y1_s, x2_s, y2_s = box
                            x1, y1 = int(x1_s * self.scale_x), int(y1_s * self.scale_y)
                            x2, y2 = int(x2_s * self.scale_x), int(y2_s * self.scale_y)
                            
                            # Enhanced validation
                            if not validate_box((x1, y1, x2, y2), DISPLAY_W, DISPLAY_H):
                                continue
                            
                            if (x2 - x1) < MIN_PILL_SIZE or (y2 - y1) < MIN_PILL_SIZE:
                                continue
                            
                            cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1
                            
                            # Check if pill is inside a pack
                            parent_pack = None
                            for pack in active_packs:
                                if is_point_in_box((cx, cy), pack['box']):
                                    parent_pack = pack
                                    break
                            
                            final_name = "Unknown"
                            final_score = 0.0
                            is_wrong_drug = False
                            is_verified = False
                            
                            if parent_pack:
                                # Inherit from parent pack
                                final_name = parent_pack['label']
                                final_score = parent_pack['score']
                                is_wrong_drug = parent_pack['is_wrong']
                                is_verified = parent_pack['verified']
                                
                                clean_name = parent_pack['clean_name']
                                if not is_wrong_drug:
                                    self.consistency_counter[clean_name] += 1
                                    found_in_this_frame.add(clean_name)
                                    prescription_state.record_detection(clean_name, final_score)
                            else:
                                # Independent pill detection
                                crop = frame_HD[y1:y2, x1:x2]
                                if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                                    real_name, real_score = trinity_inference(
                                        crop, is_pill=True,
                                        session_pills=matrix_pills,
                                        session_pills_lbl=pills_lbls,
                                        session_packs=matrix_packs,
                                        session_packs_lbl=packs_lbls
                                    )
                                    
                                    final_name = real_name
                                    final_score = real_score
                                    
                                    # Prescription validation
                                    if self.is_rx_mode and real_score >= SCORE_PASS_PILL:
                                        clean_real = real_name.lower().strip()
                                        allowed_drugs = [d.lower() for d in prescription_state.get_all_drugs()]
                                        match_found = False
                                        
                                        for allowed in allowed_drugs:
                                            if (allowed == clean_real or
                                                allowed in clean_real or
                                                clean_real in allowed or
                                                allowed.split()[0] == clean_real.split()[0]):
                                                match_found = True
                                                final_name = allowed
                                                break
                                        
                                        if not match_found and "?" not in real_name and "Unknown" not in real_name:
                                            final_name = f"WRONG: {real_name}"
                                            final_score = 0.0
                                            is_wrong_drug = True
                                    
                                    clean_name = final_name.lower()
                                    
                                    # Update consistency counter
                                    if (not is_wrong_drug and
                                        "?" not in final_name and
                                        "Unknown" not in final_name and
                                        final_score >= SCORE_PASS_PILL):
                                        
                                        self.consistency_counter[clean_name] += 1
                                        found_in_this_frame.add(clean_name)
                                        prescription_state.record_detection(clean_name, final_score)
                                        
                                        # Auto-verification
                                        if self.consistency_counter[clean_name] >= CONSISTENCY_THRESHOLD:
                                            prescription_state.verify_drug(clean_name)
                                    
                                    is_verified = prescription_state.is_verified(clean_name)
                            
                            pill_data = {
                                'label': final_name,
                                'score': final_score,
                                'type': 'pill',
                                'verified': is_verified,
                                'box': (x1, y1, x2, y2),
                                'is_wrong': is_wrong_drug,
                                'parent_pack': parent_pack['clean_name'] if parent_pack else None,
                                'detection_time': time.time()
                            }
                            
                            final_detections.append(pill_data)
                            
                            # Log detection
                            self.detection_log.append({
                                'type': 'pill',
                                'name': final_name,
                                'score': final_score,
                                'verified': is_verified,
                                'wrong': is_wrong_drug,
                                'parent': parent_pack['clean_name'] if parent_pack else None,
                                'time': time.time()
                            })
                
                # Clean up consistency counter
                all_tracked = list(self.consistency_counter.keys())
                for k in all_tracked:
                    if k not in found_in_this_frame:
                        self.consistency_counter[k] = max(0, self.consistency_counter[k] - 1)
                        if self.consistency_counter[k] == 0:
                            del self.consistency_counter[k]
                
                # Update results
                with self.lock:
                    self.results = final_detections
                
                # Update performance stats
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                frame_count += 1
                
                self.performance_stats['total_frames'] = frame_count
                self.performance_stats['avg_processing_time'] = sum(processing_times) / len(processing_times) if processing_times else 0
                
                if frame_count % 30 == 0:
                    logger.debug(f"Processing: pack={pack_time:.3f}s, pill={pill_time:.3f}s, total={processing_time:.3f}s")
            
            except Exception as e:
                logger.error(f"AI processing error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(0.1)
    
    def stop(self):
        self.stopped = True
        logger.info("AI Processor stopped")

# ================= ENHANCED UI DRAWING =================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_SCALE_SMALL = 0.6
FONT_SCALE_LARGE = 1.2
THICKNESS = 2
THICKNESS_BOX = 3
CHECKBOX_SIZE = 25

RGB_GREEN = (0, 255, 0)
RGB_RED = (255, 0, 0)
RGB_BLUE = (0, 0, 255)
RGB_YELLOW = (255, 255, 0)
RGB_WHITE = (255, 255, 255)
RGB_GRAY = (50, 50, 50)
RGB_BLACK = (0, 0, 0)
RGB_ORANGE = (255, 165, 0)
RGB_PURPLE = (255, 0, 255)

def draw_status_panel(frame, patient_info, ai_processor):
    """Draw comprehensive status panel"""
    h, w = frame.shape[:2]
    panel_h = 200
    panel_y = h - panel_h
    
    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = panel_y + 30
    
    if patient_info:
        # Patient info
        cv2.putText(frame, f"Patient: {patient_info.get('name', 'N/A')}", 
                   (20, y_offset), FONT, FONT_SCALE, RGB_GREEN, THICKNESS)
        cv2.putText(frame, f"HN: {patient_info.get('hn', 'N/A')}", 
                   (20, y_offset + 30), FONT, FONT_SCALE_SMALL, RGB_WHITE, 1)
        
        # Prescription summary
        summary = prescription_state.get_summary()
        cv2.putText(frame, f"Drugs: {summary['verified']}/{summary['total']} verified", 
                   (20, y_offset + 60), FONT, FONT_SCALE_SMALL, RGB_YELLOW, 1)
        
        # Progress bar
        bar_width = 200
        bar_height = 15
        bar_x = 20
        bar_y = y_offset + 90
        progress = summary['verified'] / summary['total'] if summary['total'] > 0 else 0
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     RGB_GRAY, -1)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_width * progress), bar_y + bar_height), 
                     RGB_GREEN, -1)
        cv2.putText(frame, f"{progress:.0%}", 
                   (bar_x + bar_width + 10, bar_y + bar_height - 2), 
                   FONT, FONT_SCALE_SMALL, RGB_WHITE, 1)
        
        # Remaining drugs
        remaining = prescription_state.get_remaining_drugs()
        if remaining:
            drugs_text = "Remaining: " + ", ".join(remaining[:3])
            if len(remaining) > 3:
                drugs_text += f" (+{len(remaining)-3})"
            cv2.putText(frame, drugs_text, 
                       (20, y_offset + 120), FONT, FONT_SCALE_SMALL, RGB_ORANGE, 1)
    else:
        cv2.putText(frame, "No patient selected", 
                   (20, y_offset), FONT, FONT_SCALE, RGB_YELLOW, THICKNESS)
        cv2.putText(frame, "Press 'R' to reload or 'L' to list patients", 
                   (20, y_offset + 30), FONT, FONT_SCALE_SMALL, RGB_WHITE, 1)
    
    # System info
    system_info = get_system_info()
    cv2.putText(frame, f"CPU: {system_info['cpu_temp']}", 
               (w - 200, y_offset), FONT, FONT_SCALE_SMALL, RGB_WHITE, 1)
    cv2.putText(frame, f"Mem: {system_info['memory'].split('(')[-1].replace(')', '')}", 
               (w - 200, y_offset + 25), FONT, FONT_SCALE_SMALL, RGB_WHITE, 1)

def draw_enhanced_boxes(frame, results):
    """Draw enhanced bounding boxes with more information"""
    for r in results:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        score = r['score']
        obj_type = r.get('type', 'pill')
        is_verified = r.get('verified', False)
        is_wrong = r.get('is_wrong', False)
        
        # Determine color and label
        if is_wrong:
            color = RGB_RED
            label_display = f"!! {label} !!"
            thickness = THICKNESS_BOX + 1
        elif is_verified:
            color = RGB_GREEN
            label_display = f"✓ {label}"
            thickness = THICKNESS_BOX
        elif obj_type == 'pack':
            if score >= SCORE_PASS_PACK:
                color = RGB_BLUE
                thickness = THICKNESS_BOX
            else:
                color = RGB_YELLOW
                thickness = THICKNESS_BOX - 1
            label_display = label
        elif "?" in label or score < SCORE_PASS_PILL:
            color = RGB_RED
            thickness = THICKNESS_BOX - 1
            label_display = label
        else:
            color = RGB_YELLOW
            thickness = THICKNESS_BOX - 1
            label_display = label
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_size = cv2.getTextSize(f"{label_display} {score:.0%}", 
                                    FONT, FONT_SCALE_SMALL, THICKNESS)[0]
        label_y = max(y1 - 10, 20)
        cv2.rectangle(frame, 
                     (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0] + 10, label_y + 5),
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, f"{label_display} {score:.0%}", 
                   (x1 + 5, label_y), 
                   FONT, FONT_SCALE_SMALL, RGB_WHITE, 1)
        
        # Draw type indicator
        if obj_type == 'pack':
            cv2.circle(frame, (x2 - 10, y1 + 10), 5, RGB_PURPLE, -1)
        else:
            cv2.circle(frame, (x2 - 10, y1 + 10), 3, RGB_ORANGE, -1)

# ================= ENHANCED MAIN =================
def list_patients(his_db):
    """List available patients"""
    print("\n" + "="*50)
    print("AVAILABLE PATIENTS")
    print("="*50)
    for hn, data in his_db.items():
        print(f"HN: {hn}")
        print(f"Name: {data['name']}")
        print(f"Drugs: {', '.join(data['drugs'][:5])}")
        if len(data['drugs']) > 5:
            print(f"     + {len(data['drugs'])-5} more")
        print("-"*50)
    
    if not his_db:
        print("No patients found in database")
    print("="*50 + "\n")

def mouse_callback(event, x, y, flags, param):
    """Enhanced mouse callback with visual feedback"""
    if event == cv2.EVENT_LBUTTONDOWN:
        clickable_areas, ai_processor = param
        
        # Check status panel clicks
        h, w = 720, 1280
        panel_h = 200
        panel_y = h - panel_h
        
        if y > panel_y:
            # Click in status panel - could add more functionality
            pass
        
        # Check drug verification boxes
        for area in clickable_areas:
            x1, y1, x2, y2 = area['box']
            if x1 <= x <= x2 and y1 <= y <= y2:
                drug = area['drug']
                old_state = prescription_state.is_verified(drug)
                prescription_state.toggle_drug(drug)
                new_state = prescription_state.is_verified(drug)
                
                action = "verified" if new_state else "unverified"
                logger.info(f"Manual {action}: {drug}")
                break

def main():
    # Initialization
    logger.info("="*60)
    logger.info("PillTrack Senior Edition - Enhanced Version")
    logger.info(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Load configuration
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    # Get target HN from config or user input
    TARGET_HN = config.get('default_patient', 'HN-101')
    
    # Initialize camera
    cam = WebcamStream().start()
    if cam is None:
        logger.error("Failed to initialize camera")
        return
    
    # Initialize AI processor
    ai = AIProcessor().start()
    
    # Load patient database
    his_db = HISLoader.load_database(HIS_FILE_PATH)
    
    # List available patients
    list_patients(his_db)
    
    # Load initial patient
    if TARGET_HN in his_db:
        d = his_db[TARGET_HN].copy()
        d['hn'] = TARGET_HN
        ai.load_patient(d)
        logger.info(f"Loaded initial patient: {d['name']} ({TARGET_HN})")
    else:
        logger.warning(f"Patient {TARGET_HN} not found in database")
        if his_db:
            first_hn = list(his_db.keys())[0]
            d = his_db[first_hn].copy()
            d['hn'] = first_hn
            ai.load_patient(d)
            logger.info(f"Loaded first patient: {d['name']} ({first_hn})")
    
    # Wait for camera
    logger.info("Waiting for camera feed...")
    for _ in range(100):
        if cam.read() is not None:
            break
        time.sleep(0.1)
    
    if cam.read() is None:
        logger.error("Camera feed not available")
        cam.stop()
        return
    
    # Create window
    window_name = "PillTrack Senior Edition - Enhanced"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback(window_name, mouse_callback, ([], ai))
    
    logger.info(f"System running with ZOOM FACTOR: {ZOOM_FACTOR}x")
    
    # FPS calculation
    fps = 0
    prev_time = time.perf_counter()
    fps_buffer = deque(maxlen=30)
    TARGET_FPS = 15
    FRAME_TIME = 1.0 / TARGET_FPS
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.perf_counter()
            frame_count += 1
            
            # Read and process frame
            frame_rgb = cam.read()
            if frame_rgb is None:
                logger.warning("Empty frame from camera")
                time.sleep(0.01)
                continue
            
            # Apply digital zoom
            if ZOOM_FACTOR > 1.0:
                frame_rgb = apply_digital_zoom(frame_rgb, ZOOM_FACTOR)
            
            # Update AI processor
            ai.update_frame(frame_rgb.copy())
            
            # Get results
            results, cur_patient = ai.get_results()
            
            # Draw on frame
            draw_enhanced_boxes(frame_rgb, results)
            draw_status_panel(frame_rgb, cur_patient, ai)
            
            # Update FPS
            curr_time = time.perf_counter()
            frame_time = curr_time - prev_time
            fps = 1 / frame_time if frame_time > 0 else 0
            fps_buffer.append(fps)
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            prev_time = curr_time
            
            # Display FPS and system info
            camera_fps = cam.get_fps()
            system_info = get_system_info()
            
            info_text = f"FPS: {avg_fps:.1f} (Cam: {camera_fps}) | CPU: {system_info['cpu_temp']} | Zoom: {ZOOM_FACTOR}x"
            cv2.putText(frame_rgb, info_text, (30, 50), 
                       FONT, 1.0, RGB_GREEN, THICKNESS)
            
            # Display detection count
            if results:
                pill_count = sum(1 for r in results if r['type'] == 'pill')
                pack_count = sum(1 for r in results if r['type'] == 'pack')
                count_text = f"Detections: {pill_count} pills, {pack_count} packs"
                cv2.putText(frame_rgb, count_text, (30, 85), 
                           FONT, 0.8, RGB_YELLOW, 1)
            
            # Display current time
            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame_rgb, current_time, (DISPLAY_W - 150, 50), 
                       FONT, 1.0, RGB_WHITE, THICKNESS)
            
            # Show frame
            cv2.imshow(window_name, frame_rgb)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            elif key == ord('r'):
                # Reload patient database
                his_db = HISLoader.load_database(HIS_FILE_PATH)
                if TARGET_HN in his_db:
                    d = his_db[TARGET_HN].copy()
                    d['hn'] = TARGET_HN
                    ai.load_patient(d)
                    logger.info(f"Reloaded patient: {d['name']}")
                else:
                    logger.warning(f"Patient {TARGET_HN} not found after reload")
            elif key == ord('l'):
                # List patients
                list_patients(his_db)
            elif key == ord('c'):
                # Clear current patient
                ai.load_patient(None)
                logger.info("Cleared current patient")
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                logger.info(f"Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                # Increase zoom
                global ZOOM_FACTOR
                ZOOM_FACTOR = min(3.0, ZOOM_FACTOR + 0.1)
                logger.info(f"Zoom increased to: {ZOOM_FACTOR:.1f}x")
            elif key == ord('-') or key == ord('_'):
                # Decrease zoom
                ZOOM_FACTOR = max(1.0, ZOOM_FACTOR - 0.1)
                logger.info(f"Zoom decreased to: {ZOOM_FACTOR:.1f}x")
            elif key == ord('d'):
                # Toggle debug mode
                logging.getLogger().setLevel(
                    logging.DEBUG if logging.getLogger().level != logging.DEBUG 
                    else logging.INFO
                )
                logger.info(f"Debug mode: {logging.getLogger().level == logging.DEBUG}")
            
            # Maintain target FPS
            elapsed = time.perf_counter() - frame_start
            if elapsed < FRAME_TIME:
                time.sleep(FRAME_TIME - elapsed)
    
    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping (KeyboardInterrupt)...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Calculate and display statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        logger.info("="*60)
        logger.info("SYSTEM STATISTICS")
        logger.info(f"Total runtime: {total_time:.1f} seconds")
        logger.info(f"Total frames: {frame_count}")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        
        # Get prescription summary
        summary = prescription_state.get_summary()
        logger.info(f"Prescription: {summary['verified']}/{summary['total']} drugs verified")
        
        # Log verification history
        history = prescription_state.get_verification_history()
        if history:
            logger.info("Verification History:")
            for record in history[-10:]:  # Last 10 records
                logger.info(f"  {record['time']} - {record['action']}: {record.get('drug', 'N/A')}")
        
        # Cleanup
        logger.info("Cleaning up...")
        cam.stop()
        ai.stop()
        cv2.destroyAllWindows()
        
        # Save session log
        session_log = {
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_runtime': total_time,
            'total_frames': frame_count,
            'avg_fps': avg_fps,
            'prescription_summary': summary,
            'verification_history': history[-20:]  # Last 20 records
        }
        
        try:
            with open('session_log.json', 'w') as f:
                json.dump(session_log, f, indent=2)
            logger.info("Session log saved to session_log.json")
        except Exception as e:
            logger.error(f"Error saving session log: {e}")
        
        logger.info("👋 System shutdown complete")

if __name__ == "__main__":
    main()