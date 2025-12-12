#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  PILLTRACK: ENHANCED VERSION v2.0                            ║
║  - Optimized Performance & Memory Management                 ║
║  - Robust Error Handling & Recovery                          ║
║  - Improved Tracking Algorithm                               ║
║  - Clean Code Architecture                                   ║
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
from contextlib import contextmanager

import numpy as np
import cv2
import torch
import pickle
from torchvision import models, transforms
from ultralytics import YOLO

# ================= 📝 LOGGING SETUP =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= ⚙️ CONFIGURATION =================
@dataclass
class Config:
    """Centralized configuration with validation"""
    
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
    
    # Display & Processing
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416
    
    # UI Zone
    UI_ZONE_X_START: int = 900
    UI_ZONE_Y_END: int = 220
    
    # Detection Thresholds
    CONF_THRESHOLD: float = 0.55  # เพิ่มเกณฑ์ให้สูงขึ้น
    YOLO_CONF: float = 0.35
    
    # Feature Weights (ปรับให้ SIFT และ Color มีน้ำหนักมากขึ้น)
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'vec': 0.35,   # ลดลง - ResNet ค่อนข้าง generic
        'col': 0.35,   # เพิ่มขึ้น - สีสำคัญมาก
        'sift': 0.30   # เพิ่มขึ้น - รายละเอียดสำคัญ
    })
    
    # SIFT Configuration (ปรับให้เข้มงวดขึ้น)
    SIFT_RATIO_TEST: float = 0.70  # เข้มงวดขึ้น (ต่ำลง = เข้มงวดมากขึ้น)
    SIFT_MAX_MATCHES_NORMALIZE: int = 20  # เพิ่มเพื่อให้มีช่วง score กว้างขึ้น
    SIFT_MIN_MATCHES: int = 5  # ต้องมี matches อย่างน้อย
    
    # Stability & Tracking
    STABILITY_HISTORY_LEN: int = 5
    STABILITY_CONFIRM_REQ: int = 3
    TRACKING_IOU_THRESH: float = 0.3
    MAX_MISSING_FRAMES: int = 1
    
    # Performance
    FRAME_SKIP: int = 1  # Process every N frames
    MAX_DETECTIONS: int = 20  # Limit detections per frame
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 < self.CONF_THRESHOLD <= 1.0, "CONF_THRESHOLD must be in (0, 1]"
        assert sum(self.WEIGHTS.values()) > 0, "Weights must sum to positive value"
        assert self.STABILITY_CONFIRM_REQ <= self.STABILITY_HISTORY_LEN, \
            "Confirm requirement cannot exceed history length"

CFG = Config()

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 System starting on: {device.type.upper()}")

# ================= 🧠 PRESCRIPTION STATE MANAGER =================
class PrescriptionManager:
    """Manages prescription validation and drug verification"""
    
    def __init__(self, prescription_file: str = CFG.PRESCRIPTION_FILE):
        self.prescription_file = Path(prescription_file)
        self.patient_name: str = "Unknown"
        self.allowed_drugs: List[str] = []
        self.verified_drugs: Set[str] = set()
        self._lock = threading.Lock()
        
        self.load_prescription()
    
    def load_prescription(self) -> bool:
        """Load prescription with error handling"""
        if not self.prescription_file.exists():
            logger.warning(f"Prescription file not found: {self.prescription_file}")
            return False
        
        try:
            with open(self.prescription_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 3:
                        self.patient_name = parts[1].strip()
                        raw_drugs = parts[2].split(',')
                        self.allowed_drugs = [
                            drug.strip().lower() 
                            for drug in raw_drugs 
                            if drug.strip()
                        ]
                        logger.info(f"✅ Loaded prescription for: {self.patient_name}")
                        logger.info(f"   Allowed drugs: {', '.join(self.allowed_drugs)}")
                        return True
            
            logger.warning("No valid prescription data found")
            return False
            
        except Exception as e:
            logger.error(f"Error loading prescription: {e}")
            return False
    
    def is_allowed(self, db_name: str) -> bool:
        """Check if drug name matches allowed prescriptions"""
        db_clean = db_name.lower().replace('_pack', '').replace('_pill', '')
        
        for allowed in self.allowed_drugs:
            if allowed in db_clean or db_clean in allowed:
                return True
        return False
    
    def verify(self, name: str) -> bool:
        """Mark drug as verified (thread-safe)"""
        clean = name.lower().replace('_pack', '').replace('_pill', '')
        
        with self._lock:
            for allowed in self.allowed_drugs:
                if allowed in clean or clean in allowed:
                    self.verified_drugs.add(allowed)
                    return True
        return False
    
    def get_verification_status(self) -> Dict[str, bool]:
        """Get verification status for all drugs"""
        with self._lock:
            return {drug: drug in self.verified_drugs for drug in self.allowed_drugs}

# ================= 🎨 FEATURE ENGINE =================
class FeatureEngine:
    """Handles feature extraction using ResNet and SIFT"""
    
    def __init__(self):
        self._initialize_resnet()
        self._initialize_sift()
    
    def _initialize_resnet(self):
        """Initialize ResNet50 for feature vectors"""
        try:
            logger.info("Loading ResNet50 model...")
            weights = models.ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
            
            # Remove classification layer
            self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
            self.model.eval()
            self.model.to(device)
            
            # Preprocessing pipeline
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("✅ ResNet50 loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ResNet50: {e}")
            raise
    
    def _initialize_sift(self):
        """Initialize SIFT detector with optimized parameters"""
        try:
            # เพิ่ม nfeatures เพื่อให้จับ keypoints ได้มากขึ้น
            self.sift = cv2.SIFT_create(
                nfeatures=500,  # เพิ่มจำนวน features
                contrastThreshold=0.03,  # ลดลงเพื่อให้ sensitive มากขึ้น
                edgeThreshold=10  # จับ edge ได้ดีขึ้น
            )
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            logger.info("✅ SIFT detector initialized (enhanced)")
        except Exception as e:
            logger.error(f"Failed to initialize SIFT: {e}")
            raise
    
    @torch.no_grad()
    def get_vector(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract feature vector from image"""
        if img_rgb is None or img_rgb.size == 0:
            return None
        
        try:
            img_tensor = self.preprocess(img_rgb).unsqueeze(0).to(device)
            features = self.model(img_tensor)
            vec = features.flatten().cpu().numpy()
            
            # L2 normalization
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-8)
            
        except Exception as e:
            logger.error(f"Error extracting vector: {e}")
            return None
    
    def get_sift_features(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT descriptors from image"""
        if img_rgb is None or img_rgb.size == 0:
            return None
        
        try:
            # Resize เพื่อให้ได้ features ที่สม่ำเสมอ
            img_resized = cv2.resize(img_rgb, (224, 224))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE เพื่อเพิ่ม contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            _, descriptors = self.sift.detectAndCompute(gray, None)
            return descriptors
            
        except Exception as e:
            logger.error(f"Error extracting SIFT features: {e}")
            return None
    
    def get_color_histogram(self, img_rgb: np.ndarray, bins: int = 32) -> Optional[np.ndarray]:
        """Extract color histogram features (HSV space)"""
        if img_rgb is None or img_rgb.size == 0:
            return None
        
        try:
            # Convert to HSV (better for color comparison)
            img_resized = cv2.resize(img_rgb, (128, 128))
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
            
            # Calculate histogram for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
            
            # Normalize
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Concatenate
            hist = np.concatenate([h_hist, s_hist, v_hist])
            
            return hist
            
        except Exception as e:
            logger.error(f"Error extracting color histogram: {e}")
            return None
    
    def match_sift(self, query_des: np.ndarray, ref_des: np.ndarray, 
                   ratio_test: float = CFG.SIFT_RATIO_TEST) -> Tuple[int, float]:
        """Match SIFT descriptors using ratio test - returns (count, quality)"""
        if query_des is None or ref_des is None:
            return 0, 0.0
        
        try:
            matches = self.bf_matcher.knnMatch(query_des, ref_des, k=2)
            
            good_matches = []
            distance_sum = 0.0
            
            for match_pair in matches:
                if len(match_pair) != 2:
                    continue
                
                m, n = match_pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)
                    distance_sum += m.distance
            
            match_count = len(good_matches)
            
            # คำนวณคุณภาพของ matches (ยิ่ง distance ต่ำยิ่งดี)
            if match_count > 0:
                avg_distance = distance_sum / match_count
                # Normalize distance (SIFT distance อยู่ประมาณ 0-512)
                match_quality = 1.0 - min(avg_distance / 300.0, 1.0)
            else:
                match_quality = 0.0
            
            return match_count, match_quality
            
        except Exception as e:
            logger.debug(f"SIFT matching error: {e}")
            return 0, 0.0

# ================= 🛡️ OBJECT STABILIZER =================
class ObjectStabilizer:
    """Tracks objects across frames and stabilizes labels using voting"""
    
    def __init__(self):
        self.tracks: Dict[int, Dict] = {}
        self.next_id: int = 0
        self._lock = threading.Lock()
    
    @staticmethod
    def calculate_iou(boxA: Tuple, boxB: Tuple) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        union_area = boxA_area + boxB_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def update(self, raw_detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections and return stable results"""
        with self._lock:
            return self._update_internal(raw_detections)
    
    def _update_internal(self, raw_detections: List[Dict]) -> List[Dict]:
        """Internal update logic (not thread-safe by itself)"""
        updated_tracks = {}
        used_indices = set()
        
        # Phase 1: Match existing tracks with new detections
        for track_id, track_data in self.tracks.items():
            best_iou = 0.0
            best_idx = -1
            last_box = track_data['box']
            
            for i, det in enumerate(raw_detections):
                if i in used_indices:
                    continue
                
                iou = self.calculate_iou(last_box, det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # Found a match
            if best_iou > CFG.TRACKING_IOU_THRESH and best_idx >= 0:
                det = raw_detections[best_idx]
                
                # Update history
                track_data['history'].append(det['label'])
                if len(track_data['history']) > CFG.STABILITY_HISTORY_LEN:
                    track_data['history'].popleft()
                
                # Update metadata
                track_data['box'] = det['box']
                track_data['contour'] = det['contour']
                track_data['missing'] = 0
                track_data['candidates'] = det.get('candidates', [])
                
                updated_tracks[track_id] = track_data
                used_indices.add(best_idx)
            
            # Track lost but give it a chance
            else:
                track_data['missing'] += 1
                if track_data['missing'] < CFG.MAX_MISSING_FRAMES:
                    updated_tracks[track_id] = track_data
        
        # Phase 2: Create new tracks for unmatched detections
        for i, det in enumerate(raw_detections):
            if i not in used_indices:
                new_id = self.next_id
                self.next_id += 1
                
                updated_tracks[new_id] = {
                    'history': collections.deque([det['label']], 
                                                maxlen=CFG.STABILITY_HISTORY_LEN),
                    'missing': 0,
                    'box': det['box'],
                    'contour': det['contour'],
                    'candidates': det.get('candidates', [])
                }
        
        self.tracks = updated_tracks
        
        # Phase 3: Generate stable output using voting
        return self._generate_stable_output()
    
    def _generate_stable_output(self) -> List[Dict]:
        """Generate stable detection results from tracks"""
        stable_results = []
        
        for track_id, track_data in self.tracks.items():
            # Skip tracks that are currently missing
            if track_data['missing'] > 0:
                continue
            
            # Vote on label
            counter = collections.Counter(track_data['history'])
            
            if not counter:
                continue
            
            top_label, count = counter.most_common(1)[0]
            
            # Determine status based on voting
            if count >= CFG.STABILITY_CONFIRM_REQ:
                final_label = top_label
                status = "confirmed" if top_label != "Unknown" else "unknown"
            else:
                final_label = f"Analyzing... ({count}/{CFG.STABILITY_CONFIRM_REQ})"
                status = "pending"
            
            stable_results.append({
                'box': track_data['box'],
                'contour': track_data['contour'],
                'label': final_label,
                'status': status,
                'candidates': track_data.get('candidates', []),
                'track_id': track_id
            })
        
        return stable_results
    
    def reset(self):
        """Reset all tracks"""
        with self._lock:
            self.tracks.clear()
            self.next_id = 0

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    """Main AI processing pipeline"""
    
    def __init__(self):
        logger.info("Initializing AI Processor...")
        
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        self.stabilizer = ObjectStabilizer()
        
        # Database storage
        self.session_db_vec: Dict[str, Tuple[str, np.ndarray]] = {}
        self.session_db_sift: Dict[str, List[np.ndarray]] = {}
        self.session_db_color: Dict[str, List[np.ndarray]] = {}  # เพิ่ม color DB
        
        # Load databases
        self._load_databases()
        
        # Initialize YOLO
        self._initialize_yolo()
        
        # Threading
        self.latest_frame: Optional[np.ndarray] = None
        self.results: List[Dict] = []
        self.lock = threading.Lock()
        self.stopped = False
        self.frame_count = 0
        
        logger.info("✅ AI Processor initialized")
    
    def _initialize_yolo(self):
        """Initialize YOLO model with error handling"""
        try:
            model_path = Path(CFG.MODEL_PACK)
            
            if model_path.exists():
                self.yolo_model = YOLO(str(model_path))
                logger.info(f"✅ Loaded custom YOLO model: {model_path}")
            else:
                logger.warning(f"Custom model not found: {model_path}")
                self.yolo_model = YOLO('yolov8n-seg.pt')
                logger.info("✅ Loaded default YOLO model")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _load_databases(self):
        """Load feature databases with validation"""
        logger.info("Loading feature databases...")
        
        # Load vector databases
        vec_count = self._load_vector_databases()
        logger.info(f"   Loaded {vec_count} vector entries")
        
        # Load SIFT and Color databases from images
        sift_count, color_count = self._load_image_databases()
        logger.info(f"   Loaded {sift_count} SIFT entries")
        logger.info(f"   Loaded {color_count} Color entries")
    
    def _load_vector_databases(self) -> int:
        """Load vector feature databases"""
        count = 0
        
        for db_path in [CFG.DB_PILLS_VEC, CFG.DB_PACKS_VEC]:
            db_file = Path(db_path)
            if not db_file.exists():
                logger.warning(f"Vector DB not found: {db_path}")
                continue
            
            try:
                with open(db_file, 'rb') as f:
                    db_data = pickle.load(f)
                
                for drug_name, vectors in db_data.items():
                    if not self.rx_manager.is_allowed(drug_name):
                        continue
                    
                    for vec in vectors:
                        key = f"{drug_name}_{count}"
                        self.session_db_vec[key] = (drug_name, np.array(vec))
                        count += 1
                        
            except Exception as e:
                logger.error(f"Error loading {db_path}: {e}")
        
        return count
    
    def _load_image_databases(self) -> Tuple[int, int]:
        """Load SIFT and Color feature databases from images"""
        sift_count = 0
        color_count = 0
        img_db = Path(CFG.IMG_DB_FOLDER)
        
        if not img_db.exists():
            logger.warning(f"Image database folder not found: {img_db}")
            return 0, 0
        
        for drug_folder in img_db.iterdir():
            if not drug_folder.is_dir():
                continue
            
            drug_name = drug_folder.name
            if not self.rx_manager.is_allowed(drug_name):
                continue
            
            sift_descriptors_list = []
            color_histograms_list = []
            
            # Load first 5 images for better coverage
            for img_file in sorted(drug_folder.iterdir())[:5]:
                if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                    continue
                
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Extract SIFT
                    sift_des = self.engine.get_sift_features(img_rgb)
                    if sift_des is not None:
                        sift_descriptors_list.append(sift_des)
                    
                    # Extract Color Histogram
                    color_hist = self.engine.get_color_histogram(img_rgb)
                    if color_hist is not None:
                        color_histograms_list.append(color_hist)
                        
                except Exception as e:
                    logger.debug(f"Error loading {img_file}: {e}")
            
            if sift_descriptors_list:
                self.session_db_sift[drug_name] = sift_descriptors_list
                sift_count += 1
            
            if color_histograms_list:
                self.session_db_color[drug_name] = color_histograms_list
                color_count += 1
        
        return sift_count, color_count
    
    def compute_sift_score(self, query_des: Optional[np.ndarray], 
                          target_name: str) -> float:
        """Compute normalized SIFT matching score with quality consideration"""
        if query_des is None or target_name not in self.session_db_sift:
            return 0.0
        
        max_score = 0.0
        
        for ref_des in self.session_db_sift[target_name]:
            match_count, match_quality = self.engine.match_sift(query_des, ref_des)
            
            # ต้องมี matches ขั้นต่ำ
            if match_count < CFG.SIFT_MIN_MATCHES:
                continue
            
            # Normalize match count
            count_score = min(match_count / CFG.SIFT_MAX_MATCHES_NORMALIZE, 1.0)
            
            # Combined score (50% count, 50% quality)
            combined_score = 0.5 * count_score + 0.5 * match_quality
            
            max_score = max(max_score, combined_score)
        
        return max_score
    
    def compute_color_score(self, query_hist: Optional[np.ndarray], 
                           target_name: str) -> float:
        """Compute color histogram similarity score"""
        if query_hist is None or target_name not in self.session_db_color:
            return 0.0
        
        max_similarity = 0.0
        
        for ref_hist in self.session_db_color[target_name]:
            # ใช้ Correlation method (ให้ผลดีกับ color comparison)
            similarity = cv2.compareHist(query_hist, ref_hist, cv2.HISTCMP_CORREL)
            
            # Correlation อยู่ระหว่าง -1 ถึง 1, normalize เป็น 0-1
            normalized_similarity = (similarity + 1.0) / 2.0
            
            max_similarity = max(max_similarity, normalized_similarity)
        
        return max_similarity
    
    def match_drug(self, vec: np.ndarray, img_crop: np.ndarray) -> List[Tuple]:
        """Match detected object against drug database with enhanced scoring"""
        if not self.session_db_vec:
            return []
        
        # Extract all features once
        query_sift = self.engine.get_sift_features(img_crop)
        query_color = self.engine.get_color_histogram(img_crop)
        
        candidates = []
        
        for key, (drug_name, db_vec) in self.session_db_vec.items():
            # 1. Vector similarity (ResNet features)
            vec_score = float(np.dot(vec, db_vec))
            
            # 2. SIFT similarity (local features)
            sift_score = self.compute_sift_score(query_sift, drug_name)
            
            # 3. Color similarity (histogram)
            color_score = self.compute_color_score(query_color, drug_name)
            
            # Weighted combination
            final_score = (
                vec_score * CFG.WEIGHTS['vec'] +
                sift_score * CFG.WEIGHTS['sift'] +
                color_score * CFG.WEIGHTS['col']
            )
            
            # เก็บ sub-scores สำหรับ debug
            candidates.append((
                drug_name, 
                final_score, 
                vec_score, 
                sift_score, 
                color_score
            ))
        
        # Sort by final score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate and keep top 5
        unique_candidates = []
        seen_names = set()
        
        for name, final_s, vec_s, sift_s, col_s in candidates:
            if name not in seen_names:
                unique_candidates.append((name, final_s, vec_s, sift_s, col_s))
                seen_names.add(name)
            
            if len(unique_candidates) >= 5:
                break
        
        # Log top candidate for debugging
        if unique_candidates:
            top = unique_candidates[0]
            logger.debug(
                f"Top match: {top[0]} | "
                f"Final: {top[1]:.3f} | "
                f"Vec: {top[2]:.3f} | "
                f"SIFT: {top[3]:.3f} | "
                f"Color: {top[4]:.3f}"
            )
        
        return unique_candidates
    
    def process_frame(self, frame: np.ndarray):
        """Process a single frame"""
        # Resize for YOLO
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        
        # Run YOLO detection
        results = self.yolo_model(
            img_ai,
            verbose=False,
            conf=CFG.YOLO_CONF,
            imgsz=CFG.AI_SIZE,
            task='segment'
        )
        
        raw_detections = []
        res = results[0]
        
        if res.masks is None:
            # No detections
            with self.lock:
                self.results = self.stabilizer.update([])
            return
        
        # Scale factors
        scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
        scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        
        for box, mask in zip(res.boxes, res.masks):
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Scale to display size
            rx1 = int(x1 * scale_x)
            ry1 = int(y1 * scale_y)
            rx2 = int(x2 * scale_x)
            ry2 = int(y2 * scale_y)
            
            # Filter out UI zone
            cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END:
                continue
            
            # Get contour
            contour = mask.xyn[0]
            contour[:, 0] *= CFG.DISPLAY_SIZE[0]
            contour[:, 1] *= CFG.DISPLAY_SIZE[1]
            contour = contour.astype(np.int32)
            
            # Crop and validate
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue
            
            # Extract features
            vec = self.engine.get_vector(crop)
            if vec is None:
                continue
            
            # Match against database
            candidates = self.match_drug(vec, crop)
            
            # Determine label
            if candidates:
                top_name, top_score, _, _ = candidates[0]
                label = top_name if top_score > CFG.CONF_THRESHOLD else "Unknown"
            else:
                label = "Unknown"
            
            raw_detections.append({
                'box': (rx1, ry1, rx2, ry2),
                'contour': contour,
                'label': label,
                'candidates': candidates
            })
            
            # Limit detections
            if len(raw_detections) >= CFG.MAX_DETECTIONS:
                break
        
        # Update stabilizer
        final_detections = self.stabilizer.update(raw_detections)
        
        # Verify drugs
        for det in final_detections:
            label = det['label']
            if label not in ["Unknown", "Verifying..."] and "Analyzing" not in label:
                self.rx_manager.verify(label)
        
        # Store results
        with self.lock:
            self.results = final_detections
    
    def start(self):
        """Start processing thread"""
        threading.Thread(target=self._run, daemon=True).start()
        return self
    
    def _run(self):
        """Main processing loop"""
        while not self.stopped:
            with self.lock:
                frame = self.latest_frame
            
            if frame is not None:
                # Frame skipping for performance
                self.frame_count += 1
                if self.frame_count % CFG.FRAME_SKIP == 0:
                    try:
                        self.process_frame(frame)
                    except Exception as e:
                        logger.error(f"Frame processing error: {e}")
            
            time.sleep(0.01)

# ================= 📷 CAMERA =================
class Camera:
    """Camera interface supporting both PiCamera and USB cameras"""
    
    def __init__(self):
        self.cap = None
        self.picam = None
        self.use_pi = False
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with fallback"""
        # Try PiCamera first
        try:
            from picamera2 import Picamera2
            
            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(
                main={"size": CFG.DISPLAY_SIZE, "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            
            self.use_pi = True
            logger.info("📷 Using PiCamera2 (RGB888)")
            return
            
        except ImportError:
            logger.info("PiCamera2 not available, trying USB camera")
        except Exception as e:
            logger.warning(f"PiCamera2 init failed: {e}")
        
        # Fallback to USB camera
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.DISPLAY_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.DISPLAY_SIZE[1])
            
            # Test capture
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture from USB camera")
            
            self.use_pi = False
            logger.info("📷 Using USB Camera (BGR→RGB)")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
    
    def get(self) -> Optional[np.ndarray]:
        """Get frame from camera"""
        try:
            if self.use_pi:
                return self.picam.capture_array()
            else:
                ret, frame = self.cap.read()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None
                
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
    
    def stop(self):
        """Release camera resources"""
        try:
            if self.use_pi and self.picam:
                self.picam.stop()
            elif self.cap:
                self.cap.release()
            logger.info("Camera stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")

# ================= 🖥️ UI RENDERER =================
def draw_ui(frame: np.ndarray, results: List[Dict], rx_manager: PrescriptionManager):
    """Draw detection results and dashboard on frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw detections
    for det in results:
        contour = det['contour']
        label = det['label']
        status = det['status']
        
        # Color coding
        color_map = {
            'confirmed': (0, 255, 0),    # Green
            'unknown': (255, 0, 0),       # Red
            'pending': (255, 255, 0)      # Yellow
        }
        color = color_map.get(status, (255, 255, 0))
        border_width = 2 if status in ['confirmed', 'unknown'] else 1
        
        # Draw filled contour
        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(overlay, [contour], True, color, border_width)
        
        # Draw label
        top_y = int(np.min(contour[:, 1]))
        top_x = int(contour[np.argmin(contour[:, 1]), 0])
        
        # Text background
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (top_x, top_y - text_h - 10),
            (top_x + text_w + 10, top_y),
            color,
            -1
        )
        
        # Text
        text_color = (0, 0, 0) if status != 'unknown' else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (top_x + 5, top_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    
    # Draw dashboard
    db_x, db_y = CFG.UI_ZONE_X_START, 10
    db_w = w - db_x - 10
    db_h = CFG.UI_ZONE_Y_END
    
    # Dashboard background
    db_roi = frame[db_y:db_y + db_h, db_x:db_x + db_w]
    white_bg = np.ones(db_roi.shape, dtype=np.uint8) * 40
    cv2.addWeighted(db_roi, 0.4, white_bg, 0.6, 0, db_roi)
    frame[db_y:db_y + db_h, db_x:db_x + db_w] = db_roi
    
    # Dashboard border
    cv2.rectangle(frame, (db_x, db_y), (db_x + db_w, db_y + db_h), (0, 255, 0), 2)
    
    # Patient name
    cv2.putText(
        frame,
        f"Patient: {rx_manager.patient_name}",
        (db_x + 10, db_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    # Drug list
    y_offset = 60
    verification_status = rx_manager.get_verification_status()
    
    for drug, is_verified in verification_status.items():
        icon = " ✓" if is_verified else " ○"
        color = (0, 255, 0) if is_verified else (180, 180, 180)
        
        cv2.putText(
            frame,
            f"• {drug}{icon}",
            (db_x + 10, db_y + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        y_offset += 25

# ================= 🚀 MAIN =================
def main():
    """Main application entry point"""
    logger.info("=" * 60)
    logger.info("PillTrack Enhanced v2.0 - Starting...")
    logger.info("=" * 60)
    
    # Initialize components
    try:
        camera = Camera()
        ai_processor = AIProcessor().start()
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return 1
    
    # Wait for first frame
    logger.info("Waiting for camera feed...")
    while camera.get() is None:
        time.sleep(0.1)
    
    logger.info("✨ System ready! Press 'q' to quit")
    
    # Create display window
    window_name = "PillTrack Enhanced v2.0"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *CFG.DISPLAY_SIZE)
    
    fps_counter = collections.deque(maxlen=30)
    last_time = time.time()
    
    try:
        while True:
            # Get frame
            frame = camera.get()
            if frame is None:
                logger.warning("Failed to get frame")
                continue
            
            # Update AI processor
            ai_processor.latest_frame = frame.copy()
            
            # Draw UI
            with ai_processor.lock:
                results = ai_processor.results.copy()
            
            draw_ui(frame, results, ai_processor.rx_manager)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_time + 1e-6)
            fps_counter.append(fps)
            avg_fps = sum(fps_counter) / len(fps_counter)
            last_time = current_time
            
            # Draw FPS
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display
            cv2.imshow(window_name, frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit requested")
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Runtime error: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        camera.stop()
        ai_processor.stopped = True
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())