import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# 1. Setup Environment
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

print("--- 🚀 STARTING SIMPLE TEST ---")

# 2. Load Model (ลองใช้ .pt ตามที่คุณบอกว่าเปลี่ยนแล้ว)
# ⚠️ เช็คชื่อไฟล์ดีๆ ว่าอยู่โฟลเดอร์ไหน
MODEL_PATH = 'models/pills.pt' 
print(f"Loading model: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 3. Setup Camera
print("Opening Camera...")
try:
    picam2 = Picamera2()
    # ลองใช้ BGR888 ไปเลย เพื่อตัดปัญหาเรื่องสีเพี้ยนกับ YOLO
    config = picam2.create_preview_configuration(
        main={"size": (640, 640), "format": "BGR888"}, 
        controls={"FrameDurationLimits": (33333, 33333)}
    )
    picam2.configure(config)
    picam2.start()
    
    # Warm up camera
    print("Warming up camera (2s)...")
    time.sleep(2.0)
    
    # 4. Capture & Predict Loop (5 ครั้งพอ)
    for i in range(5):
        print(f"\n--- Test Round {i+1} ---")
        
        # Capture
        frame = picam2.capture_array()
        
        # 🔥 CRITICAL FIX: ต้องทำ copy() เพื่อให้ Memory Contiguous ไม่งั้น YOLO เอ๋อ
        frame_clean = frame.copy()
        
        # Save image to check what AI sees
        cv2.imwrite(f"debug_frame_{i}.jpg", frame_clean)
        print(f"Saved 'debug_frame_{i}.jpg' (Check this file!)")
        
        # Predict
        results = model(frame_clean, conf=0.10, verbose=True) # conf ต่ำๆ
        
        # Check result
        if len(results[0].boxes) > 0:
            print(f"🎉 FOUND {len(results[0].boxes)} OBJECTS!")
            for box in results[0].boxes:
                print(f"   - Class: {int(box.cls)} | Conf: {float(box.conf):.2f}")
        else:
            print("💀 No detection.")
            
        time.sleep(1)

    picam2.stop()
    picam2.close()

except Exception as e:
    print(f"❌ Camera/Runtime Error: {e}")

print("--- END TEST ---")