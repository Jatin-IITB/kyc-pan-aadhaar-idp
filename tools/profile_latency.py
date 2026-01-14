import time
import cv2
import numpy as np
from apps.workers.pipeline_loader import get_pipeline

def profile():
    # Load Pipeline
    print("⏳ Loading Pipeline...")
    pipe = get_pipeline()
    
    # Create Dummy Image (4MB like a phone camera)
    img = np.zeros((3000, 4000, 3), dtype=np.uint8) + 255
    
    print("🚀 Starting Profiling (Warmup)...")
    pipe.extract_from_bgr(img, "aadhaar") # Warmup
    
    print("🚀 Profiling 10 runs...")
    times = []
    for _ in range(10):
        t0 = time.time()
        pipe.extract_from_bgr(img, "aadhaar")
        t1 = time.time()
        times.append(t1 - t0)
        
    avg = sum(times) / len(times)
    print(f"📊 Average Latency: {avg*1000:.2f} ms")

if __name__ == "__main__":
    profile()
