import cv2
import os
import time

SAVE_DIR = "dataset/images/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

img_count = 0
print("Capturing images... Press Ctrl+C to stop")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 320))
        filename = f"img_{img_count:04d}.jpg"
        cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)

        print(f"Saved {filename}")
        img_count += 1
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Stopping capture")

cap.release()
