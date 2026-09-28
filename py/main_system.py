from flask import Flask, Response, jsonify
import cv2
import numpy as np
import serial
import time
import threading
import os
from pathlib import Path
from dotenv import load_dotenv

# ==============================
# CONFIG
# ==============================

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ROWS = 3
COLS = 4
BLUR_KERNEL = 101

DENSITY_THRESHOLD = 100
MOTION_THRESHOLD = 2.0

COM_PORT = os.getenv("STAMPEDE_SERIAL_PORT", "COM16")
BAUD_RATE = int(os.getenv("STAMPEDE_BAUD_RATE", "115200"))

CAM_INDEX = int(os.getenv("STAMPEDE_CAMERA_INDEX", "0"))

# ==============================
# GLOBAL STATE
# ==============================

app = Flask(__name__)

latest_frame = None
metrics_data = {}
prev_gray = None

ser = serial.Serial(COM_PORT, BAUD_RATE)
time.sleep(2)

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("Camera not accessible")
    exit()


# ==============================
# PROCESSING LOOP
# ==============================

def process_loop():
    global latest_frame, metrics_data, prev_gray

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        cell_w = w // COLS
        cell_h = h // ROWS

        # ==============================
        # DENSITY
        # ==============================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

        lap = cv2.Laplacian(gray_blur, cv2.CV_32F)
        lap = np.abs(lap)
        lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        density_map = cv2.GaussianBlur(lap, (BLUR_KERNEL, BLUR_KERNEL), 0)
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

        # ==============================
        # OPTICAL FLOW MOTION
        # ==============================
        motion_map = np.zeros_like(gray_blur, dtype=np.float32)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray_blur,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            motion_map = magnitude

        prev_gray = gray_blur.copy()

        overlay = frame.copy()
        grid_density = np.zeros((ROWS, COLS))
        grid_motion = np.zeros((ROWS, COLS))

        # ==============================
        # GRID ANALYSIS
        # ==============================
        for r in range(ROWS):
            for c in range(COLS):

                y1 = r * cell_h
                y2 = (r+1) * cell_h
                x1 = c * cell_w
                x2 = (c+1) * cell_w

                grid_density[r,c] = np.mean(density_map[y1:y2, x1:x2])
                grid_motion[r,c] = np.mean(motion_map[y1:y2, x1:x2])

                grid_id = r*COLS + c + 1

                cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),1)

                cv2.putText(overlay,f"G{grid_id}",
                            (x1+5,y1+18),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,
                            (0,255,255),1)

                cv2.putText(overlay,
                            f"D:{int(grid_density[r,c])}",
                            (x1+5,y1+35),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,
                            (255,255,255),1)

                cv2.putText(overlay,
                            f"M:{round(grid_motion[r,c],1)}",
                            (x1+5,y1+50),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,
                            (0,255,0),1)

        # ==============================
        # RISK LOGIC
        # ==============================
        max_density = np.max(grid_density)
        max_motion = np.max(grid_motion)

        evacuation_required = (
            max_density > DENSITY_THRESHOLD
            or max_motion > MOTION_THRESHOLD
        )

        led_states = ["0"] * (ROWS*COLS)

        if evacuation_required:

            max_r, max_c = np.unravel_index(
                np.argmax(grid_density),
                grid_density.shape
            )

            boundary = []
            for c in range(COLS):
                boundary.append((0,c))
                boundary.append((ROWS-1,c))
            for r in range(ROWS):
                boundary.append((r,0))
                boundary.append((r,COLS-1))

            boundary = list(set(boundary))
            safe_r, safe_c = min(boundary,
                                 key=lambda x: grid_density[x])

            # Highlight source
            cv2.rectangle(overlay,
                          (max_c*cell_w,max_r*cell_h),
                          ((max_c+1)*cell_w,(max_r+1)*cell_h),
                          (0,0,255),4)

            # Highlight destination
            cv2.rectangle(overlay,
                          (safe_c*cell_w,safe_r*cell_h),
                          ((safe_c+1)*cell_w,(safe_r+1)*cell_h),
                          (0,255,0),4)

            # LED mapping
            high_index = max_r*COLS + max_c
            safe_index = safe_r*COLS + safe_c

            led_states[high_index] = "3"
            led_states[safe_index] = "1"

        data = ",".join(led_states) + "\n"
        ser.write(data.encode())

        metrics_data = {
            "max_density": int(max_density),
            "max_motion": round(float(max_motion),2),
            "evacuation_required": bool(evacuation_required)
        }

        ret, buffer = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        latest_frame = buffer.tobytes()


# ==============================
# STREAM ROUTE
# ==============================

@app.route('/video')
def video():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       latest_frame + b'\r\n')
            time.sleep(0.03)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/metrics')
def metrics():
    return jsonify(metrics_data)
@app.route('/')
def home():
    return """
    <html>
    <head>
        <title>Crowd Stream</title>      
    </head>
    <body style="margin:0;background:black;">
        <img src="/video" style="width:100%;">
    </body>
    </html>
    """

if __name__ == "__main__":
    threading.Thread(target=process_loop).start()
    app.run(host='0.0.0.0', port=5000)