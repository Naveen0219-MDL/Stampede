from flask import Flask, Response
import cv2
import numpy as np
import serial
import time

app = Flask(__name__)

# ==============================
# CONFIGURATION
# ==============================

ROWS = 3
COLS = 4
BLUR_KERNEL = 101
HIGH_THRESHOLD = 100

COM_PORT = "COM16"
BAUD_RATE = 115200

ser = serial.Serial(COM_PORT, BAUD_RATE)
time.sleep(2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

live_mode = True
frozen_frame = None
latest_led_states = None
movement_text = ""
show_movement = False


# ==============================
# FRAME GENERATOR
# ==============================
def generate_frames():
    global live_mode, frozen_frame, latest_led_states
    global movement_text, show_movement

    while True:

        if live_mode:
            ret, frame = cap.read()
            if not ret:
                break
            warped = frame.copy()
        else:
            warped = frozen_frame.copy()

        h, w, _ = warped.shape
        cell_w = w // COLS
        cell_h = h // ROWS

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap = np.abs(lap)
        lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
        lap = lap.astype(np.uint8)

        density_map = cv2.GaussianBlur(lap, (BLUR_KERNEL, BLUR_KERNEL), 0)
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

        heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(warped, 0.6, heatmap, 0.6, 0)

        grid_density = np.zeros((ROWS, COLS))

        # ==============================
        # DRAW GRID + NAMES
        # ==============================
        for r in range(ROWS):
            for c in range(COLS):

                y1 = r * cell_h
                y2 = (r+1) * cell_h
                x1 = c * cell_w
                x2 = (c+1) * cell_w

                cell = density_map[y1:y2, x1:x2]
                grid_density[r,c] = np.mean(cell)

                cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),1)

                grid_number = r * COLS + c + 1
                grid_name = f"G{grid_number}"

                # Grid Name
                cv2.putText(overlay, grid_name,
                            (x1+10, y1+20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,255),2)

                # Density value
                val = int(grid_density[r,c])
                cv2.putText(overlay, str(val),
                            (x1+10, y1+45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(255,255,255),2)

        max_density = np.max(grid_density)
        evacuation_required = max_density > HIGH_THRESHOLD

        led_states = ["0"] * (ROWS*COLS)
        movement_text = ""

        # ==============================
        # HIGH RISK LOGIC
        # ==============================
        if evacuation_required:

            max_r, max_c = np.unravel_index(np.argmax(grid_density), grid_density.shape)

            boundary = []
            for c in range(COLS):
                boundary.append((0,c))
                boundary.append((ROWS-1,c))
            for r in range(ROWS):
                boundary.append((r,0))
                boundary.append((r,COLS-1))

            boundary = list(set(boundary))
            safe_r, safe_c = min(boundary, key=lambda x: grid_density[x])

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

            # Path
            path = []
            r, c = max_r, max_c
            while (r,c)!=(safe_r,safe_c):
                path.append((r,c))
                if r < safe_r: r+=1
                elif r > safe_r: r-=1
                elif c < safe_c: c+=1
                elif c > safe_c: c-=1
            path.append((safe_r,safe_c))

            for i in range(len(path)-1):
                r1, c1 = path[i]
                r2, c2 = path[i+1]

                x1 = c1 * cell_w + cell_w // 2
                y1 = r1 * cell_h + cell_h // 2
                x2 = c2 * cell_w + cell_w // 2
                y2 = r2 * cell_h + cell_h // 2

                cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)

            # Movement message (but not shown yet)
            source_grid = max_r * COLS + max_c + 1
            dest_grid = safe_r * COLS + safe_c + 1
            movement_text = f"People from G{source_grid} will be moved to G{dest_grid}"

            # LED mapping
            high_index = max_r*COLS + max_c
            safe_index = safe_r*COLS + safe_c

            led_states[high_index] = "3"
            led_states[safe_index] = "1"

            for (pr,pc) in path:
                idx = pr*COLS + pc
                if idx != high_index and idx != safe_index:
                    led_states[idx] = "2"

        latest_led_states = ",".join(led_states) + "\n"

        # Show movement text ONLY after capture
        if show_movement and movement_text:
            cv2.putText(overlay,
                        movement_text,
                        (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,0,255),3)

        ret, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


# ==============================
# ROUTES
# ==============================

@app.route('/capture')
def capture():
    global live_mode, frozen_frame, latest_led_states, show_movement

    ret, frame = cap.read()
    if ret:
        frozen_frame = frame.copy()
        live_mode = False
        show_movement = True

        if latest_led_states:
            ser.write(latest_led_states.encode())

    return "Captured"


@app.route('/resume')
def resume():
    global live_mode, show_movement
    live_mode = True
    show_movement = False
    return "Resumed"


@app.route('/')
def index():
    return """
    <html>
    <body style="background:#111;color:white;text-align:center;">
        <h2>Stampede Monitoring System</h2>
        <button onclick="fetch('/capture')">Capture & Activate LEDs</button>
        <button onclick="fetch('/resume')">Resume</button>
        <br><br>
        <img src="/video" width="900">
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)