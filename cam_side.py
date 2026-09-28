import cv2
import numpy as np
import serial
import time

# ==============================
# CONFIGURATION
# ==============================
ROWS = 3
COLS = 4
BLUR_KERNEL = 101
HIGH_THRESHOLD = 100

WARP_W = 600
WARP_H = 450

COM_PORT = "COM16"     # CHANGE IF NEEDED
BAUD_RATE = 115200

# ==============================
# SERIAL SETUP
# ==============================
ser = serial.Serial(COM_PORT, BAUD_RATE)
time.sleep(2)

# ==============================
# SELECT 4 POINTS FOR HOMOGRAPHY
# ==============================
points = []

def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print("Point selected:", x, y)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Click 4 ground corner points:")
print("Top-Left, Top-Right, Bottom-Right, Bottom-Left")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    temp = frame.copy()
    for p in points:
        cv2.circle(temp, tuple(p), 6, (0,255,0), -1)

    cv2.imshow("Select Ground Plane", temp)
    cv2.setMouseCallback("Select Ground Plane", select_points)

    if len(points) == 4:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

src = np.float32(points)
dst = np.float32([[0,0], [WARP_W,0], [WARP_W,WARP_H], [0,WARP_H]])
matrix = cv2.getPerspectiveTransform(src, dst)

print("Perspective matrix computed")

# ==============================
# REAL-TIME LOOP
# ==============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    warped = cv2.warpPerspective(frame, matrix, (WARP_W, WARP_H))

    h, w, _ = warped.shape
    cell_w = w // COLS
    cell_h = h // ROWS

    # ===== DENSITY LOGIC =====
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.abs(lap)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    lap = lap.astype(np.uint8)

    density_map = cv2.GaussianBlur(lap, (BLUR_KERNEL, BLUR_KERNEL), 0)
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

    heatmap_color = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(warped, 0.6, heatmap_color, 0.6, 0)

    grid_density = np.zeros((ROWS, COLS))

    for r in range(ROWS):
        for c in range(COLS):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w

            cell = density_map[y1:y2, x1:x2]
            grid_density[r, c] = np.mean(cell)

    max_density_value = np.max(grid_density)
    evacuation_required = max_density_value > HIGH_THRESHOLD

    # ===== DRAW GRID =====
    for r in range(ROWS):
        for c in range(COLS):

            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w
            y2 = (r + 1) * cell_h

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), 1)

            value = int(grid_density[r, c])
            cv2.putText(overlay, str(value),
                        (x1+10, y1+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,0), 3)
            cv2.putText(overlay, str(value),
                        (x1+10, y1+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2)

    # ==========================
    # LED LOGIC
    # ==========================
    led_states = ["0"] * (ROWS * COLS)

    if evacuation_required:

        max_pos = np.unravel_index(np.argmax(grid_density), grid_density.shape)
        max_r, max_c = max_pos

        boundary_cells = []

        for c in range(COLS):
            boundary_cells.append((0,c))
            boundary_cells.append((ROWS-1,c))
        for r in range(ROWS):
            boundary_cells.append((r,0))
            boundary_cells.append((r,COLS-1))

        boundary_cells = list(set(boundary_cells))
        safe_cell = min(boundary_cells, key=lambda x: grid_density[x])
        safe_r, safe_c = safe_cell

        # Highlight danger
        cv2.rectangle(overlay,
                      (max_c*cell_w, max_r*cell_h),
                      ((max_c+1)*cell_w, (max_r+1)*cell_h),
                      (0,0,255), 4)

        # Highlight safe
        cv2.rectangle(overlay,
                      (safe_c*cell_w, safe_r*cell_h),
                      ((safe_c+1)*cell_w, (safe_r+1)*cell_h),
                      (0,255,0), 4)

        cv2.putText(overlay,
                    "HIGH RISK - EVACUATION REQUIRED",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)

        high_index = max_r * COLS + max_c
        safe_index = safe_r * COLS + safe_c

        led_states[high_index] = "3"  # FAST blink (danger)
        led_states[safe_index] = "1"  # SOLID (safe)

    else:
        cv2.putText(overlay,
                    "SAFE",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 3)

    # Send to ESP32
    data_string = ",".join(led_states) + "\n"
    ser.write(data_string.encode())

    cv2.imshow("Real-Time Stampede Analysis", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()