import cv2
import numpy as np
import serial
import time

# ==============================``
# CONFIGURATION
# ==============================

IMAGE_PATH = r"C:\Users\ASUS2024\Desktop\Stampede\sample_3.jpeg"

ROWS = 3
COLS = 4

BLUR_KERNEL = 101
HIGH_THRESHOLD = 165

COM_PORT = "COM16"   # CHANGE THIS
BAUD_RATE = 115200

# ==============================
# SERIAL SETUP
# ==============================

ser = serial.Serial(COM_PORT, BAUD_RATE)
time.sleep(2)

# ==============================
# LOAD IMAGE
# ==============================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found.")
    exit()

h, w, _ = image.shape
cell_w = w // COLS
cell_h = h // ROWS

# ==============================
# DENSITY ESTIMATION
# ==============================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

lap = cv2.Laplacian(gray, cv2.CV_32F)
lap = np.abs(lap)

lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
lap = lap.astype(np.uint8)

density_map = cv2.GaussianBlur(lap, (BLUR_KERNEL, BLUR_KERNEL), 0)
density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

heatmap_color = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.6, 0)

# ==============================
# GRID DENSITY
# ==============================

grid_density = np.zeros((ROWS, COLS))

for r in range(ROWS):
    for c in range(COLS):

        y1 = r * cell_h
        y2 = (r + 1) * cell_h
        x1 = c * cell_w
        x2 = (c + 1) * cell_w

        cell = density_map[y1:y2, x1:x2]
        grid_density[r, c] = np.mean(cell)

print("\nGrid Density Matrix:")
print(grid_density)

max_density_value = np.max(grid_density)
evacuation_required = max_density_value > HIGH_THRESHOLD

# ==============================
# DRAW GRID
# ==============================

for r in range(ROWS):
    for c in range(COLS):

        x1 = c * cell_w
        y1 = r * cell_h
        x2 = (c + 1) * cell_w
        y2 = (r + 1) * cell_h

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)

# ==============================
# LED STATE ARRAY
# ==============================

led_states = ["0"] * 12

if not evacuation_required:

    cv2.putText(overlay,
                "STATUS: SAFE - NO EVACUATION REQUIRED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 3)

    data_string = ",".join(led_states) + "\n"
    ser.write(data_string.encode())

else:

    max_pos = np.unravel_index(np.argmax(grid_density), grid_density.shape)
    max_r, max_c = max_pos

    boundary_cells = []

    for c in range(COLS):
        boundary_cells.append((0, c))
        boundary_cells.append((ROWS - 1, c))

    for r in range(ROWS):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, COLS - 1))

    boundary_cells = list(set(boundary_cells))

    safe_cell = min(boundary_cells, key=lambda x: grid_density[x])
    safe_r, safe_c = safe_cell

    # Draw highlight boxes
    cv2.rectangle(overlay,
                  (max_c * cell_w, max_r * cell_h),
                  ((max_c + 1) * cell_w, (max_r + 1) * cell_h),
                  (0, 0, 255), 4)

    cv2.rectangle(overlay,
                  (safe_c * cell_w, safe_r * cell_h),
                  ((safe_c + 1) * cell_w, (safe_r + 1) * cell_h),
                  (0, 255, 0), 4)

    # Manhattan Path
    path = []
    r, c = max_r, max_c

    while (r, c) != (safe_r, safe_c):
        path.append((r, c))

        if r < safe_r:
            r += 1
        elif r > safe_r:
            r -= 1
        elif c < safe_c:
            c += 1
        elif c > safe_c:
            c -= 1

    path.append((safe_r, safe_c))

    # Draw path lines
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        x1 = c1 * cell_w + cell_w // 2
        y1 = r1 * cell_h + cell_h // 2
        x2 = c2 * cell_w + cell_w // 2
        y2 = r2 * cell_h + cell_h // 2

        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)

    cv2.putText(overlay,
                "STATUS: HIGH RISK - EVACUATION REQUIRED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    # LED LOGIC
    high_index = max_r * COLS + max_c
    safe_index = safe_r * COLS + safe_c

    led_states[high_index] = "3"  # FAST blink (danger)
    led_states[safe_index] = "2"  # SLOW blink (safe)

    for (pr, pc) in path:
        idx = pr * COLS + pc
        if idx != high_index and idx != safe_index:
            led_states[idx] = "1"  # SOLID ON (path)

    data_string = ",".join(led_states) + "\n"
    ser.write(data_string.encode())

cv2.imshow("Stampede Risk Analysis", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

ser.close()