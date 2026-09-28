import cv2
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
IMAGE_PATH = r"C:\Users\ASUS2024\Desktop\Stampede\sample_9.jpeg"

ROWS = 8
COLS = 8

BLUR_KERNEL = 101  # must be odd
HIGH_THRESHOLD = 165  # you may need to retune slightly

# ==============================
# LOAD IMAGE+00000
# ==============================
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found.")
    exit()

h, w, _ = image.shape
cell_w = w // COLS
cell_h = h // ROWS

# ==============================
# STEP 1: LAPLACIAN-BASED DENSITY
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
# STEP 2: GRID DENSITY
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

# ==============================
# STEP 3: DECISION LOGIC
# ==============================

evacuation_required = max_density_value > HIGH_THRESHOLD

# ==============================
# STEP 4: DRAW GRID + VALUES
# ==============================

for r in range(ROWS):
    for c in range(COLS):
        x1 = c * cell_w
        y1 = r * cell_h
        x2 = (c + 1) * cell_w
        y2 = (r + 1) * cell_h

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)

        value = int(grid_density[r, c])

        # To reduce clutter, show values only if significant
        if value > HIGH_THRESHOLD * 0.6:
            cv2.putText(overlay, str(value),
                        (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 2)

            cv2.putText(overlay, str(value),
                        (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1)

# ==============================
# STEP 5: SAFE CASE
# ==============================

if not evacuation_required:

    cv2.putText(overlay,
                "STATUS: SAFE - NO EVACUATION REQUIRED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

# ==============================
# STEP 6: HIGH RISK CASE
# ==============================

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

    # Highlight dangerous grid
    x1 = max_c * cell_w
    y1 = max_r * cell_h
    x2 = (max_c + 1) * cell_w
    y2 = (max_r + 1) * cell_h
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Highlight safe grid
    x1 = safe_c * cell_w
    y1 = safe_r * cell_h
    x2 = (safe_c + 1) * cell_w
    y2 = (safe_r + 1) * cell_h
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Manhattan path
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

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        x1 = c1 * cell_w + cell_w // 2
        y1 = r1 * cell_h + cell_h // 2
        x2 = c2 * cell_w + cell_w // 2
        y2 = r2 * cell_h + cell_h // 2

        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.putText(overlay,
                "STATUS: HIGH RISK - EVACUATION REQUIRED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

# ==============================
# DISPLAY
# ==============================

cv2.imshow("Stampede Risk Analysis (8x8)", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()