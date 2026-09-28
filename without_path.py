import cv2
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
IMAGE_PATH = r"C:\Users\ASUS2024\Desktop\Stampede\sample_2.jpeg"
ROWS = 3
COLS = 4

BLUR_KERNEL = 101
HIGH_THRESHOLD = 165

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
evacuation_required = max_density_value > HIGH_THRESHOLD

# ==============================
# DRAW GRID + VALUES
# ==============================
for r in range(ROWS):
    for c in range(COLS):

        x1 = c * cell_w
        y1 = r * cell_h
        x2 = (c + 1) * cell_w
        y2 = (r + 1) * cell_h

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)

        value = int(grid_density[r, c])

        cv2.putText(overlay, str(value),
                    (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3)

        cv2.putText(overlay, str(value),
                    (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

# ==============================
# SAFE CASE
# ==============================
if not evacuation_required:

    cv2.putText(overlay,
                "STATUS: SAFE - NO EVACUATION REQUIRED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 3)

# ==============================
# HIGH RISK CASE (NO PATH)
# ==============================
else:

    # Most crowded grid
    max_pos = np.unravel_index(np.argmax(grid_density), grid_density.shape)
    max_r, max_c = max_pos

    # Safest boundary grid
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

    # Highlight danger grid (RED)
    cv2.rectangle(overlay,
                  (max_c * cell_w, max_r * cell_h),
                  ((max_c + 1) * cell_w, (max_r + 1) * cell_h),
                  (0, 0, 255), 4)

    # Highlight safe grid (GREEN)
    cv2.rectangle(overlay,
                  (safe_c * cell_w, safe_r * cell_h),
                  ((safe_c + 1) * cell_w, (safe_r + 1) * cell_h),
                  (0, 255, 0), 4)

    cv2.putText(overlay,
                "STATUS: HIGH RISK - EVACUATION REQUIRED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

# ==============================
# DISPLAY
# ==============================
cv2.imshow("Stampede Risk Analysis", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()