import cv2
import numpy as np
import time

# ==============================
# CONFIGURATION
# ==============================
IMAGE_PATH = r"C:\Users\ASUS2024\Desktop\Stampede\sample_8.jpeg"
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
base_overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.6, 0)

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
# DISPLAY DENSITY LOGIC
# ==============================
if evacuation_required:

    flat_indices_desc = np.argsort(grid_density, axis=None)[::-1]
    sorted_desc = [np.unravel_index(idx, grid_density.shape) for idx in flat_indices_desc]
    danger_cells = sorted_desc[:2]

    display_density = grid_density.copy()
    highest_value = display_density[danger_cells[0]]
    display_density[danger_cells[1]] = highest_value

else:
    display_density = grid_density.copy()

# ==============================
# DRAW GRID + VALUES
# ==============================
overlay = base_overlay.copy()

for r in range(ROWS):
    for c in range(COLS):

        x1 = c * cell_w
        y1 = r * cell_h
        x2 = (c + 1) * cell_w
        y2 = (r + 1) * cell_h

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)

        value = int(display_density[r, c])

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

    cv2.imshow("Stampede Risk Analysis", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==============================
# HIGH RISK CASE
# ==============================
else:

    # ---- Safest boundary grids ----
    boundary_cells = []

    for c in range(COLS):
        boundary_cells.append((0, c))
        boundary_cells.append((ROWS - 1, c))

    for r in range(ROWS):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, COLS - 1))

    boundary_cells = list(set(boundary_cells))
    boundary_sorted = sorted(boundary_cells, key=lambda x: grid_density[x])
    safe_cells = boundary_sorted[:2]

    # ---- Draw danger grids ----
    for (dr, dc) in danger_cells:
        cv2.rectangle(overlay,
                      (dc * cell_w, dr * cell_h),
                      ((dc + 1) * cell_w, (dr + 1) * cell_h),
                      (0, 0, 255), 4)

    # ---- Draw safe grids ----
    for (sr, sc) in safe_cells:
        cv2.rectangle(overlay,
                      (sc * cell_w, sr * cell_h),
                      ((sc + 1) * cell_w, (sr + 1) * cell_h),
                      (0, 255, 0), 4)

    cv2.putText(overlay,
                "STATUS: DUAL HIGH RISK - SEQUENTIAL EVACUATION",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    cv2.imshow("Stampede Risk Analysis", overlay)
    cv2.waitKey(500)

    # ==============================
    # SEQUENTIAL PATH ANIMATION
    # ==============================
    for i in range(2):

        dr, dc = danger_cells[i]
        sr, sc = safe_cells[i]

        path = []
        r, c = dr, dc

        while (r, c) != (sr, sc):
            path.append((r, c))
            if r < sr:
                r += 1
            elif r > sr:
                r -= 1
            elif c < sc:
                c += 1
            elif c > sc:
                c -= 1

        path.append((sr, sc))

        pixel_path = []
        for (pr, pc) in path:
            x = pc * cell_w + cell_w // 2
            y = pr * cell_h + cell_h // 2
            pixel_path.append((x, y))

        pixel_path = pixel_path[::-1]

        for shift in range(0, 40):

            temp = overlay.copy()

            for j in range(len(pixel_path) - 1):
                x1, y1 = pixel_path[j]
                x2, y2 = pixel_path[j + 1]

                if (j + shift) % 6 < 3:
                    cv2.line(temp, (x1, y1), (x2, y2), (255, 0, 0), 4)

            cv2.imshow("Stampede Risk Analysis", temp)
            cv2.waitKey(80)

        time.sleep(0.3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()