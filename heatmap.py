import cv2
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
IMAGE_PATH = r"C:\Users\ASUS2024\Desktop\Stampede\sample_2.jpeg"
ROWS = 3
COLS = 4
BLUR_KERNEL = 101

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
# GRID DENSITY CALCULATION
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
# DISPLAY
# ==============================
cv2.imshow("Heatmap + Grid (3x4)", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()