import cv2
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
IMAGE_PATH = r"C:\Users\ASUS2024\Desktop\Stampede\sample_2.jpeg"
ROWS = 3
COLS = 4

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

overlay = image.copy()

# ==============================
# DRAW GRID
# ==============================
for r in range(ROWS):
    for c in range(COLS):

        x1 = c * cell_w
        y1 = r * cell_h
        x2 = (c + 1) * cell_w
        y2 = (r + 1) * cell_h

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

# ==============================
# DISPLAY
# ==============================
cv2.imshow("Grid Mapping (3x4)", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()