import cv2
import numpy as np

# ==============================
# CONFIG
# ==============================
ROWS = 3
COLS = 4
BLUR_KERNEL = 101

LOW_THRESHOLD = 90
HIGH_THRESHOLD = 130

LAST_ROW = ROWS - 1

# ==============================
# ANALYSIS FUNCTION
# ==============================
def analyze_frame(image):

    h, w, _ = image.shape
    cell_w = w // COLS
    cell_h = h // ROWS

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.abs(lap)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    lap = lap.astype(np.uint8)

    density_map = cv2.GaussianBlur(lap, (BLUR_KERNEL, BLUR_KERNEL), 0)
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

    grid_density = np.zeros((ROWS, COLS))

    for r in range(ROWS):
        for c in range(COLS):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w

            cell = density_map[y1:y2, x1:x2]
            grid_density[r, c] = np.mean(cell)

    return grid_density, density_map


# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Press 'c' to analyze")
print("Press 'q' to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):

        grid_density, density_map = analyze_frame(frame.copy())

        h, w, _ = frame.shape
        cell_w = w // COLS
        cell_h = h // ROWS

        heatmap_color = cv2.applyColorMap(density_map.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.6, 0)

        # Draw grid
        for r in range(ROWS):
            for c in range(COLS):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = (c + 1) * cell_w
                y2 = (r + 1) * cell_h

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), 1)

                val = int(grid_density[r,c])
                cv2.putText(overlay, str(val),
                            (x1+10, y1+30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,0), 3)
                cv2.putText(overlay, str(val),
                            (x1+10, y1+30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2)

        # ==========================
        # LAST ROW LOGIC ONLY
        # ==========================
        last_row_values = grid_density[LAST_ROW]
        max_value = np.max(last_row_values)
        max_c = np.argmax(last_row_values)

        # Determine risk level
        if max_value < LOW_THRESHOLD:
            risk_level = "LOW"
        elif max_value < HIGH_THRESHOLD:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"

        # Highlight max cell
        x1 = max_c * cell_w
        y1 = LAST_ROW * cell_h
        x2 = (max_c + 1) * cell_w
        y2 = (LAST_ROW + 1) * cell_h

        if risk_level == "LOW":
            color = (0,255,0)
        elif risk_level == "MODERATE":
            color = (0,165,255)   # Orange
        else:
            color = (0,0,255)

        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, 4)

        # ==========================
        # HIGH RISK → PATH
        # ==========================
        if risk_level == "HIGH":

            safe_c = np.argmin(last_row_values)

            path_cols = range(min(max_c, safe_c), max(max_c, safe_c)+1)

            for c in path_cols:
                x_center = c * cell_w + cell_w//2
                y_center = LAST_ROW * cell_h + cell_h//2
                cv2.circle(overlay, (x_center, y_center), 6, (255,0,0), -1)

            status_text = "HIGH RISK - SHIFT REQUIRED"

        elif risk_level == "MODERATE":
            status_text = "MODERATE RISK - MONITOR"
        else:
            status_text = "LOW RISK - SAFE"

        cv2.putText(overlay,
                    status_text,
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 3)

        cv2.imshow("Analysis Result", overlay)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()