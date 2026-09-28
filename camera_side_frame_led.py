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

COM_PORT = "COM16"
BAUD_RATE = 115200

ser = serial.Serial(COM_PORT, BAUD_RATE)
time.sleep(2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Press:")
print("r - Resume live feed")
print("c - Capture and send to LEDs")
print("q - Quit")

live_mode = True
frozen_frame = None

while True:

    if live_mode:
        ret, frame = cap.read()
        if not ret:
            break
        warped = frame.copy()   # Direct full frame
    else:
        warped = frozen_frame.copy()

    h, w, _ = warped.shape
    cell_w = w // COLS
    cell_h = h // ROWS

    # ==============================
    # DENSITY ESTIMATION
    # ==============================
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
    # GRID + DENSITY VALUES
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

            val = int(grid_density[r,c])
            cv2.putText(overlay,str(val),(x1+10,y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3)
            cv2.putText(overlay,str(val),(x1+10,y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    max_density = np.max(grid_density)
    evacuation_required = max_density > HIGH_THRESHOLD

    led_states = ["0"] * (ROWS*COLS)

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

        # Compute path
        path = []
        r, c = max_r, max_c
        while (r,c)!=(safe_r,safe_c):
            path.append((r,c))
            if r < safe_r: r+=1
            elif r > safe_r: r-=1
            elif c < safe_c: c+=1
            elif c > safe_c: c-=1
        path.append((safe_r,safe_c))

        # Draw path
        for i in range(len(path)-1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]

            x1 = c1 * cell_w + cell_w // 2
            y1 = r1 * cell_h + cell_h // 2
            x2 = c2 * cell_w + cell_w // 2
            y2 = r2 * cell_h + cell_h // 2

            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # LED Mapping
        high_index = max_r*COLS + max_c
        safe_index = safe_r*COLS + safe_c

        led_states[high_index] = "3"   # FAST blink
        led_states[safe_index] = "1"   # SOLID

        for (pr,pc) in path:
            idx = pr*COLS + pc
            if idx != high_index and idx != safe_index:
                led_states[idx] = "2"  # SLOW blink

    cv2.imshow("Stampede System", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        print("Frame Captured - LEDs Activated")
        frozen_frame = warped.copy()
        live_mode = False

        data = ",".join(led_states) + "\n"
        ser.write(data.encode())

    if key == ord('r'):
        print("Live Mode Resumed")
        live_mode = True

    if key == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()