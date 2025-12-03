import cv2
import numpy as np
import os

# ---------- USER PARAMETERS (edit for your experiment) ----------
# Baseline between left and right camera positions (cm)
BASELINE_CM = 7.0

# Focal length of your camera in mm (approx from EXIF or phone spec)
FOCAL_MM = 4.2           # e.g. iPhone wide lens ~4.2 mm

# Sensor width of your camera in mm
SENSOR_WIDTH_MM = 5.6    # e.g. iPhone 13 main sensor ~5.6 mm

# True width of the object in cm (for comparison)
TRUE_WIDTH_CM = 8.5

# ----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stereo_dir = os.path.join(BASE_DIR, "stereo_demo")

left_path  = os.path.join(stereo_dir, "left.jpg")
right_path = os.path.join(stereo_dir, "right.jpg")

left  = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

if left is None or right is None:
    raise FileNotFoundError("Could not load left/right images. Check paths.")

if left.shape != right.shape:
    raise ValueError("Left and right images must have the same size.")

h, w = left.shape

# ---- 1) Approximate fx (focal length in pixels) from sensor width ----
fx_pixels = (FOCAL_MM * w) / SENSOR_WIDTH_MM
print(f"Image width = {w}px, fx ≈ {fx_pixels:.1f} px")

# ---- 2) Compute disparity map (very simple block matcher) ----
# Convert to 8-bit if needed
left_u8  = cv2.normalize(left,  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
right_u8 = cv2.normalize(right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

num_disparities = 16 * 8   # must be multiple of 16
block_size = 11            # odd

stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
disp_raw = stereo.compute(left_u8, right_u8).astype(np.float32)  # 16x scaled

disp = disp_raw / 16.0     # real disparity values

# ---- 3) Let the user choose two points (object width) ----
print("Click two points on the LEFT image to define the object width.")
print("Close the window when done.")

clone = cv2.cvtColor(left_u8, cv2.COLOR_GRAY2BGR)
points = []

def on_mouse(event, x, y, flags, param):
    global points, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(clone, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("Left image - click 2 points", clone)

cv2.namedWindow("Left image - click 2 points", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Left image - click 2 points", on_mouse)
cv2.imshow("Left image - click 2 points", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 2:
    raise RuntimeError("You must click exactly two points.")

(x1, y1), (x2, y2) = points

# Pixel distance (width in image)
dx = x2 - x1
dy = y2 - y1
pixel_length = np.sqrt(dx*dx + dy*dy)
print(f"Pixel distance between points = {pixel_length:.2f}px")

# ---- 4) Estimate depth Z using disparity at mid-point ----
xm = int(round((x1 + x2) / 2.0))
ym = int(round((y1 + y2) / 2.0))

local_patch = disp[max(0, ym-3):ym+4, max(0, xm-3):xm+4]
valid = local_patch[local_patch > 0]

if valid.size == 0:
    raise RuntimeError("No valid disparity near the selected segment.")

d = float(np.median(valid))  # disparity in pixels
print(f"Median disparity at object = {d:.2f}px")

# Z in mm: Z = f * B / d
BASELINE_MM = BASELINE_CM * 10.0
Z_mm = fx_pixels * BASELINE_MM / d
Z_cm = Z_mm / 10.0
print(f"Estimated depth Z ≈ {Z_cm:.2f} cm")

# ---- 5) Convert pixel length to real-world length ----
X_mm = (Z_mm * pixel_length) / fx_pixels
X_cm = X_mm / 10.0
print(f"Estimated object width ≈ {X_cm:.2f} cm (true = {TRUE_WIDTH_CM:.2f} cm)")

error_pct = abs(X_cm - TRUE_WIDTH_CM) / TRUE_WIDTH_CM * 100.0
print(f"Absolute error ≈ {error_pct:.1f}%")

# ---- 6) Save an overlay image to show on webpage ----
overlay = cv2.cvtColor(left_u8, cv2.COLOR_GRAY2BGR)
cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.putText(
    overlay,
    f"Width ≈ {X_cm:.2f} cm, Z ≈ {Z_cm:.1f} cm",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2,
)

result_path = os.path.join(stereo_dir, "result_overlay.png")
cv2.imwrite(result_path, overlay)
print(f"Saved overlay to {result_path}")
