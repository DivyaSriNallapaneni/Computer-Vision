import cv2
import numpy as np
import os

# === Resolve paths relative to this script ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # ...\Final CV\module3\scripts
MODULE3_DIR = os.path.dirname(SCRIPT_DIR)                      # ...\Final CV\module3

input_folder = os.path.join(MODULE3_DIR, "static", "images_original")
output_folder = os.path.join(MODULE3_DIR, "static", "results")

print("[DEBUG] Input folder :", input_folder)
print("[DEBUG] Output folder:", output_folder)

# Create results folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process 10 images
for i in range(1, 11):
    img_path = os.path.join(input_folder, f"img{i}.jpg")
    print(f"[DEBUG] Reading: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Image not found -> {img_path}")
        continue  # Skip missing image

    # Compute gradients
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)

    mag_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ang_img = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mag_path = os.path.join(output_folder, f"img{i}_gradient_magnitude.jpg")
    ang_path = os.path.join(output_folder, f"img{i}_gradient_angle.jpg")

    cv2.imwrite(mag_path, mag_img)
    cv2.imwrite(ang_path, ang_img)

    print(f"Saved: {mag_path} and {ang_path}")
