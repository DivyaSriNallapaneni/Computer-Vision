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

os.makedirs(output_folder, exist_ok=True)

for i in range(1, 11):
    img_path = os.path.join(input_folder, f"img{i}.jpg")
    print(f"[DEBUG] Reading: {img_path}")

    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Image not found -> {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 127, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boundary_img = img.copy()
    cv2.drawContours(boundary_img, contours, -1, (0, 255, 0), 2)

    save_path = os.path.join(output_folder, f"img{i}_boundary.jpg")
    cv2.imwrite(save_path, boundary_img)

    print(f"Saved boundary for img{i} -> {save_path}")
