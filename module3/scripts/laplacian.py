import cv2
import numpy as np
import os

# === Resolve correct paths relative to this script ===
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
        continue

    # --- Gaussian Blur + Laplacian ---
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)

    # Normalize for saving
    log_img = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save output
    save_path = os.path.join(output_folder, f"img{i}_LoG.jpg")
    cv2.imwrite(save_path, log_img)

    print(f"Saved: {save_path}")
