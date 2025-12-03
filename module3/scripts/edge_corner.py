import cv2
import numpy as np
import os

# === Resolve correct module3 paths automatically ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../Final CV/module3/scripts
MODULE3_DIR = os.path.dirname(SCRIPT_DIR)                      # .../Final CV/module3

input_folder = os.path.join(MODULE3_DIR, "static", "images_original")
output_folder = os.path.join(MODULE3_DIR, "static", "results")

print("[DEBUG] Input folder :", input_folder)
print("[DEBUG] Output folder:", output_folder)

os.makedirs(output_folder, exist_ok=True)

# === Process 10 images ===
for i in range(1, 11):
    img_path = os.path.join(input_folder, f"img{i}.jpg")
    print(f"[DEBUG] Reading: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image not found → {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ================================
    # 🔹 EDGE DETECTION — CANNY
    # ================================
    edges = cv2.Canny(gray, 100, 200)
    save_edges = os.path.join(output_folder, f"img{i}_edges.jpg")
    cv2.imwrite(save_edges, edges)

    # ================================
    # 🔹 CORNER DETECTION — HARRIS
    # ================================
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    corner_img = img.copy()
    corner_img[dst > 0.01 * dst.max()] = [0, 0, 255]

    save_corners = os.path.join(output_folder, f"img{i}_corners.jpg")
    cv2.imwrite(save_corners, corner_img)

    print(f"Saved: {save_edges} and {save_corners}")
