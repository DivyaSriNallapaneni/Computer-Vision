import cv2
import numpy as np
import os

# === Resolve correct module3 paths automatically ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../Final CV/module3/scripts
MODULE3_DIR = os.path.dirname(SCRIPT_DIR)                      # .../Final CV/module3

# Correct ArUco input and output folders
input_folder = os.path.join(MODULE3_DIR, "static", "images_aruco")
output_folder = os.path.join(MODULE3_DIR, "static", "results")

print("[DEBUG] Input folder :", input_folder)
print("[DEBUG] Output folder:", output_folder)

os.makedirs(output_folder, exist_ok=True)

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Load images
image_files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if not image_files:
    print("No images found in input folder!")
else:
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        print(f"[DEBUG] Reading: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

        if corners:
            pts = np.vstack(corners).reshape(-1, 2)
            hull = cv2.convexHull(pts).astype(int)
            cv2.polylines(img, [hull], True, (255, 0, 0), 2)
        else:
            print(f"No markers detected in {img_file}")

        save_path = os.path.join(
            output_folder,
            f"{os.path.splitext(img_file)[0]}_segmented.jpg"
        )
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")
