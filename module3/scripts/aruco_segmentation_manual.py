import cv2
import numpy as np
import os

# -------------------------
# Paths
# -------------------------
input_folder = r"C:\Users\divya\Downloads\object_analysis_project_updated\static\images_original"
aruco_folder = r"C:\Users\divya\Downloads\object_analysis_project_updated\static\images_aruco"
os.makedirs(aruco_folder, exist_ok=True)

# -------------------------
# ArUco dictionary
# -------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 50  # pixels

# Marker positions (example: 4 corners of object)
marker_positions = [
    (50, 50),
    (800, 50),
    (50, 800),
    (800, 800)
]

# -------------------------
# Process all images in folder
# -------------------------
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("No images found in the folder!")
else:
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        aruco_img = img.copy()

        # Paste ArUco markers
        for idx, (x, y) in enumerate(marker_positions):
          marker_img = cv2.aruco.generateImageMarker(aruco_dict, idx, marker_size)
          aruco_img[y:y+marker_size, x:x+marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)


        # Save new image
        save_path = os.path.join(aruco_folder, f"{os.path.splitext(img_file)[0]}_aruco.jpg")
        cv2.imwrite(save_path, aruco_img)
        print(f"Saved ArUco image: {save_path}")
