import cv2
import numpy as np
import os

from utils.stitching_utils import stitch_images
from utils.sift_from_scratch import detect_keypoints, compute_descriptors
from utils.feature_matching import match_descriptors
from utils.ransac import ransac_homography

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("outputs_task2", exist_ok=True)

# -----------------------------
# Resize function (keep aspect ratio)
# -----------------------------
def resize_image(img, width=900):
    h, w = img.shape[:2]
    ratio = width / w
    return cv2.resize(img, (width, int(h * ratio)))

# -----------------------------
# Load and resize images
# -----------------------------
images = [cv2.imread(f"images_task2/img{i}.jpg") for i in range(1, 5)]
images = [resize_image(img) for img in images]

# -----------------------------
# 1️⃣ Custom SIFT stitching
# -----------------------------
stitched_custom = images[0]

for i in range(1, len(images)):
    img2 = images[i]

    # Detect keypoints + descriptors
    kp1 = detect_keypoints(stitched_custom)
    kp2 = detect_keypoints(img2)
    des1 = compute_descriptors(stitched_custom, kp1)
    des2 = compute_descriptors(img2, kp2)

    # Match descriptors (looser ratio)
    matches = match_descriptors(des1, des2, ratio=0.9)

    # Draw matches for visualization
    if len(matches) > 0:
        img_matches = cv2.drawMatches(
            stitched_custom, kp1,
            img2, kp2,
            [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _distance=0) for m in matches],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(f"outputs_task2/matches_custom_{i}.jpg", img_matches)

    if len(matches) < 4:
        print(f"[Custom SIFT] Skipping image {i}: not enough matches ({len(matches)})")
        continue

    # Compute homography using RANSAC
    H, inliers = ransac_homography(np.float32([kp1[m[0]].pt for m in matches]),
                                   np.float32([kp2[m[1]].pt for m in matches]),
                                   max_iterations=3000, threshold=6)

    if H is None or np.isnan(H).any() or np.isinf(H).any():
        print(f"[Custom SIFT] Skipping image {i}: invalid homography")
        continue

    stitched_custom = stitch_images(stitched_custom, img2, H)
    stitched_custom = resize_image(stitched_custom)  # prevent huge size

cv2.imwrite("outputs_task2/stitched_sift_task2_custom.jpg", stitched_custom)
print("Custom SIFT panorama saved: outputs_task2/stitched_sift_task2_custom.jpg")

# -----------------------------
# 2️⃣ OpenCV SIFT stitching
# -----------------------------
stitched_opencv = images[0]
sift = cv2.SIFT_create()

for i in range(1, len(images)):
    img2 = images[i]
    gray1 = cv2.cvtColor(stitched_opencv, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher + ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Draw matches
    if len(good_matches) > 0:
        img_matches = cv2.drawMatches(
            stitched_opencv, kp1,
            img2, kp2,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(f"outputs_task2/matches_opencv_{i}.jpg", img_matches)

    if len(good_matches) < 4:
        print(f"[OpenCV SIFT] Skipping image {i}: not enough matches ({len(good_matches)})")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        print(f"[OpenCV SIFT] Skipping image {i}: Homography failed")
        continue

    stitched_opencv = stitch_images(stitched_opencv, img2, H)
    stitched_opencv = resize_image(stitched_opencv)

cv2.imwrite("outputs_task2/stitched_sift_task2_opencv.jpg", stitched_opencv)
print("OpenCV SIFT panorama saved: outputs_task2/stitched_sift_task2_opencv.jpg")
