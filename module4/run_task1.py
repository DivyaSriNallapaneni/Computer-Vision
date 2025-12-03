import cv2
import numpy as np
import os
from utils.stitching_utils import stitch_images, cylindrical_projection

# --- CONFIG (balanced) ---
WORK_WIDTH = 1600            # resize width for processing (balanced)
FOCAL_SCALE = 1.0            # focal = FOCAL_SCALE * image_width
MAX_CANVAS_W = 4000          # maximum output canvas width
MAX_CANVAS_H = 2000          # maximum output canvas height
# --------------------------

os.makedirs("outputs_task1", exist_ok=True)

def resize_image(img, width=WORK_WIDTH):
    h, w = img.shape[:2]
    if w <= width:
        return img
    ratio = width / w
    return cv2.resize(img, (width, int(h * ratio)))

# Load up to 8 images
images = []
for i in range(1, 9):
    path = f"images_task1/img{i}.jpg"
    if not os.path.exists(path):
        continue
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: can't load {path}. Skipping.")
        continue
    img = resize_image(img, WORK_WIDTH)
    images.append(img)

if len(images) < 2:
    print("Need at least 2 valid images in images_task1/. Exiting.")
    exit()

# cylindrical projection of all images
proj_images = []
for idx, img in enumerate(images):
    h, w = img.shape[:2]
    focal = max(1, int(FOCAL_SCALE * w))  # focal length in px
    cyl = cylindrical_projection(img, focal)
    proj_images.append((cyl, focal))
    print(f"Image {idx+1}: original ({w}x{h}), projected ({cyl.shape[1]}x{cyl.shape[0]}), focal={focal}")

# Now iterative stitching: base is first projected image
stitched = proj_images[0][0].copy()

sift = cv2.SIFT_create(nfeatures=5000)
bf = cv2.BFMatcher()

for i in range(1, len(proj_images)):
    img2 = proj_images[i][0]
    # detect
    gray1 = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print(f"Skipping image {i}: not enough keypoints/descriptors.")
        continue

    # match
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Image {i}: good matches = {len(good)}")
    if len(good) < 8:
        print(f"Skipping image {i}: too few good matches ({len(good)})")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None or not np.all(np.isfinite(H)):
        print(f"Skipping image {i}: invalid homography")
        continue

    # Sanity: compute transformed corners to ensure canvas won't explode
    h2, w2 = img2.shape[:2]
    corners = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    transformed = cv2.perspectiveTransform(corners, H)
    all_x = transformed[:,0,0]
    all_y = transformed[:,0,1]
    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()
    projected_w = int(np.ceil(xmax - xmin))
    projected_h = int(np.ceil(ymax - ymin))
    print(f"Projected region size for image {i}: {projected_w}x{projected_h}")

    if projected_w > MAX_CANVAS_W or projected_h > MAX_CANVAS_H:
        print(f"Skipping image {i}: projected region too large ({projected_w}x{projected_h})")
        continue

    # stitch with bounded canvas and blending
    stitched = stitch_images(stitched, img2, H, max_width=MAX_CANVAS_W, max_height=MAX_CANVAS_H)
    print(f"Successfully stitched image {i}")

# Save
out_path = "outputs_task1/stitched_task1.jpg"
cv2.imwrite(out_path, stitched)
print(f"Task 1 result saved to {out_path}")
