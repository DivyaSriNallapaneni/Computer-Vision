import cv2
import numpy as np

def stitch_images(img1, img2, H):
    """
    Warp img2 onto img1 using homography H.
    Handles translation and resizing to prevent huge empty images.
    """
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute corner points
    corners_img2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]], dtype=np.float32).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)

    all_corners = np.vstack((np.array([[0,0],[0,h1],[w1,h1],[w1,0]], dtype=np.float32).reshape(-1,1,2), warped_corners))

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation
    translation = np.array([[1, 0, -xmin],
                            [0, 1, -ymin],
                            [0, 0, 1]])

    result = cv2.warpPerspective(img2, translation.dot(H), (xmax - xmin, ymax - ymin))
    result[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1

    return result
