import numpy as np
import cv2

def ransac_homography(pts1, pts2, max_iterations=2000, threshold=6):
    """
    Compute homography using RANSAC.
    pts1, pts2: Nx2 points
    max_iterations: number of RANSAC iterations
    threshold: inlier distance threshold
    """
    best_H = None
    best_inliers = []

    num_points = pts1.shape[0]
    if num_points < 4:
        return None, []

    for _ in range(max_iterations):
        idx = np.random.choice(num_points, 4, replace=False)
        src_pts = pts1[idx]
        dst_pts = pts2[idx]
        H = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
        pts1_hom = np.concatenate([pts1, np.ones((num_points,1))], axis=1)  # Nx3
        projected = (H @ pts1_hom.T).T
        projected /= projected[:,2].reshape(-1,1)
        dists = np.linalg.norm(projected[:,:2] - pts2, axis=1)
        inliers = np.where(dists < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    if best_H is None:
        return None, []
    return best_H, best_inliers

