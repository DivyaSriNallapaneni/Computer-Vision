import numpy as np

def match_descriptors(des1, des2, ratio=0.75):
    """
    Match descriptors using ratio test (Lowe's ratio).
    des1, des2: descriptors
    ratio: threshold for ratio test
    """
    matches = []
    for i, d1 in enumerate(des1):
        distances = np.linalg.norm(des2 - d1, axis=1)
        if len(distances) < 2:
            continue
        sorted_idx = np.argsort(distances)
        if distances[sorted_idx[0]] < ratio * distances[sorted_idx[1]]:
            matches.append((i, sorted_idx[0]))
    return matches

