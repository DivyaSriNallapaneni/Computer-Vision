"""task1_template_matching.py
Task 1: Correlation-based template matching for 10 objects in a single scene.

Assumptions:
- Scene image (with all 10 objects) is at:  task1/scene.jpg
- 10 template images (captured from different scenes) are at:
    templates_task1/template1.jpg ... templates_task1/template10.jpg
- This script draws bounding boxes for each template found in the scene,
  and saves a combined result image as: task1/result_all.jpg

You should attach this code (or your modified version) with your report.
"""

import cv2
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_path = os.path.join(base_dir, "task1", "scene.jpg")
    templates_dir = os.path.join(base_dir, "templates_task1")
    out_path = os.path.join(base_dir, "task1", "result_all.jpg")

    if not os.path.exists(scene_path):
        print("Scene image not found at", scene_path)
        return

    scene = cv2.imread(scene_path)
    if scene is None:
        print("Failed to read scene image.")
        return

    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # We'll look for template1.jpg ... template10.jpg
    detections = []
    match_threshold = 0.6  # you can tune this based on your images

    for i in range(1, 11):
        tpl_name = f"template{i}.jpg"
        tpl_path = os.path.join(templates_dir, tpl_name)
        if not os.path.exists(tpl_path):
            print(f"Template {tpl_name} not found in {templates_dir}, skipping.")
            continue

        tpl = cv2.imread(tpl_path)
        if tpl is None:
            print(f"Failed to read {tpl_path}, skipping.")
            continue

        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        if scene_gray.shape[0] < tpl_gray.shape[0] or scene_gray.shape[1] < tpl_gray.shape[1]:
            print(f"Template {tpl_name} is larger than scene, skipping.")
            continue

        # Correlation-based template matching
        result = cv2.matchTemplate(scene_gray, tpl_gray, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        print(f"{tpl_name}: max correlation = {max_val:.3f} at location {max_loc}")

        if max_val >= match_threshold:
            x, y = max_loc
            h, w = tpl_gray.shape
            detections.append((x, y, w, h, max_val, tpl_name))

    # Draw all detections on a copy of the scene
    vis = scene.copy()
    for (x, y, w, h, score, name) in detections:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(out_path, vis)
    print("Saved combined result image to:", out_path)
    print("Number of templates detected above threshold:", len(detections))

if __name__ == "__main__":
    main()
