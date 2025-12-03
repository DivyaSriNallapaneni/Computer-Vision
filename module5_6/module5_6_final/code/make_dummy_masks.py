import cv2
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(base_dir, "sam2_tracker", "sam2_input.mp4")
    masks_path = os.path.join(base_dir, "sam2_tracker", "sam2_masks.npz")

    print("[INFO] Reading video:", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open sam2_input.mp4")
        return

    masks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]

        # ---------- DUMMY MASK (simple rectangle in center) ----------
        mask = np.zeros((H, W), dtype=np.uint8)
        cx1, cy1 = W//4, H//4
        cx2, cy2 = 3*W//4, 3*H//4
        mask[cy1:cy2, cx1:cx2] = 1
        # -------------------------------------------------------------

        masks.append(mask)
        frame_idx += 1

    cap.release()

    masks = np.array(masks)

    np.savez(masks_path, masks=masks)
    print(f"[DONE] Created sam2_masks.npz with {len(masks)} masks at:\n{masks_path}")

if __name__ == "__main__":
    main()
