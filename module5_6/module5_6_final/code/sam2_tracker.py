import cv2
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(base_dir, "sam2_tracker", "sam2_input.mp4")
    masks_path = os.path.join(base_dir, "sam2_tracker", "sam2_masks.npz")
    output_path = os.path.join(base_dir, "sam2_tracker", "sam2_tracked.avi")

    print("[INFO] Input video:", input_path)
    print("[INFO] Masks file:", masks_path)
    print("[INFO] Output (AVI):", output_path)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video:", input_path)
        return

    # Load masks
    if not os.path.exists(masks_path):
        print("Error: sam2_masks.npz not found at", masks_path)
        return

    data = np.load(masks_path)
    if "masks" not in data:
        print("Error: 'masks' array not found inside sam2_masks.npz")
        return
    masks = data["masks"]

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 20.0

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # --- VideoWriter for AVI (MJPG codec) ---
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    if not out.isOpened():
        print("[ERROR] Could not create AVI output file")
        cap.release()
        return

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Reached end of video.")
            break

        if frame_idx >= len(masks):
            print("Warning: ran out of masks at frame", frame_idx)
            break

        mask = masks[frame_idx]

        # Make sure mask size matches frame size
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        # Binarize mask
        mask_bin = (mask > 0).astype(np.uint8)

        # Create green overlay where mask == 1
        color = np.zeros_like(frame)
        color[:, :] = (0, 255, 0)
        overlay = frame.copy()
        overlay[mask_bin == 1] = cv2.addWeighted(
            frame[mask_bin == 1], 0.3,
            color[mask_bin == 1], 0.7,
            0
        )

        # Draw contour of segmented region
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # Add label text
        cv2.putText(overlay, "SAM2-based segmentation tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Write frame to output video
        out.write(overlay)
        written += 1
        frame_idx += 1

        # Optional live preview (press 'q' to stop early)
        cv2.imshow("SAM2 Tracking (preview)", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Stopped early by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved SAM2 tracked video to: {output_path}")
    print(f"Total frames written: {written}")

if __name__ == "__main__":
    main()
