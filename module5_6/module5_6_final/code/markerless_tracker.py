import cv2
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_path = os.path.join(base_dir, "markerless_tracker", "markerless_input.mp4")
    output_path = os.path.join(base_dir, "markerless_tracker", "markerless_tracked.avi")

    print("[INFO] Input video:", input_path)
    print("[INFO] Output video will be saved as AVI:", output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[ERROR] Could not open input video")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame")
        return

    # Select region on the first frame
    roi = cv2.selectROI("Select object to track", first_frame, False, False)
    cv2.destroyWindow("Select object to track")

    x, y, w, h = roi
    roi_frame = first_frame[y:y+h, x:x+w]

    roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(
        roi_gray, maxCorners=200, qualityLevel=0.01, minDistance=5
    )

    if features is None:
        print("[ERROR] No features detected in ROI")
        return

    features[:, 0, 0] += x
    features[:, 0, 1] += y

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = features

    # 🌟 Save output using AVI + MJPG (most compatible)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 20.0

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, frame_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

        cv2.putText(frame, "Markerless KLT Tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        out.write(frame)
        frame_count += 1

        if frame_count % 20 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

        prev_gray = frame_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()
    print(f"[DONE] Wrote {frame_count} frames to {output_path}")

if __name__ == "__main__":
    main()
