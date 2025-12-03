import cv2
import numpy as np
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(base_dir, "marker_tracker", "marker_input.mp4")
    output_path = os.path.join(base_dir, "marker_tracker", "marker_tracked.avi")

    print("[INFO] Input video:", input_path)
    print("[INFO] Output video:", output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[ERROR] Could not open input video")
        return

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # 🔥 SIMPLEST STABLE FORMAT
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.putText(frame, "Marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "No marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        out.write(frame)
        frame_count += 1

        if frame_count % 20 == 0:
            print("[INFO] Processed", frame_count)

    cap.release()
    out.release()
    print("[DONE] Wrote", frame_count, "frames")
    print("[SAVED]:", output_path)

if __name__ == "__main__":
    main()
