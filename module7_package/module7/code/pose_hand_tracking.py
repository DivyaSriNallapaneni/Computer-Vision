
import cv2
import csv
import os

try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "mediapipe is not installed. Install it with:\n"
        "    pip install mediapipe opencv-python\n"
    )

def main():
    """
    Real-time pose and hand tracking demo using MediaPipe.

    - Opens the default webcam (index 0).
    - Draws pose and hand landmarks on each frame.
    - Saves an annotated video and CSV files with numeric keypoints.

    Output files:
        module7_pose_hand_demo.mp4  (in ../demo/)
        pose_keypoints.csv          (in ../csv/)
        hand_keypoints.csv          (in ../csv/)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(os.path.dirname(base_dir), "csv")
    demo_dir = os.path.join(os.path.dirname(base_dir), "demo")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)

    pose_csv_path = os.path.join(csv_dir, "pose_keypoints.csv")
    hand_csv_path = os.path.join(csv_dir, "hand_keypoints.csv")
    video_out_path = os.path.join(demo_dir, "module7_pose_hand_demo.mp4")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam (index 0).")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        cap.release()
        return

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out_path, fourcc, 20.0, (width, height))

    pose_csv_file = open(pose_csv_path, mode="w", newline="")
    hand_csv_file = open(hand_csv_path, mode="w", newline="")
    pose_writer = csv.writer(pose_csv_file)
    hand_writer = csv.writer(hand_csv_file)

    pose_writer.writerow(["frame_id", "landmark_id", "x", "y", "z", "visibility"])
    hand_writer.writerow(["frame_id", "hand_label", "landmark_id", "x", "y", "z"])

    frame_id = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        print("Press 'q' to stop recording.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or cannot read frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    pose_writer.writerow([frame_id, idx, lm.x, lm.y, lm.z, lm.visibility])

            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    hands_results.multi_hand_landmarks,
                    hands_results.multi_handedness
                ):
                    label = handedness.classification[0].label
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        hand_writer.writerow([frame_id, label, idx, lm.x, lm.y, lm.z])

            cv2.putText(
                frame,
                "Module 7: Pose & Hand Tracking (press 'q' to stop)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Module 7 – Pose & Hand Tracking", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

    cap.release()
    out.release()
    pose_csv_file.close()
    hand_csv_file.close()
    cv2.destroyAllWindows()

    print(f"Saved demo video to: {video_out_path}")
    print(f"Saved pose CSV to: {pose_csv_path}")
    print(f"Saved hand CSV to: {hand_csv_path}")


if __name__ == "__main__":
    main()
