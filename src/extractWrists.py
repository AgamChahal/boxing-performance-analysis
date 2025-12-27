import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0)  # 0 = no limit
    

    args = ap.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    fieldnames = [
        "frame",
        "time_s",
        "left_wrist_x",
        "left_wrist_y",
        "left_wrist_vis",
        "right_wrist_x",
        "right_wrist_y",
        "right_wrist_vis",
    ]

    frame_idx = 0
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            # default: blank when not detected
            lwx = lwy = lwv = ""
            rwx = rwy = rwv = ""

            if result.pose_landmarks is not None:
                lm = result.pose_landmarks.landmark
                lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]   # 15
                rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]  # 16
                lwx, lwy, lwv = lw.x, lw.y, lw.visibility
                rwx, rwy, rwv = rw.x, rw.y, rw.visibility

            time_s = (frame_idx - 1) / fps if fps else ""

            w.writerow(
                {
                    "frame": frame_idx,
                    "time_s": time_s,
                    "left_wrist_x": lwx,
                    "left_wrist_y": lwy,
                    "left_wrist_vis": lwv,
                    "right_wrist_x": rwx,
                    "right_wrist_y": rwy,
                    "right_wrist_vis": rwv,
                }
            )

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

    pose.close()
    cap.release()
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()


