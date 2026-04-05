"""POC: Rider-level tracking in trial run videos.

Tests three approaches:
1. YOLO object detection (pretrained COCO)
2. Camera motion compensation + frame differencing
3. Position-based rider ordering (assigns car numbers by spatial order)

Usage:
    python3 scripts/poc_rider_tracking.py /tmp/trial_video_poc/iizuka_20260404_R01_trial.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Approach 1: YOLO detection
# ---------------------------------------------------------------------------

def test_yolo_detection(video_path: str, output_dir: Path) -> None:
    print("\n=== Approach 1: YOLO Detection ===")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed, skipping")
        return

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    sample_times = [3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    results_frames: list[tuple[float, np.ndarray, list]] = []

    for target_t in sample_times:
        target_frame = int(target_t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False, conf=0.15)

        annotated = frame.copy()
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                if cls_name in ("motorcycle", "person", "bicycle", "car", "truck"):
                    detections.append((cls_name, conf, x1, y1, x2, y2))
                    color = (0, 255, 0) if cls_name == "motorcycle" else (255, 165, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        results_frames.append((target_t, annotated, detections))
        moto = sum(1 for d in detections if d[0] == "motorcycle")
        person = sum(1 for d in detections if d[0] == "person")
        print(f"  t={target_t:.1f}s: motorcycle={moto}, person={person}, other={len(detections)-moto-person}")

    cap.release()

    if results_frames:
        h, w = results_frames[0][1].shape[:2]
        cols = min(len(results_frames), 4)
        rows = (len(results_frames) + cols - 1) // cols
        sheet = np.zeros((rows * (h + 30), cols * w, 3), dtype=np.uint8)

        for i, (t, annotated, _) in enumerate(results_frames):
            r, c = divmod(i, cols)
            y_off = r * (h + 30)
            x_off = c * w
            sheet[y_off:y_off + h, x_off:x_off + w] = annotated
            cv2.putText(sheet, f"t={t:.1f}s", (x_off + 5, y_off + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        path = str(output_dir / "yolo_detection.jpg")
        cv2.imwrite(path, sheet)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Approach 2: Camera motion compensation + frame differencing
# ---------------------------------------------------------------------------

def test_camera_compensation(video_path: str, output_dir: Path) -> None:
    print("\n=== Approach 2: Camera Motion Compensation ===")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    sample_times = [6.0, 10.0, 15.0, 20.0, 25.0]
    sample_frame_indices = {int(t * fps) for t in sample_times}
    results_frames: list[tuple[float, np.ndarray, np.ndarray, int]] = []

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if idx in sample_frame_indices:
            kp1, des1 = orb.detectAndCompute(prev_gray, None)
            kp2, des2 = orb.detectAndCompute(gray, None)

            rider_mask = np.zeros_like(gray)
            n_riders = 0

            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda m: m.distance)[:50]

                if len(matches) >= 4:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if H is not None:
                        h_frame, w_frame = gray.shape
                        warped_prev = cv2.warpPerspective(prev_gray, H, (w_frame, h_frame))
                        diff = cv2.absdiff(gray, warped_prev)
                        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        rider_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                        rider_mask = cv2.morphologyEx(rider_mask, cv2.MORPH_CLOSE, kernel)

                        contours, _ = cv2.findContours(
                            rider_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                        )
                        big_contours = [c for c in contours if cv2.contourArea(c) > 300]
                        n_riders = len(big_contours)

                        annotated = frame.copy()
                        for j, cnt in enumerate(big_contours):
                            x, y, w, h_ = cv2.boundingRect(cnt)
                            cv2.rectangle(annotated, (x, y), (x + w, y + h_), (0, 255, 0), 2)
                            cv2.putText(annotated, f"#{j+1}", (x, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        annotated = frame.copy()
                else:
                    annotated = frame.copy()
            else:
                annotated = frame.copy()

            t = idx / fps
            results_frames.append((t, annotated, rider_mask, n_riders))
            print(f"  t={t:.1f}s: {n_riders} riders detected after camera compensation")

        prev_gray = gray
        idx += 1

    cap.release()

    if results_frames:
        h, w = results_frames[0][1].shape[:2]
        sheet = np.zeros((len(results_frames) * (h + 30), w * 2, 3), dtype=np.uint8)
        for i, (t, annotated, mask_img, count) in enumerate(results_frames):
            y_off = i * (h + 30)
            sheet[y_off:y_off + h, 0:w] = annotated
            sheet[y_off:y_off + h, w:w * 2] = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(sheet, f"t={t:.1f}s | {count} riders",
                        (10, y_off + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        path = str(output_dir / "camera_compensated.jpg")
        cv2.imwrite(path, sheet)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Approach 3: Full tracking with position-based car number assignment
# ---------------------------------------------------------------------------

def test_tracking_with_ordering(video_path: str, output_dir: Path) -> None:
    print("\n=== Approach 3: YOLO Tracking + Position-based Ordering ===")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed, skipping")
        return

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    first_lap_start = int(3.0 * fps)
    first_lap_end = int(12.0 * fps)

    rider_tracks: dict[int, list[tuple[int, int, int]]] = {}
    sample_annotated: list[tuple[float, np.ndarray]] = []
    sample_interval = int(fps * 2)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if first_lap_start <= idx <= first_lap_end or idx % sample_interval == 0:
            results = model(frame, verbose=False, conf=0.15,
                            classes=[0, 1, 3])  # person, bicycle, motorcycle

            detections_in_frame = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    area = (x2 - x1) * (y2 - y1)
                    if area > 200 and cy > h_frame * 0.15:
                        detections_in_frame.append((cx, cy, x1, y1, x2, y2, area))

            if first_lap_start <= idx <= first_lap_start + int(fps * 2):
                detections_sorted = sorted(detections_in_frame, key=lambda d: d[0])
                for rank, det in enumerate(detections_sorted):
                    car_num = rank + 1
                    if car_num not in rider_tracks:
                        rider_tracks[car_num] = []
                    rider_tracks[car_num].append((idx, det[0], det[1]))

            if idx % sample_interval == 0 and idx >= first_lap_start:
                annotated = frame.copy()
                detections_sorted = sorted(detections_in_frame, key=lambda d: d[0])
                for rank, det in enumerate(detections_sorted):
                    cx, cy, x1, y1, x2, y2, area = det
                    car_label = f"#{rank+1}"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, car_label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                sample_annotated.append((idx / fps, annotated))

        idx += 1

    cap.release()

    print(f"  Tracked {len(rider_tracks)} riders in first lap window")
    for car_num, track in sorted(rider_tracks.items()):
        print(f"    Car #{car_num}: {len(track)} position samples")

    if sample_annotated:
        h, w = sample_annotated[0][1].shape[:2]
        cols = min(len(sample_annotated), 4)
        rows = min((len(sample_annotated) + cols - 1) // cols, 5)
        max_frames = rows * cols
        sheet = np.zeros((rows * (h + 30), cols * w, 3), dtype=np.uint8)
        for i, (t, annotated) in enumerate(sample_annotated[:max_frames]):
            r, c = divmod(i, cols)
            y_off = r * (h + 30)
            x_off = c * w
            sheet[y_off:y_off + h, x_off:x_off + w] = annotated
            cv2.putText(sheet, f"t={t:.1f}s", (x_off + 5, y_off + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        path = str(output_dir / "tracking_ordered.jpg")
        cv2.imwrite(path, sheet)
        print(f"  Saved: {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = Path(video_path).parent
    name = Path(video_path).stem

    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")

    test_yolo_detection(video_path, output_dir)
    test_camera_compensation(video_path, output_dir)
    test_tracking_with_ordering(video_path, output_dir)

    print(f"\n{'='*60}")
    print("All approaches complete. Check output images:")
    print(f"  {output_dir}/yolo_detection.jpg")
    print(f"  {output_dir}/camera_compensated.jpg")
    print(f"  {output_dir}/tracking_ordered.jpg")


if __name__ == "__main__":
    main()
