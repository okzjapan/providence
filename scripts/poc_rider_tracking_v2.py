"""POC v2: Improved rider detection for small objects in trial run videos.

Approach improvements:
1. YOLO with upscaled input (imgsz=1280) for small object detection
2. Phase correlation for robust camera motion estimation
3. Track-region detection: find dark objects on the light track surface

Usage:
    python3 scripts/poc_rider_tracking_v2.py /tmp/trial_video_poc/iizuka_20260404_R01_trial.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def get_sample_frames(video_path: str, times: list[float]) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if ret:
            frames.append((t, frame))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Approach 1: YOLO with high-res input
# ---------------------------------------------------------------------------

def test_yolo_highres(video_path: str, output_dir: Path) -> None:
    print("\n=== YOLO with imgsz=1280 (upscaled for small objects) ===")
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")

    times = [5.0, 8.0, 12.0, 18.0, 24.0, 28.0]
    frames = get_sample_frames(video_path, times)

    results_out = []
    for t, frame in frames:
        results = model(frame, verbose=False, conf=0.1, imgsz=1280)

        annotated = frame.copy()
        counts = {}
        for r in results:
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                counts[cls_name] = counts.get(cls_name, 0) + 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        results_out.append((t, annotated))
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  t={t:.1f}s: {summary or 'nothing detected'}")

    _save_grid(results_out, output_dir / "yolo_highres.jpg", cols=3)


# ---------------------------------------------------------------------------
# Approach 2: Phase correlation camera compensation
# ---------------------------------------------------------------------------

def test_phase_correlation(video_path: str, output_dir: Path) -> None:
    print("\n=== Phase Correlation Camera Compensation ===")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    sample_times = [5.0, 8.0, 12.0, 18.0, 24.0, 28.0]
    sample_indices = {int(t * fps) for t in sample_times}
    results_out = []

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float64)

        if idx in sample_indices:
            shift, _ = cv2.phaseCorrelate(prev_gray, gray_f)
            dx, dy = shift

            h, w = gray.shape
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            warped_prev = cv2.warpAffine(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), M, (w, h),
            )
            diff = cv2.absdiff(gray, warped_prev)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
            clean = cv2.dilate(clean, kernel, iterations=1)

            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            big = [c for c in contours if cv2.contourArea(c) > 100]

            annotated = frame.copy()
            for j, cnt in enumerate(big):
                x, y, cw, ch = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

            t = idx / fps
            results_out.append((t, annotated))
            print(f"  t={t:.1f}s: shift=({dx:.1f},{dy:.1f})px, {len(big)} objects")

        prev_gray = gray_f
        prev_frame = frame.copy()
        idx += 1

    cap.release()
    _save_grid(results_out, output_dir / "phase_corr.jpg", cols=3)


# ---------------------------------------------------------------------------
# Approach 3: Track-surface dark-object detection
# ---------------------------------------------------------------------------

def test_track_detection(video_path: str, output_dir: Path) -> None:
    print("\n=== Track-surface Dark Object Detection ===")

    times = [5.0, 8.0, 12.0, 18.0, 24.0, 28.0]
    frames = get_sample_frames(video_path, times)
    results_out = []

    for t, frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_frame, w_frame = gray.shape

        track_mask = np.zeros_like(gray)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        track_mask[(s_channel < 80) & (v_channel > 80) & (v_channel < 220)] = 255

        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_CLOSE, kernel_large)
        track_mask = cv2.morphologyEx(track_mask, cv2.MORPH_OPEN, kernel_large)

        track_mask[:int(h_frame * 0.15), :] = 0
        track_mask[int(h_frame * 0.85):, :] = 0

        dark_on_track = np.zeros_like(gray)
        dark_on_track[(gray < 100) & (track_mask > 0)] = 255

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_on_track = cv2.morphologyEx(dark_on_track, cv2.MORPH_OPEN, kernel_small)
        dark_on_track = cv2.morphologyEx(dark_on_track, cv2.MORPH_CLOSE, kernel_small)
        dark_on_track = cv2.dilate(dark_on_track, kernel_small, iterations=2)

        contours, _ = cv2.findContours(dark_on_track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        riders = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = ch / max(cw, 1)
            if 80 < area < 5000 and 0.3 < aspect < 4.0:
                riders.append((x, y, cw, ch, area))

        riders.sort(key=lambda r: r[0])

        annotated = frame.copy()
        for j, (x, y, cw, ch, area) in enumerate(riders):
            cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
            cv2.putText(annotated, f"#{j+1} a={area}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        h_half = h_frame // 2
        mask_vis = np.zeros((h_frame, w_frame, 3), dtype=np.uint8)
        mask_vis[:, :, 1] = track_mask
        mask_vis[:, :, 2] = dark_on_track

        combined = np.hstack([annotated, mask_vis])
        results_out.append((t, combined))
        print(f"  t={t:.1f}s: {len(riders)} rider candidates")

    _save_grid_wide(results_out, output_dir / "track_detection.jpg")


def _save_grid(frames: list[tuple[float, np.ndarray]], path: Path, cols: int = 3) -> None:
    if not frames:
        return
    h, w = frames[0][1].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * (h + 25), cols * w, 3), dtype=np.uint8)
    for i, (t, img) in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = r * (h + 25), c * w
        sheet[y:y + h, x:x + w] = img
        cv2.putText(sheet, f"t={t:.1f}s", (x + 5, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), sheet)
    print(f"  Saved: {path}")


def _save_grid_wide(frames: list[tuple[float, np.ndarray]], path: Path) -> None:
    if not frames:
        return
    h, w = frames[0][1].shape[:2]
    sheet = np.zeros((len(frames) * (h + 25), w, 3), dtype=np.uint8)
    for i, (t, img) in enumerate(frames):
        y = i * (h + 25)
        sheet[y:y + h, :w] = img
        cv2.putText(sheet, f"t={t:.1f}s", (5, y + h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), sheet)
    print(f"  Saved: {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = Path(video_path).parent

    test_yolo_highres(video_path, output_dir)
    test_phase_correlation(video_path, output_dir)
    test_track_detection(video_path, output_dir)


if __name__ == "__main__":
    main()
