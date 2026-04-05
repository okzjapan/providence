"""POC: Extract key frames from trial run videos for visual inspection.

Extracts frames at regular intervals and at the start of each lap,
creating a contact sheet for each video. Also attempts basic rider
detection using background subtraction.

Usage:
    python3 scripts/poc_frame_extract.py /tmp/trial_video_poc/*.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def extract_keyframes(video_path: str, interval_sec: float = 2.0) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    interval_frames = int(fps * interval_sec)
    frames: list[tuple[float, np.ndarray]] = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval_frames == 0:
            t = idx / fps
            frames.append((t, frame))
        idx += 1

    cap.release()
    print(f"  Extracted {len(frames)} frames from {duration:.1f}s video ({video_path})")
    return frames


def create_contact_sheet(
    frames: list[tuple[float, np.ndarray]],
    output_path: str,
    cols: int = 5,
    thumb_width: int = 400,
) -> None:
    if not frames:
        return

    h_orig, w_orig = frames[0][1].shape[:2]
    scale = thumb_width / w_orig
    thumb_h = int(h_orig * scale)

    rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * (thumb_h + 30), cols * thumb_width, 3), dtype=np.uint8)

    for i, (t, frame) in enumerate(frames):
        r, c = divmod(i, cols)
        thumb = cv2.resize(frame, (thumb_width, thumb_h))

        y_off = r * (thumb_h + 30)
        x_off = c * thumb_width
        sheet[y_off:y_off + thumb_h, x_off:x_off + thumb_width] = thumb

        label = f"t={t:.1f}s"
        cv2.putText(
            sheet, label,
            (x_off + 5, y_off + thumb_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    cv2.imwrite(output_path, sheet)
    print(f"  Contact sheet saved: {output_path}")


def detect_riders_simple(video_path: str, output_path: str) -> None:
    """Use background subtraction to detect moving objects (riders)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=int(fps * 2), varThreshold=50, detectShadows=False,
    )

    sample_times = [5.0, 10.0, 15.0, 20.0, 25.0]
    sample_frames = [int(t * fps) for t in sample_times]
    results: list[tuple[float, np.ndarray, np.ndarray, int]] = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = bg_sub.apply(frame)

        if idx in sample_frames:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            big_contours = [c for c in contours if cv2.contourArea(c) > 200]

            annotated = frame.copy()
            for j, cnt in enumerate(big_contours):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"#{j+1}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            t = idx / fps
            results.append((t, annotated, clean_mask, len(big_contours)))
        idx += 1

    cap.release()

    if not results:
        return

    h, w = results[0][1].shape[:2]
    sheet = np.zeros((len(results) * (h + 40), w * 2, 3), dtype=np.uint8)

    for i, (t, annotated, mask_img, count) in enumerate(results):
        y_off = i * (h + 40)
        sheet[y_off:y_off + h, 0:w] = annotated
        mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        sheet[y_off:y_off + h, w:w * 2] = mask_bgr

        label = f"t={t:.1f}s | Detected objects: {count}"
        cv2.putText(
            sheet, label,
            (10, y_off + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

    cv2.imwrite(output_path, sheet)
    print(f"  Rider detection sheet saved: {output_path}")
    for t, _, _, count in results:
        print(f"    t={t:.1f}s: {count} objects detected")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path> [video_path ...]")
        sys.exit(1)

    for video_path in sys.argv[1:]:
        p = Path(video_path)
        print(f"\n{'='*60}")
        print(f"Processing: {p.name}")
        print(f"{'='*60}")

        frames = extract_keyframes(video_path, interval_sec=2.0)

        contact_path = str(p.parent / f"{p.stem}_contact_sheet.jpg")
        create_contact_sheet(frames, contact_path)

        detect_path = str(p.parent / f"{p.stem}_rider_detection.jpg")
        detect_riders_simple(video_path, detect_path)


if __name__ == "__main__":
    main()
