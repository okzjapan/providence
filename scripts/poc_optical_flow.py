"""POC: Optical flow analysis of autorace trial run video.

Usage:
    python3 scripts/poc_optical_flow.py /tmp/trial_video_poc/iizuka_20260404_R01_trial.mp4

Outputs:
    - Magnitude time-series plot (saved as PNG)
    - Flow heatmap of a sample frame (saved as PNG)
    - Summary statistics printed to stdout
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_optical_flow_stats(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.1f}s")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    magnitudes_mean: list[float] = []
    magnitudes_max: list[float] = []
    magnitudes_std: list[float] = []
    sample_flow = None
    sample_frame_idx = total_frames // 2

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        magnitudes_mean.append(float(np.mean(mag)))
        magnitudes_max.append(float(np.max(mag)))
        magnitudes_std.append(float(np.std(mag)))

        if frame_idx == sample_frame_idx:
            sample_flow = flow.copy()

        prev_gray = gray
        frame_idx += 1

    cap.release()

    mag_arr = np.array(magnitudes_mean)
    print(f"\nOptical Flow Statistics:")
    print(f"  Mean magnitude (avg): {mag_arr.mean():.4f}")
    print(f"  Mean magnitude (std): {mag_arr.std():.4f}")
    print(f"  Mean magnitude (min): {mag_arr.min():.4f}")
    print(f"  Mean magnitude (max): {mag_arr.max():.4f}")

    return {
        "fps": fps,
        "total_frames": total_frames,
        "magnitudes_mean": magnitudes_mean,
        "magnitudes_max": magnitudes_max,
        "magnitudes_std": magnitudes_std,
        "sample_flow": sample_flow,
        "width": width,
        "height": height,
    }


def plot_magnitude_timeseries(stats: dict, output_path: str) -> None:
    fps = stats["fps"]
    mag_mean = np.array(stats["magnitudes_mean"])
    mag_max = np.array(stats["magnitudes_max"])
    times = np.arange(len(mag_mean)) / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(times, mag_mean, linewidth=0.8, color="steelblue")
    ax1.set_ylabel("Mean Magnitude")
    ax1.set_title("Optical Flow - Mean Magnitude over Time")
    ax1.grid(True, alpha=0.3)

    window = int(fps)
    if len(mag_mean) > window:
        smoothed = np.convolve(mag_mean, np.ones(window) / window, mode="valid")
        ax1.plot(
            times[window - 1:], smoothed,
            linewidth=2, color="red", label=f"Moving avg ({window} frames)",
        )
        ax1.legend()

    ax2.plot(times, mag_max, linewidth=0.8, color="coral")
    ax2.set_ylabel("Max Magnitude")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Optical Flow - Max Magnitude over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved time-series plot: {output_path}")


def plot_flow_heatmap(stats: dict, output_path: str) -> None:
    flow = stats["sample_flow"]
    if flow is None:
        print("No sample flow available for heatmap")
        return

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((stats["height"], stats["width"], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(mag, cmap="hot")
    ax1.set_title("Flow Magnitude (mid-video frame)")
    ax1.axis("off")
    cbar = fig.colorbar(ax1.images[0], ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Pixel displacement")

    ax2.imshow(rgb)
    ax2.set_title("Flow Direction + Magnitude (HSV)")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved flow heatmap: {output_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = Path(video_path).parent

    stats = compute_optical_flow_stats(video_path)
    plot_magnitude_timeseries(stats, str(output_dir / "optical_flow_timeseries.png"))
    plot_flow_heatmap(stats, str(output_dir / "optical_flow_heatmap.png"))

    print("\n--- Summary ---")
    print(f"Frames processed: {len(stats['magnitudes_mean'])}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
