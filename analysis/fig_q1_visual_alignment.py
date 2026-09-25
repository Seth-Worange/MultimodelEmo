"""用真实视频帧、面部关键点和音轨生成问题一对齐示意图。"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from analysis.style import setup_style
from scripts.features_q1 import FACE_POINTS, SAMPLE_RATE, decode_audio


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "-3g5yACwYnA_13"
VIDEO = next((ROOT / "data" / "附件1-数据集原始多模态样本").rglob("-3g5yACwYnA/13.mp4"))
FEATURES = ROOT / "outputs" / "features_q1_face_pose" / f"{SAMPLE}.npz"
FIGURES = ROOT.parent / "thesis" / "figures"


def video_frame(second: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取视频：{VIDEO}")
    capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    okay, frame = capture.read()
    capture.release()
    if not okay:
        raise RuntimeError(f"无法读取{second:.2f}秒的帧")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detected_points(frame: np.ndarray) -> np.ndarray:
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=(ROOT / "task" / "face_landmarker.task").read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
    if not result.face_landmarks:
        raise RuntimeError("示例视频帧未检测到人脸，请更换采样时刻")
    h, w = frame.shape[:2]
    return np.array([(result.face_landmarks[0][i].x * w,
                      result.face_landmarks[0][i].y * h) for i in FACE_POINTS])


def main() -> None:
    setup_style(base_fontsize=8.5)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    sample = np.load(FEATURES, allow_pickle=False)
    words = sample["words"].astype(str)
    times = sample["times"].astype(float)
    focus = 5  # solutions
    second = float(times[focus].mean())
    frame = video_frame(second)
    points = detected_points(frame)
    audio = decode_audio(VIDEO)
    stride = max(1, len(audio) // 5000)
    t = np.arange(0, len(audio), stride) / SAMPLE_RATE
    wave = audio[::stride]

    fig = plt.figure(figsize=(6.42, 3.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.65], height_ratios=[1, .92])
    face = fig.add_subplot(grid[:, 0])
    waveform = fig.add_subplot(grid[0, 1])
    alignment = fig.add_subplot(grid[1, 1], sharex=waveform)

    face.imshow(frame)
    face.scatter(points[:, 0], points[:, 1], s=16, facecolors="#E69F00",
                 edgecolors="#1D3D48", linewidths=.55)
    for a, b in [(5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                 (13, 15), (15, 14), (0, 1), (1, 2), (2, 3)]:
        face.plot(points[[a, b], 0], points[[a, b], 1], color="#00A88F", linewidth=1.1)
    face.set_axis_off()
    face.set_title(f"a 真实视频帧与16个面部锚点  {second:.2f}s", loc="left", fontsize=8.5)

    waveform.plot(t, wave, color="#4D788E", linewidth=.42, rasterized=True)
    waveform.axvspan(*times[focus], color="#E69F00", alpha=.27)
    waveform.axvline(second, color="#C34D39", linewidth=.8, linestyle="--")
    waveform.set_ylabel("振幅")
    waveform.set_title("b 原视频音轨波形与所选词时段", loc="left", fontsize=8.5)
    waveform.grid(axis="x", color="#E5EBEE", linewidth=.7)
    waveform.tick_params(labelbottom=False)

    for index, (word, (start, end)) in enumerate(zip(words, times)):
        color = "#E69F00" if index == focus else ("#0072B2" if index % 2 == 0 else "#5C9AB8")
        alignment.barh(0, end - start, left=start, height=.58, color=color,
                       edgecolor="white", linewidth=.8)
        alignment.text((start + end) / 2, .39 + .12 * (index % 2), word,
                       rotation=60, ha="left", va="bottom", fontsize=6.5, color="#273E49")
    alignment.set_ylim(-.65, 1.4)
    alignment.set_yticks([])
    alignment.set_xlim(max(0, times[0, 0] - .18), times[-1, 1] + .25)
    alignment.set_xlabel("视频时间（秒）")
    alignment.set_title("c WhisperX词级区间；声画帧按区间池化", loc="left", fontsize=8.5)
    alignment.spines[["top", "right", "left"]].set_visible(False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(FIGURES / f"fig_q1_face_wave_alignment.{suffix}", **kwargs)
    plt.close(fig)
    print(f"sample={SAMPLE} frame={second:.3f}s words={len(words)} face_points={len(points)}")


if __name__ == "__main__":
    main()
