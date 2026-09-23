"""Publication-oriented diagnostic visualization for one processed sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .io_utils import read_json, write_csv


DETAIL_FIELDS = (
    "sample_id", "word_index", "original_word", "start_s", "end_s",
    "alignment_mask", "alignment_failure_reason", "text_mask", "audio_mask",
    "visual_mask", "audio_frame_indices", "audio_frame_count",
    "visual_frame_indices", "visual_original_frame_indices", "visual_frame_count",
    "visual_valid_face_count", "visual_status",
)


def _decode_frames(video_path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    import av

    output: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in indices:
                output[index] = frame.to_ndarray(format="rgb24")
            if len(output) == len(indices):
                break
    return output


def visualize_sample(
    video_path: Path,
    sample_dir: Path,
    output_path: Path,
    *,
    max_keyframes: int = 4,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization") from exc
    fused_path = sample_dir / "fused_features.npz"
    if not fused_path.is_file():
        raise FileNotFoundError(fused_path)
    metadata = read_json(sample_dir / "fused_metadata.json")
    audio_metadata = read_json(sample_dir / "audio_features_metadata.json")
    with np.load(fused_path, allow_pickle=False) as arrays:
        sample_id = str(arrays["sample_id"].item())
        words = arrays["original_words"].astype(str)
        starts = arrays["word_start_s"]
        ends = arrays["word_end_s"]
        alignment = arrays["alignment_mask"].astype(bool)
        text_mask = arrays["text_mask"].astype(bool)
        audio_mask = arrays["audio_mask"].astype(bool)
        visual_mask = arrays["visual_mask"].astype(bool)
        audio_times = arrays["audio_frame_times_s"]
        audio_frames = arrays["audio_frame_features"]
        visual_times = arrays["visual_frame_times_s"]
        visual_detection = arrays["visual_detection_mask"].astype(bool)
        visual_original = arrays["visual_original_frame_indices"].astype(int)
        audio_counts = arrays["audio_word_frame_counts"].astype(int)
        visual_counts = arrays["visual_word_frame_counts"].astype(int)
        visual_valid_counts = arrays["visual_word_valid_face_counts"].astype(int)

    audio_indices = metadata["audio_word_frame_indices"]
    visual_indices = metadata["visual_word_frame_indices"]
    visual_original_by_word = metadata["visual_word_original_frame_indices"]
    visual_status = metadata["visual_word_status"]
    alignment_reasons = metadata["alignment_failure_reasons"]
    rows = []
    for index, word in enumerate(words):
        rows.append({
            "sample_id": sample_id,
            "word_index": index,
            "original_word": word,
            "start_s": starts[index],
            "end_s": ends[index],
            "alignment_mask": int(alignment[index]),
            "alignment_failure_reason": alignment_reasons[index],
            "text_mask": int(text_mask[index]),
            "audio_mask": int(audio_mask[index]),
            "visual_mask": int(visual_mask[index]),
            "audio_frame_indices": ";".join(map(str, audio_indices[index])),
            "audio_frame_count": audio_counts[index],
            "visual_frame_indices": ";".join(map(str, visual_indices[index])),
            "visual_original_frame_indices": ";".join(map(str, visual_original_by_word[index])),
            "visual_frame_count": visual_counts[index],
            "visual_valid_face_count": visual_valid_counts[index],
            "visual_status": visual_status[index],
        })
    detail_path = output_path.with_name(output_path.stem + "_alignment_detail.csv")
    write_csv(detail_path, rows, DETAIL_FIELDS)

    chosen: list[tuple[int, int]] = []
    for word_index, frame_list in enumerate(visual_original_by_word):
        if visual_mask[word_index] and frame_list:
            chosen.append((word_index, int(frame_list[0])))
        if len(chosen) >= max_keyframes:
            break
    decoded = _decode_frames(video_path, {frame_index for _, frame_index in chosen}) if chosen else {}

    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(5, max(1, max_keyframes), height_ratios=[1.6, 1.4, 0.8, 0.8, 2.0])
    ax_words = figure.add_subplot(grid[0, :])
    ax_audio = figure.add_subplot(grid[1, :], sharex=ax_words)
    ax_video = figure.add_subplot(grid[2, :], sharex=ax_words)
    ax_masks = figure.add_subplot(grid[3, :])

    for index, (word, start, end, valid) in enumerate(zip(words, starts, ends, alignment)):
        if valid:
            ax_words.broken_barh([(start, max(0.001, end - start))], (0.1, 0.8), facecolors="#4C78A8", alpha=0.65)
            ax_words.text((start + end) / 2.0, 0.5, word, rotation=35, ha="center", va="center", fontsize=7)
        else:
            ax_words.text(index / max(1, len(words) - 1), 0.92, f"{index}:{word}", transform=ax_words.transAxes,
                          color="#D62728", fontsize=6, ha="center")
    ax_words.set_ylim(0, 1.05)
    ax_words.set_yticks([])
    ax_words.set_ylabel("Words")
    ax_words.set_title(
        f"{sample_id}: automatic MFA alignment diagnostic (not human-confirmed ground truth)"
    )

    feature_names = audio_metadata["feature_names"]
    rms_index = feature_names.index("rms")
    if len(audio_times):
        ax_audio.plot(audio_times, audio_frames[:, rms_index], color="#F58518", linewidth=1.1)
    ax_audio.set_ylabel("RMS energy")
    ax_audio.grid(alpha=0.2)

    if len(visual_times):
        colors = np.where(visual_detection, "#54A24B", "#E45756")
        ax_video.scatter(visual_times, np.zeros_like(visual_times), c=colors, marker="|", s=130)
    ax_video.set_yticks([])
    ax_video.set_ylabel("Video\nframes")
    ax_video.set_xlabel("Unified sample-relative time (s)")
    ax_video.text(0.01, 0.8, "green=face detected; red=detection failed", transform=ax_video.transAxes, fontsize=8)

    mask_matrix = np.vstack([text_mask, audio_mask, visual_mask]).astype(float)
    ax_masks.imshow(mask_matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax_masks.set_yticks([0, 1, 2], ["text", "audio", "visual"])
    ax_masks.set_xlabel("Original word index")
    ax_masks.set_title("Word-level validity masks (visual identity remains separately unverified)")

    for column in range(max_keyframes):
        axis = figure.add_subplot(grid[4, column])
        if column < len(chosen):
            word_index, frame_index = chosen[column]
            image = decoded.get(frame_index)
            if image is not None:
                axis.imshow(image)
            axis.set_title(f"word {word_index}: {words[word_index]}\noriginal frame {frame_index}", fontsize=9)
        else:
            axis.text(0.5, 0.5, "No verified in-interval\nface frame", ha="center", va="center")
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path, detail_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure, detail = visualize_sample(args.video, args.sample_dir, args.output)
    print(f"wrote {figure} and {detail}")


if __name__ == "__main__":
    main()
