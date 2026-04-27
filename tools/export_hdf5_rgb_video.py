#!/usr/bin/env python3
"""
Export RGB frames from EgoVLA simulator HDF5 episodes and write preview videos.

Examples
--------
Export a single episode:
    python tools/export_hdf5_rgb_video.py \
        --input data/EgoVLA_SIM/Open-Laptop/episode_0.hdf5

Export selected episodes under a task directory:
    python tools/export_hdf5_rgb_video.py \
        --input data/EgoVLA_SIM/Open-Laptop \
        --episodes 0 5 9 \
        --fps 30

Export the same episode ids across every task under EgoVLA_SIM:
    python tools/export_hdf5_rgb_video.py \
        --input data/EgoVLA_SIM \
        --episodes 0 1
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: h5py. Install it in your EgoVLA environment first, "
        "for example with `pip install h5py`."
    ) from exc

import numpy as np


DEFAULT_IMAGE_KEYS = (
    "observations/images/main",
    "observation/images/main",
    "obs/images/main",
    "images/main",
    "observations/rgb_obs",
    "rgb_obs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RGB frames from HDF5 episodes into videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to one .hdf5 file, a single task directory, or the dataset root "
            "that contains multiple task subdirectories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory used to store exported videos and optional frame folders.",
    )
    parser.add_argument(
        "--pattern",
        default="episode_*.hdf5",
        help="Glob pattern used when --input points to a directory.",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Only export specific episode ids, such as `--episodes 0 5 9`. "
            "When --input is the dataset root, the filter is applied across all tasks "
            "under that root."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only export the first N matched HDF5 files.",
    )
    parser.add_argument(
        "--image-key",
        default=None,
        help="Explicit HDF5 dataset key. If omitted, the script auto-detects it.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Inclusive starting frame index.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Exclusive ending frame index.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every N-th frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of exported frames after striding.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Optional output frame width.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Optional output frame height.",
    )
    parser.add_argument(
        "--codec",
        choices=("mpeg4", "libx264"),
        default="mpeg4",
        help="FFmpeg video codec.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="FFmpeg executable.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save decoded RGB frames to a sibling frame directory.",
    )
    parser.add_argument(
        "--frame-format",
        choices=("ppm", "png", "jpg"),
        default="ppm",
        help="Image format used when --save-frames is enabled.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Only export frames, do not write a video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print progress every N frames.",
    )
    args = parser.parse_args()

    if args.stride <= 0:
        parser.error("--stride must be a positive integer.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    if (args.resize_width is None) ^ (args.resize_height is None):
        parser.error("Please set both --resize-width and --resize-height together.")
    if args.no_video and not args.save_frames:
        parser.error("Nothing to do: enable --save-frames or remove --no-video.")
    return args


def natural_hdf5_key(path: Path) -> Tuple[int, str]:
    return (extract_episode_id(path), path.name)


def extract_episode_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def relative_parent_key(path: Path, root: Path) -> str:
    try:
        return path.parent.relative_to(root).as_posix()
    except ValueError:
        return path.parent.as_posix()


def collect_hdf5_files(input_path: Path, pattern: str, episodes: List[int] | None, limit: int | None) -> List[Path]:
    if input_path.is_file():
        if episodes is not None:
            raise ValueError("--episodes can only be used when --input points to a directory.")
        if input_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Expected an HDF5 file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    all_files = sorted(
        [path for path in input_path.rglob(pattern) if path.is_file()],
        key=lambda path: (relative_parent_key(path, input_path), natural_hdf5_key(path)),
    )

    if episodes:
        requested_episode_ids = set(episodes)
        files = [
            path for path in all_files
            if extract_episode_id(path) in requested_episode_ids
        ]
    else:
        files = all_files

    if limit is not None:
        files = files[:limit]

    if not files:
        episode_note = ""
        if episodes:
            episode_note = f" and episodes {sorted(set(episodes))}"
        raise FileNotFoundError(
            f"No HDF5 files found under {input_path} with pattern {pattern!r}{episode_note}."
        )
    return files


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "preview_videos"
    return input_path / "preview_videos"


def is_image_like_dataset(dataset: h5py.Dataset) -> bool:
    if dataset.ndim not in (3, 4):
        return False
    shape = dataset.shape
    if dataset.ndim == 4:
        return shape[-1] in (1, 3, 4) or shape[1] in (1, 3, 4)
    return shape[-1] in (1, 3, 4) or shape[0] in (1, 3, 4)


def dataset_priority(name: str, dataset: h5py.Dataset) -> int:
    score = 0
    if name in DEFAULT_IMAGE_KEYS:
        score += 1000 - DEFAULT_IMAGE_KEYS.index(name) * 10
    lower_name = name.lower()
    if "image" in lower_name:
        score += 100
    if "rgb" in lower_name:
        score += 60
    if lower_name.endswith("/main"):
        score += 20
    if dataset.ndim == 4:
        score += 20
    return score


def list_image_datasets(h5_file: h5py.File) -> List[Tuple[str, h5py.Dataset]]:
    candidates: List[Tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and is_image_like_dataset(obj):
            candidates.append((name, obj))

    h5_file.visititems(visitor)
    candidates.sort(key=lambda item: dataset_priority(item[0], item[1]), reverse=True)
    return candidates


def resolve_image_dataset(h5_file: h5py.File, explicit_key: str | None) -> Tuple[str, h5py.Dataset]:
    if explicit_key is not None:
        if explicit_key not in h5_file:
            available = "\n".join(
                f"  - {name} shape={dataset.shape} dtype={dataset.dtype}"
                for name, dataset in list_image_datasets(h5_file)
            )
            raise KeyError(
                f"Dataset key {explicit_key!r} not found.\nPossible image datasets:\n{available}"
            )
        dataset = h5_file[explicit_key]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"HDF5 object at {explicit_key!r} is not a dataset.")
        return explicit_key, dataset

    for key in DEFAULT_IMAGE_KEYS:
        if key in h5_file:
            dataset = h5_file[key]
            if isinstance(dataset, h5py.Dataset):
                return key, dataset

    candidates = list_image_datasets(h5_file)
    if not candidates:
        raise KeyError(
            "Could not auto-detect an image dataset in this HDF5 file. "
            "Try passing --image-key explicitly."
        )
    return candidates[0]


def frame_count(dataset: h5py.Dataset) -> int:
    if dataset.ndim == 4:
        return int(dataset.shape[0])
    if dataset.ndim == 3:
        return 1
    raise ValueError(f"Unsupported image dataset rank: {dataset.ndim}")


def frame_indices(total_frames: int, start: int, end: int | None, stride: int, max_frames: int | None) -> range:
    start = max(0, start)
    end = total_frames if end is None else min(end, total_frames)
    if end <= start:
        return range(0, 0)
    indices = range(start, end, stride)
    if max_frames is None:
        return indices
    limited_end = start + min(len(indices), max_frames) * stride
    return range(start, limited_end, stride)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)

    if frame.ndim == 2:
        frame = frame[:, :, None]
    elif frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]

    if frame.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image frame, got shape {frame.shape}")

    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    elif frame.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got frame shape {frame.shape}")

    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(np.nanmax(frame)) if frame.size > 0 else 0.0
        if max_value <= 1.0 + 1e-6:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

    return np.ascontiguousarray(frame)


def resize_frame_nearest(frame: np.ndarray, target_width: int | None, target_height: int | None) -> np.ndarray:
    if target_width is None or target_height is None:
        return frame

    src_height, src_width = frame.shape[:2]
    if src_width == target_width and src_height == target_height:
        return frame

    y_idx = np.minimum(
        (np.arange(target_height, dtype=np.float32) * (src_height / target_height)).astype(np.int64),
        src_height - 1,
    )
    x_idx = np.minimum(
        (np.arange(target_width, dtype=np.float32) * (src_width / target_width)).astype(np.int64),
        src_width - 1,
    )
    resized = frame[y_idx][:, x_idx]
    return np.ascontiguousarray(resized)


def write_ppm(frame: np.ndarray, output_path: Path) -> None:
    height, width, _ = frame.shape
    with output_path.open("wb") as file_obj:
        file_obj.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        file_obj.write(frame.tobytes())


def save_frame_image(frame: np.ndarray, output_path: Path, frame_format: str) -> None:
    if frame_format == "ppm":
        write_ppm(frame, output_path)
        return

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"Saving {frame_format} frames requires Pillow. "
            f"Install it with `pip install pillow`, or use `--frame-format ppm`."
        ) from exc

    image = Image.fromarray(frame)
    if frame_format == "jpg":
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def ffmpeg_command(
    ffmpeg_bin: str,
    width: int,
    height: int,
    fps: float,
    codec: str,
    output_path: Path,
    overwrite: bool,
) -> List[str]:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
    ]

    if codec == "libx264":
        command += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        command += ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"]

    command.append(str(output_path))
    return command


def output_stem(hdf5_path: Path) -> str:
    parent_name = hdf5_path.parent.name
    return f"{parent_name}__{hdf5_path.stem}"


def read_frame(dataset: h5py.Dataset, index: int) -> np.ndarray:
    if dataset.ndim == 4:
        return dataset[index]
    return dataset[()]


def export_episode(hdf5_path: Path, output_dir: Path, args: argparse.Namespace) -> Tuple[Path | None, Path | None]:
    frame_dir: Path | None = None
    video_path: Path | None = None

    with h5py.File(hdf5_path, "r") as h5_file:
        image_key, image_dataset = resolve_image_dataset(h5_file, args.image_key)
        total_frames = frame_count(image_dataset)
        indices = frame_indices(
            total_frames=total_frames,
            start=args.start_frame,
            end=args.end_frame,
            stride=args.stride,
            max_frames=args.max_frames,
        )

        if len(indices) == 0:
            raise ValueError(
                f"No frames selected from {hdf5_path}. "
                f"Requested range start={args.start_frame}, end={args.end_frame}, stride={args.stride}."
            )

        stem = output_stem(hdf5_path)
        write_video = not args.no_video
        write_frames = args.save_frames

        if not args.no_video:
            video_path = output_dir / f"{stem}.mp4"
            if video_path.exists() and not args.overwrite:
                print(f"[skip] Video already exists: {video_path}")
                write_video = False

        if args.save_frames:
            frame_dir = output_dir / f"{stem}_frames"
            if frame_dir.exists() and args.overwrite:
                shutil.rmtree(frame_dir)
            elif frame_dir.exists() and not args.overwrite:
                print(f"[skip] Frame directory already exists: {frame_dir}")
                write_frames = False

            if write_frames:
                frame_dir.mkdir(parents=True, exist_ok=True)

        if not write_video and not write_frames:
            return video_path, frame_dir

        print(
            f"[info] {hdf5_path.name}: dataset={image_key!r}, shape={image_dataset.shape}, "
            f"dtype={image_dataset.dtype}, selected_frames={len(indices)}"
        )

        iterator = iter(indices)
        first_index = next(iterator)
        first_frame = normalize_frame(read_frame(image_dataset, first_index))
        first_frame = resize_frame_nearest(first_frame, args.resize_width, args.resize_height)

        ffmpeg_process: subprocess.Popen[bytes] | None = None

        if write_video:
            if shutil.which(args.ffmpeg_bin) is None:
                raise RuntimeError(
                    f"Could not find FFmpeg executable {args.ffmpeg_bin!r}. "
                    "Install FFmpeg or pass --ffmpeg-bin with the correct path."
                )

            height, width = first_frame.shape[:2]
            command = ffmpeg_command(
                ffmpeg_bin=args.ffmpeg_bin,
                width=width,
                height=height,
                fps=args.fps,
                codec=args.codec,
                output_path=video_path,
                overwrite=args.overwrite,
            )
            ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        try:
            exported = 0
            all_indices: Iterable[int] = [first_index]
            all_indices = list(all_indices) + list(iterator)

            for local_idx, frame_index in enumerate(all_indices):
                frame = first_frame if local_idx == 0 else normalize_frame(read_frame(image_dataset, frame_index))
                if local_idx != 0:
                    frame = resize_frame_nearest(frame, args.resize_width, args.resize_height)

                if write_frames and frame_dir is not None:
                    frame_path = frame_dir / f"frame_{frame_index:06d}.{args.frame_format}"
                    save_frame_image(frame, frame_path, args.frame_format)

                if ffmpeg_process is not None and ffmpeg_process.stdin is not None:
                    ffmpeg_process.stdin.write(frame.tobytes())

                exported += 1
                if args.log_every > 0 and (exported % args.log_every == 0 or exported == len(indices)):
                    print(f"[info] {hdf5_path.name}: wrote {exported}/{len(indices)} frames")

            if ffmpeg_process is not None and ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.close()
                ffmpeg_stderr = ffmpeg_process.stderr.read() if ffmpeg_process.stderr is not None else b""
                return_code = ffmpeg_process.wait()
                if return_code != 0:
                    raise RuntimeError(
                        "FFmpeg failed while encoding "
                        f"{video_path}:\n{ffmpeg_stderr.decode('utf-8', errors='replace')}"
                    )
        finally:
            if ffmpeg_process is not None and ffmpeg_process.poll() is None:
                ffmpeg_process.kill()

    return video_path, frame_dir


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        hdf5_files = collect_hdf5_files(
            input_path=input_path,
            pattern=args.pattern,
            episodes=args.episodes,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    processed_count = 0
    for file_index, hdf5_path in enumerate(hdf5_files, start=1):
        print(f"[run] ({file_index}/{len(hdf5_files)}) exporting {hdf5_path}")
        try:
            video_path, frame_dir = export_episode(hdf5_path, output_dir, args)
        except Exception as exc:
            print(f"[error] Failed to export {hdf5_path}: {exc}", file=sys.stderr)
            return 1

        processed_count += 1
        if video_path is not None:
            print(f"[done] video:  {video_path}")
        if frame_dir is not None:
            print(f"[done] frames: {frame_dir}")

    print(f"[summary] Processed {processed_count} episode(s) into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
