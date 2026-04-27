#!/usr/bin/env python3
"""
Analyze within-task episode similarity for EgoVLA simulator hand/action data.

The main metric follows the post-training supervision signal more closely than the
raw HDF5 action: for each frame sample, it reconstructs the future hand / EE label
sequence used by training (3D EE translation, the supervised MANO DOF dimensions,
and EE rotation for both hands over future horizons), then compares episode-level
trajectories across episodes from the same task.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


MANO_PER_DIM_MIN = np.concatenate(
    [
        np.array([-1.0, 1.5, -2.0, -3.0, -1.5, -1.0], dtype=np.float32),
        -4.0 * np.ones(9, dtype=np.float32),
    ]
)
MANO_PER_DIM_MAX = np.concatenate(
    [
        np.array([2.2, 3.5, 1.0, 0.5, 4.0, 5.0], dtype=np.float32),
        4.0 * np.ones(9, dtype=np.float32),
    ]
)
MANO_RANGE = MANO_PER_DIM_MAX - MANO_PER_DIM_MIN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze within-task episode similarity on EgoVLA hand/action data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default="/root/gpufree-data/EgoVLA_Release/data/EgoVLA_SIM_Processed/hand_FIXED_SET_MIX_train_parquets",
        help=(
            "Path to the processed hand parquet directory used for robot training. "
            "Passing the HF_hand directory also works if the sibling parquet directory exists."
        ),
    )
    parser.add_argument(
        "--future-steps",
        type=int,
        default=30,
        help="Number of future supervision steps used by training.",
    )
    parser.add_argument(
        "--future-index",
        type=int,
        default=1,
        help="Future stride used by training label construction.",
    )
    parser.add_argument(
        "--hand-loss-dim",
        type=int,
        default=6,
        help="Number of MANO DOF dimensions per hand that enter the training loss.",
    )
    parser.add_argument(
        "--resample-steps",
        type=int,
        default=32,
        help="Number of normalized progress points used for episode-level signatures.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task filter, such as `--tasks Open-Laptop Pour-Balls`.",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Optional cap per task, useful for quick debugging.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full analysis as JSON.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress line every N episodes while building episode summaries.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path).expanduser().resolve()
    if path.is_dir() and any(path.glob("*.parquet")):
        return path

    if path.is_dir() and path.name.startswith("HF_hand_"):
        candidate = path.parent / f"{path.name.replace('HF_hand_', 'hand_')}_parquets"
        if candidate.is_dir() and any(candidate.glob("*.parquet")):
            return candidate

    raise FileNotFoundError(
        f"Could not find a parquet dataset at {path}. "
        "Pass the hand_*_parquets directory or the matching HF_hand directory."
    )


def row_value(row: object, key: str) -> object:
    if isinstance(row, dict):
        return row[key]
    return getattr(row, key)


def norm_hand_dof_np(hand_dof: np.ndarray) -> np.ndarray:
    return (hand_dof - MANO_PER_DIM_MIN) / MANO_RANGE


def reshape_feature(value: Iterable[float], dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1, dim)


def build_current_vector(row: Dict[str, object], hand_loss_dim: int) -> np.ndarray:
    left_trans = np.asarray(row_value(row, "current_left_mano_trans"), dtype=np.float32).reshape(3)
    right_trans = np.asarray(row_value(row, "current_right_mano_trans"), dtype=np.float32).reshape(3)

    left_hand = norm_hand_dof_np(
        np.asarray(row_value(row, "current_left_mano_parameters"), dtype=np.float32).reshape(15)
    )[:hand_loss_dim]
    right_hand = norm_hand_dof_np(
        np.asarray(row_value(row, "current_right_mano_parameters"), dtype=np.float32).reshape(15)
    )[:hand_loss_dim]

    left_rot = np.asarray(row_value(row, "current_left_mano_rot"), dtype=np.float32).reshape(3)
    right_rot = np.asarray(row_value(row, "current_right_mano_rot"), dtype=np.float32).reshape(3)

    return np.concatenate(
        [left_trans, right_trans, left_hand, right_hand, left_rot, right_rot],
        dtype=np.float32,
    )


def build_future_label_vector(
    row: Dict[str, object],
    future_steps: int,
    future_index: int,
    hand_loss_dim: int,
) -> np.ndarray:
    left_trans = reshape_feature(row_value(row, "future_left_mano_trans"), 3)
    right_trans = reshape_feature(row_value(row, "future_right_mano_trans"), 3)
    left_rot = reshape_feature(row_value(row, "future_left_mano_rot"), 3)
    right_rot = reshape_feature(row_value(row, "future_right_mano_rot"), 3)
    left_hand = norm_hand_dof_np(reshape_feature(row_value(row, "future_left_mano_parameters"), 15))
    right_hand = norm_hand_dof_np(reshape_feature(row_value(row, "future_right_mano_parameters"), 15))

    max_len = left_trans.shape[0]
    if max_len == 0:
        raise ValueError(f"Found empty future trajectory in sample {row_value(row, 'seq_name')}")

    target_indices = np.minimum(
        np.arange(1, future_steps + 1, dtype=np.int64) * future_index,
        max_len - 1,
    )

    ee_3d = np.concatenate([left_trans[target_indices], right_trans[target_indices]], axis=1)
    hand = np.concatenate(
        [
            left_hand[target_indices, :hand_loss_dim],
            right_hand[target_indices, :hand_loss_dim],
        ],
        axis=1,
    )
    ee_rot = np.concatenate([left_rot[target_indices], right_rot[target_indices]], axis=1)

    return np.concatenate([ee_3d, hand, ee_rot], axis=1).reshape(-1).astype(np.float32)


def build_action_vector(row: Dict[str, object]) -> np.ndarray:
    return np.asarray(row_value(row, "action"), dtype=np.float32).reshape(-1)


def resample_sequence(sequence: np.ndarray, target_steps: int) -> np.ndarray:
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {sequence.shape}")

    num_steps, feat_dim = sequence.shape
    if num_steps == target_steps:
        return sequence.astype(np.float32, copy=False)
    if num_steps == 1:
        return np.repeat(sequence.astype(np.float32), target_steps, axis=0)

    positions = np.linspace(0.0, num_steps - 1, target_steps, dtype=np.float32)
    left_idx = np.floor(positions).astype(np.int64)
    right_idx = np.ceil(positions).astype(np.int64)
    alpha = (positions - left_idx).reshape(-1, 1).astype(np.float32)

    left = sequence[left_idx]
    right = sequence[right_idx]
    return (1.0 - alpha) * left + alpha * right


def cosine_similarity_matrix(signatures: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized = signatures / norms
    return normalized @ normalized.T


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.empty(0, dtype=np.float32)
    tri = np.triu_indices(matrix.shape[0], k=1)
    return matrix[tri]


def summarize_pairwise_cosine(signatures: np.ndarray) -> Dict[str, float]:
    cosine_matrix = cosine_similarity_matrix(signatures)
    pair_values = upper_triangle_values(cosine_matrix)

    if pair_values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(pair_values.mean()),
        "median": float(np.median(pair_values)),
        "min": float(pair_values.min()),
        "max": float(pair_values.max()),
    }


def summarize_prototype(signatures: np.ndarray) -> Dict[str, float]:
    prototype = signatures.mean(axis=0, keepdims=True)
    cosine = cosine_similarity_matrix(np.concatenate([signatures, prototype], axis=0))
    prototype_cos = cosine[:-1, -1]
    return {
        "mean_cosine": float(prototype_cos.mean()),
        "min_cosine": float(prototype_cos.min()),
        "max_cosine": float(prototype_cos.max()),
    }


def summarize_rank(signatures: np.ndarray) -> Dict[str, float]:
    centered = signatures - signatures.mean(axis=0, keepdims=True)
    if centered.shape[0] <= 1 or np.allclose(centered, 0.0):
        return {"pcs90": 0, "effective_rank": 0.0}

    gram = centered @ centered.T
    eigenvalues = np.linalg.eigvalsh(gram)
    energy = np.clip(eigenvalues, 0.0, None)[::-1]
    total = energy.sum()
    if total <= 1e-12:
        return {"pcs90": 0, "effective_rank": 0.0}

    probs = energy / total
    cumulative = np.cumsum(probs)
    pcs90 = int(np.searchsorted(cumulative, 0.90) + 1)
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    effective_rank = float(np.exp(entropy))
    return {"pcs90": pcs90, "effective_rank": effective_rank}


def summarize_episode_lengths(lengths: List[int]) -> Dict[str, float]:
    array = np.asarray(lengths, dtype=np.float32)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def flatten_signature_dict(
    episode_summaries: Dict[str, Dict[str, np.ndarray]],
    key: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    signatures = []
    tasks = []
    episodes = []
    for episode_name, summary in episode_summaries.items():
        signatures.append(summary[key])
        tasks.append(summary["task"])
        episodes.append(episode_name)
    return np.stack(signatures), tasks, episodes


def global_pairwise_summary(signatures: np.ndarray, tasks: List[str]) -> Dict[str, float]:
    cosine_matrix = cosine_similarity_matrix(signatures)
    pair_mask = np.triu(np.ones_like(cosine_matrix, dtype=bool), k=1)

    task_array = np.asarray(tasks)
    same_task_mask = (task_array[:, None] == task_array[None, :]) & pair_mask
    diff_task_mask = (task_array[:, None] != task_array[None, :]) & pair_mask

    same_values = cosine_matrix[same_task_mask]
    diff_values = cosine_matrix[diff_task_mask]

    return {
        "same_task_mean": float(same_values.mean()) if same_values.size else float("nan"),
        "different_task_mean": float(diff_values.mean()) if diff_values.size else float("nan"),
        "same_task_median": float(np.median(same_values)) if same_values.size else float("nan"),
        "different_task_median": float(np.median(diff_values)) if diff_values.size else float("nan"),
    }


def build_episode_summaries(args: argparse.Namespace) -> Dict[str, Dict[str, np.ndarray]]:
    dataset_path = resolve_dataset_path(args.dataset_path)
    task_filter = set(args.tasks) if args.tasks else None

    columns = [
        "seq_name",
        "frame_count",
        "action",
        "current_left_mano_trans",
        "current_right_mano_trans",
        "current_left_mano_parameters",
        "current_right_mano_parameters",
        "current_left_mano_rot",
        "current_right_mano_rot",
        "future_left_mano_trans",
        "future_right_mano_trans",
        "future_left_mano_parameters",
        "future_right_mano_parameters",
        "future_left_mano_rot",
        "future_right_mano_rot",
    ]
    dataframe = pd.read_parquet(dataset_path, columns=columns)
    dataframe["task"] = dataframe["seq_name"].str.split("/").str[0]

    if task_filter is not None:
        dataframe = dataframe[dataframe["task"].isin(task_filter)]

    if args.max_episodes_per_task is not None:
        allowed_episodes: List[str] = []
        for _task_name, group in dataframe.groupby("task", sort=False):
            allowed_episodes.extend(group["seq_name"].drop_duplicates().iloc[: args.max_episodes_per_task].tolist())
        dataframe = dataframe[dataframe["seq_name"].isin(set(allowed_episodes))]

    summaries: Dict[str, Dict[str, np.ndarray]] = {}
    grouped = dataframe.groupby("seq_name", sort=False)
    for episode_index, (episode_name, episode_df) in enumerate(grouped, start=1):
        episode_df = episode_df.sort_values("frame_count", kind="stable")
        rows = list(episode_df.itertuples(index=False))
        current_seq = np.stack(
            [build_current_vector(row, args.hand_loss_dim) for row in rows],
            axis=0,
        )
        future_seq = np.stack(
            [
                build_future_label_vector(
                    row=row,
                    future_steps=args.future_steps,
                    future_index=args.future_index,
                    hand_loss_dim=args.hand_loss_dim,
                )
                for row in rows
            ],
            axis=0,
        )
        action_seq = np.stack([build_action_vector(row) for row in rows], axis=0)

        summaries[episode_name] = {
            "task": str(episode_df["task"].iloc[0]),
            "num_frames": int(len(rows)),
            "current_signature": resample_sequence(current_seq, args.resample_steps).reshape(-1),
            "future_signature": resample_sequence(future_seq, args.resample_steps).reshape(-1),
            "action_signature": resample_sequence(action_seq, args.resample_steps).reshape(-1),
        }

        if args.progress_every > 0 and episode_index % args.progress_every == 0:
            print(
                f"[progress] processed {episode_index}/{grouped.ngroups} episodes"
            )

    return summaries


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    episode_summaries = build_episode_summaries(args)
    task_to_episodes: Dict[str, List[str]] = defaultdict(list)
    for episode_name, summary in episode_summaries.items():
        task_to_episodes[summary["task"]].append(episode_name)

    per_task: Dict[str, Dict[str, object]] = {}
    for task_name in sorted(task_to_episodes):
        episode_names = sorted(task_to_episodes[task_name])
        frame_lengths = [episode_summaries[name]["num_frames"] for name in episode_names]

        current_signatures = np.stack(
            [episode_summaries[name]["current_signature"] for name in episode_names],
            axis=0,
        )
        future_signatures = np.stack(
            [episode_summaries[name]["future_signature"] for name in episode_names],
            axis=0,
        )
        action_signatures = np.stack(
            [episode_summaries[name]["action_signature"] for name in episode_names],
            axis=0,
        )

        per_task[task_name] = {
            "episodes": len(episode_names),
            "episode_names": episode_names,
            "frame_lengths": summarize_episode_lengths(frame_lengths),
            "current_similarity": summarize_pairwise_cosine(current_signatures),
            "future_similarity": summarize_pairwise_cosine(future_signatures),
            "action_similarity": summarize_pairwise_cosine(action_signatures),
            "future_prototype": summarize_prototype(future_signatures),
            "future_rank": summarize_rank(future_signatures),
            "current_rank": summarize_rank(current_signatures),
            "action_rank": summarize_rank(action_signatures),
        }

    global_summary: Dict[str, Dict[str, float]] = {}
    for key in ("current_signature", "future_signature", "action_signature"):
        signatures, tasks, _episodes = flatten_signature_dict(episode_summaries, key)
        global_summary[key] = global_pairwise_summary(signatures, tasks)

    return {
        "config": {
            "dataset_path": str(resolve_dataset_path(args.dataset_path)),
            "future_steps": args.future_steps,
            "future_index": args.future_index,
            "hand_loss_dim": args.hand_loss_dim,
            "resample_steps": args.resample_steps,
            "tasks": sorted(args.tasks) if args.tasks else None,
            "max_episodes_per_task": args.max_episodes_per_task,
        },
        "num_episodes": len(episode_summaries),
        "tasks": per_task,
        "global_similarity": global_summary,
    }


def print_report(result: Dict[str, object]) -> None:
    print("Global pairwise cosine summary")
    for key, summary in result["global_similarity"].items():
        print(
            f"  {key}: same_task_mean={summary['same_task_mean']:.4f}, "
            f"different_task_mean={summary['different_task_mean']:.4f}"
        )

    print("\nPer-task summary")
    header = (
        "task,episodes,frames_mean,"
        "future_within_mean,future_proto_mean,future_pcs90,"
        "current_within_mean,action_within_mean"
    )
    print(header)
    for task_name, summary in result["tasks"].items():
        print(
            f"{task_name},{summary['episodes']},{summary['frame_lengths']['mean']:.1f},"
            f"{summary['future_similarity']['mean']:.4f},"
            f"{summary['future_prototype']['mean_cosine']:.4f},"
            f"{summary['future_rank']['pcs90']},"
            f"{summary['current_similarity']['mean']:.4f},"
            f"{summary['action_similarity']['mean']:.4f}"
        )


def main() -> int:
    args = parse_args()
    result = analyze(args)
    print_report(result)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
