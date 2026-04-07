import os
import json
import numpy as np

from loguru import logger
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Any, Iterable


ASPECT_RATIO_BUCKETS = (
    (1, 1),
    (1, 4),
    (1, 8),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 1),
    (4, 3),
    (4, 5),
    (5, 4),
    (8, 1),
    (9, 16),
    (16, 9),
    (21, 9),
)


# ---------------- Loading Method for Different File Types ---------------- #
def load_jsonl_file(jsonl_path: str | None) -> list[dict[str, Any]]:
    if jsonl_path is None or not Path(jsonl_path).exists():
        logger.error(f"{jsonl_path=} is None or not existing, return empty data sample list.")
        return []

    sample_lines = Path(jsonl_path).read_text().splitlines()
    return [json.loads(l) for l in sample_lines]


def load_json_file(json_path: str | None) -> dict[str, Any]:
    if json_path is None or not Path(json_path).exists():
        logger.error(f"{json_path=} is None or not existing, return empty data sample list.")
        return {}

    with open(json_path, "r") as f:
        samples = json.load(f)
    return samples


def get_centercrop_params(
    source_shape: Iterable[int],
    target_shape: Iterable[int],
) -> tuple[int]:
    w, h = source_shape
    tw, th = target_shape

    i = (h - th) // 2
    j = (w - tw) // 2
    return i, j, th, tw


def bucket_aspect_ratios(
    dataset_indices: str,
    output_indices: str,
    buckets: tuple[tuple[int, int]] = ASPECT_RATIO_BUCKETS,
):
    samples = load_jsonl_file(dataset_indices)
    ar_buckets = np.array([w / h for w, h in buckets])

    updated_samples = {i: [] for i in range(len(buckets))}
    for sample in tqdm(samples, desc="Bucket Classify"):
        if not sample["is_align"]:
            continue
        updated_sample = {}
        updated_sample.update(sample)
        target_path = sample["target_path"]
        target = Image.open(target_path).convert("RGB")
        ar_image = target.width / target.height
        closest_bucket_idx = np.argmin(np.abs(ar_buckets - ar_image)).item()
        updated_sample["bucket"] = closest_bucket_idx
        updated_sample["target_aspect_ratio"] = buckets[closest_bucket_idx]
        updated_samples[closest_bucket_idx].append(updated_sample)

    with open(output_indices, "w") as f:
        json.dump(updated_samples, f, indent=2)

    logger.info(f"Bucket indices and dataset saved to {output_indices}.")
