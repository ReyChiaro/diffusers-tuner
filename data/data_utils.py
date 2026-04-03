import os
from pathlib import Path
import json
from typing import Any

from loguru import logger


def load_jsonl_file(jsonl_path: str | None) -> list[dict[str, Any]]:
    if jsonl_path is None or not Path(jsonl_path).exists():
        logger.error(f"{jsonl_path=} is None or not existing, return empty data sample list.")
        return []

    sample_lines = Path(jsonl_path).read_text().splitlines()
    return [json.loads(l) for l in sample_lines]
