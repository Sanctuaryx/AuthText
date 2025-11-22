from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: str | pathlib.Path) -> List[Dict[str, Any]]:
    path = pathlib.Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_xy(
    path: str | pathlib.Path,
    text_key: str = "text",
    label_key: str = "label",
    group_key: Optional[str] = None,
) -> Tuple[List[str], List[int], Optional[List[Any]]]:
    rows = read_jsonl(path)
    X = [row[text_key] for row in rows]
    y = [int(row[label_key]) for row in rows]
    groups = [row[group_key] for row in rows] if group_key is not None else None
    return X, y, groups
