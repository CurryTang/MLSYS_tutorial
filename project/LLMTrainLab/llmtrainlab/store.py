"""JSON snapshot of the simulated cluster, so `llmctl` commands compose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path(".llmtrainlab")
STATE_NAME = "state.json"


def state_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else Path.cwd()
    return base / DEFAULT_DIR / STATE_NAME


def load_state(root: Path | None = None) -> dict[str, Any]:
    path = state_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"未找到集群状态 {path}。先运行 `llmctl cluster init`。"
        )
    return json.loads(path.read_text())


def save_state(state: dict[str, Any], root: Path | None = None) -> Path:
    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return path
