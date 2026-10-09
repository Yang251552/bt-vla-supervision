"""Append-only JSONL event log: one line per event, control step + monotonic wall time."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .contracts import Ev


class EventLog:
    def __init__(self, path, cfg_hash: str, **meta):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)  # line-buffered: a crash keeps every full line
        self(Ev.EPISODE, -1, cfg_hash=cfg_hash, **meta)

    def __call__(self, ev: str, t: int, **fields) -> None:
        rec = {"ev": ev, "t": int(t), "wall": time.monotonic(), **fields}
        self._f.write(json.dumps(rec, default=_jsonable) + "\n")

    def close(self) -> None:
        self._f.close()


def _jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "value"):  # Enum
        return x.value
    raise TypeError(f"not JSON-serialisable: {type(x)}")


def read(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
