"""Read back the JSON caches written by ``scripts/prepare_data.py`` into
dataclass examples."""
from __future__ import annotations

import json
from pathlib import Path

from biasneut.data.schema import DetectionExample, EditExample


def load_detection_examples(path: str | Path) -> list[DetectionExample]:
    data = json.loads(Path(path).read_text())
    return [DetectionExample(**row) for row in data]


def load_edit_examples(path: str | Path) -> list[EditExample]:
    data = json.loads(Path(path).read_text())
    return [EditExample(**row) for row in data]
