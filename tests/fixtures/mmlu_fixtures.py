from pathlib import Path

import pytest

__all__ = [
    "mmlu_dataset_path",
]


@pytest.fixture
def mmlu_dataset_path(datasets_path: Path) -> Path:
    return datasets_path / "mmlu"
