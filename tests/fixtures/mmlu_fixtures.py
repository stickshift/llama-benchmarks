from pathlib import Path

import pytest
import torch

__all__ = [
    "default_dtype",
    "mmlu_dataset_path",
]


@pytest.fixture(autouse=True)
def default_dtype():
    torch.set_default_dtype(torch.bfloat16)


@pytest.fixture
def mmlu_dataset_path(datasets_path: Path) -> Path:
    return datasets_path / "mmlu"
