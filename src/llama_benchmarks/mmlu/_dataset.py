from pathlib import Path

import pandas as pd
from pandas import DataFrame

__all__ = [
    "load_dataset",
]


def load_dataset(dataset_path: Path) -> tuple[DataFrame, DataFrame]:
    """Load MMLU examples and questions."""

    examples = _load_segment("dev", dataset_path=dataset_path)
    questions = _load_segment("test", dataset_path=dataset_path)

    return examples, questions


def _load_segment(segment: str, dataset_path: Path) -> DataFrame:
    """Load segment of MMLU dataset."""

    column_names = ["question", "A", "B", "C", "D", "answer"]

    # Sort paths to ensure consistent order
    paths = sorted(path for path in dataset_path.glob(f"{segment}/*.csv"))

    dataset = None
    for path in paths:
        # Load csv
        df = pd.read_csv(path, names=column_names)

        # Infer category from file name: x_y_z_test.csv -> x y z
        df["category"] = " ".join(path.stem.split("_")[0:-1])

        # Append
        dataset = df if dataset is None else pd.concat([dataset, df], ignore_index=True)

    # Pandas parses the word "None" as a NaN. Replace these with explicit string "None"
    dataset = dataset.fillna("None")

    return dataset
