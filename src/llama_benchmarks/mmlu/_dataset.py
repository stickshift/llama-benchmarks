from importlib.metadata import distribution
from pathlib import Path

import pandas as pd
from pandas import DataFrame

__all__ = [
    "OPTIONS",
    "load_dataset",
    "answer_distribution",
    "swap_answers",
]

OPTIONS = ["A", "B", "C", "D"]


def load_dataset(dataset_path: Path) -> tuple[DataFrame, DataFrame]:
    """Load MMLU examples and questions."""

    examples = _load_segment("dev", dataset_path=dataset_path)
    questions = _load_segment("test", dataset_path=dataset_path)

    return examples, questions


def swap_answers(questions: DataFrame, option: str) -> DataFrame:
    """Swap answers for all questions to option."""

    # Validate
    if option not in OPTIONS:
        raise ValueError(f"Invalid option: {option}")

    # Since the columns we're switching are different for each row, we have to swap them one by one
    rows = []
    for _, input_row in questions.iterrows():
        # Clone input row
        output_row = input_row.copy()

        value = output_row[option]
        output_row[option] = output_row[output_row.answer]
        output_row[output_row.answer] = value
        output_row.answer = option

        rows.append(output_row)

    return DataFrame(rows)


def answer_distribution(questions: DataFrame) -> dict[str, int]:
    """Calculate answer distribution for questions."""
    distribution = {
        option: questions[questions.answer == option].answer.count() for option in OPTIONS
    }
    return distribution


#-------------------------------------------------------------------------------
# Utilities
#-------------------------------------------------------------------------------

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
