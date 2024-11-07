from pathlib import Path

import numpy as np

import llama_benchmarks as llb


def test_load_dataset(datasets_path: Path):
    #
    # Givens
    #

    # Path to mmlu dataset
    dataset_path = datasets_path / "mmlu"

    #
    # Whens
    #

    # I load mmlu dataset
    examples, questions = llb.mmlu.load_dataset(dataset_path)

    #
    # Thens
    #

    # Indices should be unique
    assert examples.index.is_unique
    assert questions.index.is_unique

    # Columns should be populated
    expected_columns = {
        "category",
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
    }
    assert set(examples.columns) == expected_columns
    assert set(questions.columns) == expected_columns

    # There should be 57 categories
    assert len(examples.category.unique()) == 57

    # Each category should have 5 examples
    assert np.all(examples.category.value_counts() == 5)

    # None of the columns should have NaNs
    for column in expected_columns:
        assert examples[column].isna().sum() == 0
        assert questions[column].isna().sum() == 0
