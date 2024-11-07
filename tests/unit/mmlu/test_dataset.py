from pathlib import Path

import numpy as np

import llama_benchmarks as llb


def test_load_dataset(mmlu_dataset_path: Path):

    #
    # Whens
    #

    # I load mmlu dataset
    examples, questions = llb.mmlu.load_dataset(mmlu_dataset_path)

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


def test_swap_answers(mmlu_dataset_path: Path):

    #
    # Whens
    #

    # I load mmlu dataset
    examples, questions = llb.mmlu.load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # Answers should be distributed across all options
    distribution = llb.mmlu.answer_distribution(questions)
    assert np.all(distribution[option] > 0 for option in llb.mmlu.OPTIONS)

    #
    # Whens
    #

    # I swap answers to option "A"
    questions_a = llb.mmlu.swap_answers(questions, "A")

    #
    # Thens
    #

    # Answers should all be A
    assert (questions_a.answer == "A").all()
