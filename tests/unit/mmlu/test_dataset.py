from collections import Counter
from pathlib import Path

import llama_benchmarks as llb
from llama_benchmarks.mmlu import OPTIONS


def test_load_dataset(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load mmlu dataset
    examples, questions = llb.mmlu.load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # There should be 14,042 questions
    assert len(questions) == 14042

    # There should be 57 categories
    assert len(set(q.category for q in examples)) == 57

    # Each category should have 5 examples
    counts = Counter(q.category for q in examples)
    assert all(count == 5 for count in counts.values())


def test_load_dataset_sample(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # Sample size is 16
    n_questions = 16

    #
    # Whens
    #

    # I load sample of mmlu dataset
    examples, questions = llb.mmlu.load_dataset(mmlu_dataset_path, n_questions=n_questions)

    #
    # Thens
    #

    # There should be 16 questions
    assert len(questions) == n_questions

    # Examples should be limited to the relevant categories
    relevant_categories = set(q.category for q in questions)
    assert all(q.category in relevant_categories for q in examples)


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
    assert all(distribution[option] > 0 for option in OPTIONS)

    #
    # Whens
    #

    # I swap answers to option "A"
    questions_a = llb.mmlu.swap_answers(questions, "A")

    #
    # Thens
    #

    # Answers should all be A
    assert all(q.answer == "A" for q in questions_a)
