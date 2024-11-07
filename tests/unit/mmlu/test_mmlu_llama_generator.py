from pathlib import Path

import pytest

import llama_benchmarks as llb
from llama_benchmarks.mmlu import MMLULlamaGenerator, OPTIONS
from llama_benchmarks.models import llama


def test_mmlu_llama_generator(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # Sample size of 4
    n_questions = 4

    # I loaded question sample from mmlu dataset
    examples, questions = llb.mmlu.load_dataset(mmlu_dataset_path, n_questions=n_questions)

    # I created a Llama 3.2 3B MMLU generator
    generator = MMLULlamaGenerator(llama.config("Llama3.2-3B"))

    #
    # Whens
    #

    # I generate answers for each question
    for answer in generator(examples, questions):

        #
        # Thens
        #

        # Expected answer should match question
        assert answer.expected == questions[answer.qid].answer

        # Actual answer should be valid option
        assert answer.actual in OPTIONS

        # Logits should include all options
        assert all(option in answer.logits for option in OPTIONS)

        # Correct should be True if expected matches actual
        assert answer.correct == (answer.expected == answer.actual)
