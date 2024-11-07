from pathlib import Path

import pytest

import llama_benchmarks as llb
from llama_benchmarks.mmlu import LlamaGenerator, OPTIONS


@pytest.mark.wip
def test_llama_generator(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # Sample size of 16
    n_questions = 16

    # I loaded question sample from mmlu dataset
    examples, questions = llb.mmlu.load_dataset(mmlu_dataset_path, n_questions=n_questions)

    # Llama 3.2 3B config
    config = llb.models.llama.config("Llama3.2-3B")

    # I created an MMLU LlamaGenerator
    generator = LlamaGenerator(config)

    #
    # Whens
    #

    # I generate answers
    answers = generator(examples, questions)

    #
    # Thens
    #

    # There should be 16 answers
    assert len(answers) == n_questions

    # For each question...
    for answer in answers:

        # Expected answer should match question
        assert answer.expected == questions[answer.qid].answer

        # Actual answer should be valid option
        assert answer.actual in OPTIONS

        # Logits should include all options
        assert all(option in answer.logits for option in OPTIONS)

        # Correct should be True if expected matches actual
        assert answer.correct == (answer.expected == answer.actual)
