import logging
from pathlib import Path
from random import sample
from time import perf_counter_ns as timer

import llama_benchmarks as llb
from llama_benchmarks.mmlu import OPTIONS, MMLULlamaGenerator
from llama_benchmarks.models import llama

logger = logging.getLogger(__name__)


def test_mmlu_llama_generator(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # Sample size of 8
    n_questions = 8

    # I loaded question sample from mmlu dataset
    questions = sample(llb.mmlu.load_dataset(mmlu_dataset_path), n_questions)

    # I created a Llama 3.2 3B MMLU generator
    generator = MMLULlamaGenerator(llama.config("Llama3.2-3B"))

    #
    # Whens
    #

    # I start timer
    start_time = timer()

    # I generate answers for each question
    correct = 0
    for answer in generator(questions):
        #
        # Thens
        #

        # Expected answer should match question
        question = next(q for q in questions if q.qid == answer.qid)
        assert answer.expected == question.answer

        # Actual answer should be valid option
        assert answer.actual in OPTIONS

        # Logits should include all options
        assert all(option in answer.scores for option in OPTIONS)

        # Correct should be True if expected matches actual
        assert answer.correct == (answer.expected == answer.actual)

        if answer.correct:
            correct += 1

    # I end timer
    duration = timer() - start_time

    # I calculate metrics
    accuracy = correct / n_questions
    rps = 1000000000 * n_questions / duration

    logger.info(f"Accuracy: {accuracy:.2f}")
    logger.info(f"RPS: {rps:.2f}")
