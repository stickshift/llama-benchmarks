import logging
from pathlib import Path
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
    examples, questions = llb.mmlu.load_dataset(mmlu_dataset_path, n_questions=n_questions)

    # I created a Llama 3.2 3B MMLU generator
    generator = MMLULlamaGenerator(llama.config("Llama3.2-3B"))

    #
    # Whens
    #

    # I start timer
    start_time = timer()

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

    # I end timer
    duration = timer() - start_time

    # I calculate metrics
    rps = 1000000000 * n_questions / duration

    logger.info(f"RPS: {rps:.2f}")
