import csv
from pathlib import Path
import random
from typing import NamedTuple, Sequence

from llama_benchmarks.tools import executor, default_arg

__all__ = [
    "OPTIONS",
    "Answer",
    "Answers",
    "Question",
    "Questions",
    "answer_distribution",
    "generate_prompt",
    "load_dataset",
    "swap_answers",
    "debias_example_answers",
    "debias_question_answers",
]

OPTIONS = tuple(["A", "B", "C", "D"])


class Question(NamedTuple):
    """Represents an MMLU question."""

    qid: int

    category: str

    question: str

    A: str

    B: str

    C: str

    D: str

    answer: str


Questions = Sequence[Question]


class Answer(NamedTuple):
    """Represents an answer to MMLU question."""

    qid: int

    expected: str

    actual: str

    scores: dict[str, float]

    correct: bool


Answers = Sequence[Answer]


def load_dataset(
    dataset_path: Path,
    n_questions: int | None = None,
) -> tuple[Questions, Questions]:
    """Load MMLU examples and questions."""
    examples = _load_segment("dev", dataset_path=dataset_path)

    questions = _load_segment("test", dataset_path=dataset_path)

    # Sample questions
    if n_questions is not None:
        questions = random.sample(questions, n_questions)

        categories = {q.category for q in questions}
        examples = tuple(e for e in examples if e.category in categories)

    return examples, questions


def swap_answers(questions: Questions, option: str) -> Questions:
    """Swap answers for all questions to option."""
    # Validate
    if option not in OPTIONS:
        raise ValueError(f"Invalid option: {option}")

    # Since the columns we're switching are different for each row, we have to swap them one by one
    results = []
    for question in questions:
        # Convert to mutable dict
        data = question._asdict()

        # Swap values
        value = data[option]
        data[option] = data[question.answer]
        data[question.answer] = value
        data["answer"] = option

        # Append
        results.append(Question(**data))

    return tuple(results)


def answer_distribution(questions: Questions) -> dict[str, int]:
    """Calculate answer distribution for questions."""
    distribution = {option: sum(1 for q in questions if q.answer == option) for option in OPTIONS}
    return distribution


def generate_prompt(examples: Questions, question: Question, n_shots: int | None = None, header: bool | None = None):
    """Generate prompt for specified question."""
    # Defaults
    header = default_arg(header, True)

    # Select examples for category
    selected_examples = [e for e in examples if e.category == question.category]

    # Deterministically select n_shots if specified
    if n_shots is not None:
        selected_examples = selected_examples[:n_shots]

    content = ""

    # Start with examples
    if header:
        content += f"The following are multiple choice questions (with answers) about {question.category}.\n\n"

    for row in selected_examples:
        content += (
            f"Question: {row.question}\n\nA) {row.A}\nB) {row.B}\nC) {row.C}\nD) {row.D}\n\nAnswer: {row.answer}\n\n"
        )

    # Pose question
    content += (
        f"Question: {question.question}\n"
        f"\n"
        f"A) {question.A}\n"
        f"B) {question.B}\n"
        f"C) {question.C}\n"
        f"D) {question.D}\n"
        f"\n"
        f"Answer: "
    )

    return content


def debias_example_answers(examples: Questions) -> Questions:
    """Evenly distribute example answers across options for each category."""
    categories = tuple(e.category for e in examples)

    # Deterministically select 4 examples per category
    results = ()
    for category in categories:
        population = tuple(e for e in examples if e.category == category)

        # First 4 examples
        selection = population[:4]

        # Move 25% of answers to each option
        segment_size = 1
        for i, option in enumerate(OPTIONS):
            segment = swap_answers(selection[i * segment_size : (i + 1) * segment_size], option)
            results += segment

    return results


def debias_question_answers(questions: Questions) -> Questions:
    """Evenly distribute question answers across options."""
    chunk_size = len(OPTIONS)

    # Deterministically select maximal subset of questions that is multiple of chunk size
    n_questions = chunk_size * (len(questions) // chunk_size)
    questions = questions[:n_questions]

    # Move 25% of answers to each option
    normalized = ()
    segment_size = int(n_questions / chunk_size)
    for i, option in enumerate(OPTIONS):
        segment = swap_answers(questions[i * segment_size: (i + 1) * segment_size], option)
        normalized += segment

    return normalized


# -------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------


def _load_data_file(path: Path) -> Sequence[Question]:
    """Load a single MMLU data file."""
    # Infer category from file name: x_y_z_test.csv -> x y z
    category = " ".join(path.stem.split("_")[0:-1])

    with open(path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        questions = tuple(Question(i, category, *row) for i, row in enumerate(reader))

    return questions


def _load_segment(segment: str, dataset_path: Path) -> Sequence[Question]:
    """Load segment of MMLU dataset."""
    # Sort paths to ensure consistent order
    paths = sorted(path for path in dataset_path.glob(f"{segment}/*.csv"))

    # Load data files in parallel
    futures = [executor.submit(_load_data_file, path) for path in paths]

    # Collect results
    collected = ()
    for future in futures:
        collected += future.result()

    # Reassign ids
    questions = []
    for i, question in enumerate(collected):
        questions.append(Question(i, *question[1:]))

    return tuple(questions)
