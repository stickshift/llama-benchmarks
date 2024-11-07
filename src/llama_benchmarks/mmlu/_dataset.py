import csv
from pathlib import Path
import random
from typing import NamedTuple, Sequence

from llama_benchmarks.tools import executor

__all__ = [
    "Question",
    "Questions",
    "Answer",
    "Answers",
    "OPTIONS",
    "load_dataset",
    "answer_distribution",
    "swap_answers",
    "generate_prompt",
]

OPTIONS = ["A", "B", "C", "D"]


class Question(NamedTuple):
    """Represents an MMLU question."""
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

    logits: dict[str, float]

    correct: bool


Answers = Sequence[Answer]


def load_dataset(dataset_path: Path, n_questions: int | None = None,) -> tuple[Questions, Questions]:
    """Load MMLU examples and questions."""

    examples = _load_segment("dev", dataset_path=dataset_path)

    questions = _load_segment("test", dataset_path=dataset_path)

    # Sample questions
    if n_questions is not None:
        questions = random.sample(questions, n_questions)

        categories = set(q.category for q in questions)
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
    distribution = {
        option: sum(1 for q in questions if q.answer == option)
        for option in OPTIONS
    }
    return distribution


def generate_prompt(examples: Questions, question: Question, n_shots: int | None = None):
    """Generate prompt for specified question."""
    # Select examples for category
    selected_examples = [e for e in examples if e.category == question.category]

    # Select n_shots if specified
    if n_shots is not None:
        selected_examples = random.sample(selected_examples, n_shots)

    # Start with examples
    content = f"The following are multiple choice questions (with answers) about {question.category}.\n\n"
    for row in selected_examples:
        content += (
            f"Question: {row.question}\n"
            f"\n"
            f"A) {row.A}\n"
            f"B) {row.B}\n"
            f"C) {row.C}\n"
            f"D) {row.D}\n"
            f"\n"
            f"Answer: {row.answer}\n"
            f"\n"
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


# -------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------



def _load_data_file(path: Path) -> Sequence[Question]:
    """Load a single MMLU data file."""

    # Infer category from file name: x_y_z_test.csv -> x y z
    category = " ".join(path.stem.split("_")[0:-1])

    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        questions = tuple(Question(category, *row) for row in reader)

    return questions


def _load_segment(segment: str, dataset_path: Path) -> Sequence[Question]:
    """Load segment of MMLU dataset."""

    # Sort paths to ensure consistent order
    paths = sorted(path for path in dataset_path.glob(f"{segment}/*.csv"))

    # Load data files in parallel
    futures = [executor.submit(_load_data_file, path) for path in paths]

    # Collect results
    questions = ()
    for future in futures:
        questions += future.result()

    return questions
