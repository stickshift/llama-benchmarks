from llama_benchmarks.models.llama import Config

from ._dataset import Questions, Answers

__all__ = [
    "LlamaGenerator",
]


class LlamaGenerator:
    """Generates MMLU answers using a Llama model."""

    def __init__(self, config: Config):
        pass

    def __call__(self, examples: Questions, questions: Questions) -> Answers:
        pass
