import logging
from typing import Any, Iterator

from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import torch
from torch import Tensor, nn
from torch.nn.functional import softmax

from llama_benchmarks.models.llama import (
    Config,
    LlamaLayer,
    rope_frequencies,
)
from llama_benchmarks.tools import trace

from ._dataset import OPTIONS, Answer, Questions, generate_prompt

__all__ = [
    "MMLULlamaGenerator",
]

logger = logging.getLogger(__name__)


class MMLULlamaHead(nn.Module):
    """Custom Llama head for MMLU."""

    def __init__(self, config: Config, checkpoint: dict[str, Any], tokenizer: Tokenizer):
        super().__init__()

        self.config = config

        # Head normalization
        self.normalize_head = RMSNorm(config.d_model, config.rms_norm_eps).to(config.device)
        self.normalize_head.load_state_dict({
            "weight": checkpoint["norm.weight"],
        })

        # Output projection
        self.w_head = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False,
            device=config.device,
        )
        self.w_head.load_state_dict({
            "weight": checkpoint["output.weight"],
        })

        # Calculate token ids for each MMLU option
        self.token_ids = {option: tokenizer.encode(option, bos=False, eos=False)[0] for option in OPTIONS}

    def forward(self, x: Tensor):
        # Normalize head inputs
        x = self.normalize_head(x)

        # Use last embedding to represent the entire sequence
        x = x[-1]

        # Project outputs to token space
        x = self.w_head(x)

        # Select logits for MMLU options
        logits = torch.tensor([x[self.token_ids[option]] for option in OPTIONS], device=x.device)

        # Convert to scores
        scores = softmax(logits, dim=-1)

        # Map options to scores
        scores = {option: scores[i] for i, option in enumerate(OPTIONS)}

        # Convert scores back to floats
        scores = {k: v.item() for k, v in scores.items()}

        return scores


class MMLULlamaModel(nn.Module):
    def __init__(self, config: Config, tokenizer: Tokenizer):
        super().__init__()

        # Load checkpoint
        checkpoint = torch.load(
            config.checkpoint_path / "consolidated.00.pth",
            weights_only=True,
        )

        self.config = config

        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            device=config.device,
        )
        self.embeddings.load_state_dict({"weight": checkpoint["tok_embeddings.weight"]})

        self.layers = nn.ModuleList(LlamaLayer(config, checkpoint, layer_id) for layer_id in range(config.n_layers))

        self.head = MMLULlamaHead(config, checkpoint, tokenizer)

    def forward(self, x: Tensor, r_cos: Tensor, r_sin: Tensor):
        # Map tokens to embeddings
        x = self.embeddings(x)

        # Transform token embeddings to semantic embeddings
        for layer in self.layers:
            x = layer(x, r_cos, r_sin)

        # Head
        scores = self.head(x)

        # Calculate answer
        actual = max(scores, key=scores.get)

        return scores, actual


class MMLULlamaGenerator:
    """Custom Llama generative model for MMLU."""

    def __init__(self, config: Config):
        self.config = config

        self.tokenizer = Tokenizer(str(config.checkpoint_path / "tokenizer.model"))

        self.model = MMLULlamaModel(config, self.tokenizer).to(config.device)

    def __call__(
        self,
        questions: Questions,
        n_shots: int | None = None,
        examples: Questions | None = None,
    ) -> Iterator[Answer]:
        """Generate answers."""
        # Prepare model
        self.model.eval()

        with torch.no_grad():
            for question in questions:
                with trace(logger, f"Answering question {question.qid}"):
                    # Generate prompt
                    prompt = generate_prompt(question, n_shots=n_shots, examples=examples)

                    # Split raw text into tokens
                    token_ids = self.tokenizer.encode(prompt, bos=True, eos=False)

                    # Compute cos and sin rotation matrices
                    r_cos, r_sin = rope_frequencies(self.config, len(token_ids))

                    # Load token ids into a tensor
                    x = torch.tensor(token_ids, device=self.config.device)

                    # Generate answer
                    scores, actual = self.model(x, r_cos, r_sin)

                # Yield answer
                yield Answer(
                    qid=question.qid,
                    expected=question.answer,
                    actual=actual,
                    scores=scores,
                    correct=(actual == question.answer),
                )
