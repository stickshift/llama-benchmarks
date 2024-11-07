import logging
from typing import Any, Iterator

from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import torch
from torch import Tensor, nn

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

        # Lookup logits for MMLU token ids
        logits = {option: x[token_id] for option, token_id in self.token_ids.items()}

        return logits


class MMLULlamaGenerator:
    """Custom Llama generative model for MMLU."""

    def __init__(self, config: Config):
        # Load checkpoint
        checkpoint = torch.load(
            config.checkpoint_path / "consolidated.00.pth",
            weights_only=True,
            map_location=config.device,
        )

        self.config = config

        self.tokenizer = Tokenizer(str(config.checkpoint_path / "tokenizer.model"))

        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            device=config.device,
        )
        self.embeddings.load_state_dict({"weight": checkpoint["tok_embeddings.weight"]})

        self.layers = nn.ModuleList(LlamaLayer(config, checkpoint, layer_id) for layer_id in range(config.n_layers))

        self.head = MMLULlamaHead(config, checkpoint, self.tokenizer)

    def __call__(self, examples: Questions, questions: Questions) -> Iterator[Answer]:
        """Generate answers."""
        for qid, question in enumerate(questions):
            with trace(logger, f"{qid}: Generating prompt"):
                # Generate prompt
                prompt = generate_prompt(examples, question)

            with trace(logger, f"{qid}: Tokenizing"):
                # Split raw text into tokens
                token_ids = self.tokenizer.encode(prompt, bos=True, eos=False)

            with trace(logger, f"{qid}: Computing rope frequencies"):
                # Compute cos and sin rotation matrices
                r_cos, r_sin = rope_frequencies(self.config, len(token_ids))

            with trace(logger, f"{qid}: Loading tokens"):
                # Load token ids into a tensor
                x = torch.tensor(token_ids, device=self.config.device)

            with trace(logger, f"{qid}: Embeddings"):
                # Map tokens to embeddings
                x = self.embeddings(x)

            with trace(logger, f"{qid}: Context Layers"):
                # Transform token embeddings to semantic embeddings
                for layer in self.layers:
                    x = layer(x, r_cos, r_sin)

            with trace(logger, f"{qid}: Head"):
                # Head
                logits = self.head(x)

            # Calculate answer
            actual = max(logits, key=logits.get)

            # Yield answer
            yield Answer(
                qid=qid,
                expected=question.answer,
                actual=actual,
                logits=logits,
                correct=(actual == question.answer),
            )

            logger.info(f"Answered question {qid}: {question.question}")
