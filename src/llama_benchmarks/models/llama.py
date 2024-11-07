"""Minimalistic Llama3 utilities based on Meta's reference implementation."""

import json
import logging
from pathlib import Path
from typing import Any, Iterator, NamedTuple

from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.functional import silu, softmax

from llama_benchmarks.tools import default_arg
from llama_benchmarks.tools import device as torch_device

__all__ = [
    "Config",
    "LlamaGenerator",
    "LlamaHead",
    "LlamaLayer",
    "config",
    "rope_frequencies",
]

logger = logging.getLogger(__name__)


class Config(NamedTuple):
    """Custom Llama3 config."""

    device: torch.device

    checkpoint_path: Path

    vocab_size: int

    d_model: int

    d_head: int

    d_ffn: int

    n_layers: int

    n_heads: int

    n_kv_heads: int

    rms_norm_eps: float

    rope_theta: float

    max_seq_len: int

    temperature: float | None = 0.6

    top_k: int = 50

    top_p: float = 0.9

    max_completion_tokens: int = 64


def config(
    checkpoint_name: str,
    device: torch.device | None = None,
    max_seq_len: int | None = None,
    **kwargs,
) -> Config:
    """Load Llama3 config from checkpoint."""
    # Defaults
    device = default_arg(device, default_factory=torch_device)
    max_seq_len = default_arg(max_seq_len, 8192)

    # Build checkpoint_path
    checkpoints_path = Path("~/.llama/checkpoints").expanduser()
    checkpoint_path = checkpoints_path / checkpoint_name

    # Load hyperparameters
    hparams_path = checkpoint_path / "params.json"
    hparams = json.loads(hparams_path.read_text())

    # Calculate d_ffn from 8/3 * d_model rounded to nearest multiple_of
    d_model = hparams["dim"]
    ffn_dim_multiplier = hparams["ffn_dim_multiplier"]
    multiple_of = hparams["multiple_of"]
    d_ffn = int(8 / 3 * d_model * ffn_dim_multiplier)
    d_ffn = multiple_of * ((d_ffn + multiple_of - 1) // multiple_of)

    defaults = {
        "device": device,
        "checkpoint_path": checkpoint_path,
        "vocab_size": hparams["vocab_size"],
        "d_model": hparams["dim"],
        "n_layers": hparams["n_layers"],
        "rms_norm_eps": hparams["norm_eps"],
        "n_heads": hparams["n_heads"],
        "d_head": int(hparams["dim"] / hparams["n_heads"]),
        "n_kv_heads": hparams["n_kv_heads"],
        "rope_theta": hparams["rope_theta"],
        "d_ffn": d_ffn,
        "max_seq_len": max_seq_len,
    }

    # Override with kwargs
    data = defaults | kwargs

    return Config(**data)


def rope_frequencies(config: Config, n: int):
    """Compute RoPE cos and sin rotation matrices."""
    # Hyperparameters
    base = config.rope_theta
    d = config.d_head

    # Calculate thetas
    i = torch.arange(d // 2, device=config.device)
    thetas = base ** (-2 * i / d)

    # Duplicate each theta, e.g. [theta_0, theta_1] -> [theta_0, theta_0, theta_1, theta_1]
    thetas = thetas.repeat_interleave(2)

    # Repeat thetas for each position from 0 to n and stack in an (n, d_head) matrix
    theta_stack = torch.stack([m * thetas for m in range(n)])

    # Apply cos, sin
    r_cos = torch.cos(theta_stack)
    r_sin = torch.sin(theta_stack)

    # Sanity check
    assert r_cos.shape[0] == n and r_cos.shape[1] == config.d_head
    assert r_sin.shape[0] == n and r_sin.shape[1] == config.d_head

    return r_cos, r_sin


def rope_swap(x):
    """Maps [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]."""
    # Preserve original shape
    s = x.shape

    # Split into pairs, swap, and restore shape
    x = x.reshape(-1, 2).flip(-1).view(s)

    # Multiply every even index along the last dimension by -1
    #   e.g. [x0, x1, x2, x3] -> [-x0, x1, -x2, x3]
    x[..., ::2] *= -1

    return x


def rope_rotate(x, r_cos, r_sin):
    """Rotate embeddings using RoPE transform."""
    return (x * r_cos) + (rope_swap(x) * r_sin)


class LlamaLayer(nn.Module):
    def __init__(self, config: Config, checkpoint: dict[str, Any], layer_id: int):
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        # Attention normalization
        self.normalize_attention = RMSNorm(config.d_model, config.rms_norm_eps).to(config.device)
        self.normalize_attention.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.attention_norm.weight"],
        })

        # Query projection
        self.w_q = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_heads * config.d_head,
            bias=False,
            device=config.device,
        )
        self.w_q.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.attention.wq.weight"],
        })

        # Key projection
        self.w_k = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=config.device,
        )
        self.w_k.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.attention.wk.weight"],
        })

        # Value projection
        self.w_v = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_kv_heads * config.d_head,
            bias=False,
            device=config.device,
        )
        self.w_v.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.attention.wv.weight"],
        })

        # Attention output projection
        self.w_a = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_model,
            bias=False,
            device=config.device,
        )
        self.w_a.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.attention.wo.weight"],
        })

        # FFN normalization
        self.normalize_ffn = RMSNorm(config.d_model, config.rms_norm_eps).to(config.device)
        self.normalize_ffn.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.ffn_norm.weight"],
        })

        # SwiGLU FFN
        self.w_h = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=config.device,
        )
        self.w_h.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.feed_forward.w3.weight"],
        })

        self.w_g = nn.Linear(
            in_features=config.d_model,
            out_features=config.d_ffn,
            bias=False,
            device=config.device,
        )
        self.w_g.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.feed_forward.w1.weight"],
        })

        # FFN output projection
        self.w_f = nn.Linear(
            in_features=config.d_ffn,
            out_features=config.d_model,
            bias=False,
            device=config.device,
        )
        self.w_f.load_state_dict({
            "weight": checkpoint[f"layers.{layer_id}.feed_forward.w2.weight"],
        })

    def forward(self, x: Tensor, r_cos: Tensor, r_sin: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        #
        # Attention
        #
        residual = x

        # Normalize attention inputs
        x = self.normalize_attention(x)

        # Project embeddings to query, key, value spaces
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Split attention heads
        q = self._split_heads(q, self.config.n_heads)
        k = self._split_heads(k, self.config.n_kv_heads)
        v = self._split_heads(v, self.config.n_kv_heads)

        # Expand key/value groups
        reps = self.config.n_heads // self.config.n_kv_heads
        k = k.repeat_interleave(reps, dim=0)
        v = v.repeat_interleave(reps, dim=0)

        # Encode positions by rotating queries and keys
        q = rope_rotate(q, r_cos, r_sin)
        k = rope_rotate(k, r_cos, r_sin)

        # Compute masked attention bias M
        n = len(x)
        mask = torch.ones(n, n, dtype=torch.bool, device=self.config.device).tril(diagonal=0)
        m = torch.zeros(n, n, device=self.config.device).masked_fill_(mask.logical_not(), float("-inf"))

        # Compute attention for all heads in parallel
        a = softmax(q @ k.transpose(-2, -1) / np.sqrt(self.config.d_head) + m, dim=-1) @ v

        # Combine attention heads
        a = self._combine_heads(a)

        # Project attention representations back to model space
        a = self.w_a(a)

        # Combine attention representations with residual embeddings
        x = residual + a

        #
        # FFN
        #

        residual = x

        # Normalize FFN inputs
        x = self.normalize_ffn(x)

        # Apply SwiGLU transform
        f = silu(self.w_g(x)) * self.w_h(x)

        # Project FFN representations back to model space
        f = self.w_f(f)

        # Combine FFN representations with residual embeddings
        x = residual + f

        return x

    def _split_heads(self, x: Tensor, n_heads: int):
        return x.view(-1, n_heads, self.config.d_head).transpose(-3, -2)

    def _combine_heads(self, x):
        return x.transpose(-3, -2).contiguous().view(-1, int(self.config.n_heads * self.config.d_head))


class LlamaHead(nn.Module):
    """General purpose Llama head."""

    def __init__(self, config: Config, checkpoint: dict[str, Any]):
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

    def forward(self, x: Tensor) -> int:
        # Normalize head inputs
        x = self.normalize_head(x)

        # Use last embedding to represent the entire sequence
        x = x[-1]

        # Project outputs to token space
        x = self.w_head(x)

        #
        # Temperature
        #

        # Apply temperature
        x = x / self.config.temperature

        #
        # Ranking
        #

        # Convert logits to probabilities
        probs = softmax(x, dim=-1)

        # Sort probabilities in descending order
        probs, indices = probs.sort(descending=True)

        #
        # Top K
        #

        # Retain top k tokens
        probs = probs[: self.config.top_k]

        #
        # Top P
        #

        # Find cutoff where cumulative probability exceeds top_p
        cumulative_mask = probs.cumsum(dim=-1) > self.config.top_p
        threshold_index = torch.argmax(cumulative_mask).item()

        # Only apply threshold if top_p was exceeded
        if cumulative_mask.any():
            probs = probs[: threshold_index + 1]

        #
        # Random Selection
        #

        # Sample from remaining tokens weighted by probability
        sampled_index = torch.multinomial(probs, 1)

        # Convert sampled_index to original logits
        token_id = indices[sampled_index]

        return token_id.item()


class LlamaModel(nn.Module):
    """Container module for embeddings, layers, and head."""

    def __init__(self, config: Config):
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

        self.head = LlamaHead(config, checkpoint)

    def forward(self, x: Tensor, r_cos: Tensor, r_sin: Tensor) -> int:
        # Map tokens to embeddings
        x = self.embeddings(x)

        # Transform token embeddings to semantic embeddings
        for layer in self.layers:
            x = layer(x, r_cos, r_sin)

        # Head
        token_id = self.head(x)

        return token_id


class LlamaGenerator:
    """General purpose Llama generative model."""

    def __init__(self, config: Config):
        self.config = config

        self.tokenizer = Tokenizer(str(config.checkpoint_path / "tokenizer.model"))

        self.model = LlamaModel(config).to(config.device)

    def __call__(self, prompt: str) -> Iterator[int]:
        """Generate tokens from prompt."""
        # Prepare model
        self.model.eval()

        # Split raw text into tokens
        token_ids = self.tokenizer.encode(prompt, bos=True, eos=False, allowed_special="all")

        with torch.no_grad():
            # Generate output until we get a stop token or we exceed max_output_tokens.
            for _ in range(self.config.max_completion_tokens):
                # Compute cos and sin rotation matrices once for entire sequence
                r_cos, r_sin = rope_frequencies(self.config, len(token_ids))

                # Load token ids into a tensor
                x = torch.tensor(token_ids, device=self.config.device)

                # Generate next token
                token_id = self.model(x, r_cos, r_sin)

                # Check stopping criteria
                if token_id in self.tokenizer.stop_tokens:
                    break

                # Decode token_id
                token = self.tokenizer.decode([token_id])

                # Yield token
                yield token

                # Append to end of sequence
                token_ids.append(token_id)
