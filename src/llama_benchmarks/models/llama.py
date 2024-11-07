"""Minimalistic Llama3 utilities based on Meta's reference implementation."""

import json
from pathlib import Path

from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.reference_impl.model import RMSNorm
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch
from torch import nn
from torch.nn.functional import silu, softmax

from llama_benchmarks.tools import default_arg, take, device as torch_device

__all__ = [
    "Config",
    "config",
    "load_state",
    "tokenizer",
    "embeddings",
    "context_layers",
    "head",
]


class Config(BaseModel):
    """Custom Llama3 config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    max_output_tokens: int = 500


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


def tokenizer(config: Config) -> Tokenizer:
    """Load Llama3 tokenizer from checkpoint."""

    # Load tokenizer model from checkpoint
    return Tokenizer(str(config.checkpoint_path / "tokenizer.model"))


def embeddings(x, *, config: Config, checkpoint):
    # Initialize embeddings lookup table
    layer = nn.Embedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.d_model,
        device=config.device,
    )

    # Load pre-trained state
    load_state(layer, "embeddings", checkpoint=checkpoint)

    return layer(x)


def load_state(*args, checkpoint, layer=None):
    # Defaults
    layer = default_arg(layer, lambda: 0)

    for module, key in take(2, args):
        match key:
            # Embeddings
            case "embeddings":
                module.load_state_dict({
                    "weight": checkpoint["tok_embeddings.weight"],
                })

            # Attention
            case "normalize_attention":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention_norm.weight"],
                })
            case "w_q":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wq.weight"],
                })
            case "w_k":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wk.weight"],
                })
            case "w_v":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wv.weight"],
                })
            case "w_a":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.attention.wo.weight"],
                })

            # FFN
            case "normalize_ffn":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.ffn_norm.weight"],
                })
            case "w_g":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w1.weight"],
                })
            case "w_h":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w3.weight"],
                })
            case "w_f":
                module.load_state_dict({
                    "weight": checkpoint[f"layers.{layer}.feed_forward.w2.weight"],
                })

            # Head
            case "normalize_head":
                module.load_state_dict({
                    "weight": checkpoint["norm.weight"],
                })
            case "w_head":
                module.load_state_dict({
                    "weight": checkpoint["output.weight"],
                })
            case _:
                raise ValueError(f"Unexpected key {key}")


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


def split_heads(config: Config, x, n_heads):
    return x.view(-1, n_heads, config.d_head).transpose(-3, -2)


def combine_heads(config: Config, x):
    return (
        x.transpose(-3, -2).contiguous().view(-1, int(config.n_heads * config.d_head))
    )


def context_layers(x, *, config: Config, checkpoint):
    # Compute cos and sin rotation matrices
    r_cos, r_sin = rope_frequencies(config, len(x))

    # Attention normalization
    normalize_attention = RMSNorm(config.d_model, config.rms_norm_eps).to(config.device)

    # Query, key, value projections
    w_q = nn.Linear(
        in_features=config.d_model,
        out_features=config.n_heads * config.d_head,
        bias=False,
        device=config.device,
    )
    w_k = nn.Linear(
        in_features=config.d_model,
        out_features=config.n_kv_heads * config.d_head,
        bias=False,
        device=config.device,
    )
    w_v = nn.Linear(
        in_features=config.d_model,
        out_features=config.n_kv_heads * config.d_head,
        bias=False,
        device=config.device,
    )

    # Attention output projection
    w_a = nn.Linear(
        in_features=config.d_model,
        out_features=config.d_model,
        bias=False,
        device=config.device,
    )

    # FFN normalization
    normalize_ffn = RMSNorm(config.d_model, config.rms_norm_eps).to(config.device)

    # SwiGLU FFN
    w_h = nn.Linear(
        in_features=config.d_model,
        out_features=config.d_ffn,
        bias=False,
        device=config.device,
    )
    w_g = nn.Linear(
        in_features=config.d_model,
        out_features=config.d_ffn,
        bias=False,
        device=config.device,
    )

    # FFN output projection
    w_f = nn.Linear(
        in_features=config.d_ffn,
        out_features=config.d_model,
        bias=False,
        device=config.device,
    )

    # Apply layer logic in a loop
    for layer in range(config.n_layers):
        # Load pre-trained state for layer
        load_state(
            normalize_attention,
            "normalize_attention",
            w_q,
            "w_q",
            w_k,
            "w_k",
            w_v,
            "w_v",
            w_a,
            "w_a",
            normalize_ffn,
            "normalize_ffn",
            w_g,
            "w_g",
            w_h,
            "w_h",
            w_f,
            "w_f",
            checkpoint=checkpoint,
            layer=layer,
        )

        #
        # Attention
        #

        # Normalize attention inputs
        residual = x
        x = normalize_attention(x)

        # Project embeddings to query, key, value spaces
        q = w_q(x)
        k = w_k(x)
        v = w_v(x)

        # Split attention heads
        q = split_heads(config, q, config.n_heads)
        k = split_heads(config, k, config.n_kv_heads)
        v = split_heads(config, v, config.n_kv_heads)

        # Expand key/value groups
        reps = config.n_heads // config.n_kv_heads
        k = k.repeat_interleave(reps, dim=0)
        v = v.repeat_interleave(reps, dim=0)

        # Encode positions by rotating queries and keys
        q = rope_rotate(q, r_cos, r_sin)
        k = rope_rotate(k, r_cos, r_sin)

        # Compute masked attention bias M
        n = len(x)
        mask = torch.ones(n, n, dtype=torch.bool, device=config.device).tril(diagonal=0)
        m = torch.zeros(n, n, device=config.device).masked_fill_(
            mask.logical_not(), float("-inf")
        )

        # Compute attention for all heads in parallel
        a = softmax(q @ k.transpose(-2, -1) / np.sqrt(config.d_head) + m, dim=-1) @ v

        # Combine attention heads
        a = combine_heads(config, a)

        # Project attention representations back to model space
        a = w_a(a)

        # Combine attention representations with residual embeddings
        x = residual + a

        #
        # FFN
        #

        # Normalize FFN inputs
        residual = x
        x = normalize_ffn(x)

        # Apply SwiGLU transform
        f = silu(w_g(x)) * w_h(x)

        # Project FFN representations back to model space
        f = w_f(f)

        # Combine FFN representations with residual embeddings
        x = residual + f

    return x


def head(x, *, config: Config, checkpoint):
    # Head normalization
    normalize_head = RMSNorm(config.d_model, config.rms_norm_eps).to(config.device)

    # Output projection
    w_head = nn.Linear(
        in_features=config.d_model,
        out_features=config.vocab_size,
        bias=False,
        device=config.device,
    )

    # Load pre-trained weights
    load_state(
        normalize_head,
        "normalize_head",
        w_head,
        "w_head",
        checkpoint=checkpoint,
    )

    # Normalize head inputs
    x = normalize_head(x)

    # Use last embedding to represent the entire sequence
    x = x[-1]

    # Project outputs to token space
    x = w_head(x)

    #
    # Temperature
    #

    # Apply temperature
    x = x / config.temperature

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
    probs = probs[: config.top_k]

    #
    # Top P
    #

    # Find cutoff where cumulative probability exceeds top_p
    cumulative_mask = probs.cumsum(dim=-1) > config.top_p
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
