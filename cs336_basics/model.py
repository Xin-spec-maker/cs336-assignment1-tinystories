"""Transformer language model built from basic PyTorch tensor operations."""

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """Bias-free projection; weights have shape (out_features, in_features)."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """Look up a learned vector for each integer token ID."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize the last dimension using at least float32 accumulation."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight).to(x.dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Gated feed-forward network with independent activation and gate branches."""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil((8 * d_model / 3) / 64) * 64
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Stable softmax along any axis, with float32 work for low-precision input."""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    exponentials = torch.exp(work - work.amax(dim=dim, keepdim=True))
    return (exponentials / exponentials.sum(dim=dim, keepdim=True)).to(x.dtype)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """Batched attention; True mask entries are visible. Fully masked rows return zero."""
    query = Q.float() if Q.dtype in (torch.float16, torch.bfloat16) else Q
    key = K.to(query.dtype)
    scores = (query @ key.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
        # Avoid -inf - -inf in softmax, including in backward, for empty rows.
        scores = torch.where(mask.any(dim=-1, keepdim=True), scores, 0.0)
    probabilities = softmax(scores, dim=-1)
    if mask is not None:
        probabilities = probabilities.masked_fill(~mask, 0.0)
    return (probabilities @ V.to(probabilities.dtype)).to(V.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent coordinate pairs using fixed position-dependent angles."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k <= 0 or d_k % 2:
            raise ValueError("RoPE requires a positive even head dimension")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("RoPE theta and maximum sequence length must be positive")
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = torch.arange(max_seq_len, device=device, dtype=torch.float32)[:, None] * frequencies
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError("Input's final dimension must match RoPE d_k")
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalMultiHeadSelfAttention(nn.Module):
    """Three batched Q/K/V projections, optional RoPE, and causal attention."""

    def __init__(
        self, d_model: int, num_heads: int,
        max_seq_len: int | None = None, theta: float | None = None,
        device=None, dtype=None,
    ):
        super().__init__()
        if num_heads <= 0 or d_model <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if theta is not None:
            if max_seq_len is None:
                raise ValueError("max_seq_len is required when using RoPE")
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]

        def split_heads(projected: Tensor) -> Tensor:
            return projected.reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-3, -2)

        query = split_heads(self.q_proj(x))
        key = split_heads(self.k_proj(x))
        value = split_heads(self.v_proj(x))
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            # Insert a broadcast axis for heads; each head uses the same positions.
            positions = token_positions.unsqueeze(-2)
            query = self.rope(query, positions)
            key = self.rope(key, positions)
        indices = torch.arange(sequence_length, device=x.device)
        causal_mask = indices[:, None] >= indices[None, :]
        attended = scaled_dot_product_attention(query, key, value, causal_mask)
        combined = attended.transpose(-3, -2).reshape(x.shape)
        return self.output_proj(combined)


class TransformerBlock(nn.Module):
    """Pre-norm attention and SwiGLU, each followed by a residual addition."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int | None,
        max_seq_len: int, theta: float = 10000.0, device=None, dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    """Token IDs -> embeddings -> Transformer blocks -> RMSNorm -> logits.

    Input and output embeddings are independent. Softmax belongs in the loss
    or sampling code, not in this forward pass.
    """

    def __init__(
        self, vocab_size: int, context_length: int, d_model: int,
        num_layers: int, num_heads: int, d_ff: int | None = None,
        rope_theta: float = 10000.0, device=None, dtype=None,
    ):
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive")
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        if not 0 < token_ids.shape[-1] <= self.context_length:
            raise ValueError("Sequence length must be between 1 and context_length")
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
