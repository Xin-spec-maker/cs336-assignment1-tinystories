"""Model configuration and the architecture ablations specified in the handout."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model import Linear, TransformerBlock, TransformerLM, silu


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    norm_style: str = "pre"
    position_encoding: str = "rope"
    activation: str = "swiglu"

    def validate(self):
        for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.norm_style not in ("pre", "post", "none"):
            raise ValueError("norm_style must be pre, post, or none")
        if self.position_encoding not in ("rope", "none"):
            raise ValueError("position_encoding must be rope or none")
        if self.activation not in ("swiglu", "silu"):
            raise ValueError("activation must be swiglu or silu")
        if self.position_encoding == "rope" and (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")


class Identity(nn.Module):
    def forward(self, x):
        return x


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model, device=None):
        super().__init__()
        # The handout uses 4*d_model to approximately match SwiGLU parameters.
        self.w1 = Linear(d_model, 4 * d_model, device=device)
        self.w2 = Linear(4 * d_model, d_model, device=device)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))


class PostNormBlock(TransformerBlock):
    def forward(self, x, token_positions=None):
        x = self.ln1(x + self.attn(x, token_positions))
        return self.ln2(x + self.ffn(x))


def build_model(config: ModelConfig, device="cpu") -> TransformerLM:
    config.validate()
    kwargs = {key: value for key, value in asdict(config).items()
              if key not in ("norm_style", "position_encoding", "activation")}
    # NoPE avoids constructing an unused RoPE module (including its even-width restriction).
    if config.position_encoding == "none":
        kwargs["rope_theta"] = None
    model = TransformerLM(**kwargs, device=device)
    for index, block in enumerate(model.layers):
        if config.norm_style == "post":
            block = PostNormBlock(
                config.d_model, config.num_heads, config.d_ff, config.context_length,
                kwargs["rope_theta"], device=device,
            )
            model.layers[index] = block
        elif config.norm_style == "none":
            block.ln1, block.ln2 = Identity(), Identity()
        if config.activation == "silu":
            block.ffn = SiLUFeedForward(config.d_model, device=device)
    if config.norm_style == "none":
        model.ln_final = Identity()
    return model


def load_model_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" not in checkpoint:
        raise ValueError("Generation needs a training-run checkpoint containing model_config")
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    return model, config
