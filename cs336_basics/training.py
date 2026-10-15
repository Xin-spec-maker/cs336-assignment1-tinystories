"""Training primitives for the CS336 Transformer language model."""

import math
from collections.abc import Iterable
from os import PathLike
from typing import BinaryIO

import numpy as np
import torch
from torch import Tensor, nn


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Mean next-token loss for logits (..., vocab_size) and targets (...).

    Keep half-precision inputs' loss computation in float32 to avoid overflow.
    Working with shifted logits also avoids taking log of a tiny probability.
    """
    if inputs.shape[:-1] != targets.shape:
        raise ValueError("Target shape must match the logits' batch dimensions")
    logits = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_partition = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_partition - target_logits).mean()


@torch.no_grad()
def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """Clip the combined L2 norm of all available gradients, in place."""
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be nonnegative")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    norms = [
        torch.linalg.vector_norm(
            grad, dtype=torch.float32 if grad.dtype in (torch.float16, torch.bfloat16) else grad.dtype,
        ).to(gradients[0].device)
        for grad in gradients
    ]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in gradients:
            grad.mul_(scale.to(device=grad.device, dtype=grad.dtype))


class AdamW(torch.optim.Optimizer):
    """AdamW following Algorithm 1 of the 2026 handout.

    Decay uses the original learning rate and precedes the moment update.
    Bias correction is applied to the step size; epsilon is added to sqrt(v).
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            if not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError("Learning rate must be finite and nonnegative")
            if len(group["betas"]) != 2 or not all(0 <= beta < 1 for beta in group["betas"]):
                raise ValueError("AdamW requires two beta values in [0, 1)")
            if not math.isfinite(group["eps"]) or group["eps"] < 0:
                raise ValueError("epsilon must be finite and nonnegative")
            if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
                raise ValueError("Weight decay must be finite and nonnegative")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse or parameter.is_complex():
                    raise RuntimeError("AdamW supports dense real-valued gradients only")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                moment, variance = state["exp_avg"], state["exp_avg_sq"]
                parameter.mul_(1 - lr * weight_decay)
                moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                parameter.addcdiv_(moment, variance.sqrt().add_(eps), value=-step_size)
        return loss


def get_lr_cosine_schedule(
    it: int, max_learning_rate: float, min_learning_rate: float,
    warmup_iters: int, cosine_cycle_iters: int,
) -> float:
    """Linear warmup, cosine decay until the specified final iteration, then a floor."""
    if it < 0 or warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError("Require it >= 0 and 0 <= warmup_iters < cosine_cycle_iters")
    if not 0 <= min_learning_rate <= max_learning_rate:
        raise ValueError("Require 0 <= min_learning_rate <= max_learning_rate")
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Uniformly sample valid windows, loading only batch-sized slices of a memmap."""
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError("dataset must be a one-dimensional integer array")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    if len(dataset) <= context_length:
        raise ValueError("dataset needs at least context_length + 1 tokens")
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length)
    inputs = torch.tensor(dataset[indices], dtype=torch.long, device=device)
    targets = torch.tensor(dataset[indices + 1], dtype=torch.long, device=device)
    return inputs, targets


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
    out: str | PathLike | BinaryIO,
) -> None:
    """Save model weights, optimizer moments/hyperparameters, and completed iterations."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(
    src: str | PathLike | BinaryIO, model: nn.Module, optimizer: torch.optim.Optimizer,
) -> int:
    """Restore into existing objects and return the saved iteration number.

    Loading via CPU allows checkpoints from another device to be read; the
    state-dict loaders move tensors to the model/optimizer parameters' devices.
    """
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
