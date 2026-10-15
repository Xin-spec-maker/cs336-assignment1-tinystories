"""Train and evaluate the existing from-scratch model; JSONL logs and resumable checkpoints."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .experiment import ModelConfig, build_model
from .prepare_tinystories import sha256_file
from .training import AdamW, cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 8
    learning_rate: float = 0.001
    min_learning_rate: float = 0.0001
    warmup_steps: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_tokens: int = 32768
    log_interval: int = 25
    seed: int = 336
    device: str = "mps"

    def validate(self):
        if min(self.steps, self.batch_size, self.eval_interval, self.eval_tokens, self.log_interval) <= 0:
            raise ValueError("steps, batch size and log/evaluation intervals must be positive")
        get_lr_cosine_schedule(0, self.learning_rate, self.min_learning_rate, self.warmup_steps, self.steps)
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, max_tokens=None):
    """Token-weighted NLL; deterministic nonoverlapping windows, no RNG mutation.

    Periodic evaluations use equally spaced whole windows across the split.
    With max_tokens=None, score every next-token target, including the final tail.
    """
    if len(data) < 2:
        raise ValueError("Evaluation requires at least two tokens")
    total_windows = (len(data) - 1) // context_length
    count = total_windows if max_tokens is None else min(total_windows, max_tokens // context_length)
    if max_tokens is not None and count == 0:
        raise ValueError("eval_tokens and dataset must allow a full context window")
    selected = np.linspace(0, total_windows - 1, count, dtype=np.int64) if count else []
    was_training = model.training
    model.eval()
    nll, tokens = 0.0, 0
    try:
        for offset in range(0, count, batch_size):
            starts = selected[offset:offset + batch_size] * context_length
            indices = starts[:, None] + np.arange(context_length)
            inputs = torch.tensor(data[indices], dtype=torch.long, device=device)
            targets = torch.tensor(data[indices + 1], dtype=torch.long, device=device)
            value = cross_entropy(model(inputs), targets).item()
            nll += value * targets.numel()
            tokens += targets.numel()
        tail = (len(data) - 1) % context_length
        if max_tokens is None and tail:
            start = total_windows * context_length
            inputs = torch.tensor(np.array(data[start:-1]), dtype=torch.long, device=device)[None, :]
            targets = torch.tensor(np.array(data[start + 1:]), dtype=torch.long, device=device)[None, :]
            nll += cross_entropy(model(inputs), targets).item() * tail
            tokens += tail
    finally:
        model.train(was_training)
    loss = nll / tokens
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite validation loss")
    return {"loss": loss, "perplexity": math.exp(loss), "evaluated_tokens": tokens}


def rng_state(device):
    state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
              "torch": torch.get_rng_state()}
    if str(device).startswith("mps"):
        result["mps"] = torch.mps.get_rng_state()
    elif str(device).startswith("cuda"):
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if str(device).startswith("mps") and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    elif str(device).startswith("cuda") and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, destination):
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_data(data_dir, model_config):
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if model_config.vocab_size != manifest["vocab_size"]:
        raise ValueError("Model vocabulary differs from prepared tokenizer")
    arrays = {}
    for split in ("train", "valid"):
        path = data_dir / f"{split}.npy"
        if sha256_file(path) != manifest[split]["sha256"]:
            raise ValueError(f"Data checksum mismatch: {split}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 1 or array.dtype != np.uint16 or len(array) <= model_config.context_length:
            raise ValueError(f"Invalid token data: {split}")
        if int(array.max()) >= model_config.vocab_size:
            raise ValueError(f"Out-of-vocabulary token in {split}")
        arrays[split] = array
    for name, expected in manifest["tokenizer_sha256"].items():
        if sha256_file(data_dir / name) != expected:
            raise ValueError(f"Tokenizer checksum mismatch: {name}")
    return arrays, manifest, sha256_file(manifest_path)


def run(config, data_dir, output_dir, resume=None, stop_after=None):
    model_config, train_config = ModelConfig(**config["model"]), TrainConfig(**config["training"])
    model_config.validate()
    train_config.validate()
    device = train_config.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; choose cpu explicitly")
    data, manifest, data_hash = load_data(Path(data_dir), model_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    if log_path.exists() and resume is None:
        raise ValueError("Run already has logs; resume explicitly or use a new output directory")
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    # Initialize on CPU for consistent initial weights across CPU/MPS; train in float32.
    model = build_model(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate,
                      betas=(train_config.beta1, train_config.beta2), weight_decay=train_config.weight_decay)
    iteration, elapsed_before, best_loss = 0, 0.0, float("inf")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != asdict(model_config) or checkpoint["train_config"] != asdict(train_config):
            raise ValueError("Resume requires identical model/training configuration, including the LR schedule")
        if checkpoint["data_manifest_sha256"] != data_hash:
            raise ValueError("Resume dataset/tokenizer differs from checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        iteration = checkpoint["iteration"]
        elapsed_before, best_loss = checkpoint["elapsed_seconds"], checkpoint["best_validation_loss"]
        restore_rng(checkpoint["rng"], device)
        if log_path.exists():
            recorded_steps = [json.loads(line)["step"] for line in log_path.read_text().splitlines()]
            if max(recorded_steps, default=0) > iteration:
                raise ValueError("Logs extend beyond checkpoint; use a new output directory for this resume")
    final_step = train_config.steps if stop_after is None else min(stop_after, train_config.steps)
    if final_step <= iteration:
        raise ValueError("No training steps remain")
    full_config = {"model": asdict(model_config), "training": asdict(train_config)}
    (output_dir / "config.json").write_text(json.dumps(full_config, indent=2) + "\n")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    synchronize(device)
    start = time.perf_counter()

    def record(kind, step, **values):
        row = {"event": kind, "step": step, "elapsed_seconds": elapsed_before + time.perf_counter() - start,
               **values}
        line = json.dumps(row, allow_nan=False)
        with log_path.open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
        return row

    def validation(step, full=False):
        metrics = {}
        for split in ("train", "valid"):
            # Full final validation; train metric remains a fixed diagnostic subset.
            cap = None if full and split == "valid" else train_config.eval_tokens
            metrics[split] = evaluate(model, data[split], train_config.batch_size,
                                      model_config.context_length, device, cap)
        record("validation_full" if full else "validation", step, **metrics)
        return metrics

    def save(step, metrics):
        nonlocal best_loss
        improved = metrics["valid"]["loss"] < best_loss
        best_loss = min(best_loss, metrics["valid"]["loss"])
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": step,
                   "model_config": asdict(model_config), "train_config": asdict(train_config),
                   "data_manifest_sha256": data_hash, "tokenizer_sha256": manifest["tokenizer_sha256"],
                   "rng": rng_state(device), "best_validation_loss": best_loss,
                   "elapsed_seconds": elapsed_before + time.perf_counter() - start, "metrics": metrics}
        atomic_save(payload, output_dir / "last.pt")
        if improved:
            atomic_save(payload, output_dir / "best.pt")

    record("resume" if resume else "start", iteration, parameters=parameter_count, device=device,
           dtype="float32", torch_version=str(torch.__version__), data_manifest_sha256=data_hash)
    if not resume:
        validation(0)
    losses = []
    model.train()
    for step in range(iteration + 1, final_step + 1):
        learning_rate = get_lr_cosine_schedule(step, train_config.learning_rate, train_config.min_learning_rate,
                                               train_config.warmup_steps, train_config.steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        inputs, targets = get_batch(data["train"], train_config.batch_size, model_config.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        gradient_clipping(model.parameters(), train_config.grad_clip)
        optimizer.step()
        losses.append(value)
        if step == 1 or step % train_config.log_interval == 0 or step == final_step:
            synchronize(device)
            record("train", step, loss=sum(losses) / len(losses), mean_over_steps=len(losses),
                   learning_rate=learning_rate, tokens_seen=step * train_config.batch_size * model_config.context_length)
            losses = []
        if step % train_config.eval_interval == 0 or step == final_step:
            metrics = validation(step)
            save(step, metrics)
    final_metrics = validation(final_step, full=True)
    synchronize(device)
    # Include the final evaluation in resume timing without comparing the full
    # split's score against the best checkpoint's fixed-window score.
    checkpoint = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)
    checkpoint["full_validation"] = final_metrics["valid"]
    checkpoint["elapsed_seconds"] = elapsed_before + time.perf_counter() - start
    atomic_save(checkpoint, output_dir / "last.pt")
    summary = record("complete" if final_step == train_config.steps else "paused", final_step,
                     parameters=parameter_count, train=final_metrics["train"], valid=final_metrics["valid"],
                     tokens_seen=final_step * train_config.batch_size * model_config.context_length,
                     checkpoint="last.pt", data_manifest_sha256=data_hash)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="Stop after this global step without changing the LR schedule")
    args = parser.parse_args(argv)
    run(json.loads(args.config.read_text()), args.data_dir, args.output_dir, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
