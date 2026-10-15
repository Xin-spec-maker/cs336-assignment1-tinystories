"""Temperature/top-p decoding. Run with python -m cs336_basics.generate --help."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .bpe import BPETokenizer
from .experiment import load_model_checkpoint
from .model import softmax


def sampling_probabilities(logits, temperature=1.0, top_p=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive for sampling probabilities")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not torch.isfinite(logits).all():
        raise ValueError("Model logits must be finite")
    work = logits.float()
    probabilities = softmax((work - work.amax(dim=-1, keepdim=True)) / temperature)
    if top_p == 1:
        return probabilities
    sorted_probs, indices = probabilities.sort(dim=-1, descending=True)
    # Keep the smallest prefix reaching p, including the token crossing p.
    preceding_mass = sorted_probs.cumsum(dim=-1) - sorted_probs
    kept = sorted_probs.masked_fill(preceding_mass >= top_p, 0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probabilities).scatter(-1, indices, kept)


@torch.inference_mode()
def generate(model, prompt_ids, max_new_tokens=256, temperature=1.0, top_p=0.9,
             eos_token_id=None, generator=None):
    """Return prompt plus completion; crop only the model's input context window."""
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token")
    if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("max_new_tokens and temperature must be nonnegative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    ids = list(prompt_ids)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            inputs = torch.tensor([ids[-model.context_length:]], device=device, dtype=torch.long)
            logits = model(inputs)[0, -1]
            if not torch.isfinite(logits).all():
                raise ValueError("Model logits must be finite")
            if temperature == 0:
                token = int(logits.argmax().item())
            else:
                probs = sampling_probabilities(logits, temperature, top_p)
                token = int(torch.multinomial(probs, 1, generator=generator).item())
            ids.append(token)
            if token == eos_token_id:
                break
    finally:
        model.train(was_training)
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 selects greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, help="Save exact token IDs, sampling settings and checkpoint hash")
    args = parser.parse_args(argv)
    if args.output_json is not None and args.output_json.exists():
        parser.error("Output already exists; choose another path to preserve previous samples")
    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, args.special_token or [args.eos_token])
    model, config = load_model_checkpoint(args.checkpoint, args.device)
    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for name, path in (("vocab.json", args.vocab), ("merges.json", args.merges)):
        expected = metadata.get("tokenizer_sha256", {}).get(name)
        with open(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f"Tokenizer does not match the training checkpoint: {name}")
    if set(tokenizer.vocab) != set(range(config.vocab_size)):
        raise ValueError("Tokenizer IDs must cover exactly the model vocabulary")
    eos = [index for index, value in tokenizer.vocab.items() if value == args.eos_token.encode()]
    if not eos:
        raise ValueError("EOS token is missing from the tokenizer vocabulary")
    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    prompt_ids = tokenizer.encode(args.prompt)
    start = time.perf_counter()
    ids = generate(model, prompt_ids, args.max_new_tokens,
                   args.temperature, args.top_p, eos[0], rng)
    seconds = time.perf_counter() - start
    text = tokenizer.decode(ids)
    print(text)
    if args.output_json is not None:
        completion = ids[len(prompt_ids):]
        with open(args.checkpoint, "rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        result = {"prompt": args.prompt, "text": text, "prompt_ids": prompt_ids, "completion_ids": completion,
                  "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                  "max_new_tokens": args.max_new_tokens, "generated_tokens": len(completion),
                  "eos_token_id": eos[0], "stop_reason": "eos" if completion and completion[-1] == eos[0] else "length",
                  "device": args.device, "generation_seconds": seconds, "checkpoint_sha256": checkpoint_hash,
                  "checkpoint_iteration": metadata.get("iteration"), "torch_version": str(torch.__version__)}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
