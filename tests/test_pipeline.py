"""Integration checks for orchestration; these synthetic tests are not training results."""

import json

import numpy as np
import pytest
import torch

from cs336_basics.experiment import ModelConfig, build_model, load_model_checkpoint
from cs336_basics.generate import generate, sampling_probabilities
from cs336_basics.prepare_tinystories import EOS, download_prefix, sha256_file
from cs336_basics.train import evaluate, run


def prepared_data(path):
    path.mkdir()
    for split, seed in (("train", 1), ("valid", 2)):
        np.save(path / f"{split}.npy", np.random.default_rng(seed).integers(0, 16, 83, dtype=np.uint16))
    for name in ("vocab.json", "merges.json"):
        (path / name).write_text("{}")
    manifest = {"vocab_size": 16,
                "tokenizer_sha256": {name: sha256_file(path / name) for name in ("vocab.json", "merges.json")},
                **{split: {"sha256": sha256_file(path / f"{split}.npy")} for split in ("train", "valid")}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def small_config():
    return {"model": {"vocab_size": 16, "context_length": 8, "d_model": 8, "num_layers": 1,
                      "num_heads": 2, "d_ff": 16},
            "training": {"steps": 4, "batch_size": 2, "warmup_steps": 1, "eval_interval": 2,
                         "eval_tokens": 32, "log_interval": 2, "device": "cpu"}}


def test_resume_matches_uninterrupted_model_optimizer_and_next_batch(tmp_path):
    data = prepared_data(tmp_path / "data")
    config = small_config()
    run(config, data, tmp_path / "continuous")
    full = torch.load(tmp_path / "continuous/last.pt", weights_only=True)
    run(config, data, tmp_path / "resumed", stop_after=2)
    run(config, data, tmp_path / "resumed", resume=tmp_path / "resumed/last.pt")
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert full["iteration"] == resumed["iteration"] == 4
    for name, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
    for index, state in full["optimizer"]["state"].items():
        for name, value in state.items():
            other = resumed["optimizer"]["state"][index][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other
    assert full["rng"]["numpy"] == resumed["rng"]["numpy"]
    torch.testing.assert_close(full["rng"]["torch"], resumed["rng"]["torch"], rtol=0, atol=0)
    loaded, _ = load_model_checkpoint(tmp_path / "resumed/last.pt")
    expected = build_model(ModelConfig(**config["model"]))
    expected.load_state_dict(resumed["model"])
    inputs = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(loaded(inputs), expected(inputs))
    with pytest.raises(ValueError, match="already has logs"):
        run(config, data, tmp_path / "resumed")
    np.save(data / "train.npy", np.zeros(83, dtype=np.uint16))
    with pytest.raises(ValueError, match="checksum"):
        run(config, data, tmp_path / "bad-data")


def test_evaluation_weights_tail_and_restores_mode_without_rng_changes():
    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(*x.shape, 16)

    model = Uniform().train()
    data = np.arange(20, dtype=np.uint16) % 16
    before = np.random.get_state()
    metrics = evaluate(model, data, 2, 8, "cpu")
    assert metrics["evaluated_tokens"] == 19
    assert metrics["loss"] == pytest.approx(np.log(16))
    assert metrics["perplexity"] == pytest.approx(16)
    assert model.training
    assert np.array_equal(before[1], np.random.get_state()[1])
    sampled = evaluate(model, data, 2, 8, "cpu", max_tokens=8)
    assert sampled["evaluated_tokens"] == 8


def test_sampling_temperature_nucleus_and_eos():
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1]))
    torch.testing.assert_close(sampling_probabilities(logits, top_p=0.7), torch.tensor([2/3, 1/3, 0.0]))
    assert sampling_probabilities(logits, temperature=0.5)[0] > sampling_probabilities(logits, temperature=2)[0]

    class AlwaysEOS(torch.nn.Module):
        context_length = 2

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            assert x.shape[-1] <= self.context_length
            return torch.tensor([[[0.0, 100.0, 0.0]]]).expand(1, x.shape[-1], 3)

    model = AlwaysEOS().train()
    assert generate(model, [0, 2, 0], temperature=0, eos_token_id=1) == [0, 2, 0, 1]
    assert model.training
    assert len(generate(model, [0], max_new_tokens=3, temperature=0, eos_token_id=None)) == 4
    for temperature, top_p in ((0, 1), (1, 0), (1, 1.1)):
        with pytest.raises(ValueError):
            sampling_probabilities(logits, temperature, top_p)


def test_streamed_download_stops_at_complete_document(tmp_path):
    text = ("first\n" + EOS + "\nsecond\n" + EOS + "\nthird" + EOS).encode()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            for offset in range(0, len(text), 7):
                yield text[offset:offset + 7]

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    destination = tmp_path / "raw.txt"
    download_prefix(Session(), "https://example.invalid", destination, 2)
    assert destination.read_bytes() == ("first\n" + EOS + "\nsecond\n" + EOS).encode()
