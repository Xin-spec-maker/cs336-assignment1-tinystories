"""Additional numerical, optimizer, data, and resume checks; no official tests are changed."""

import numpy as np
import pytest
import torch

from cs336_basics.model import Linear
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def test_cross_entropy_batch_dimensions_and_gradient():
    torch.manual_seed(336)
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 5, (2, 3))
    actual = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
    torch.testing.assert_close(actual, expected)
    actual_gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.autograd.gradcheck(lambda value: cross_entropy(value, targets), (logits,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cross_entropy_extreme_logits(dtype):
    logits = torch.tensor([[60000.0, -60000.0]], dtype=dtype, requires_grad=True)
    loss = cross_entropy(logits, torch.tensor([1]))
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, logits.float()[0, 0] - logits.float()[0, 1])
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[1.0, -1.0]], dtype=dtype))


def test_clipping_uses_global_norm_and_preserves_parameters():
    a, b, frozen = (torch.nn.Parameter(torch.tensor([value])) for value in (1.0, 2.0, 3.0))
    a.grad, b.grad = torch.tensor([3.0]), torch.tensor([4.0])
    original_gradient = a.grad
    gradient_clipping(iter([a, frozen, b]), 2.5)
    assert a.grad is original_gradient
    torch.testing.assert_close(a.grad, torch.tensor([1.5]))
    torch.testing.assert_close(b.grad, torch.tensor([2.0]))
    assert frozen.grad is None
    assert [p.item() for p in (a, b, frozen)] == [1.0, 2.0, 3.0]


def test_clipping_empty_zero_and_below_threshold():
    gradient_clipping([], 1.0)
    p = torch.nn.Parameter(torch.ones(3))
    gradient_clipping([p], 1.0)
    p.grad = torch.tensor([0.1, 0.2, 0.3])
    expected = p.grad.clone()
    gradient_clipping([p], 1.0)
    assert torch.equal(p.grad, expected)
    gradient_clipping([p], 0.0)
    assert p.grad.count_nonzero() == 0


def test_clipping_low_precision_large_gradients():
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.float16))
    p.grad = torch.full_like(p, 40000)
    gradient_clipping([p], 1.0)
    assert torch.isfinite(p.grad).all()
    assert 0.99 < p.grad.float().norm() < 1.01


def test_adamw_handout_epsilon_and_decay_order():
    p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = AdamW([p], lr=0.1, betas=(0.5, 0.75), eps=0.2, weight_decay=0.1)
    p.grad = torch.tensor([4.0], dtype=torch.float64)
    optimizer.step()
    # First-step m=2, v=4, corrected learning rate=0.1; decay gives 1.98.
    torch.testing.assert_close(p.detach(), torch.tensor([1.98 - 0.2 / 2.2], dtype=torch.float64))
    assert optimizer.state[p]["step"] == 1
    assert p.grad.item() == 4.0


def test_adamw_parameter_groups_and_missing_gradient():
    a, b, frozen = (torch.nn.Parameter(torch.tensor([2.0])) for _ in range(3))
    optimizer = AdamW([
        {"params": [a], "lr": 0.1},
        {"params": [b, frozen], "lr": 0.2},
    ], betas=(0.0, 0.0), eps=0, weight_decay=0.5)
    a.grad, b.grad = torch.zeros_like(a), torch.ones_like(b)
    # Nonzero epsilon is needed for the all-zero gradient's denominator.
    optimizer.param_groups[0]["eps"] = 1e-8
    optimizer.step()
    torch.testing.assert_close(a.detach(), torch.tensor([1.9]))
    torch.testing.assert_close(b.detach(), torch.tensor([1.6]))
    assert frozen.item() == 2.0 and frozen not in optimizer.state
    a.grad = None
    optimizer.step()
    assert optimizer.state[a]["step"] == 1
    assert optimizer.state[b]["step"] == 2


def test_adamw_closure_runs_with_gradients_enabled():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = AdamW([p], lr=0.1)

    def closure():
        optimizer.zero_grad()
        loss = p.square().sum()
        loss.backward()
        return loss

    assert optimizer.step(closure).item() == 4.0
    assert p.item() < 2.0


def test_schedule_zero_warmup_and_boundaries():
    def schedule(it):
        return get_lr_cosine_schedule(it, 1.0, 0.1, 0, 10)

    assert schedule(0) == 1.0
    assert schedule(5) == pytest.approx(0.55)
    assert schedule(10) == schedule(100) == 0.1
    values = [get_lr_cosine_schedule(it, 0.2, 0.02, 3, 9) for it in range(12)]
    assert values[0] == 0.0 and values[3] == 0.2 and values[9] == 0.02
    assert values[:4] == sorted(values[:4])
    assert values[3:] == sorted(values[3:], reverse=True)
    with pytest.raises(ValueError):
        get_lr_cosine_schedule(0, 1, 0, 3, 3)


def test_batch_single_valid_window_and_integer_dtype():
    dataset = np.array([7, 11, 13, 19], dtype=np.uint16)
    inputs, targets = get_batch(dataset, 3, 3, "cpu")
    assert inputs.dtype == targets.dtype == torch.long
    torch.testing.assert_close(inputs, torch.tensor([[7, 11, 13]]).expand(3, 3))
    torch.testing.assert_close(targets, torch.tensor([[11, 13, 19]]).expand(3, 3))
    inputs.zero_()
    assert dataset.tolist() == [7, 11, 13, 19]
    assert targets[0, 0] == 11


def test_batch_readonly_memmap_and_reproducibility(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(1024, dtype=np.uint16))
    dataset = np.load(path, mmap_mode="r")
    np.random.seed(336)
    inputs, targets = get_batch(dataset, 4, 7, "cpu")
    np.random.seed(336)
    repeated, _ = get_batch(dataset, 4, 7, "cpu")
    assert torch.equal(inputs, repeated)
    assert torch.equal(inputs + 1, targets)
    assert inputs.device.type == targets.device.type == "cpu"


@pytest.mark.parametrize("dataset,batch_size,context_length", [
    (np.arange(3), 1, 3),
    (np.arange(4), 0, 2),
    (np.arange(4), 1, 0),
    (np.ones((2, 3), dtype=np.int64), 1, 2),
    (np.ones(4), 1, 2),
])
def test_batch_rejects_invalid_inputs(dataset, batch_size, context_length):
    with pytest.raises(ValueError):
        get_batch(dataset, batch_size, context_length, "cpu")


@pytest.mark.parametrize("use_stream", [False, True])
def test_checkpoint_resumes_identical_next_update(tmp_path, use_stream):
    import io

    torch.manual_seed(336)
    model = Linear(3, 2, dtype=torch.float64)
    optimizer = AdamW(model.parameters(), lr=0.03, betas=(0.7, 0.9), weight_decay=0.2)
    inputs = torch.randn(4, 3, dtype=torch.float64)
    targets = torch.tensor([0, 1, 1, 0])

    def step(current_model, current_optimizer):
        current_optimizer.zero_grad()
        loss = cross_entropy(current_model(inputs), targets)
        loss.backward()
        gradient_clipping(current_model.parameters(), 1.0)
        current_optimizer.step()

    for _ in range(3):
        step(model, optimizer)
    destination = io.BytesIO() if use_stream else tmp_path / "checkpoint.pt"
    save_checkpoint(model, optimizer, 3, destination)
    if use_stream:
        assert not destination.closed
        destination.seek(0)
    restored = Linear(3, 2, dtype=torch.float64)
    restored_optimizer = AdamW(restored.parameters(), lr=0.001)
    assert load_checkpoint(destination, restored, restored_optimizer) == 3
    assert restored_optimizer.param_groups[0]["lr"] == 0.03
    step(model, optimizer)
    step(restored, restored_optimizer)
    torch.testing.assert_close(model.weight, restored.weight, rtol=0, atol=0)
    for key, value in optimizer.state[model.weight].items():
        restored_value = restored_optimizer.state[restored.weight][key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, restored_value, rtol=0, atol=0)
        else:
            assert value == restored_value
