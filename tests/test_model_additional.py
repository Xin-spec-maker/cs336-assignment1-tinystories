"""Model invariants beyond the unmodified official snapshot tests."""

import math

import pytest
import torch

from cs336_basics.model import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerLM,
    scaled_dot_product_attention,
    silu,
    softmax,
)


def test_parameter_initialization():
    torch.manual_seed(336)
    linear = Linear(128, 64)
    embedding = Embedding(256, 64)
    std = math.sqrt(2 / (128 + 64))
    assert set(dict(linear.named_parameters())) == {"weight"}
    assert linear.weight.abs().max() <= 3 * std
    assert abs(linear.weight.std().item() / std - 1) < 0.07
    assert embedding.weight.abs().max() <= 3
    assert abs(embedding.weight.std().item() - 1) < 0.07


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_basic_layers_batch_shapes_and_gradients(dtype):
    linear = Linear(4, 6, device="cpu", dtype=dtype)
    x = torch.randn(2, 3, 5, 4, dtype=dtype, requires_grad=True)
    y = linear(x)
    assert y.shape == (2, 3, 5, 6)
    assert y.dtype == dtype
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None and torch.isfinite(linear.weight.grad).all()

    embedding = Embedding(8, 4, dtype=dtype)
    ids = torch.tensor([[2, 2, 5]])
    embedded = embedding(ids)
    assert embedded.shape == (1, 3, 4)
    assert embedded.dtype == dtype
    embedded.sum().backward()
    torch.testing.assert_close(embedding.weight.grad[2], torch.full((4,), 2.0, dtype=dtype))
    assert embedding.weight.grad[0].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_low_precision_does_not_overflow(dtype):
    module = RMSNorm(4, dtype=dtype)
    x = torch.tensor([[1000.0, -2000.0, 3000.0, -4000.0]], dtype=dtype, requires_grad=True)
    output = module(x)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert output.abs().max() > 1
    torch.testing.assert_close(output.float().square().mean(), torch.tensor(1.0), atol=0.01, rtol=0.01)
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(module.weight.detach(), torch.ones(4, dtype=dtype))


def test_silu_extreme_inputs_and_swiglu_backward():
    x = torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True)
    silu(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.0, 0.5, 1.0]))
    module = SwiGLU(12)
    assert module.d_ff % 64 == 0
    output = module(torch.randn(2, 3, 12))
    assert output.shape == (2, 3, 12)
    output.square().sum().backward()
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_softmax_large_logits_arbitrary_axis_and_backward():
    x = (torch.randn(2, 3, 4, dtype=torch.float64) + 10000).requires_grad_()
    output = softmax(x, dim=1)
    torch.testing.assert_close(output.sum(dim=1), torch.ones(2, 4, dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda value: softmax(value, 1), (x,))
    half = torch.tensor([[60000.0, 60000.0, -60000.0]], dtype=torch.float16)
    torch.testing.assert_close(softmax(half), torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float16))


def test_attention_mask_broadcast_and_fully_masked_row_backward():
    torch.manual_seed(336)
    q = torch.randn(2, 3, 4, 6, requires_grad=True)
    k = torch.randn(2, 3, 5, 6, requires_grad=True)
    v = torch.randn(2, 3, 5, 7, requires_grad=True)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[1:, :2] = True
    output = scaled_dot_product_attention(q, k, v, mask)
    assert output.shape == (2, 3, 4, 7)
    assert output[..., 0, :].count_nonzero() == 0
    changed = v.detach().clone()
    changed[..., 2:, :] += 10000
    torch.testing.assert_close(output, scaled_dot_product_attention(q, k, changed, mask))
    output.square().sum().backward()
    for tensor in (q, k, v):
        assert torch.isfinite(tensor.grad).all()
    assert k.grad[..., 2:, :].count_nonzero() == 0
    assert v.grad[..., 2:, :].count_nonzero() == 0


def test_rope_norm_and_relative_position_invariance():
    rope = RotaryPositionalEmbedding(10000, 8, 32)
    x, y = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8)
    positions = torch.tensor([[[0, 1, 2, 3]], [[3, 4, 5, 6]]])
    rotated = rope(x, positions)
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
    dots = (rope(x, positions) * rope(y, positions + 1)).sum(-1)
    shifted_dots = (rope(x, positions + 7) * rope(y, positions + 8)).sum(-1)
    torch.testing.assert_close(dots, shifted_dots, atol=2e-6, rtol=2e-5)
    assert not list(rope.parameters())
    assert not rope.state_dict()


@pytest.mark.parametrize("use_rope", [False, True])
def test_causal_attention_cannot_see_future(use_rope):
    module = CausalMultiHeadSelfAttention(16, 4, 8, 10000 if use_rope else None)
    x = torch.randn(2, 3, 6, 16, requires_grad=True)
    positions = torch.arange(6).expand(2, 3, 6)
    output = module(x, positions)
    changed = x.detach().clone()
    changed[..., 3:, :] += 100
    torch.testing.assert_close(output[..., :3, :], module(changed, positions)[..., :3, :])
    output[..., :3, :].sum().backward()
    assert x.grad[..., 3:, :].count_nonzero() == 0


def test_lm_prefix_consistency_and_all_parameter_gradients():
    torch.manual_seed(336)
    model = TransformerLM(32, 8, 16, 2, 4, 32)
    ids = torch.randint(0, 32, (2, 3, 6))
    logits = model(ids)
    assert logits.shape == (2, 3, 6, 32)
    torch.testing.assert_close(logits[..., :3, :], model(ids[..., :3]))
    changed = ids.clone()
    changed[..., 3:] = (changed[..., 3:] + 1) % 32
    torch.testing.assert_close(logits[..., :3, :], model(changed)[..., :3, :])
    assert model.token_embeddings.weight is not model.lm_head.weight
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.count_nonzero() > 0, name


def test_lm_state_roundtrip_and_context_validation():
    import io

    model = TransformerLM(16, 8, 8, 1, 2, 16, dtype=torch.float64)
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids)
    assert expected.dtype == torch.float64
    state = io.BytesIO()
    torch.save(model.state_dict(), state)
    state.seek(0)
    restored = TransformerLM(16, 8, 8, 1, 2, 16, dtype=torch.float64)
    restored.load_state_dict(torch.load(state, weights_only=True))
    torch.testing.assert_close(restored(ids), expected, rtol=0, atol=0)
    assert not any("cos" in name or "sin" in name for name in model.state_dict())
    with pytest.raises(ValueError, match="context_length"):
        model(torch.zeros(1, 9, dtype=torch.long))
    with pytest.raises(ValueError, match="context_length"):
        model(torch.zeros(1, 0, dtype=torch.long))


def test_attention_rejects_invalid_head_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        CausalMultiHeadSelfAttention(10, 3)
    with pytest.raises(ValueError, match="even"):
        CausalMultiHeadSelfAttention(12, 4, 8, 10000)
