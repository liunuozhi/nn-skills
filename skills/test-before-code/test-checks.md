# Test Checks for NN Components

**Load this reference when:** writing or reviewing tests for nn.Module subclasses, model components, or data pipelines.

## Overview

Five checks catch bugs that code review misses — each targets a silent failure mode. Audit existing test coverage against these checks and fill gaps.

**Core principle:** These are the recommended checks to cover, not the only tests you'll write. Start here, then add domain-specific tests as needed.

## Process

1. **Read** existing tests for the target module/feature
2. **Map** which of the 5 checks are already covered
3. **Report** coverage table to user (check × target, ✓/✗/NA)
4. **Write** tests for missing checks (skip NA)

## Which Checks Apply

✓ = should test, NA = not applicable

| Target                | Shape | Grad | Overfit | Batch-indep | Data |
|-----------------------|-------|------|---------|-------------|------|
| nn.Module (layer)     | ✓     | ✓    | NA      | ✓           | NA   |
| Full model + loss     | ✓     | ✓    | ✓       | ✓           | NA   |
| Dataset / DataLoader  | NA    | NA   | NA      | NA          | ✓    |

Report existing coverage as ✓ (covered), ✗ (missing), or NA (not applicable).
Only write tests for ✗ cells — skip NA entirely.

## Setup: conftest.py

Customize per project. Tests pull from these fixtures.

```python
# conftest.py
import pytest, torch

@pytest.fixture
def model():
    return MyModel()

@pytest.fixture
def input_shape():
    return (3, 32, 32)  # single sample, no batch dim

@pytest.fixture
def output_shape():
    return (10,)

@pytest.fixture
def sample_batch(input_shape):
    return torch.randn(4, *input_shape)

@pytest.fixture
def target_batch(output_shape):
    # classification: scalar targets; regression: match output_shape
    return torch.randint(0, output_shape[0], (4,))

@pytest.fixture
def dataset():
    return MyDataset(split="train")
```

## The Five Checks

### 1. Output Shape

Catches dimension mismatches. The batch=1 case catches `squeeze()` removing the batch dim.

```python
@torch.no_grad()
def test_output_shape(model, sample_batch, output_shape):
    out = model(sample_batch)
    assert out.shape == (sample_batch.shape[0], *output_shape)
    # batch=1 edge case (catches squeeze bugs)
    assert model(sample_batch[:1]).shape == (1, *output_shape)
```

### 2. Gradient Flow

Verifies every `requires_grad` parameter actually receives a gradient.

```python
def test_gradient_flow(model, sample_batch):
    model(sample_batch).mean().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name}: no grad"
            assert p.grad.abs().sum() > 0, f"{name}: zero grad"
```

### 3. Single-Batch Overfit

If a model can't memorize a tiny fixed batch, the training loop is broken.

```python
@pytest.mark.slow
def test_overfit_single_batch(model, sample_batch, target_batch):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        loss = F.cross_entropy(model(sample_batch), target_batch)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.01, f"Can't overfit: loss={loss.item()}"
```

### 4. Batch Independence

Ensures no information leaks between samples in a batch. Masks out one
sample's output, backprops, and checks that the masked sample's input got
zero gradient. Must use `model.eval()` because BatchNorm in train mode
couples samples.

```python
@pytest.mark.slow
def test_batch_independence(model, sample_batch):
    model.eval()
    x = sample_batch.clone().requires_grad_(True)
    out = model(x)
    mask = torch.ones_like(out)
    mask[0] = 0
    (out * mask).sum().backward()
    assert x.grad[0].abs().sum() == 0, "Gradient leaked across batch"
```

### 5. Data Pipeline Sanity

Catches broken normalization, wrong shapes, and accidentally deterministic augmentations.

```python
def test_data_pipeline(dataset):
    x, y = dataset[0]
    assert x.ndim >= 2, f"Expected image-like tensor, got shape {x.shape}"
    assert x.min() >= -3.0 and x.max() <= 3.0, "Values look unnormalized"
    # augmentation randomness (skip if val/test set)
    x2, _ = dataset[0]
    assert not torch.equal(x, x2), "Augmentation not active"
```

## Performance

Checks 3 (overfit) and 4 (batch independence) are slow.
When writing these, propose to the user:
- Mark with `@pytest.mark.slow`
- Configure pytest to skip slow by default (`-m "not slow"`)
- Only proceed with marking after user confirms

Checks 1, 2, and 5 are fast — no special handling.

## Common Mistakes

- Forgetting batch=1 edge case in shape test
- Using `model.train()` instead of `model.eval()` for batch independence — BatchNorm in train mode couples samples across the batch
- Overfit test too few steps or lr too low → flaky
- Not calling `model.zero_grad()` before gradient checks when reusing a model across tests
- Using `torch.randn` without seeding → flaky tests across runs
- Hardcoding tensor sizes that don't match the model's actual expected input
- Forgetting that frozen layers (requires_grad=False) should be excluded from gradient checks
