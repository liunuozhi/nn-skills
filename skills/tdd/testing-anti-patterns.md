# Testing Anti-Patterns

**Load this reference when:** writing or changing tests, setting up fixtures, or tempted to add test-only methods to production modules.

## Overview

Tests must verify real behavior, not fixture properties. Synthetic data is a means to isolate, not the thing being tested.

**Core principle:** Test what the module does, not what the test data looks like.

**Following strict TDD prevents these anti-patterns.**

## The Iron Laws

```
1. NEVER test fixture behavior instead of module behavior
2. NEVER add test-only methods to production modules
3. NEVER over-mock when small tensors suffice
```

## Anti-Pattern 1: Testing Fixture Properties

**The violation:**
```python
# BAD: Testing that synthetic data has the right shape
def test_model(model, sample_batch):
    assert sample_batch.shape == (4, 3, 32, 32)  # tests the fixture, not the model
    out = model(sample_batch)
    assert out is not None  # tests nothing meaningful
```

**Why this is wrong:**
- You're verifying the test setup, not the module
- `assert out is not None` passes for any garbage output
- Tells you nothing about real behavior

**The fix:**
```python
# GOOD: Test what the model actually produces
def test_model_output_shape(model, sample_batch):
    out = model(sample_batch)
    assert out.shape == (sample_batch.shape[0], 10)
```

### Gate Function

```
BEFORE writing an assertion:
  Ask: "Am I testing the module's behavior or just my test setup?"

  IF testing setup:
    STOP - Delete the assertion or replace with a behavioral check

  Test real behavior instead
```

## Anti-Pattern 2: Test-Only Methods in Production Modules

**The violation:**
```python
# BAD: _get_intermediate() only used in tests
class Encoder(nn.Module):
    def _get_intermediate(self, x):
        """Expose intermediate activations for testing."""
        return self.layer1(x)

    def forward(self, x):
        h = self.layer1(x)
        return self.layer2(h)
```

**Why this is wrong:**
- Production module polluted with test-only code
- Someone might call `_get_intermediate()` in production
- Violates YAGNI and separation of concerns

**The fix:**
```python
# GOOD: Use hooks to inspect internals without modifying the module
def test_intermediate_activations(model, sample_batch):
    activations = {}
    def hook(module, input, output):
        activations["layer1"] = output

    model.layer1.register_forward_hook(hook)
    model(sample_batch)
    assert activations["layer1"].shape == (4, 64, 16, 16)
```

### Gate Function

```
BEFORE adding any method to a production nn.Module:
  Ask: "Is this only used by tests?"

  IF yes:
    STOP - Use forward hooks, or test via public API
    Put helpers in conftest.py or test utilities

  Ask: "Does this module own this functionality?"

  IF no:
    STOP - Wrong module for this method
```

## Anti-Pattern 3: Over-Mocking When Small Tensors Suffice

**The violation:**
```python
# BAD: Mocking the entire forward pass
def test_training_step(mocker):
    mock_model = mocker.MagicMock()
    mock_model.return_value = torch.tensor([[0.1, 0.9]])
    mock_loss_fn = mocker.MagicMock()
    mock_loss_fn.return_value = torch.tensor(0.5, requires_grad=True)

    loss = train_step(mock_model, mock_loss_fn, inputs, targets)
    mock_model.assert_called_once()
```

**Why this is wrong:**
- Tests that mocks were called, not that training works
- Fake tensors don't flow through real computation
- Gradient behavior is completely untested
- Test passes even if train_step is broken

**The fix:**
```python
# GOOD: Use a tiny real model — fast and tests real behavior
def test_training_step():
    model = nn.Linear(4, 2)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(2, 4)
    y = torch.randint(0, 2, (2,))

    loss = train_step(model, loss_fn, x, y)
    assert loss.item() > 0
    assert model.weight.grad is not None
```

### Gate Function

```
BEFORE mocking any PyTorch module:
  STOP - Do you actually need a mock?

  In DL, real modules with small tensors are:
    - Fast (tiny inputs = microseconds)
    - Accurate (real gradients, real dtypes)
    - Simple (less setup than mock configuration)

  Use mocks ONLY for:
    - External I/O (file systems, network, databases)
    - Hardware-specific code you can't run locally
    - Third-party API calls

  NEVER mock:
    - nn.Module subclasses (use tiny versions instead)
    - Loss functions (they're pure computation)
    - Optimizers (they're fast with small param counts)
    - Tensor operations
```

## Anti-Pattern 4: Wrong Tolerance or No Tolerance

**The violation:**
```python
# BAD: Exact equality on floating-point results
def test_normalize(sample_batch):
    out = normalize(sample_batch)
    assert out.mean() == 0.0  # fails due to float precision
    assert out.std() == 1.0
```

**Why this is wrong:**
- Floating-point arithmetic is inexact
- Test is flaky or always fails
- Different devices (CPU vs GPU) produce slightly different results

**The fix:**
```python
# GOOD: Use appropriate tolerances
def test_normalize(sample_batch):
    out = normalize(sample_batch)
    assert torch.allclose(out.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(out.std(), torch.tensor(1.0), atol=1e-1)
```

### Gate Function

```
BEFORE comparing tensor values:
  Ask: "Is exact equality meaningful here?"

  IF floating-point computation is involved:
    Use torch.allclose() with explicit atol/rtol
    Choose tolerances based on:
      - float32: atol=1e-5 to 1e-6 for most ops
      - float16: atol=1e-2 to 1e-3
      - Accumulated ops (many layers): looser tolerance

  Exact equality is OK for:
    - Integer tensors (class labels, indices)
    - Shape comparisons
    - Device/dtype checks
```

## Anti-Pattern 5: Full-Size Inputs in Tests

**The violation:**
```python
# BAD: Using production-size inputs
def test_resnet_output():
    model = ResNet50()
    x = torch.randn(32, 3, 224, 224)  # 1.5GB+ memory, slow
    out = model(x)
    assert out.shape == (32, 1000)
```

**Why this is wrong:**
- Slow — seconds per test instead of milliseconds
- Memory-heavy — may OOM on CI
- Tests the same thing a small input would test
- Discourages running tests frequently

**The fix:**
```python
# GOOD: Tiny inputs test the same properties
def test_resnet_output():
    model = ResNet50()
    x = torch.randn(2, 3, 32, 32)  # minimal valid input
    out = model(x)
    assert out.shape == (2, 1000)
```

### Gate Function

```
BEFORE choosing tensor sizes for tests:
  Default to SMALL:
    - Batch: 2-4
    - Spatial: smallest valid size (e.g., 32x32 for CNNs)
    - Channels: match model requirements
    - Sequence length: 4-16 for transformers

  Use larger sizes ONLY when:
    - Testing behavior specific to size (e.g., padding at odd dimensions)
    - Explicitly stress-testing memory/performance (mark @pytest.mark.slow)
```

## Anti-Pattern 6: No Seed, Flaky Tests

**The violation:**
```python
# BAD: Random init means random test outcome
def test_model_output_range():
    model = MyModel()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.min() >= 0  # fails 1 in 20 runs
```

**Why this is wrong:**
- Test passes or fails depending on random seed
- CI becomes unreliable — "just re-run it" culture
- Hides real bugs in noise

**The fix:**
```python
# GOOD: Seed when assertions depend on values
def test_model_output_range():
    torch.manual_seed(42)
    model = MyModel()
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.min() >= 0
```

Or better — don't assert on value ranges that depend on initialization. Test invariants instead:

```python
# BETTER: Test structural invariants, not random-dependent values
def test_softmax_output_sums_to_one():
    model = MyModel()  # no seed needed
    x = torch.randn(4, 3, 32, 32)
    out = F.softmax(model(x), dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)
```

## When Mocks Become Complex — You Don't Need Them

**Warning signs:**
- Mock setup longer than test logic
- Mocking nn.Modules or loss functions
- Configuring return values for tensor operations
- Test breaks when mock changes

**In DL, the answer is almost always: use a real tiny model.**

A `nn.Linear(4, 2)` is faster to set up, faster to run, and tests real behavior.

## TDD Prevents These Anti-Patterns

**Why TDD helps:**
1. **Write test first** → Forces you to think about what you're actually testing
2. **Watch it fail** → Confirms test tests real behavior, not fixtures
3. **Minimal implementation** → No test-only methods creep in
4. **Small tensors by default** → RED phase needs fast feedback
5. **Real modules** → You see what the test needs before reaching for mocks

## Quick Reference

| Anti-Pattern | Fix |
|--------------|-----|
| Assert on fixture properties | Test module output/behavior |
| Test-only methods in modules | Use hooks or test via public API |
| Mock nn.Modules or losses | Use tiny real modules |
| Exact float equality | Use `torch.allclose` with tolerances |
| Full-size inputs | Batch=2-4, smallest valid spatial dims |
| No seed, flaky tests | Seed or test invariants, not random-dependent values |

## Red Flags

- Assertions on tensor existence (`is not None`) without checking values/shapes
- Methods only called in test files
- `MagicMock` on anything that could be a real `nn.Module`
- `assert x == y` on float tensors
- Test takes >1 second without `@pytest.mark.slow`
- "Just re-run, it's flaky"

## The Bottom Line

**In DL testing, small real models beat mocks every time.**

If TDD reveals you're testing fixture behavior or mock interactions, you've gone wrong.

Fix: Test real behavior with tiny tensors.
