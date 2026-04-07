# nn-skills

A Claude Code plugin with skills for neural network development — TDD workflow and PyTorch component test coverage.

## Why

Writing correct neural networks is hard. Silent bugs (wrong shapes, broken gradients, dtype mismatches) pass code review and only surface as degraded training. These skills encode testing discipline so the agent catches those failures early.

## Skills

- **`nn-skills:test-before-code`** — Test-driven development workflow. Write the test first, watch it fail, write minimal code to pass.
- **`nn-skills:test-nn-components`** — Audit and complete test coverage for `nn.Module` subclasses against five critical checks (output shape, gradient flow, dtype consistency, determinism, serialization).

## Install

Add the marketplace and install the plugin inside Claude Code:

```
/plugin marketplace add liunuozhi/nn-skills
/plugin install nn-skills
```

## Acknowledgments

Inspired by [Superpowers](https://github.com/obra/superpowers) by Jesse Vincent — an agentic skills framework and software development methodology for coding agents.
