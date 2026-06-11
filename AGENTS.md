# Agent guidelines for flock

## Comments

Do not write comments that narrate what code does. The code speaks for itself.

Bad:

```python
# Base class for scheduling policies
class Policy:
    pass

@dataclass(frozen=True)
class Random(Policy):
    """Run a random ready worker on each step.

    A fixed ``seed`` makes the run reproducible. ``seed=None`` draws a fresh
    seed on every run.
    """
    seed: int | None = None
```

Good:

```python
class Policy:
    pass

@dataclass(frozen=True)
class Random(Policy):
    seed: int | None = None
```

A comment is justified only when the code alone cannot convey *why*
something is done, a non-obvious trade-off, or a subtle constraint.

## Style

- Python 3.12+. Use PEP 695 generics (`class Foo[T]:`, `def bar[T]():`)
  instead of `TypeVar`.
- `@dataclass` for data, plain classes for bases with no fields.
- `ClassVar` for class-level constants on dataclasses.
- `from __future__ import annotations` in every module.
- No docstrings on classes or functions whose name and signature already
  explain what they do.

## Project structure

- `src/flock/` — library source
- `tests/` — pytest tests (`test_scheduler.py` for low-level,
  `test_api.py` for the public API)
- `justfile` — run `just check` (ruff + mypy), `just test`, `just fix`

## Tooling

- `uv` for dependency management
- `ruff` for linting and formatting
- `mypy` for type checking (src only, strict-ish)
- `pytest` for tests
- Run `just check && just test` after changes

## Design principles

- The scheduler is the only event loop. No asyncio, no threads.
- Coroutines yield `Op` objects; the scheduler interprets them.
- `contextvars` provide per-rank state (`rank()`, `world_size()`).
- The public API (`flock.distribute`, `flock.isend`, etc.) is a thin
  wrapper; all logic lives in the scheduler.
- Error messages should be clear and actionable, aimed at someone
  learning distributed programming.
