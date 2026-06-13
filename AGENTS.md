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
- `from __future__ import annotations` only when a module needs forward
  references (e.g. `Scheduler` used before its class body).
- No docstrings on classes or functions whose name and signature already
  explain what they do.

## Project structure

- `src/flock/` — library source
- `src/flock/p2p/` — point-to-point API and engine
- `src/flock/collectives/` — collectives API and engine
- `src/flock/wait.py` — `Work`, shared wait machinery
- `src/flock/scheduler/` — protocol, runtime hook, cooperative implementation
- `tests/` — pytest tests mirroring package layout:
  `test_distribute.py`, `p2p/`, `collectives/`, `scheduler/`, `typing/`
- `justfile` — run `just check` (ruff + mypy), `just test`, `just fix`

## Tooling

- `uv` for dependency management
- `ruff` for linting and formatting
- `mypy` for type checking (`src` + `tests/typing`, strict-ish)
- `pytest` for tests
- Run `just check && just test` after changes

## Design principles

- The scheduler is the only event loop. No asyncio, no threads.
- Operations register with engines via `require_runtime()`; `await work.wait()`
  yields its handle for the scheduler to dispatch.
- `contextvars` provide per-rank state (`rank()`, `world_size()`).
- The public API (`flock.distribute`, `flock.isend`, etc.) is a thin
  wrapper; logic lives in `P2PEngine` and `CollectiveEngine`, wired through
  `Runtime` (`require_runtime()`).
- Error messages should be clear and actionable, aimed at someone
  learning distributed programming.
