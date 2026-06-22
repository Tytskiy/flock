# flock

[![CI](https://github.com/Tytskiy/flock/actions/workflows/ci.yml/badge.svg)](https://github.com/Tytskiy/flock/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Learn distributed programming on your laptop.

`flock` is a tiny Python playground for multi-rank programs. Write point-to-point
messages, try collectives, break things on purpose, and get useful deadlock
diagnostics without GPUs, MPI, `torchrun`, or a cluster.

It is mainly a learning tool: small enough to read, fast enough to experiment
with, and close enough to `torch.distributed` / MPI ideas to build the right
mental model. It is also handy when you want to debug a distributed algorithm
before moving it to heavier infrastructure.

## Why flock?

- Run distributed programming examples in one normal Python process.
- Learn `send` / `recv`, `all_reduce`, `broadcast`, `gather`, `scatter`, and
  other collectives without setup pain.
- Reproduce tricky interleavings with deterministic scheduling.
- Get deadlock messages that tell you which rank is waiting for what.
- Iterate quickly on distributed algorithms before reaching for real clusters.

## Run An Example

```bash
git clone https://github.com/Tytskiy/flock.git
cd flock
uv run python examples/ring.py
```

The repo includes runnable mini-lessons:

- [`examples/ring.py`](examples/ring.py): pass messages around a ring.
- [`examples/all_reduce.py`](examples/all_reduce.py): try `all_reduce` and
  `all_gather`.
- [`examples/deadlock.py`](examples/deadlock.py): see a send/recv bug explained.
- [`examples/process_groups.py`](examples/process_groups.py): run a collective
  on a subgroup.

## Quickstart: Pass A Message Around A Ring

```python
import flock


@flock.distribute(workers=4)
async def ring():
    rank = flock.get_rank()
    world = flock.get_world_size()

    await flock.isend((rank + 1) % world, f"hello from {rank}").wait()
    return await flock.recv((rank - 1) % world)


print(ring())
# ['hello from 3', 'hello from 0', 'hello from 1', 'hello from 2']
```

`@flock.distribute(workers=4)` runs the same function as ranks `0`, `1`, `2`,
and `3`, then returns their results in rank order.

## Collectives Without A Cluster

Collectives are the operations behind a lot of distributed training and
distributed systems code. With `flock`, you can play with them directly:

```python
import flock


@flock.distribute(workers=4)
async def sum_ranks():
    rank = flock.get_rank()
    total = await flock.all_reduce(rank, "sum").wait()
    gathered = await flock.all_gather(f"rank {rank}").wait()
    return total, gathered


print(sum_ranks())
# [
#   (6, ['rank 0', 'rank 1', 'rank 2', 'rank 3']),
#   (6, ['rank 0', 'rank 1', 'rank 2', 'rank 3']),
#   (6, ['rank 0', 'rank 1', 'rank 2', 'rank 3']),
#   (6, ['rank 0', 'rank 1', 'rank 2', 'rank 3']),
# ]
```

You get the core vocabulary: point-to-point messages, barriers, gather/scatter,
broadcast, reduce, all-reduce, reduce-scatter, all-to-all, groups, tags, and
wildcard receives.

## Deadlocks That Talk Back

Distributed bugs are often boring in the worst way: the program just hangs.
`flock` tries to make those bugs teach you something.

```python
import flock


@flock.distribute(workers=2)
async def stuck():
    rank = flock.get_rank()
    if rank == 0:
        return await flock.recv(1)

    return "rank 1 never sends"


stuck()
```

Instead of hanging forever:

```text
FlockDeadlockError
deadlock: no rank can make progress.
(reproduce this ordering with Random(seed=0))
rank 0 is blocked in recv waiting for rank 1
```

Unawaited operations are reported too, so a forgotten `.wait()` does not quietly
turn into a mystery later.

## Tiny, But Not Toy-Sized

Because ranks are lightweight, you can try examples that would be annoying to
set up with real infrastructure. This 1,000-rank collective runs in one Python
process:

```python
import flock


@flock.distribute(workers=1000)
async def many_ranks():
    return await flock.all_reduce(1, "sum").wait()


assert set(many_ranks()) == {1000}
```

On a laptop, this is the kind of thing you can run while you are still thinking.

## Installation

`flock` requires Python 3.12+ and has no runtime dependencies.

With `pip`:

```bash
pip install git+https://github.com/Tytskiy/flock.git
```

With `uv`:

```bash
uv add git+https://github.com/Tytskiy/flock.git
```

For local development:

```bash
git clone https://github.com/Tytskiy/flock.git
cd flock
uv sync
```

## A Few Things To Try

- Write a ring, a fan-out, or a pipeline with `send`, `recv`, `isend`, and
  `irecv`.
- Replace hand-written message passing with `all_gather`, `broadcast`, or
  `all_reduce`.
- Create a subgroup with `await flock.new_group([0, 2])`; every rank calls it,
  then non-members skip the subgroup collective.
- Use `flock.ANY_SOURCE` or tags when message order is part of the puzzle.
- Switch scheduling policies to shake out ordering bugs.

## For Curious Readers

Each rank is an `async def` function. Operations return `Work` handles, and
`await work.wait()` gives the scheduler a chance to advance messages and
collectives. Nonblocking sends, receives, and collectives register when you call
them, so you can post communication early and wait later.

By default, `@flock.distribute` uses deterministic random scheduling:

```python
@flock.distribute(workers=4, seed=123)
async def reproducible():
    ...


@flock.distribute(workers=4, seed=None)
async def fresh_interleaving_each_run():
    ...


@flock.distribute(workers=4, policy=flock.Fifo())
async def fifo_order():
    ...
```

There are no sockets, threads, subprocesses, or `asyncio` event loops in the
runtime. The scheduler is the event loop.

## Development

```bash
just check   # ruff + mypy
just test    # pytest
just fix     # auto-fix lint and format
```

## Status

`flock` currently supports point-to-point messaging, common collectives,
subgroups, deterministic scheduling, deadlock diagnostics, and typed Python
packages. Next up: richer execution traces and more teaching examples.
