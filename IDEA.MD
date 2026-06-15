# flock

Learn distributed collectives in pure Python, no GPU.

## What is this?

A single-process simulator for distributed programs. Every rank is a
coroutine, the scheduler drives them cooperatively, and communication
happens through in-memory mailboxes. Think `torch.distributed` semantics
you can run on a laptop.

```python
import flock

@flock.distribute(workers=4)
async def ring():
    rank = flock.get_rank()
    nxt = (rank + 1) % flock.get_world_size()
    prv = (rank - 1) % flock.get_world_size()

    await flock.isend(nxt, f"hello from {rank}").wait()
    return await flock.recv(prv)

print(ring())
# ['hello from 3', 'hello from 0', 'hello from 1', 'hello from 2']
```

## How it works

1. Each rank is an `async def` coroutine.
2. Nonblocking ops (`isend`, `irecv`, collectives) **register with the runtime on
   call**. Blocking sugar (`send`, `recv`) registers when you `await` them.
3. `await work.wait()` yields the handle to the scheduler, which delegates to
   `P2PEngine` or `CollectiveEngine`.
4. When no rank can make progress, a deadlock or usage error is raised with a
   diagnostic report.

No threads, no processes, no sockets. Just coroutines and a `while` loop.

### Register on call

Calling `flock.isend(...)`, `flock.irecv(...)`, or `flock.barrier()` starts the
operation immediately — even before `.wait()`. That lets ranks overlap
communication with local work:

```python
work = flock.all_gather(flock.get_rank())
await flock.isend(nxt, flock.get_rank()).wait()  # do other comms first
result = await work.wait()
```

Every started operation must be awaited (via `.wait()` or blocking `send`/`recv`)
before a rank returns. Unawaited ops raise `FlockUsageError`.

`work.is_completed()` is a non-blocking poll: it returns `True` when `await
work.wait()` would resume the rank without blocking — i.e. the message has
arrived, the send has been received, or every other group member is already
waiting on the collective. Use it to decide whether to keep overlapping local
work or to wait now.

Collectives also advance their per-rank sequence counter on call, not on wait.
Calling `flock.barrier()` without awaiting consumes a collective slot.

## Architecture

```
src/flock/
  p2p/                  # point-to-point API and engine
  collectives/          # collectives API and engine
  scheduler/            # protocol, runtime, cooperative implementation
  context.py            # per-rank contextvars: get_rank(), get_world_size()
  decorator.py          # @distribute
  errors.py             # FlockError, FlockDeadlockError, FlockUsageError, ...
  types.py              # Rank, Group, new_group(), WORLD, ANY_SOURCE, ANY_TAG
  work.py               # Work, shared wait machinery
  __init__.py           # re-exports
```

### Point-to-point operations

| API              | Returns      | Sender blocks? | Receiver blocks? |
|------------------|--------------|----------------|------------------|
| `isend(dst, v)`  | `Work[None]` | No             | —                |
| `irecv(src)`     | `Work[Any]`  | —              | On `.wait()`     |
| `send(dst, v)`   | `None`       | Until recv     | —                |
| `recv(src)`      | `T`          | —              | Until message    |

`send` is rendezvous: the sender suspends until the receiver calls `recv`.
`isend` delivers immediately (or buffers); `.wait()` only unblocks the sender.

`recv` and `send` are `async def` sugar for simple programs. Use `irecv` /
`isend` when you need to post receives or sends before awaiting them.

A rank may target itself. `isend(rank, v)` buffers into the rank's own mailbox,
so a later `recv(rank)` succeeds. Blocking `send(rank, v)` to yourself deadlocks:
the rank suspends waiting to be received by code it can never reach.

`isend` buffers eagerly with **no bound**: a rank can post unlimited unreceived
sends, and they never block on buffer space. This is a deliberate simplification —
real transports (NCCL/MPI) have eager/rendezvous thresholds and finite buffers, so
backpressure and flow control are intentionally out of scope.

#### Tags and ANY_SOURCE

Every p2p call takes an optional `tag` (default `0`); a `recv` only matches a
message with the same `(src, tag)`. Pass `flock.ANY_SOURCE` as the source or
`flock.ANY_TAG` as the tag to match any sender / any tag — useful for ring CP and
pipeline activation passing where the partner or message ordering isn't fixed.
Matching is non-overtaking per `(src, tag)`: messages in the same stream are
received in send order, but a later `recv` may pick a different tag out of order.

### Collectives

All return `Work[...]`; complete with `await work.wait()`. Like p2p ops, they
register on call, so you can start one and overlap the wait with local work.

| API | Result type |
|-----|-------------|
| `barrier()` | `None` |
| `all_gather(value)` | `list[T]` |
| `all_reduce(value, op)` | `T` |
| `reduce(value, op, dst)` | `T` on `dst`, else `None` |
| `broadcast(value, src)` | `T` |
| `gather(value, dst)` | `list[T]` on `dst`, else `None` |
| `scatter(values, src)` | `T` |
| `reduce_scatter(values, op)` | `T` |
| `all_to_all(values)` | `list[T]` |

`op` accepts a named op (`"sum"`, `ReduceOp.PROD`, …) or any binary callable
`(a, b) -> c` — e.g. `torch.minimum` for tensor reductions.

Groups: `flock.new_group([0, 2])` or omit `group` / pass `flock.WORLD` for all
ranks. Mismatched collective kinds at the same site raise `FlockCollectiveMismatch`.

### Scheduling policies

`@flock.distribute` uses `Random(seed=...)` by default:

- `seed=0` (default) — deterministic ordering
- `seed=None` — fresh seed each run

Pass `policy=` to choose another policy, e.g. `@flock.distribute(workers=4,
policy=flock.Fifo())` for a deterministic round-robin order. An explicit `policy`
overrides `seed`.

### Deadlock detection

When the runnable queue empties but not all ranks are done, the scheduler
reports which ranks are stuck and why. Unawaited operations are listed first
(`FlockUsageError`). Otherwise `FlockDeadlockError` with P2P and collective
details. If a random seed was used, it's included so you can reproduce the
exact interleaving.

## Vision: a distributed-training course

flock is the runtime under a "build distributed training from scratch" course, in
the spirit of the Tensor / GPU / Triton Puzzles: each puzzle is a spec + expected
output + a body to fill, graded automatically. The deterministic scheduler and
deadlock diagnostics are the teaching superpower — students *see* interleavings,
hangs, and overlap that real frameworks hide.

### Substrate decisions

- **Tensors & autograd: real (CPU) `torch`.** Payloads are torch tensors; forward
  and backward use torch autograd. This keeps the lessons transferable to actual
  `torch.distributed`.
- **Core stays dependency-free.** The flock runtime is just coroutines + mailboxes
  and does not import torch. Torch is a *course/puzzle* dependency group; the engine
  treats a tensor as any other value.

### What flock provides vs. what students build

flock is the *substrate*: only the primitives that need scheduler/engine support
and can't be built from outside. Everything expressible in terms of them is a
puzzle, not a library feature.

- **flock provides:** p2p (`send`/`recv`/`isend`/`irecv`), the built-in
  collectives, the cooperative scheduler, deadlock/usage diagnostics, deterministic
  seeds, `Group`/`new_group`, `tag` / `ANY_SOURCE` / `ANY_TAG` matching, and the
  per-rank context (`rank`, `world_size`). Still to add: the trace/cost model.
- **Students build (the puzzles):** `DeviceMesh` (an n-d reshape of the world over
  `new_group`), `Placements` (`Shard`/`Replicate`/`Partial`), `DTensor` +
  `redistribute` (each placement change is a collective), and the parallelisms
  (DDP, FSDP, TP, CP, EP, PP).

The point is pedagogical: shipping `DeviceMesh` or `redistribute` would hand
students the punchline. flock stays minimal so the high-level structure is theirs
to discover.

### Two ideas the whole course is built around

1. **Collectives are p2p algorithms.** Implement `all_reduce` (ring vs recursive
   halving/doubling), `broadcast` (flat vs tree), etc., from `send`/`recv`, and
   compare them by message count / rounds / bytes.
2. **Redistribution is collectives.** Once DTensor exists, every placement change is
   a collective:
   - `Shard → Replicate` = `all_gather`
   - `Replicate → Shard` = local slice (no comm)
   - `Partial → Replicate` = `all_reduce`
   - `Shard(i) → Shard(j)` = `all_to_all`

### Tiers

0. **Mental model** — ranks, the scheduler, deadlocks (plain Python values).
1. **P2P → collectives → algorithms** — semantics, then implement collectives from
   p2p (ring/tree), with a trace/cost model to compare them.
2. **Tensors + autograd → DeviceMesh / Placements / DTensor** — `redistribute` as
   collectives; DeviceMesh is an n-d reshape of the world handing back a `Group`
   per dim.
3. **Data parallel** — DDP (all_reduce grads), bucketing + comm/compute overlap,
   then ZeRO 1/2/3 as the ramp into sharding.
4. **Model parallelisms** — FSDP, TP, CP, EP, PP, and a 2D/3D composition capstone
   (TP × DP × PP) — the real DeviceMesh payoff.

### What each tier needs from flock

Most tiers need *nothing new* — they're built on the existing primitives. flock's
own remaining work is small.

- Tier 1: full collective set (done) + a trace/cost model (messages, rounds, bytes)
  to compare hand-rolled algorithms.
- Tier 2: nothing new — `DeviceMesh`/`Placements`/`redistribute` are student work
  over `new_group` and the collectives.
- Tier 3–4: nothing new — `tag` / `ANY_SOURCE` `recv` (available) cover ring CP and
  PP activation passing. PP is p2p of activations (best showcase for visualizing
  GPipe vs 1F1B bubbles); EP is `all_to_all`.

### Topics not to forget

- Comm/compute overlap & prefetch (DDP buckets, FSDP all-gather prefetch) — built on
  "post early, await late" + `work.is_completed()`.
- Gradient accumulation / micro-batching (prereq for PP).
- Sharded checkpoint save/load (DTensor state dict).
- Numerics & debugging: all_reduce float ordering, shape/rank-mismatch diagnostics.

### Engine roadmap

- Trace / debug mode (log which rank ran which op; communication volume).
- Process / socket backends (real isolation, closer to torchrun).
