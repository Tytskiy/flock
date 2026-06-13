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
    rank = flock.rank()
    nxt = (rank + 1) % flock.world_size()
    prv = (rank - 1) % flock.world_size()

    await flock.isend(nxt, f"hello from {rank}")
    msg = await flock.recv(prv)
    return msg

print(ring())
# ['hello from 3', 'hello from 0', 'hello from 1', 'hello from 2']
```

## How it works

1. Each rank is an `async def` coroutine.
2. Operations like `isend`, `send`, `recv`, and `barrier` register with
   the runtime immediately; `await request` yields the handle for the
   scheduler to dispatch.
3. The scheduler picks a runnable rank (random or FIFO), resumes it, and
   delegates waits to `P2PEngine` or `CollectiveEngine`.
4. When no rank can make progress, a deadlock error is raised with a
   diagnostic report.

No threads, no processes, no sockets. Just coroutines and a `while` loop.

## Architecture

```
src/flock/
  p2p/                 # point-to-point API and engine
  collectives/          # collectives API, engine, groups, futures
  scheduler/            # protocol, runtime hook, cooperative implementation
  context.py            # Per-rank contextvars: rank(), world_size()
  decorator.py          # distribute decorator
  errors.py             # FlockError, FlockDeadlockError, FlockUsageError
  types.py              # Rank, Group, shared types
  wait.py               # Request
  __init__.py           # Re-exports
```

### Point-to-point operations

| API             | Sender blocks? | Receiver blocks? |
|-----------------|---------------|-----------------|
| `isend(dst, v)` | No            | —               |
| `send(dst, v)`  | Until recv    | —               |
| `recv(src)`     | —             | Until message   |

`send` is synchronous (rendezvous): the sender is suspended until the
receiver actually calls `recv`. `isend` is fire-and-forget.

### Scheduling policies

- `Random(seed=0)` — deterministic random ordering (default)
- `Random(seed=None)` — fresh seed each run, for fuzzing
- `Fifo()` — always pick the oldest runnable rank

### Deadlock detection

When the runnable queue empties but not all ranks are done, the scheduler
reports which ranks are stuck and why. If a random seed was used, it's
included so you can reproduce the exact interleaving.

## Roadmap

### Collectives

Implemented:

- `barrier()` — collective matching via per-group sequence counters
- `Request` async API — register on call, block on `await request`

Planned, in this order:
2. `broadcast(value, src=0)` — one sender, many receivers
3. `all_gather(value)` — everyone contributes, everyone receives
4. `reduce(value, op, dst=0)` / `all_reduce(value, op)` — reductions

Each collective needs a monotonic **call id** so different collective
sites in the same program don't mix messages. Mismatched collectives
(rank 0 at `barrier`, rank 1 at `all_gather`) should raise
`FlockCollectiveMismatch`.

### Later

- Ring / tree algorithm implementations (educational)
- Trace / debug mode (log which rank ran which op)
- Process backend (real isolation, closer to torchrun)
- Socket backend (real networking)
