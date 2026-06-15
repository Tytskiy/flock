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
  types.py              # Rank, Group, new_group(), WORLD
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

### Collectives

All return `Work[...]`; complete with `await work.wait()`. Like p2p ops, they
register on call, so you can start one and overlap the wait with local work.

| API | Result type |
|-----|-------------|
| `barrier()` | `None` |
| `all_gather(value)` | `list[T]` |
| `all_reduce(value, op)` | `T` |
| `scatter(values, src)` | `T` |

Groups: `flock.new_group([0, 2])` or omit `group` / pass `flock.WORLD` for all
ranks. Mismatched collective kinds at the same site raise `FlockCollectiveMismatch`.

### Scheduling policies

`@flock.distribute` uses `Random(seed=...)` by default:

- `seed=0` (default) — deterministic ordering
- `seed=None` — fresh seed each run, for fuzzing

Pass `policy=` to choose another policy, e.g. `@flock.distribute(workers=4,
policy=flock.Fifo())` for a deterministic round-robin order. An explicit `policy`
overrides `seed`.

### Deadlock detection

When the runnable queue empties but not all ranks are done, the scheduler
reports which ranks are stuck and why. Unawaited operations are listed first
(`FlockUsageError`). Otherwise `FlockDeadlockError` with P2P and collective
details. If a random seed was used, it's included so you can reproduce the
exact interleaving.

## Roadmap

### Collectives — planned

- `broadcast(value, src=0)` — one sender, many receivers
- `reduce(value, op, dst=0)` — reduction to one rank

### Later

- Ring / tree algorithm implementations (educational)
- Trace / debug mode (log which rank ran which op)
- Process backend (real isolation, closer to torchrun)
- Socket backend (real networking)
