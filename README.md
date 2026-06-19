# flock

Learn distributed programming in pure Python — no GPU, no cluster.

Every rank is a coroutine. A cooperative scheduler drives them; communication
goes through in-memory mailboxes. Same mental model as `torch.distributed`, runnable
on a laptop.

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

## API at a glance

**Post early, await late** — `isend`, `irecv`, and collectives register with the
runtime when you call them. Complete them with `await work.wait()`.

| API | Returns | When it registers |
|-----|---------|-------------------|
| `isend(dst, v)` | `Work[None]` | on call |
| `irecv(src)` | `Work[Any]` | on call |
| `send(dst, v)` | `None` (async) | when awaited |
| `recv(src)` | `T` (async) | when awaited |
| `barrier()`, `all_gather()`, … | `Work[...]` | on call |

Blocking sugar: `await flock.send(...)` and `await flock.recv(...)` for simple
programs. Overlap communication with computation via `isend` / `irecv` / collectives
and `.wait()`. Poll without blocking using `work.is_completed()`.

## Development

```bash
just check   # ruff + mypy
just test    # pytest
just fix     # auto-format
```

## More detail

See [IDEA.MD](IDEA.MD) for architecture, scheduling, and deadlock diagnostics.
Course content lives in the sibling [dist-puzzles](../dist-puzzles) project.
