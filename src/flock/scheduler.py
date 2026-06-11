from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Coroutine, Sequence
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import Any

from flock.errors import FlockDeadlockError
from flock.ops import ISendOp, RecvOp, SendOp

Rank = int


@dataclass
class Message[T]:
    src: Rank
    value: T
    ack: bool = False


@dataclass
class Worker[R]:
    coro: Coroutine[Any, Any, R]
    context: Context = field(default_factory=copy_context)


class Policy:
    pass


@dataclass(frozen=True)
class Random(Policy):
    seed: int | None = None


@dataclass(frozen=True)
class Fifo(Policy):
    pass


DEFAULT_POLICY = Random(seed=0)


class Scheduler[R]:
    def __init__(
        self,
        workers: Sequence[Worker[R]],
        policy: Policy = DEFAULT_POLICY,
    ) -> None:
        self.workers = list(workers)
        self.world_size = len(self.workers)

        self.seed: int | None
        self.rng: random.Random | None
        match policy:
            case Random(seed=seed):
                self.seed = random.randrange(2**32) if seed is None else seed
                self.rng = random.Random(self.seed)
            case Fifo():
                self.seed = None
                self.rng = None
            case _:
                raise TypeError(f"unknown policy: {policy!r}")

        self.runnable: list[tuple[Rank, Any]] = [(i, None) for i in range(self.world_size)]
        self.suspended: dict[Rank, Rank] = {}
        self.mailboxes: defaultdict[Rank, deque[Message[Any]]] = defaultdict(deque)
        self.done: dict[Rank, R] = {}

    def run(self) -> list[R]:
        while self.runnable:
            self._tick()

        if len(self.done) != self.world_size:
            raise FlockDeadlockError(self._deadlock_report())

        return [self.done[rank] for rank in range(self.world_size)]

    def _deadlock_report(self) -> str:
        lines = ["deadlock: no rank can make progress."]

        for rank, src in sorted(self.suspended.items()):
            lines.append(f"rank {rank} is blocked in recv waiting for rank {src}")

        for dst, mailbox in sorted(self.mailboxes.items()):
            for message in mailbox:
                if message.ack:
                    lines.append(f"rank {message.src} is blocked in send waiting for rank {dst}")

        if self.seed is not None:
            lines.append(f"(reproduce this ordering with Random(seed={self.seed}))")

        return "\n".join(lines)

    def _resume(self, rank: Rank, value: Any) -> Any:
        worker = self.workers[rank]
        return worker.context.run(worker.coro.send, value)

    def _tick(self) -> None:
        rank, resume_value = self._pick()

        try:
            op = self._resume(rank, resume_value)
        except StopIteration as stop:
            self.done[rank] = stop.value
            return

        match op:
            case ISendOp(dst=dst, value=value):
                self.runnable.append((rank, None))

                if self.suspended.get(dst) == rank:
                    del self.suspended[dst]
                    self.runnable.append((dst, value))
                else:
                    self.mailboxes[dst].append(Message(src=rank, value=value))

            case SendOp(dst=dst, value=value):
                if self.suspended.get(dst) == rank:
                    del self.suspended[dst]
                    self.runnable.append((dst, value))
                    self.runnable.append((rank, None))
                else:
                    self.mailboxes[dst].append(Message(src=rank, value=value, ack=True))

            case RecvOp(src=src):
                if not self.mailboxes[rank]:
                    self.suspended[rank] = src
                    return

                message = self.mailboxes[rank].popleft()

                if message.src != src:
                    raise FlockDeadlockError(
                        f"Expected message from rank {src}, got message from rank {message.src}"
                    )

                self.runnable.append((rank, message.value))

                if message.ack:
                    self.runnable.append((message.src, None))

    def _pick(self) -> tuple[Rank, Any]:
        if self.rng is None:
            return self.runnable.pop(0)

        choice = self.rng.randint(0, len(self.runnable) - 1)

        self.runnable[choice], self.runnable[-1] = (
            self.runnable[-1],
            self.runnable[choice],
        )
        return self.runnable.pop()
