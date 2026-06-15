from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from flock.collectives.engine import CollectiveEngine
from flock.errors import FlockDeadlockError, FlockUsageError
from flock.p2p.engine import P2PEngine
from flock.scheduler.port import SchedulePort
from flock.scheduler.protocol import Fifo, Policy, Random, Worker
from flock.scheduler.runtime import Runtime, activate
from flock.types import Rank


class _SchedulePort:
    def __init__(self, scheduler: CooperativeScheduler[Any]) -> None:
        self._scheduler = scheduler

    @property
    def world_size(self) -> int:
        return self._scheduler.world_size

    def resume(self, rank: Rank, value: Any = None) -> None:
        self._scheduler.runnable.append((rank, value))


class CooperativeScheduler[R]:
    def __init__(
        self,
        workers: Sequence[Worker[R]],
        policy: Policy,
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
        self.done: dict[Rank, R] = {}

        port: SchedulePort = _SchedulePort(self)
        self.runtime = Runtime(
            p2p=P2PEngine(port),
            collectives=CollectiveEngine(port, world_size=self.world_size),
        )
        for worker in self.workers:
            worker.context.run(activate, self.runtime)

    def run(self) -> list[R]:
        while self.runnable:
            self._tick()

        if len(self.done) != self.world_size:
            blocking = self._blocking_report()
            pending = self.runtime.pending.report_lines()
            if pending:
                lines = "\n".join(f"  - {line}" for line in pending)
                raise FlockUsageError(f"unawaited operations:\n{lines}\n\n{blocking}")
            raise FlockDeadlockError(blocking)

        return [self.done[rank] for rank in range(self.world_size)]

    def _blocking_report(self) -> str:
        lines = ["deadlock: no rank can make progress."]

        if self.seed is not None:
            lines.append(f"(reproduce this ordering with Random(seed={self.seed}))")

        lines.extend(self.runtime.p2p.deadlock_lines())
        lines.extend(self.runtime.collectives.deadlock_lines())

        return "\n".join(lines)

    def _resume(self, rank: Rank, value: Any) -> Any:
        worker = self.workers[rank]
        return worker.context.run(worker.coro.send, value)

    def _tick(self) -> None:
        rank, resume_value = self._pick()

        try:
            handle = self._resume(rank, resume_value)
        except StopIteration as stop:
            self.runtime.pending.check(rank)
            self.done[rank] = stop.value
            return

        self.runtime.wait(rank, handle)

    def _pick(self) -> tuple[Rank, Any]:
        assert self.runnable, "no runnable ranks"

        if self.rng is None:
            return self.runnable.pop(0)

        choice = self.rng.randint(0, len(self.runnable) - 1)

        self.runnable[choice], self.runnable[-1] = (
            self.runnable[-1],
            self.runnable[choice],
        )
        return self.runnable.pop()
