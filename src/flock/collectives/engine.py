from collections import defaultdict
from dataclasses import dataclass, field

from flock.collectives.handle import CollectiveHandle
from flock.collectives.ops import CollectiveCall, CollectiveState
from flock.context import get_rank as get_current_rank
from flock.errors import FlockCollectiveMismatch, FlockUsageError
from flock.scheduler.port import SchedulePort
from flock.tracer import Tracer
from flock.types import WORLD, Group, Rank, _default_world_group


@dataclass
class CollectiveSlot:
    members: Group
    state: CollectiveState
    arrived: set[Rank] = field(default_factory=set)
    waiting: set[Rank] = field(default_factory=set)


class CollectiveEngine:
    def __init__(self, port: SchedulePort, *, world_size: int, tracer: Tracer | None = None) -> None:
        self._port = port
        self._world_size = world_size
        self._tracer = tracer
        self.counters: defaultdict[tuple[Rank, Group], int] = defaultdict(int)
        self.slots: dict[tuple[Group, int], CollectiveSlot] = {}
        self.blocked: dict[Rank, CollectiveHandle] = {}
        self._request_counter = 0

    def begin(self, group: Group, call: CollectiveCall) -> CollectiveHandle:
        rank = get_current_rank()
        members = self._resolve_group(group, rank)
        index = self.counters[(rank, members)]
        key = (members, index)

        slot = self.slots.get(key)
        if slot is None:
            state = call.begin(rank)
            slot = CollectiveSlot(members=members, state=state)
            self.slots[key] = slot

        if call.kind != slot.state.kind:
            raise FlockCollectiveMismatch(
                f"rank {rank} called {call.kind!r} at collective #{index}, "
                f"but other ranks in the group already started {slot.state.kind!r} there."
            )

        call.enter(slot.state, rank, members)
        slot.arrived.add(rank)
        self.counters[(rank, members)] += 1
        if self._tracer is not None:
            self._tracer.collective_enter(
                rank,
                call.kind,
                index,
                len(members),
                nbytes=slot.state.rank_payload_bytes(rank),
            )
        request_id = self._request_counter
        self._request_counter += 1
        return CollectiveHandle(
            group=members,
            index=index,
            rank=rank,
            kind=call.kind,
            request_id=request_id,
        )

    def wait(self, rank: Rank, handle: CollectiveHandle) -> None:
        key = (handle.group, handle.index)
        slot = self.slots.get(key)
        if slot is None:
            raise FlockUsageError(
                f"rank {rank} tried to wait on collective #{handle.index} without entering it on this rank."
            )

        if rank not in slot.arrived:
            raise FlockUsageError(
                f"rank {rank} tried to wait on collective #{handle.index} without entering it on this rank."
            )

        slot.waiting.add(rank)

        if slot.waiting == set(slot.members):
            if self._tracer is not None:
                self._tracer.collective_sync(
                    handle.kind,
                    handle.index,
                    len(slot.members),
                    total_payload_bytes=slot.state.payload_bytes(),
                )
            del self.slots[key]
            for waiting_rank in sorted(slot.waiting):
                if waiting_rank in self.blocked:
                    del self.blocked[waiting_rank]
                result = slot.state.complete(slot.members, waiting_rank)
                self._port.resume(waiting_rank, result)
            return

        self.blocked[rank] = handle

    def is_complete(self, handle: CollectiveHandle) -> bool:
        slot = self.slots.get((handle.group, handle.index))
        if slot is None:
            return True
        members = set(slot.members)
        return slot.arrived == members and slot.waiting == members - {handle.rank}

    def deadlock_lines(self) -> list[str]:
        lines: list[str] = []

        for rank, collective in sorted(self.blocked.items()):
            lines.append(
                f"rank {rank} is blocked in {collective.kind} "
                f"(collective #{collective.index}, group size {len(collective.group)})"
            )

        for (_, index), slot in sorted(self.slots.items()):
            missing = set(slot.members) - slot.arrived
            if missing:
                lines.append(
                    f"collective #{index} ({slot.state.kind}) is waiting for ranks {sorted(missing)} to enter"
                )

            registered = slot.arrived - slot.waiting
            if slot.arrived == set(slot.members) and registered:
                lines.append(
                    f"ranks {sorted(registered)} entered collective #{index} ({slot.state.kind}) "
                    "but have not awaited yet"
                )

        return lines

    def _resolve_group(self, group: Group, rank: Rank) -> Group:
        members = _default_world_group(self._world_size) if group is WORLD else group

        if not members:
            raise FlockUsageError("a group must contain at least one rank.")

        if rank not in members:
            raise FlockUsageError(f"rank {rank} is not a member of this group (members: {sorted(members)}).")

        for member in members:
            if member < 0 or member >= self._world_size:
                raise FlockUsageError(
                    f"group member {member} is out of range for world_size={self._world_size}."
                )

        return members
