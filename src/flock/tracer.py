from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from flock.types import Rank


@dataclass(frozen=True)
class TraceEvent:
    step: int


@dataclass(frozen=True)
class P2PBegin(TraceEvent):
    rank: Rank
    peer: Rank
    op: str
    tag: int
    nbytes: int


@dataclass(frozen=True)
class P2PDeliver(TraceEvent):
    src: Rank
    dst: Rank
    nbytes: int
    tag: int = 0


@dataclass(frozen=True)
class CollectiveEnter(TraceEvent):
    rank: Rank
    kind: str
    index: int
    group_size: int
    nbytes: int


@dataclass(frozen=True)
class CollectiveSync(TraceEvent):
    kind: str
    index: int
    group_size: int
    nbytes: int


@dataclass(frozen=True)
class TraceSummary:
    world_size: int
    total_bytes: int
    total_messages: int
    bytes_sent: tuple[int, ...]
    bytes_received: tuple[int, ...]
    messages_sent: tuple[int, ...]
    messages_received: tuple[int, ...]
    bytes_matrix: tuple[tuple[int, ...], ...]
    collective_counts: tuple[tuple[str, int], ...]


class Tracer:
    def __init__(self, *, world_size: int = 0) -> None:
        self._world_size = world_size
        self._step = 0
        self._events: list[TraceEvent] = []
        self._summary: TraceSummary | None = None

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def p2p_begin(self, rank: Rank, op: str, peer: Rank, tag: int, *, nbytes: int) -> None:
        self._record(
            P2PBegin(
                step=self._next_step(),
                rank=rank,
                peer=peer,
                op=op,
                tag=tag,
                nbytes=nbytes,
            )
        )

    def p2p_deliver(self, src: Rank, dst: Rank, *, nbytes: int, tag: int = 0) -> None:
        self._record(
            P2PDeliver(
                step=self._next_step(),
                src=src,
                dst=dst,
                tag=tag,
                nbytes=nbytes,
            )
        )

    def collective_enter(
        self,
        rank: Rank,
        collective_kind: str,
        collective_index: int,
        group_size: int,
        *,
        nbytes: int,
    ) -> None:
        self._record(
            CollectiveEnter(
                step=self._next_step(),
                rank=rank,
                kind=collective_kind,
                index=collective_index,
                group_size=group_size,
                nbytes=nbytes,
            )
        )

    def collective_sync(
        self,
        collective_kind: str,
        collective_index: int,
        group_size: int,
        *,
        total_payload_bytes: int,
    ) -> None:
        self._record(
            CollectiveSync(
                step=self._next_step(),
                kind=collective_kind,
                index=collective_index,
                group_size=group_size,
                nbytes=total_payload_bytes,
            )
        )

    def summary(self) -> TraceSummary:
        if self._summary is None:
            self._summary = self._compute_summary()
        return self._summary

    def format(self) -> str:
        return _format_summary(self.summary())

    def timeline(self, *, max_events: int | None = 50) -> str:
        return _format_timeline(self.events, max_events=max_events)

    def __str__(self) -> str:
        return self.format()

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def _record(self, event: TraceEvent) -> None:
        self._summary = None
        self._events.append(event)

    def _compute_summary(self) -> TraceSummary:
        world_size = self._world_size
        bytes_sent = [0] * world_size
        bytes_received = [0] * world_size
        messages_sent = [0] * world_size
        messages_received = [0] * world_size
        bytes_matrix = [[0] * world_size for _ in range(world_size)]
        collective_counter: Counter[str] = Counter()
        total_messages = 0

        for event in self._events:
            match event:
                case P2PDeliver(src=src, dst=dst, nbytes=nbytes):
                    bytes_sent[src] += nbytes
                    bytes_received[dst] += nbytes
                    messages_sent[src] += 1
                    messages_received[dst] += 1
                    bytes_matrix[src][dst] += nbytes
                    total_messages += 1
                case CollectiveSync(kind=kind):
                    collective_counter[kind] += 1

        return TraceSummary(
            world_size=world_size,
            total_bytes=sum(bytes_sent),
            total_messages=total_messages,
            bytes_sent=tuple(bytes_sent),
            bytes_received=tuple(bytes_received),
            messages_sent=tuple(messages_sent),
            messages_received=tuple(messages_received),
            bytes_matrix=tuple(tuple(row) for row in bytes_matrix),
            collective_counts=tuple(sorted(collective_counter.items())),
        )


def _format_summary(summary: TraceSummary) -> str:
    lines = [
        "communication trace",
        _rule(),
        "",
        f"p2p  ·  {summary.world_size} ranks",
        f"  messages  {summary.total_messages}",
        f"  bytes     {summary.total_bytes}",
    ]

    if summary.world_size:
        lines.extend(["", *_format_rank_table(summary), "", *_format_delivery_matrix(summary)])

    lines.extend(["", *_format_collectives(summary.collective_counts)])
    return "\n".join(lines)


def _format_rank_table(summary: TraceSummary) -> list[str]:
    rank_width = max(1, len(str(summary.world_size - 1)))
    sent_b = max(6, len(str(max(summary.bytes_sent, default=0))))
    recv_b = max(6, len(str(max(summary.bytes_received, default=0))))
    sent_m = max(8, len(str(max(summary.messages_sent, default=0))))
    recv_m = max(8, len(str(max(summary.messages_received, default=0))))
    columns = (
        ("rank", rank_width + 2),
        ("sent B", sent_b),
        ("recv B", recv_b),
        ("sent msg", sent_m),
        ("recv msg", recv_m),
    )
    header = "  ".join(f"{label:>{width}}" for label, width in columns)
    rows = [header, _rule(len(header))]
    for rank in range(summary.world_size):
        rows.append(
            "  ".join(
                [
                    f"{rank:>{rank_width + 2}}",
                    f"{summary.bytes_sent[rank]:>{sent_b}}",
                    f"{summary.bytes_received[rank]:>{recv_b}}",
                    f"{summary.messages_sent[rank]:>{sent_m}}",
                    f"{summary.messages_received[rank]:>{recv_m}}",
                ]
            )
        )
    return ["per rank", *rows]


def _format_delivery_matrix(summary: TraceSummary) -> list[str]:
    world_size = summary.world_size
    rank_width = max(1, len(str(world_size - 1)))
    max_bytes = max((value for row in summary.bytes_matrix for value in row), default=0)
    cell_width = max(1, len(str(max_bytes)))

    header = " " * (rank_width + 3) + " ".join(f"{dst:>{cell_width}}" for dst in range(world_size))
    rows = [
        "delivery matrix (bytes, src row → dst column)",
        header,
        _rule(len(header)),
    ]
    for src in range(world_size):
        cells = []
        for dst in range(world_size):
            value = summary.bytes_matrix[src][dst]
            cells.append(f"{'·':>{cell_width}}" if value == 0 else f"{value:>{cell_width}}")
        rows.append(f"{src:>{rank_width}} →  " + " ".join(cells))
    return rows


def _format_collectives(collective_counts: tuple[tuple[str, int], ...]) -> list[str]:
    lines = ["collectives"]
    if not collective_counts:
        lines.append("  none")
        return lines
    kind_width = max(len(kind) for kind, _ in collective_counts)
    for kind, count in collective_counts:
        lines.append(f"  {kind:<{kind_width}}  {count}")
    return lines


def _format_timeline(events: tuple[TraceEvent, ...], *, max_events: int | None) -> str:
    selected = events if max_events is None else events[:max_events]
    lines = [
        "timeline",
        _rule(),
        f"  {'#':>4}  {'event':<12}  detail",
        _rule(28),
        *(_format_event(event) for event in selected),
    ]
    if max_events is not None and len(events) > max_events:
        lines.append(f"  ... {len(events) - max_events} more events")
    return "\n".join(lines)


def _format_event(event: TraceEvent) -> str:
    match event:
        case P2PBegin(step=step, rank=rank, op=op, peer=peer, nbytes=nbytes):
            detail = f"rank {rank} {op}→{peer} ({nbytes} B)"
            return f"  {step:>4}  {'begin':<12}  {detail}"
        case P2PDeliver(step=step, src=src, dst=dst, nbytes=nbytes):
            detail = f"{src}→{dst} ({nbytes} B)"
            return f"  {step:>4}  {'deliver':<12}  {detail}"
        case CollectiveEnter(step=step, rank=rank, kind=kind, index=index, nbytes=nbytes):
            detail = f"rank {rank} {kind} #{index} ({nbytes} B)"
            return f"  {step:>4}  {'enter':<12}  {detail}"
        case CollectiveSync(step=step, kind=kind, index=index, nbytes=nbytes):
            detail = f"{kind} #{index} ({nbytes} B)"
            return f"  {step:>4}  {'sync':<12}  {detail}"
        case _:
            return f"  {event.step:>4}  {type(event).__name__:<12}"


def _rule(width: int = 19) -> str:
    return "─" * width


def normalize_rank_values(
    values: Sequence[int] | Any,
    *,
    world_size: int,
    label: str,
) -> tuple[int, ...]:
    if hasattr(values, "values"):
        values = values.values
    if len(values) != world_size:
        raise ValueError(f"{label} must have one entry per rank ({world_size}), got {len(values)}")
    return tuple(int(value) for value in values)
