from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from flock.context import get_rank as get_current_rank
from flock.errors import FlockUsageError
from flock.p2p.handle import P2PHandle
from flock.p2p.ops import Irecv, Isend, P2PCall, Recv, Send
from flock.scheduler.port import SchedulePort
from flock.types import Rank, RequestId


@dataclass
class Message:
    src: Rank
    value: Any
    ack: bool = False
    send_id: int | None = None


@dataclass
class P2PRequest:
    handle: P2PHandle
    done: bool = False


class P2PEngine:
    def __init__(self, port: SchedulePort) -> None:
        self._port = port
        self.suspended: dict[Rank, Rank] = {}
        self.mailboxes: defaultdict[Rank, defaultdict[Rank, deque[Message]]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self.requests: dict[RequestId, P2PRequest] = {}
        self.blocked: dict[Rank, P2PHandle] = {}
        self._request_counter = 0

    def begin(self, call: P2PCall) -> P2PHandle:
        rank = get_current_rank()
        self._validate_peer(call.peer)
        handle = self._new_handle(call.kind, rank, call.peer)

        match call:
            case Isend(value=value):
                self._deliver(rank, call.peer, value, ack=False)
                self.requests[handle.request_id] = P2PRequest(handle=handle, done=True)

            case Send(value=value):
                if self.suspended.get(call.peer) == rank:
                    del self.suspended[call.peer]
                    recv_handle = self.blocked.pop(call.peer)
                    self.requests.pop(recv_handle.request_id)
                    self.requests[handle.request_id] = P2PRequest(handle=handle, done=True)
                    self._port.resume(call.peer, value)
                else:
                    self.requests[handle.request_id] = P2PRequest(handle=handle)
                    self.mailboxes[call.peer][rank].append(
                        Message(src=rank, value=value, ack=True, send_id=handle.request_id)
                    )

            case Recv() | Irecv():
                self.requests[handle.request_id] = P2PRequest(handle=handle)

            case _:
                raise TypeError(f"unknown p2p call: {call!r}")

        return handle

    def wait(self, rank: Rank, handle: P2PHandle) -> None:
        request = self.requests.get(handle.request_id)
        if request is None:
            raise FlockUsageError(
                f"rank {rank} tried to wait on {handle.kind} without starting it on this rank."
            )

        match handle.kind:
            case "isend":
                self.requests.pop(handle.request_id)
                self._port.resume(rank)

            case "send":
                if request.done:
                    self.requests.pop(handle.request_id)
                    self._port.resume(rank)
                else:
                    self.blocked[rank] = handle

            case "recv" | "irecv":
                message = self._take_message(rank, handle.peer)
                if message is None:
                    self.suspended[rank] = handle.peer
                    self.blocked[rank] = handle
                    return

                self.requests.pop(handle.request_id)
                if message.ack and message.send_id is not None:
                    self._complete_send(message.send_id)
                self._port.resume(rank, message.value)

    def deadlock_lines(self) -> list[str]:
        lines: list[str] = []

        seen_send: set[Rank] = set()

        for rank, handle in sorted(self.blocked.items()):
            lines.append(f"rank {rank} is blocked in {handle.kind} waiting for rank {handle.peer}")
            if handle.kind == "send":
                seen_send.add(rank)

        for dst, by_src in sorted(self.mailboxes.items()):
            for mailbox in by_src.values():
                for message in mailbox:
                    if message.ack and message.send_id is not None:
                        request = self.requests.get(message.send_id)
                        if request is not None and not request.done and message.src is not seen_send:
                            lines.append(f"rank {message.src} is blocked in send waiting for rank {dst}")

        return lines

    def _new_handle(self, kind: str, rank: Rank, peer: Rank) -> P2PHandle:
        request_id = self._request_counter
        self._request_counter += 1
        return P2PHandle(kind=kind, rank=rank, peer=peer, request_id=request_id)

    def _validate_peer(self, peer: Rank) -> None:
        world_size = self._port.world_size
        if peer < 0 or peer >= world_size:
            raise FlockUsageError(f"rank {peer} is out of range for world_size={world_size}.")

    def _deliver(self, src: Rank, dst: Rank, value: Any, *, ack: bool, send_id: int | None = None) -> None:
        if self.suspended.get(dst) == src:
            del self.suspended[dst]
            recv_handle = self.blocked.pop(dst)
            self.requests.pop(recv_handle.request_id)
            self._port.resume(dst, value)
            return

        self.mailboxes[dst][src].append(Message(src=src, value=value, ack=ack, send_id=send_id))

    def _take_message(self, dst: Rank, src: Rank) -> Message | None:
        mailbox = self.mailboxes[dst].get(src)
        if not mailbox:
            return None
        return mailbox.popleft()

    def _complete_send(self, send_id: int) -> None:
        request = self.requests.get(send_id)
        if request is None:
            return

        request.done = True
        rank = request.handle.rank
        if rank in self.blocked and self.blocked[rank].request_id == send_id:
            del self.blocked[rank]
            self.requests.pop(send_id)
            self._port.resume(rank)
