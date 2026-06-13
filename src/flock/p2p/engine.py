from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from flock.context import rank as current_rank
from flock.errors import FlockDeadlockError, FlockUsageError
from flock.p2p.handle import P2PHandle
from flock.scheduler.port import SchedulePort
from flock.types import Rank


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
        self.mailboxes: defaultdict[Rank, deque[Message]] = defaultdict(deque)
        self.requests: dict[Rank, P2PRequest] = {}
        self.blocked: dict[Rank, P2PHandle] = {}
        self._next_request_id = 0

    def begin_isend(self, dst: Rank, value: Any) -> P2PHandle:
        rank = current_rank()
        handle = self._new_handle("isend", rank, dst)
        self._deliver(rank, dst, value, ack=False)
        self.requests[handle.request_id] = P2PRequest(handle=handle, done=True)
        return handle

    def begin_send(self, dst: Rank, value: Any) -> P2PHandle:
        rank = current_rank()
        handle = self._new_handle("send", rank, dst)

        if self.suspended.get(dst) == rank:
            del self.suspended[dst]
            recv_handle = self.blocked.pop(dst)
            self.requests.pop(recv_handle.request_id)
            self.requests[handle.request_id] = P2PRequest(handle=handle, done=True)
            self._port.resume(dst, value)
        else:
            self.requests[handle.request_id] = P2PRequest(handle=handle)
            self.mailboxes[dst].append(Message(src=rank, value=value, ack=True, send_id=handle.request_id))

        return handle

    def begin_recv(self, src: Rank) -> P2PHandle:
        rank = current_rank()
        handle = self._new_handle("recv", rank, src)
        self.requests[handle.request_id] = P2PRequest(handle=handle)
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

            case "recv":
                if not self.mailboxes[rank]:
                    self.suspended[rank] = handle.peer
                    self.blocked[rank] = handle
                    return

                message = self.mailboxes[rank].popleft()

                if message.src != handle.peer:
                    raise FlockDeadlockError(
                        f"Expected message from rank {handle.peer}, got message from rank {message.src}"
                    )

                self.requests.pop(handle.request_id)
                if message.ack and message.send_id is not None:
                    self._complete_send(message.send_id)
                self._port.resume(rank, message.value)

    def deadlock_lines(self) -> list[str]:
        lines: list[str] = []

        for rank, handle in sorted(self.blocked.items()):
            lines.append(f"rank {rank} is blocked in {handle.kind} waiting for rank {handle.peer}")

        for dst, mailbox in sorted(self.mailboxes.items()):
            for message in mailbox:
                if message.ack and message.send_id is not None:
                    request = self.requests.get(message.send_id)
                    if request is not None and not request.done:
                        lines.append(f"rank {message.src} is blocked in send waiting for rank {dst}")

        return lines

    def _new_handle(self, kind: str, rank: Rank, peer: Rank) -> P2PHandle:
        request_id = self._next_request_id
        self._next_request_id += 1
        return P2PHandle(kind=kind, rank=rank, peer=peer, request_id=request_id)

    def _deliver(self, src: Rank, dst: Rank, value: Any, *, ack: bool, send_id: int | None = None) -> None:
        if self.suspended.get(dst) == src:
            del self.suspended[dst]
            recv_handle = self.blocked.pop(dst)
            self.requests.pop(recv_handle.request_id)
            self._port.resume(dst, value)
            return

        self.mailboxes[dst].append(Message(src=src, value=value, ack=ack, send_id=send_id))

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
