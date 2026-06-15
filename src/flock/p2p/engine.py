from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from flock.context import get_rank as get_current_rank
from flock.errors import FlockUsageError
from flock.p2p.handle import P2PHandle
from flock.p2p.ops import Irecv, Isend, P2PCall, Recv, Send
from flock.scheduler.port import SchedulePort
from flock.types import ANY_SOURCE, ANY_TAG, Rank, RequestId


@dataclass
class Message:
    src: Rank
    value: Any
    tag: int = 0
    ack: bool = False
    send_id: int | None = None


@dataclass
class P2PRequest:
    handle: P2PHandle
    done: bool = False


class P2PEngine:
    def __init__(self, port: SchedulePort) -> None:
        self._port = port
        self.mailboxes: defaultdict[Rank, deque[Message]] = defaultdict(deque)
        self.requests: dict[RequestId, P2PRequest] = {}
        self.blocked: dict[Rank, P2PHandle] = {}
        self._request_counter = 0

    def begin(self, call: P2PCall) -> P2PHandle:
        rank = get_current_rank()
        self._validate_peer(call)
        handle = self._new_handle(call.kind, rank, call.peer, call.tag)

        match call:
            case Isend(value=value):
                self._deliver(rank, call.peer, value, call.tag, ack=False)
                self.requests[handle.request_id] = P2PRequest(handle=handle, done=True)

            case Send(value=value):
                self.requests[handle.request_id] = P2PRequest(handle=handle)
                self._deliver(rank, call.peer, value, call.tag, ack=True, send_id=handle.request_id)

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
                message = self._take_message(rank, handle)
                if message is None:
                    self.blocked[rank] = handle
                    return

                self.requests.pop(handle.request_id)
                if message.ack and message.send_id is not None:
                    self._complete_send(message.send_id)
                self._port.resume(rank, message.value)

    def deadlock_lines(self) -> list[str]:
        lines = []
        for rank, handle in sorted(self.blocked.items()):
            peer = "any source" if handle.peer == ANY_SOURCE else f"rank {handle.peer}"
            lines.append(f"rank {rank} is blocked in {handle.kind} waiting for {peer}")
        return lines

    def is_complete(self, handle: P2PHandle) -> bool:
        request = self.requests.get(handle.request_id)
        if request is None:
            return True
        if handle.kind in ("isend", "send"):
            return request.done
        mailbox = self.mailboxes.get(handle.rank)
        if not mailbox:
            return False
        return any(self._matches(handle, message) for message in mailbox)

    def _new_handle(self, kind: str, rank: Rank, peer: Rank, tag: int) -> P2PHandle:
        request_id = self._request_counter
        self._request_counter += 1
        return P2PHandle(kind=kind, rank=rank, peer=peer, tag=tag, request_id=request_id)

    def _validate_peer(self, call: P2PCall) -> None:
        peer = call.peer
        if call.kind in ("recv", "irecv") and peer == ANY_SOURCE:
            return
        world_size = self._port.world_size
        if peer < 0 or peer >= world_size:
            raise FlockUsageError(f"rank {peer} is out of range for world_size={world_size}.")

    def _deliver(
        self, src: Rank, dst: Rank, value: Any, tag: int, *, ack: bool, send_id: int | None = None
    ) -> None:
        receiver = self._waiting_recv(dst, src, tag)
        if receiver is not None:
            del self.blocked[dst]
            self.requests.pop(receiver.request_id)
            if ack and send_id is not None:
                self._complete_send(send_id)
            self._port.resume(dst, value)
            return

        self.mailboxes[dst].append(Message(src=src, value=value, tag=tag, ack=ack, send_id=send_id))

    def _waiting_recv(self, dst: Rank, src: Rank, tag: int) -> P2PHandle | None:
        handle = self.blocked.get(dst)
        if handle is None or handle.kind not in ("recv", "irecv"):
            return None
        if self._matches(handle, Message(src=src, value=None, tag=tag)):
            return handle
        return None

    def _take_message(self, dst: Rank, handle: P2PHandle) -> Message | None:
        mailbox = self.mailboxes.get(dst)
        if not mailbox:
            return None
        for index, message in enumerate(mailbox):
            if self._matches(handle, message):
                del mailbox[index]
                return message
        return None

    @staticmethod
    def _matches(handle: P2PHandle, message: Message) -> bool:
        src_ok = handle.peer == ANY_SOURCE or handle.peer == message.src
        tag_ok = handle.tag == ANY_TAG or handle.tag == message.tag
        return src_ok and tag_ok

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
