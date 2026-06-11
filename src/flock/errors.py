from __future__ import annotations


class FlockError(Exception):
    pass


class FlockDeadlockError(FlockError):
    pass


class FlockUsageError(FlockError):
    pass
