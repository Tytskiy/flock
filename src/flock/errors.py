class FlockError(Exception):
    pass


class FlockDeadlockError(FlockError):
    pass


class FlockUsageError(FlockError):
    pass


class FlockCollectiveMismatch(FlockError):
    pass
