"""Domain errors raised by the MLOps outer loop."""


class MLOpsLoopError(Exception):
    """Base error for deterministic loop failures."""


class ConcurrencyConflict(MLOpsLoopError):
    """The caller acted on a stale run version."""


class IdempotencyConflict(MLOpsLoopError):
    """An idempotency key was reused with a different payload."""


class InvalidTransition(MLOpsLoopError):
    """A domain state transition violated the lifecycle contract."""


class InvariantViolation(MLOpsLoopError):
    """A persistent domain invariant would have been broken."""


class LeaseConflict(MLOpsLoopError):
    """An attempt was accessed without its current fencing lease."""
