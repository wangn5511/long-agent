"""Deterministic attempt-level retry policy."""

from dataclasses import dataclass, field

from openhands.mlops_loop.models import TaskResult, ValidationDisposition


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    reason: str
    failure_class: str | None = None
    spec_diff: dict[str, object] = field(default_factory=dict)
    next_attempt_number: int | None = None


@dataclass(frozen=True)
class RetryPolicy:
    retryable_failure_classes: frozenset[str]

    def decide(
        self,
        *,
        result: TaskResult,
        disposition: ValidationDisposition,
        attempt_number: int,
        max_attempts: int,
    ) -> RetryDecision:
        if disposition is not ValidationDisposition.RETRYABLE:
            return RetryDecision(False, f'validator disposition is {disposition.value}')
        proposal = result.next_attempt_proposal
        if proposal is None:
            return RetryDecision(False, 'retryable result has no next-attempt proposal')
        if proposal.failure_class not in self.retryable_failure_classes:
            return RetryDecision(
                False,
                f'failure class {proposal.failure_class!r} is not retryable',
                proposal.failure_class,
            )
        if proposal.failure_class == 'cuda-oom-with-valid-proposal':
            batch_size = proposal.spec_diff.get('batch_size')
            if (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
                or batch_size < 1
            ):
                return RetryDecision(
                    False,
                    'CUDA OOM retry requires a positive batch_size spec diff',
                    proposal.failure_class,
                )
        if attempt_number >= max_attempts:
            return RetryDecision(
                False,
                f'attempt budget exhausted ({attempt_number}/{max_attempts})',
                proposal.failure_class,
            )
        return RetryDecision(
            True,
            proposal.reason,
            proposal.failure_class,
            dict(proposal.spec_diff),
            attempt_number + 1,
        )
