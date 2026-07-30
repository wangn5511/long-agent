"""Independent machine validators for task-agent results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from openhands.mlops_loop.artifact_store import ArtifactStore
from openhands.mlops_loop.errors import InvariantViolation
from openhands.mlops_loop.models import (
    CompletionContract,
    MetricGate,
    TaskResult,
    TaskSpec,
    ValidationDisposition,
    ValidationResult,
)

Validator = Callable[[TaskSpec, TaskResult], ValidationResult]


def _compare(actual: Any, gate: MetricGate) -> bool:
    expected = gate.value
    if gate.op == 'eq':
        return actual == expected
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        return False
    if gate.op == 'gte':
        return actual >= expected
    if gate.op == 'gt':
        return actual > expected
    if gate.op == 'lte':
        return actual <= expected
    if gate.op == 'lt':
        return actual < expected
    raise AssertionError(f'unhandled gate operator: {gate.op}')


class ContractValidator:
    """Validate the structured result against files and the task contract.

    It never reads the Agent's natural-language summary as evidence.
    """

    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def __call__(self, task: TaskSpec, result: TaskResult) -> ValidationResult:
        validation_id = f'validation:{uuid4().hex}'
        if result.status == 'cancelled':
            return self._result(
                validation_id,
                ValidationDisposition.TERMINAL,
                ['task agent cancelled the attempt'],
            )
        if result.status == 'waiting_external':
            return self._result(
                validation_id,
                ValidationDisposition.TERMINAL,
                ['waiting_external is a scheduling signal, not a final result'],
            )
        if result.status == 'needs_decision':
            return self._result(
                validation_id,
                ValidationDisposition.NEEDS_DECISION,
                ['task agent requested a separate semantic decision'],
            )
        if result.status == 'failed':
            if result.next_attempt_proposal is None:
                return self._result(
                    validation_id,
                    ValidationDisposition.TERMINAL,
                    ['failed task omitted its required next-attempt proposal'],
                )
            if not result.evidence_refs:
                return self._result(
                    validation_id,
                    ValidationDisposition.TERMINAL,
                    ['failed task did not provide machine-readable evidence'],
                )
            return self._result(
                validation_id,
                ValidationDisposition.RETRYABLE,
                [
                    'failure proposal is structurally valid; retry policy must '
                    'independently authorize its failure class and budget'
                ],
            )
        return self._validate_success(task.completion_contract, result, validation_id)

    def _validate_success(
        self,
        contract: CompletionContract,
        result: TaskResult,
        validation_id: str,
    ) -> ValidationResult:
        reasons: list[str] = []
        if (
            result.metrics.get('skip_training') is True
            and not contract.allow_skip_training
        ):
            reasons.append('completion contract does not allow skipping training')
        ids = [item.candidate_id for item in result.artifact_candidates]
        if len(ids) != len(set(ids)):
            reasons.append('artifact candidate ids are not unique')
        kinds = {item.kind for item in result.artifact_candidates}
        missing_kinds = sorted(set(contract.required_artifact_kinds) - kinds)
        if missing_kinds:
            reasons.append(f'missing required artifact kinds: {missing_kinds}')
        for candidate in result.artifact_candidates:
            if candidate.validation_uri is not None:
                reasons.append(
                    f'artifact {candidate.candidate_id} supplied a reserved '
                    'validation URI'
                )
                continue
            try:
                self.artifact_store.verify(candidate)
            except (OSError, ValueError, InvariantViolation) as exc:
                reasons.append(
                    f'artifact {candidate.candidate_id} failed hash verification: {exc}'
                )
        if contract.required_evidence and not result.evidence_refs:
            reasons.append('completion contract requires evidence refs')
        for gate in contract.gates:
            actual = result.metrics.get(gate.metric)
            if actual is None:
                reasons.append(f'missing required metric: {gate.metric}')
            elif not _compare(actual, gate):
                reasons.append(
                    f'metric gate failed: {gate.metric}={actual!r} '
                    f'{gate.op} {gate.value!r}'
                )
        if reasons:
            return self._result(
                validation_id,
                ValidationDisposition.TERMINAL,
                reasons,
                metrics=result.metrics,
            )
        return ValidationResult(
            validation_id=validation_id,
            disposition=ValidationDisposition.ACCEPTED,
            reasons=['all artifact, evidence, and metric gates passed'],
            verified_metrics=result.metrics,
            accepted_candidate_ids=ids,
        )

    @staticmethod
    def _result(
        validation_id: str,
        disposition: ValidationDisposition,
        reasons: list[str],
        *,
        metrics: dict[str, Any] | None = None,
    ) -> ValidationResult:
        return ValidationResult(
            validation_id=validation_id,
            disposition=disposition,
            reasons=reasons,
            verified_metrics=metrics or {},
        )


class ValidatorRegistry:
    """Named validator boundary used by LoopSpec task definitions."""

    def __init__(self, default: Validator):
        self.default = default
        self._validators: dict[str, Validator] = {}

    def register(self, name: str, validator: Validator) -> None:
        if name in self._validators:
            raise ValueError(f'validator is already registered: {name}')
        self._validators[name] = validator

    def validate(self, task: TaskSpec, result: TaskResult) -> ValidationResult:
        return self._validators.get(task.validator, self.default)(task, result)
