"""Typed protocol shared by the Loop Controller and task agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TASK_RESULT_PROTOCOL: Literal['mlops.openhands.dev/task-result/v1'] = (
    'mlops.openhands.dev/task-result/v1'
)


class RunState(StrEnum):
    CREATED = 'CREATED'
    WAITING_FOR_AGENT = 'WAITING_FOR_AGENT'
    WAITING_EXTERNAL = 'WAITING_EXTERNAL'
    VALIDATING_RESULT = 'VALIDATING_RESULT'
    RETRY_SCHEDULED = 'RETRY_SCHEDULED'
    NEEDS_INTERVENTION = 'NEEDS_INTERVENTION'
    COMPLETED = 'COMPLETED'
    COMPLETED_NO_TRAINING = 'COMPLETED_NO_TRAINING'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class TaskStatus(StrEnum):
    PENDING = 'PENDING'
    DISPATCHED = 'DISPATCHED'
    RUNNING = 'RUNNING'
    WAITING_EXTERNAL = 'WAITING_EXTERNAL'
    RESULT_SUBMITTED = 'RESULT_SUBMITTED'
    VALIDATING = 'VALIDATING'
    SUCCEEDED = 'SUCCEEDED'
    RETRY_SCHEDULED = 'RETRY_SCHEDULED'
    NEEDS_DECISION = 'NEEDS_DECISION'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class AttemptStatus(StrEnum):
    CREATED = 'CREATED'
    DISPATCHED = 'DISPATCHED'
    RUNNING = 'RUNNING'
    WAITING_EXTERNAL = 'WAITING_EXTERNAL'
    RESULT_SUBMITTED = 'RESULT_SUBMITTED'
    ACCEPTED = 'ACCEPTED'
    REJECTED = 'REJECTED'
    TIMED_OUT = 'TIMED_OUT'
    CRASHED = 'CRASHED'
    CANCELLED = 'CANCELLED'


class ValidationDisposition(StrEnum):
    ACCEPTED = 'accepted'
    RETRYABLE = 'retryable'
    TERMINAL = 'terminal'
    NEEDS_DECISION = 'needs_decision'


class ArtifactCandidate(BaseModel):
    """An untrusted artifact proposal produced inside one task attempt."""

    model_config = ConfigDict(extra='forbid')

    candidate_id: str = Field(min_length=1)
    kind: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    size_bytes: int = Field(ge=0)
    media_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # The storage implementation may retain the verified staging URI while it
    # returns a remote sealed URI.  It is deliberately excluded from every
    # wire/persistence representation and rejected by validators when supplied
    # by an Agent.
    validation_uri: str | None = Field(default=None, exclude=True)


class LineageProposal(BaseModel):
    """A proposed provenance edge, persisted only after validation."""

    model_config = ConfigDict(extra='forbid')

    input_artifact_id: str = Field(min_length=1)
    output_candidate_id: str = Field(min_length=1)
    transform_kind: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    code_git_sha: str | None = Field(default=None, pattern=r'^[a-f0-9]{40,64}$')
    container_image_digest: str | None = Field(
        default=None, pattern=r'^sha256:[a-f0-9]{64}$'
    )
    config_artifact_id: str | None = None

    @model_validator(mode='after')
    def reject_self_edge(self) -> LineageProposal:
        if self.input_artifact_id == self.output_candidate_id:
            raise ValueError('lineage self-edges are not allowed')
        return self


class NextAttemptProposal(BaseModel):
    """Agent advice; it never allocates or dispatches the next attempt."""

    model_config = ConfigDict(extra='forbid')

    failure_class: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    spec_diff: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class TaskResult(BaseModel):
    """The structured completion or scheduling message accepted from an Agent."""

    model_config = ConfigDict(extra='forbid')

    protocol: Literal['mlops.openhands.dev/task-result/v1'] = TASK_RESULT_PROTOCOL
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    status: Literal[
        'succeeded',
        'failed',
        'needs_decision',
        'cancelled',
        'waiting_external',
    ]
    summary: str = Field(min_length=1)
    artifact_candidates: list[ArtifactCandidate] = Field(default_factory=list)
    lineage_proposals: list[LineageProposal] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    next_attempt_proposal: NextAttemptProposal | None = None

    @model_validator(mode='after')
    def validate_status_payload(self) -> TaskResult:
        if self.status == 'succeeded' and self.next_attempt_proposal is not None:
            raise ValueError('a successful result cannot propose another attempt')
        if self.status == 'failed' and self.next_attempt_proposal is None:
            raise ValueError('a failed result must include a failure proposal')
        if self.status == 'waiting_external' and (
            self.artifact_candidates
            or self.lineage_proposals
            or self.next_attempt_proposal is not None
        ):
            raise ValueError(
                'a waiting result cannot publish artifacts, lineage, or a retry'
            )
        return self


class MetricGate(BaseModel):
    model_config = ConfigDict(extra='forbid')

    metric: str = Field(min_length=1)
    op: Literal['gte', 'gt', 'lte', 'lt', 'eq']
    value: float | int | bool | str


class CompletionContract(BaseModel):
    """Machine-verifiable facts required before a task can succeed."""

    model_config = ConfigDict(extra='forbid')

    required_artifact_kinds: list[str] = Field(default_factory=list)
    required_evidence: bool = False
    gates: list[MetricGate] = Field(default_factory=list)
    allow_skip_training: bool = False


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    kind: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    objective: str = Field(min_length=1)
    agent_profile: str = Field(min_length=1)
    validator: str = Field(min_length=1)
    max_attempts: int = Field(ge=1, le=20)
    timeout_seconds: int = Field(ge=1)
    completion_contract: CompletionContract = Field(default_factory=CompletionContract)
    requires_human_confirmation: bool = False


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    validation_id: str = Field(min_length=1)
    disposition: ValidationDisposition
    reasons: list[str] = Field(min_length=1)
    verified_metrics: dict[str, float | int | bool | str | None] = Field(
        default_factory=dict
    )
    accepted_candidate_ids: list[str] = Field(default_factory=list)


class ArtifactRef(BaseModel):
    """Read-only reference to an artifact already sealed by the outer loop."""

    model_config = ConfigDict(extra='forbid')

    id: str = Field(min_length=1)
    kind: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    size_bytes: int = Field(ge=0)
    media_type: str | None = None


class TaskContext(BaseModel):
    """Fenced, task-local context injected into a fresh conversation."""

    model_config = ConfigDict(extra='forbid')

    protocol: Literal['mlops.openhands.dev/task-context/v1'] = (
        'mlops.openhands.dev/task-context/v1'
    )
    run_id: str
    run_version: int = Field(ge=1)
    task_id: str
    task_kind: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    attempt_id: str
    attempt_number: int = Field(ge=1)
    lease_id: str
    objective: str
    run_spec: dict[str, Any] = Field(default_factory=dict)
    input_artifact_ids: list[str] = Field(default_factory=list)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    completion_contract: CompletionContract
    attempt_spec: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    domain_refs: dict[str, str] = Field(default_factory=dict)

    @field_validator('lease_id')
    @classmethod
    def lease_must_be_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('lease_id is required')
        return value
