"""Loop Engineering outer controller."""

from __future__ import annotations

import time
from typing import Any

from openhands.mlops_loop.artifact_store import ArtifactStore
from openhands.mlops_loop.dispatcher import (
    ConversationDispatcher,
    conversation_id_for_dispatch,
)
from openhands.mlops_loop.errors import InvariantViolation
from openhands.mlops_loop.ledger import SQLiteLedger
from openhands.mlops_loop.models import (
    ArtifactRef,
    AttemptStatus,
    CompletionContract,
    RunState,
    TaskContext,
    TaskResult,
    TaskSpec,
    ValidationDisposition,
)
from openhands.mlops_loop.retry_policy import RetryPolicy
from openhands.mlops_loop.validators import ValidatorRegistry

TERMINAL_RUN_STATES = {
    RunState.COMPLETED,
    RunState.COMPLETED_NO_TRAINING,
    RunState.NEEDS_INTERVENTION,
    RunState.FAILED,
    RunState.CANCELLED,
}


class LoopController:
    """Own the deterministic lifecycle around task-local OpenHands agents."""

    def __init__(
        self,
        *,
        ledger: SQLiteLedger,
        artifact_store: ArtifactStore,
        dispatcher: ConversationDispatcher,
        validators: ValidatorRegistry,
        retry_policy: RetryPolicy,
        controller_id: str = 'loop-controller-1',
        candidate_registry_required: bool = False,
    ):
        self.ledger = ledger
        self.artifact_store = artifact_store
        self.dispatcher = dispatcher
        self.validators = validators
        self.retry_policy = retry_policy
        self.controller_id = controller_id
        self.candidate_registry_required = candidate_registry_required

    def create_run(
        self,
        *,
        loop_id: str,
        watermark: str,
        run_spec: dict[str, Any],
        tasks: list[TaskSpec],
        initial_artifact_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.ledger.create_run(
            loop_id=loop_id,
            watermark=watermark,
            spec=run_spec,
            tasks=tasks,
            idempotency_key=f'create-run:{loop_id}:{watermark}',
            initial_artifact_ids=initial_artifact_ids or [],
        )

    def bootstrap_artifact(
        self,
        *,
        path: str,
        kind: str,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register an operator-supplied immutable input through validation."""

        candidate = self.artifact_store.stage(
            path,
            kind=kind,
            media_type=media_type,
            metadata=metadata,
        )
        task_spec = TaskSpec(
            name=f'bootstrap-{kind}',
            kind='bootstrap-artifact',
            objective=f'validate and seal operator supplied {kind}',
            agent_profile='default',
            validator='bootstrap-artifact-v1',
            max_attempts=1,
            timeout_seconds=300,
            completion_contract=CompletionContract(
                required_artifact_kinds=[kind], required_evidence=True
            ),
        )
        run = self.create_run(
            loop_id='artifact-bootstrap',
            watermark=f'{kind}:sha256:{candidate.sha256}',
            run_spec={
                'bootstrap': {
                    'kind': kind,
                    'sha256': candidate.sha256,
                    'size_bytes': candidate.size_bytes,
                    'media_type': media_type,
                    'metadata': metadata or {},
                }
            },
            tasks=[task_spec],
        )
        artifacts = self.ledger.list_artifacts(run['id'])
        if artifacts:
            return artifacts[0]
        task = self.ledger.current_task(run['id'])
        if task is None:
            raise RuntimeError('bootstrap run has no current task')
        attempt = self.ledger.latest_attempt(task['id'])
        if attempt is None:
            attempt = self.ledger.create_first_attempt(
                run_id=run['id'],
                task_id=task['id'],
                attempt_spec={},
                expected_version=run['version'],
                idempotency_key=f'first-attempt:{task["id"]}',
            )
        if attempt['status'] == AttemptStatus.CREATED:
            current = self.ledger.get_run(run['id'])
            lease_id = self.ledger.acquire_attempt_lease(
                run_id=run['id'],
                attempt_id=attempt['id'],
                agent_instance_id='operator-bootstrap',
                ttl_seconds=300,
                expected_version=current['version'],
                idempotency_key=f'lease:{attempt["id"]}:bootstrap',
            )
            current = self.ledger.get_run(run['id'])
            self.ledger.dispatch_attempt(
                run_id=run['id'],
                attempt_id=attempt['id'],
                conversation_id=f'operator-bootstrap:{attempt["id"]}',
                dispatch_key=f'operator-bootstrap:{attempt["id"]}',
                lease_id=lease_id,
                expected_version=current['version'],
                idempotency_key=f'link-conversation:{attempt["id"]}',
            )
            attempt = self.ledger.get_attempt(attempt['id'])
        lease = self.ledger.get_lease(attempt['id'])
        if attempt['status'] == AttemptStatus.RUNNING:
            if self.candidate_registry_required:
                self.ledger.record_artifact_candidate(
                    attempt_id=attempt['id'],
                    lease_id=lease['lease_id'],
                    candidate=candidate,
                )
            result = TaskResult(
                task_id=task['id'],
                attempt_id=attempt['id'],
                status='succeeded',
                summary='operator supplied immutable bootstrap artifact',
                artifact_candidates=[candidate],
                evidence_refs=[f'operator-bootstrap:sha256:{candidate.sha256}'],
            )
            self.submit_result(run['id'], result, lease['lease_id'])
        self.validate_current_result(run['id'])
        artifacts = self.ledger.list_artifacts(run['id'])
        if len(artifacts) != 1:
            raise RuntimeError('bootstrap artifact did not pass independent validation')
        return artifacts[0]

    def dispatch_current_task(self, run_id: str) -> str | None:
        run = self.ledger.get_run(run_id)
        if RunState(run['state']) in TERMINAL_RUN_STATES:
            return None
        if self.enforce_run_policy(run_id):
            return None
        run = self.ledger.get_run(run_id)
        task_record = self.ledger.current_task(run_id)
        if task_record is None:
            return None
        task = TaskSpec.model_validate(task_record['spec'])
        if task.requires_human_confirmation and not self.ledger.is_task_approved(
            task_record['id']
        ):
            return None
        if task.kind == 'deploy' and not self.ledger.list_deployments(run_id):
            releases = self.ledger.list_releases(run_id)
            if not releases:
                raise RuntimeError('deploy task requires a sealed release')
            deployment_spec = run['spec'].get('spec', {}).get('deployment', {})
            self.ledger.request_deployment(
                run_id=run_id,
                release_id=releases[-1]['id'],
                environment=deployment_spec.get('environment', 'staging'),
                initial_traffic_percent=float(
                    deployment_spec.get('initialCanaryPercent', 0)
                ),
                expected_version=run['version'],
                idempotency_key=f'deploy:{releases[-1]["id"]}:canary',
            )
            run = self.ledger.get_run(run_id)
        attempt = self.ledger.latest_attempt(task_record['id'])
        if attempt is None:
            attempt = self.ledger.create_first_attempt(
                run_id=run_id,
                task_id=task_record['id'],
                attempt_spec={},
                expected_version=run['version'],
                idempotency_key=f'first-attempt:{task_record["id"]}',
            )
            run = self.ledger.get_run(run_id)
        if attempt['status'] == AttemptStatus.WAITING_EXTERNAL:
            return self._dispatch_attempt_continuation(
                run=run,
                task_record=task_record,
                task=task,
                attempt=attempt,
            )
        if attempt['status'] == AttemptStatus.RUNNING:
            conversation_id = attempt.get('conversation_id')
            dispatch_key = attempt.get('dispatch_key')
            if conversation_id and dispatch_key:
                self.dispatcher.start(
                    conversation_id=conversation_id,
                    dispatch_key=dispatch_key,
                )
            return conversation_id
        if attempt['status'] in {
            AttemptStatus.RESULT_SUBMITTED,
            AttemptStatus.ACCEPTED,
        }:
            return attempt.get('conversation_id')
        if attempt['status'] != AttemptStatus.CREATED:
            return None
        lease_key = f'lease:{attempt["id"]}:{attempt["attempt_number"]}'
        try:
            lease = self.ledger.get_lease(attempt['id'])
            if lease['expires_at'] <= time.time():
                raise KeyError(attempt['id'])
            lease_id = lease['lease_id']
        except KeyError:
            try:
                expired_lease = self.ledger.get_lease(attempt['id'])
                lease_key = (
                    f'lease:{attempt["id"]}:renew:{int(expired_lease["expires_at"])}'
                )
            except KeyError:
                pass
            lease_id = self.ledger.acquire_attempt_lease(
                run_id=run_id,
                attempt_id=attempt['id'],
                agent_instance_id=self.controller_id,
                ttl_seconds=max(task.timeout_seconds + 300, 600),
                expected_version=run['version'],
                idempotency_key=lease_key,
            )
            run = self.ledger.get_run(run_id)
        dispatch_key = f'dispatch:{attempt["id"]}'
        outbox = self.ledger.enqueue_conversation_dispatch(
            run_id=run_id,
            attempt_id=attempt['id'],
            conversation_id=conversation_id_for_dispatch(dispatch_key),
            dispatch_key=dispatch_key,
            lease_id=lease_id,
            expected_version=run['version'],
            idempotency_key=f'queue-dispatch:{attempt["id"]}',
        )
        if outbox['status'] == 'DELIVERED':
            return outbox['external_id']
        if outbox['available_at'] > time.time():
            return None
        run = self.ledger.get_run(run_id)
        context = self._build_task_context(
            run=run,
            task_record=task_record,
            attempt=attempt,
            lease_id=lease_id,
        )
        try:
            conversation_id = self.dispatcher.dispatch(
                context=context, task=task, dispatch_key=dispatch_key
            )
        except Exception as exc:
            current = self.ledger.get_run(run_id)
            self.ledger.record_conversation_dispatch_failure(
                run_id=run_id,
                outbox_id=outbox['id'],
                error=type(exc).__name__,
                expected_version=current['version'],
            )
            raise
        current = self.ledger.get_run(run_id)
        self.ledger.complete_conversation_dispatch(
            run_id=run_id,
            outbox_id=outbox['id'],
            conversation_id=conversation_id,
            lease_id=lease_id,
            expected_version=current['version'],
        )
        self.dispatcher.start(
            conversation_id=conversation_id,
            dispatch_key=dispatch_key,
        )
        return conversation_id

    def get_task_context(self, attempt_id: str, lease_id: str) -> TaskContext:
        """Return only the fenced context for the attempt's current Agent."""

        self.ledger.assert_attempt_lease(attempt_id, lease_id)
        attempt = self.ledger.get_attempt(attempt_id)
        task_record = self.ledger.get_task(attempt['task_id'])
        run = self.ledger.get_run(task_record['run_id'])
        return self._build_task_context(
            run=run,
            task_record=task_record,
            attempt=attempt,
            lease_id=lease_id,
        )

    def submit_result(self, run_id: str, result: TaskResult, lease_id: str) -> RunState:
        run = self.ledger.get_run(run_id)
        attempt = self.ledger.get_attempt(result.attempt_id)
        continuation = attempt['attempt_spec'].get('_continuation', {})
        yield_count = (
            continuation.get('yield_count', 0) if isinstance(continuation, dict) else 0
        )
        if not isinstance(yield_count, int) or isinstance(yield_count, bool):
            raise InvariantViolation('attempt continuation count is invalid')
        running_job_ids = [
            job['id']
            for job in self.ledger.list_job_attempts(run_id)
            if job['task_attempt_id'] == result.attempt_id
            and job['status'] == 'running'
        ]
        should_yield = result.status == 'waiting_external' or (
            result.status == 'cancelled' and bool(running_job_ids)
        )
        episode_number = yield_count
        if attempt['status'] == AttemptStatus.WAITING_EXTERNAL and result.status in {
            'waiting_external',
            'cancelled',
        }:
            # A caller may retry after the yield commit but before receiving
            # its response. That request still belongs to the previous
            # Conversation episode.
            episode_number -= 1
        if episode_number < 0:
            raise InvariantViolation('attempt continuation episode is invalid')
        idempotency_key = f'task-result:{result.attempt_id}:episode:{episode_number}'
        if should_yield:
            if not running_job_ids:
                raise InvariantViolation(
                    'waiting_external requires a running job owned by the attempt'
                )
            if (
                result.artifact_candidates
                or result.lineage_proposals
                or result.next_attempt_proposal is not None
            ):
                raise InvariantViolation(
                    'a yielded Conversation cannot publish artifacts, lineage, '
                    'or a retry proposal'
                )
            self.ledger.yield_task_attempt(
                run_id=run_id,
                result=result,
                lease_id=lease_id,
                running_job_ids=running_job_ids,
                expected_version=run['version'],
                idempotency_key=idempotency_key,
            )
            return RunState.WAITING_EXTERNAL
        self.ledger.submit_task_result(
            run_id=run_id,
            result=result,
            lease_id=lease_id,
            expected_version=run['version'],
            idempotency_key=idempotency_key,
        )
        return RunState.VALIDATING_RESULT

    def validate_current_result(self, run_id: str) -> ValidationDisposition | None:
        run = self.ledger.get_run(run_id)
        if RunState(
            run['state']
        ) not in TERMINAL_RUN_STATES and self.enforce_run_policy(run_id):
            return None
        run = self.ledger.get_run(run_id)
        if run['state'] != RunState.VALIDATING_RESULT:
            return None
        task_record = self.ledger.current_task(run_id)
        if task_record is None:
            return None
        attempt = self.ledger.latest_attempt(task_record['id'])
        if attempt is None or attempt['result_json'] is None:
            return None
        task = TaskSpec.model_validate(task_record['spec'])
        result = TaskResult.model_validate_json(attempt['result_json'])
        validation = self.validators.validate(task, result)
        if self.candidate_registry_required and result.artifact_candidates:
            try:
                self.ledger.assert_attempt_candidates(
                    attempt['id'], result.artifact_candidates
                )
            except InvariantViolation as exc:
                validation = validation.model_copy(
                    update={
                        'disposition': ValidationDisposition.TERMINAL,
                        'reasons': [str(exc)],
                        'accepted_candidate_ids': [],
                    }
                )
        if (
            result.status == 'failed'
            and validation.disposition is ValidationDisposition.RETRYABLE
            and result.next_attempt_proposal is not None
        ):
            proposal = result.next_attempt_proposal
            jobs = [
                job
                for job in self.ledger.list_job_attempts(run_id)
                if job['task_attempt_id'] == attempt['id']
                and f'job:{job["id"]}' in result.evidence_refs
                and job['status'] == 'failed'
                and isinstance(job.get('failure'), dict)
                and job['failure'].get('failure_class') == proposal.failure_class
            ]
            if not jobs:
                validation = validation.model_copy(
                    update={
                        'disposition': ValidationDisposition.TERMINAL,
                        'reasons': [
                            'retry proposal is not backed by a failed job '
                            'observation from this attempt with the same '
                            'failure class'
                        ],
                    }
                )
        artifact_budget = (
            run['spec'].get('spec', {}).get('policy', {}).get('maxArtifactBytes')
        )
        if (
            validation.disposition is ValidationDisposition.ACCEPTED
            and isinstance(artifact_budget, int)
            and sum(item.size_bytes for item in result.artifact_candidates)
            > artifact_budget
        ):
            validation = validation.model_copy(
                update={
                    'disposition': ValidationDisposition.TERMINAL,
                    'reasons': [
                        'accepted artifact candidates exceed Run policy '
                        f'maxArtifactBytes={artifact_budget}'
                    ],
                    'accepted_candidate_ids': [],
                }
            )
        if validation.disposition is ValidationDisposition.ACCEPTED:
            sealed = [
                self.artifact_store.seal(candidate)
                for candidate in result.artifact_candidates
                if candidate.candidate_id in validation.accepted_candidate_ids
            ]
            self.ledger.accept_task_result(
                run_id=run_id,
                result=result,
                validation=validation,
                sealed_candidates=sealed,
                expected_version=run['version'],
                idempotency_key=f'validate-result:{result.attempt_id}',
            )
            if task.kind == 'monitor':
                deployment = self.ledger.list_deployments(run_id)[-1]
                champion = self.ledger.get_champion(run['loop_id'])
                if (
                    deployment['observed_state'] == 'UNHEALTHY'
                    and champion is not None
                    and deployment['release_id'] != champion['release_id']
                ):
                    current = self.ledger.get_run(run_id)
                    self.ledger.schedule_rollback_to_champion(
                        run_id=run_id,
                        expected_version=current['version'],
                        reason='validated canary observation is unhealthy',
                        evidence_refs=result.evidence_refs,
                        idempotency_key=f'rollback:{result.attempt_id}',
                    )
            return validation.disposition
        decision = self.retry_policy.decide(
            result=result,
            disposition=validation.disposition,
            attempt_number=attempt['attempt_number'],
            max_attempts=task.max_attempts,
        )
        self.ledger.reject_task_result(
            run_id=run_id,
            result=result,
            validation=validation,
            retry_decision=decision,
            expected_version=run['version'],
            idempotency_key=f'validate-result:{result.attempt_id}',
        )
        return validation.disposition

    def tick(self, run_id: str) -> RunState:
        """Make one deterministic scheduling or validation decision."""

        run = self.ledger.get_run(run_id)
        state = RunState(run['state'])
        if state in TERMINAL_RUN_STATES:
            return state
        if self.enforce_run_policy(run_id):
            return RunState(self.ledger.get_run(run_id)['state'])
        run = self.ledger.get_run(run_id)
        state = RunState(run['state'])
        if state is RunState.VALIDATING_RESULT:
            self.validate_current_result(run_id)
        else:
            self.dispatch_current_task(run_id)
        return RunState(self.ledger.get_run(run_id)['state'])

    def record_cost(
        self,
        run_id: str,
        *,
        source_type: str,
        source_id: str,
        amount_usd: float,
        metadata: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        run = self.ledger.get_run(run_id)
        entry = self.ledger.record_cost(
            run_id=run_id,
            source_type=source_type,
            source_id=source_id,
            amount_usd=amount_usd,
            metadata=metadata,
            expected_version=run['version'],
            idempotency_key=idempotency_key,
        )
        self.enforce_run_policy(run_id)
        return entry

    def enforce_run_policy(self, run_id: str) -> bool:
        run = self.ledger.get_run(run_id)
        if RunState(run['state']) in TERMINAL_RUN_STATES:
            return False
        policy = run['spec'].get('spec', {}).get('policy', {})
        max_duration = policy.get('maxRunDurationSeconds')
        if (
            isinstance(max_duration, int)
            and not isinstance(max_duration, bool)
            and time.time() - run['created_at'] >= max_duration
        ):
            return self.ledger.stop_run_for_policy(
                run_id=run_id,
                reason=f'Run exceeded maxRunDurationSeconds={max_duration}',
                policy='maxRunDurationSeconds',
                expected_version=run['version'],
                idempotency_key='policy-stop:max-run-duration',
            )
        max_cost = policy.get('maxRunCostUsd')
        if (
            isinstance(max_cost, (int, float))
            and not isinstance(max_cost, bool)
            and self.ledger.total_cost_usd(run_id) >= float(max_cost)
        ):
            total = self.ledger.total_cost_usd(run_id)
            return self.ledger.stop_run_for_policy(
                run_id=run_id,
                reason=f'Run cost ${total:.6f} reached maxRunCostUsd={max_cost}',
                policy='maxRunCostUsd',
                expected_version=run['version'],
                idempotency_key='policy-stop:max-run-cost',
            )
        return False

    def reconcile_expired_attempt(self, run_id: str) -> bool:
        """Recover a lost Agent without relying on its Conversation memory."""

        task = self.ledger.current_task(run_id)
        if task is None:
            return False
        attempt = self.ledger.latest_attempt(task['id'])
        if attempt is None or attempt['status'] != AttemptStatus.RUNNING:
            return False
        try:
            lease = self.ledger.get_lease(attempt['id'])
        except KeyError:
            return False
        if lease['expires_at'] > time.time():
            return False
        run = self.ledger.get_run(run_id)
        self.ledger.recover_expired_attempt(
            run_id=run_id,
            attempt_id=attempt['id'],
            expected_version=run['version'],
            idempotency_key=f'recover-expired:{attempt["id"]}',
        )
        return True

    def _build_task_context(
        self,
        *,
        run: dict[str, Any],
        task_record: dict[str, Any],
        attempt: dict[str, Any],
        lease_id: str,
    ) -> TaskContext:
        task = TaskSpec.model_validate(task_record['spec'])
        pending_ids = run['spec'].get('spec', {}).get('pending_artifact_ids', [])
        artifact_records = self.ledger.list_artifacts(run['id'])
        known_ids = {artifact['id'] for artifact in artifact_records}
        artifact_records.extend(
            self.ledger.get_artifacts(
                [
                    artifact_id
                    for artifact_id in pending_ids
                    if artifact_id not in known_ids
                ]
            )
        )
        releases = self.ledger.list_releases(run['id'])
        deployments = self.ledger.list_deployments(run['id'])
        rollback = self.ledger.get_active_rollback(run['id'])
        attempt_jobs = [
            job
            for job in self.ledger.list_job_attempts(run['id'])
            if job['task_attempt_id'] == attempt['id']
        ]
        return TaskContext(
            run_id=run['id'],
            run_version=run['version'],
            task_id=task_record['id'],
            task_kind=task.kind,
            attempt_id=attempt['id'],
            attempt_number=attempt['attempt_number'],
            lease_id=lease_id,
            objective=task.objective,
            run_spec=run['spec'],
            input_artifact_ids=list(
                dict.fromkeys(artifact['id'] for artifact in artifact_records)
            ),
            input_artifacts=[
                ArtifactRef(
                    id=artifact['id'],
                    kind=artifact['kind'],
                    uri=artifact['uri'],
                    sha256=artifact['sha256'],
                    size_bytes=artifact['size_bytes'],
                    media_type=artifact['media_type'],
                )
                for artifact in artifact_records
            ],
            completion_contract=task.completion_contract,
            attempt_spec=attempt['attempt_spec'],
            evidence=[
                {
                    'kind': 'job-attempt',
                    'job_id': job['id'],
                    'status': job['status'],
                    'return_code': job['return_code'],
                    'result': job['result'],
                    'failure': job['failure'],
                    'log_path': job['log_path'],
                    'log_sha256': job['log_sha256'],
                    'log_size_bytes': job['log_size_bytes'],
                }
                for job in attempt_jobs
            ],
            domain_refs={
                **({'release_id': releases[-1]['id']} if releases else {}),
                **({'deployment_id': deployments[-1]['id']} if deployments else {}),
                **(
                    {
                        'rollback_id': rollback['id'],
                        'rollback_target_release_id': rollback['target_release_id'],
                    }
                    if rollback is not None
                    else {}
                ),
            },
        )

    def _dispatch_attempt_continuation(
        self,
        *,
        run: dict[str, Any],
        task_record: dict[str, Any],
        task: TaskSpec,
        attempt: dict[str, Any],
    ) -> str:
        continuation = attempt['attempt_spec'].get('_continuation')
        if not isinstance(continuation, dict):
            raise InvariantViolation('waiting attempt has no continuation state')
        yield_count = continuation.get('yield_count')
        if (
            not isinstance(yield_count, int)
            or isinstance(yield_count, bool)
            or yield_count < 1
        ):
            raise InvariantViolation('attempt continuation count is invalid')
        lease_id, lease_generation = self.ledger.rotate_attempt_lease_for_continuation(
            run_id=run['id'],
            attempt_id=attempt['id'],
            agent_instance_id=self.controller_id,
            ttl_seconds=max(task.timeout_seconds + 300, 600),
            expected_version=run['version'],
            idempotency_key=(f'continuation-lease:{attempt["id"]}:{yield_count}'),
        )
        run = self.ledger.get_run(run['id'])
        attempt = self.ledger.get_attempt(attempt['id'])
        dispatch_key = f'dispatch:{attempt["id"]}:continuation:{yield_count}'
        if lease_generation:
            # A Conversation created with an expired lease cannot be reused:
            # its immutable TaskContext still contains the stale fence token.
            dispatch_key += f':lease-renewal:{lease_generation}'
        context = self._build_task_context(
            run=run,
            task_record=task_record,
            attempt=attempt,
            lease_id=lease_id,
        )
        conversation_id = self.dispatcher.dispatch(
            context=context,
            task=task,
            dispatch_key=dispatch_key,
        )
        run = self.ledger.get_run(run['id'])
        self.ledger.continue_task_attempt(
            run_id=run['id'],
            attempt_id=attempt['id'],
            conversation_id=conversation_id,
            dispatch_key=dispatch_key,
            lease_id=lease_id,
            expected_version=run['version'],
            idempotency_key=(f'continuation-dispatched:{attempt["id"]}:{yield_count}'),
        )
        self.dispatcher.start(
            conversation_id=conversation_id,
            dispatch_key=dispatch_key,
        )
        return conversation_id

    def approve_current_task(
        self, run_id: str, *, approved_by: str, reason: str
    ) -> None:
        run = self.ledger.get_run(run_id)
        task = self.ledger.current_task(run_id)
        if task is None:
            raise RuntimeError('run has no current task')
        self.ledger.approve_task(
            run_id=run_id,
            task_id=task['id'],
            approved_by=approved_by,
            reason=reason,
            expected_version=run['version'],
            idempotency_key=f'approve:{task["id"]}',
        )
