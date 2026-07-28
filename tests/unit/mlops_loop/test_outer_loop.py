import json
from pathlib import Path

import pytest

from openhands.mlops_loop.artifact_store import LocalArtifactStore
from openhands.mlops_loop.controller import LoopController
from openhands.mlops_loop.dispatcher import InMemoryConversationDispatcher
from openhands.mlops_loop.errors import (
    IdempotencyConflict,
    InvariantViolation,
    LeaseConflict,
)
from openhands.mlops_loop.ledger import SQLiteLedger
from openhands.mlops_loop.models import (
    CompletionContract,
    LineageProposal,
    NextAttemptProposal,
    RunState,
    TaskContext,
    TaskResult,
    TaskSpec,
)
from openhands.mlops_loop.retry_policy import RetryPolicy
from openhands.mlops_loop.rightarm_validators import (
    build_rightarm_validator_registry,
)
from openhands.mlops_loop.spec import RIGHTARM_RELEASE_REQUIRED_KINDS
from openhands.mlops_loop.validators import ContractValidator, ValidatorRegistry


@pytest.fixture
def outer_loop(
    tmp_path: Path,
) -> tuple[
    LoopController, SQLiteLedger, LocalArtifactStore, InMemoryConversationDispatcher
]:
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    dispatcher = InMemoryConversationDispatcher()
    validators = ValidatorRegistry(ContractValidator(artifacts))
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=dispatcher,
        validators=validators,
        retry_policy=RetryPolicy(
            frozenset({'transient-network', 'cuda-oom-with-valid-proposal'})
        ),
    )
    return controller, ledger, artifacts, dispatcher


def _task(name: str, kind: str, artifact_kind: str, *, max_attempts: int = 2):
    return TaskSpec(
        name=name,
        kind=kind,
        objective=f'produce {artifact_kind}',
        agent_profile='test-profile',
        validator='test-validator',
        max_attempts=max_attempts,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=[artifact_kind], required_evidence=True
        ),
    )


def _active_attempt(ledger: SQLiteLedger, run_id: str):
    task = ledger.current_task(run_id)
    assert task is not None
    attempt = ledger.latest_attempt(task['id'])
    assert attempt is not None
    lease = ledger.get_lease(attempt['id'])
    return task, attempt, lease


def _success_result(
    *,
    ledger: SQLiteLedger,
    artifacts: LocalArtifactStore,
    run_id: str,
    output: Path,
    kind: str,
    lineage_input: str | None = None,
) -> tuple[TaskResult, str]:
    task, attempt, lease = _active_attempt(ledger, run_id)
    candidate = artifacts.stage(output, kind=kind)
    lineage = []
    if lineage_input:
        lineage.append(
            LineageProposal(
                input_artifact_id=lineage_input,
                output_candidate_id=candidate.candidate_id,
                transform_kind=f'build-{kind}',
                code_git_sha='a' * 40,
                container_image_digest=f'sha256:{"b" * 64}',
            )
        )
    return (
        TaskResult(
            task_id=task['id'],
            attempt_id=attempt['id'],
            status='succeeded',
            summary='agent claims success, but the validator will check it',
            artifact_candidates=[candidate],
            lineage_proposals=lineage,
            evidence_refs=['log:sha256:evidence'],
        ),
        lease['lease_id'],
    )


def test_two_tasks_use_two_conversations_and_replay_projection(
    outer_loop, tmp_path: Path
):
    controller, ledger, artifacts, dispatcher = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-001',
        run_spec={'source': {'uri': 'tos://bucket/batch-001'}},
        tasks=[
            _task('inspect-data', 'inspect-data', 'source-manifest'),
            _task('train', 'train', 'checkpoint'),
        ],
    )
    run_id = run['id']

    first_conversation = controller.dispatch_current_task(run_id)
    source_file = tmp_path / 'source.json'
    source_file.write_text('{"batch":"001"}')
    first_result, first_lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run_id,
        output=source_file,
        kind='source-manifest',
    )
    controller.submit_result(run_id, first_result, first_lease)
    assert controller.tick(run_id) is RunState.WAITING_FOR_AGENT

    source_artifact = ledger.list_artifacts(run_id)[0]['id']
    second_conversation = controller.dispatch_current_task(run_id)
    second_context = dispatcher.conversations[
        f'dispatch:{ledger.list_attempts(run_id)[-1]["id"]}'
    ][1]
    assert second_context.input_artifact_ids == [source_artifact]
    assert second_context.input_artifacts[0].id == source_artifact
    assert second_context.input_artifacts[0].uri.startswith('file://')
    assert second_context.run_spec['source']['uri'] == 'tos://bucket/batch-001'
    model_file = tmp_path / 'model.pth'
    model_file.write_bytes(b'fake-checkpoint')
    second_result, second_lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run_id,
        output=model_file,
        kind='checkpoint',
        lineage_input=source_artifact,
    )
    controller.submit_result(run_id, second_result, second_lease)
    assert controller.tick(run_id) is RunState.COMPLETED

    assert first_conversation != second_conversation
    assert len(dispatcher.conversations) == 2
    assert len(ledger.list_lineage(run_id)) == 1
    ledger.assert_projection_matches_replay(run_id)


def test_agent_success_claim_without_artifact_cannot_advance(outer_loop):
    controller, ledger, _, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-missing-artifact',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[_task('train', 'train', 'checkpoint')],
    )
    controller.dispatch_current_task(run['id'])
    task, attempt, lease = _active_attempt(ledger, run['id'])
    result = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='succeeded',
        summary='Training finished successfully!',
        evidence_refs=['final-answer:untrusted'],
    )
    controller.submit_result(run['id'], result, lease['lease_id'])
    assert controller.tick(run['id']) is RunState.NEEDS_INTERVENTION
    assert ledger.list_artifacts(run['id']) == []


def test_outer_policy_rejects_artifacts_over_run_budget(outer_loop, tmp_path: Path):
    controller, ledger, artifacts, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-artifact-budget',
        run_spec={
            'source': {'uri': 'file:///batch'},
            'spec': {'policy': {'maxArtifactBytes': 4}},
        },
        tasks=[_task('train', 'train', 'checkpoint', max_attempts=1)],
    )
    controller.dispatch_current_task(run['id'])
    output = tmp_path / 'checkpoint.pth'
    output.write_bytes(b'too-large')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run['id'],
        output=output,
        kind='checkpoint',
    )
    controller.submit_result(run['id'], result, lease)

    assert controller.tick(run['id']) is RunState.NEEDS_INTERVENTION
    attempt = ledger.list_attempts(run['id'])[0]
    validation = json.loads(attempt['validator_result_json'])
    assert 'maxArtifactBytes=4' in validation['reasons'][0]
    assert ledger.list_artifacts(run['id']) == []


def test_run_duration_budget_fences_work_before_dispatch(outer_loop):
    controller, ledger, _, dispatcher = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='duration-budget',
        run_spec={
            'source': {'uri': 'file:///batch'},
            'spec': {'policy': {'maxRunDurationSeconds': 1}},
        },
        tasks=[_task('train', 'train', 'checkpoint')],
    )
    with ledger._connect() as connection:
        connection.execute(
            'UPDATE mlops_runs SET created_at=? WHERE id=?',
            (0, run['id']),
        )

    assert controller.tick(run['id']) is RunState.NEEDS_INTERVENTION
    assert dispatcher.conversations == {}
    assert ledger.list_attempts(run['id']) == []
    assert ledger.list_tasks(run['id'])[0]['status'] == 'CANCELLED'
    assert ledger.list_events(run['id'])[-1]['event_type'] == 'RunPolicyStopped'
    ledger.assert_projection_matches_replay(run['id'])


def test_cost_budget_is_idempotent_and_fences_running_attempt(outer_loop):
    controller, ledger, _, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='cost-budget',
        run_spec={
            'source': {'uri': 'file:///batch'},
            'spec': {'policy': {'maxRunCostUsd': 1.0}},
        },
        tasks=[_task('train', 'train', 'checkpoint')],
    )
    controller.dispatch_current_task(run['id'])
    attempt = ledger.list_attempts(run['id'])[0]
    first = controller.record_cost(
        run['id'],
        source_type='llm',
        source_id='conversation-1',
        amount_usd=0.4,
        metadata={'tokens': 100},
        idempotency_key='provider-charge-1',
    )
    duplicate = controller.record_cost(
        run['id'],
        source_type='llm',
        source_id='conversation-1',
        amount_usd=0.4,
        metadata={'tokens': 100},
        idempotency_key='provider-charge-1',
    )
    assert duplicate['id'] == first['id']
    assert ledger.total_cost_usd(run['id']) == pytest.approx(0.4)

    controller.record_cost(
        run['id'],
        source_type='executor',
        source_id='gpu-job-1',
        amount_usd=0.6,
        metadata={'gpu_seconds': 60},
        idempotency_key='executor-charge-1',
    )

    assert ledger.get_run(run['id'])['state'] == RunState.NEEDS_INTERVENTION
    assert ledger.get_attempt(attempt['id'])['status'] == 'CANCELLED'
    with pytest.raises(KeyError):
        ledger.get_lease(attempt['id'])
    assert ledger.total_cost_usd(run['id']) == pytest.approx(1.0)
    audit = ledger.audit_run(run['id'])
    assert len(audit['cost_entries']) == 2
    assert audit['total_cost_usd'] == pytest.approx(1.0)
    assert audit['outbox'][0]['status'] == 'DELIVERED'
    ledger.assert_projection_matches_replay(run['id'])


def test_retry_is_outer_loop_decision_and_gets_new_conversation(outer_loop):
    controller, ledger, _, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-oom',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[_task('train', 'train', 'checkpoint', max_attempts=2)],
    )
    first_conversation = controller.dispatch_current_task(run['id'])
    task, attempt, lease = _active_attempt(ledger, run['id'])
    _record_failed_job(ledger, attempt['id'], 'job_oom', 'cuda-oom-with-valid-proposal')
    failed = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='failed',
        summary='CUDA allocation failed',
        evidence_refs=['job:job_oom'],
        next_attempt_proposal=NextAttemptProposal(
            failure_class='cuda-oom-with-valid-proposal',
            reason='reduce the per-device batch size',
            spec_diff={'batch_size': 32},
            evidence_refs=['job:job_oom'],
        ),
    )
    controller.submit_result(run['id'], failed, lease['lease_id'])
    assert controller.tick(run['id']) is RunState.RETRY_SCHEDULED

    attempts = ledger.list_attempts(run['id'])
    assert [item['attempt_number'] for item in attempts] == [1, 2]
    assert any(
        event['event_type'] == 'RetryDecided'
        and event['payload']['failed_attempt_id'] == attempt['id']
        for event in ledger.list_events(run['id'])
    )
    second_conversation = controller.dispatch_current_task(run['id'])
    assert second_conversation != first_conversation
    assert ledger.latest_attempt(task['id'])['attempt_spec'] == {'batch_size': 32}


def test_cuda_oom_without_resource_diff_does_not_retry(outer_loop):
    controller, ledger, _, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-oom-without-fix',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint', max_attempts=2)],
    )
    controller.dispatch_current_task(run['id'])
    task, attempt, lease = _active_attempt(ledger, run['id'])
    _record_failed_job(
        ledger, attempt['id'], 'job_oom_no_diff', 'cuda-oom-with-valid-proposal'
    )
    failed = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='failed',
        summary='CUDA OOM but no actionable resource change',
        evidence_refs=['job:job_oom_no_diff'],
        next_attempt_proposal=NextAttemptProposal(
            failure_class='cuda-oom-with-valid-proposal',
            reason='retry unchanged',
            spec_diff={},
            evidence_refs=['job:job_oom_no_diff'],
        ),
    )
    controller.submit_result(run['id'], failed, lease['lease_id'])

    assert controller.tick(run['id']) is RunState.NEEDS_INTERVENTION
    assert len(ledger.list_attempts(run['id'])) == 1


def test_agent_cannot_invent_a_retryable_failure_without_job_evidence(outer_loop):
    controller, ledger, _, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='invented-oom',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint', max_attempts=2)],
    )
    controller.dispatch_current_task(run['id'])
    task, attempt, lease = _active_attempt(ledger, run['id'])
    result = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='failed',
        summary='unverified OOM claim',
        evidence_refs=['job:does-not-exist'],
        next_attempt_proposal=NextAttemptProposal(
            failure_class='cuda-oom-with-valid-proposal',
            reason='reduce batch size',
            spec_diff={'batch_size': 8},
            evidence_refs=['job:does-not-exist'],
        ),
    )

    controller.submit_result(run['id'], result, lease['lease_id'])

    assert controller.tick(run['id']) is RunState.NEEDS_INTERVENTION
    attempt = ledger.list_attempts(run['id'])[0]
    validation = json.loads(attempt['validator_result_json'])
    assert 'not backed by a failed job observation' in validation['reasons'][0]


def _record_failed_job(
    ledger: SQLiteLedger, attempt_id: str, job_id: str, failure_class: str
) -> None:
    ledger.record_job_launch(
        attempt_id=attempt_id,
        job_id=job_id,
        idempotency_key=job_id,
        argv=['rightarm-mlops', 'train'],
        status='requested',
    )
    ledger.record_job_observation(
        attempt_id=attempt_id,
        job_id=job_id,
        observation={
            'status': 'failed',
            'return_code': 1,
            'failure': {'failure_class': failure_class},
            'log_path': f'/jobs/{job_id}/job.log',
            'log_sha256': 'a' * 64,
            'log_size_bytes': 10,
        },
    )


def test_watermark_and_dispatch_are_idempotent(outer_loop):
    controller, ledger, _, dispatcher = outer_loop
    kwargs = {
        'loop_id': 'gesture-loop',
        'watermark': 'batch-idempotent',
        'run_spec': {'source': {'uri': 'tos://bucket/complete'}},
        'tasks': [_task('train', 'train', 'checkpoint')],
    }
    first = controller.create_run(**kwargs)
    second = controller.create_run(**kwargs)
    assert first['id'] == second['id']
    conversation_1 = controller.dispatch_current_task(first['id'])
    conversation_2 = controller.dispatch_current_task(first['id'])
    assert conversation_1 == conversation_2
    assert len(dispatcher.conversations) == 1

    with pytest.raises(IdempotencyConflict):
        ledger.create_run(
            loop_id='gesture-loop',
            watermark='batch-idempotent',
            spec={'source': {'uri': 'tos://different'}},
            tasks=[_task('train', 'train', 'checkpoint')],
            idempotency_key='create-run:gesture-loop:batch-idempotent',
        )


def test_pending_dispatch_outbox_recovers_after_controller_restart(outer_loop):
    controller, ledger, artifacts, _ = outer_loop

    class FailingDispatcher:
        def dispatch(
            self,
            *,
            context: TaskContext,
            task: TaskSpec,
            dispatch_key: str,
        ) -> str:
            del context, task, dispatch_key
            raise ConnectionError('response lost')

    controller.dispatcher = FailingDispatcher()
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='dispatch-outbox-recovery',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint')],
    )
    with pytest.raises(ConnectionError, match='response lost'):
        controller.dispatch_current_task(run['id'])
    outbox = ledger.list_outbox(run['id'])
    assert len(outbox) == 1
    assert outbox[0]['status'] == 'PENDING'
    assert outbox[0]['delivery_attempts'] == 1
    with ledger._connect() as connection:
        connection.execute(
            'UPDATE mlops_outbox SET available_at=0 WHERE id=?', (outbox[0]['id'],)
        )

    dispatcher = InMemoryConversationDispatcher()
    restarted = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=dispatcher,
        validators=controller.validators,
        retry_policy=controller.retry_policy,
        controller_id='loop-controller-restarted',
    )
    conversation_id = restarted.dispatch_current_task(run['id'])

    assert conversation_id is not None
    assert len(dispatcher.conversations) == 1
    assert len(ledger.list_attempts(run['id'])) == 1
    delivered = ledger.list_outbox(run['id'])[0]
    assert delivered['status'] == 'DELIVERED'
    assert delivered['external_id'] == conversation_id
    assert [
        event['event_type']
        for event in ledger.list_events(run['id'])
        if event['event_type'].startswith('ConversationDispatch')
        or event['event_type'] == 'TaskAttemptDispatched'
    ] == [
        'ConversationDispatchRequested',
        'ConversationDispatchFailed',
        'TaskAttemptDispatched',
    ]
    ledger.assert_projection_matches_replay(run['id'])


def test_attempt_is_running_before_dispatcher_starts_agent(outer_loop):
    controller, ledger, _, _ = outer_loop

    class OrderingDispatcher(InMemoryConversationDispatcher):
        status_at_start: str | None = None

        def start(self, *, conversation_id: str, dispatch_key: str) -> None:
            context = self.conversations[dispatch_key][1]
            self.status_at_start = ledger.get_attempt(context.attempt_id)['status']
            super().start(
                conversation_id=conversation_id,
                dispatch_key=dispatch_key,
            )

    dispatcher = OrderingDispatcher()
    controller.dispatcher = dispatcher
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='dispatch-start-ordering',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint')],
    )

    conversation_id = controller.dispatch_current_task(run['id'])

    assert conversation_id is not None
    assert dispatcher.status_at_start == 'RUNNING'
    assert dispatcher.started_dispatches == {
        f'dispatch:{ledger.list_attempts(run["id"])[0]["id"]}'
    }


def test_running_attempt_retries_start_after_post_commit_failure(outer_loop):
    controller, ledger, _, _ = outer_loop

    class FailOnceStartDispatcher(InMemoryConversationDispatcher):
        start_calls = 0

        def start(self, *, conversation_id: str, dispatch_key: str) -> None:
            self.start_calls += 1
            if self.start_calls == 1:
                raise ConnectionError('start response lost')
            super().start(
                conversation_id=conversation_id,
                dispatch_key=dispatch_key,
            )

    dispatcher = FailOnceStartDispatcher()
    controller.dispatcher = dispatcher
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='dispatch-start-recovery',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint')],
    )

    with pytest.raises(ConnectionError, match='start response lost'):
        controller.dispatch_current_task(run['id'])

    attempt = ledger.list_attempts(run['id'])[0]
    assert attempt['status'] == 'RUNNING'
    assert ledger.list_outbox(run['id'])[0]['status'] == 'DELIVERED'

    recovered = controller.dispatch_current_task(run['id'])

    assert recovered == attempt['conversation_id']
    assert dispatcher.start_calls == 2
    assert dispatcher.started_dispatches == {attempt['dispatch_key']}
    assert [
        event['event_type']
        for event in ledger.list_events(run['id'])
        if event['event_type'] == 'TaskAttemptDispatched'
    ] == ['TaskAttemptDispatched']
    ledger.assert_projection_matches_replay(run['id'])


def test_stale_agent_lease_is_fenced(outer_loop):
    controller, ledger, _, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-fencing',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[_task('train', 'train', 'checkpoint')],
    )
    controller.dispatch_current_task(run['id'])
    task, attempt, _ = _active_attempt(ledger, run['id'])
    result = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='succeeded',
        summary='spoofed stale result',
        evidence_refs=['stale'],
    )
    with pytest.raises(LeaseConflict):
        controller.submit_result(run['id'], result, 'lease_stale')


def test_submitted_attempt_can_no_longer_use_task_tools(outer_loop):
    controller, ledger, _, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-tools-after-submit',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint')],
    )
    controller.dispatch_current_task(run['id'])
    task, attempt, lease = _active_attempt(ledger, run['id'])
    controller.submit_result(
        run['id'],
        TaskResult(
            task_id=task['id'],
            attempt_id=attempt['id'],
            status='succeeded',
            summary='submitted',
            evidence_refs=['claim-only'],
        ),
        lease['lease_id'],
    )

    with pytest.raises(LeaseConflict, match='no longer active'):
        ledger.assert_attempt_lease(attempt['id'], lease['lease_id'])


def test_run_rejects_unresolved_template_values(outer_loop):
    controller, _, _, _ = outer_loop
    with pytest.raises(ValueError, match='unresolved placeholders'):
        controller.create_run(
            loop_id='gesture-loop',
            watermark='batch-placeholder',
            run_spec={'source': {'uri': 'tos://bucket/<batch-id>'}},
            tasks=[_task('train', 'train', 'checkpoint')],
        )


def test_run_initial_artifacts_are_sealed_visible_inputs(outer_loop, tmp_path: Path):
    controller, ledger, artifacts, dispatcher = outer_loop
    producer = controller.create_run(
        loop_id='baseline-loop',
        watermark='baseline-0',
        run_spec={'source': {'uri': 'file:///baseline'}},
        tasks=[_task('register-base', 'train', 'base-model')],
    )
    controller.dispatch_current_task(producer['id'])
    base_path = tmp_path / 'base.pth'
    base_path.write_bytes(b'sealed-baseline')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=producer['id'],
        output=base_path,
        kind='base-model',
    )
    controller.submit_result(producer['id'], result, lease)
    controller.tick(producer['id'])
    base_id = ledger.list_artifacts(producer['id'])[0]['id']

    consumer = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-with-baseline',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint')],
        initial_artifact_ids=[base_id],
    )
    controller.dispatch_current_task(consumer['id'])
    attempt = ledger.list_attempts(consumer['id'])[0]
    context = dispatcher.conversations[f'dispatch:{attempt["id"]}'][1]
    assert context.input_artifact_ids == [base_id]
    assert ledger.list_artifacts(consumer['id'])[0]['id'] == base_id
    assert ledger.list_events(consumer['id'])[0]['payload']['initial_artifact_ids'] == [
        base_id
    ]


def test_operator_bootstrap_is_content_idempotent(outer_loop, tmp_path: Path):
    controller, ledger, _, _ = outer_loop
    model = tmp_path / 'base.pth'
    model.write_bytes(b'baseline')

    first = controller.bootstrap_artifact(path=str(model), kind='base-model')
    second = controller.bootstrap_artifact(path=str(model), kind='base-model')

    assert first['id'] == second['id']
    runs = ledger.list_runs('artifact-bootstrap')
    assert len(runs) == 1
    assert runs[0]['state'] == RunState.COMPLETED


def test_run_rejects_missing_initial_artifact(outer_loop):
    controller, _, _, _ = outer_loop
    with pytest.raises(InvariantViolation, match='missing or unsealed'):
        controller.create_run(
            loop_id='gesture-loop',
            watermark='batch-missing-baseline',
            run_spec={'source': {'uri': 'file:///batch'}},
            tasks=[_task('train', 'train', 'checkpoint')],
            initial_artifact_ids=['base-model:sha256:' + '0' * 64],
        )


def test_lineage_cannot_smuggle_an_artifact_from_another_run(
    outer_loop, tmp_path: Path
):
    controller, ledger, artifacts, _ = outer_loop
    foreign_path = tmp_path / 'foreign.bin'
    foreign_path.write_bytes(b'foreign')
    foreign = controller.bootstrap_artifact(path=str(foreign_path), kind='base-model')
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='lineage-smuggling',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[_task('train', 'train', 'checkpoint')],
    )
    controller.dispatch_current_task(run['id'])
    output = tmp_path / 'checkpoint.pth'
    output.write_bytes(b'checkpoint')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run['id'],
        output=output,
        kind='checkpoint',
        lineage_input=foreign['id'],
    )
    controller.submit_result(run['id'], result, lease)

    with pytest.raises(InvariantViolation, match='not sealed into this Run'):
        controller.tick(run['id'])


def test_release_closure_and_deployment_state_are_outer_loop_owned(
    outer_loop, tmp_path: Path
):
    controller, ledger, artifacts, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-release',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[
            _task('inspect-data', 'inspect-data', 'source-manifest'),
            _task('train', 'train', 'checkpoint'),
            _task('build-release', 'package', 'release-manifest'),
        ],
    )
    run_id = run['id']

    controller.dispatch_current_task(run_id)
    source_path = tmp_path / 'source-manifest.json'
    source_path.write_text('{}')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run_id,
        output=source_path,
        kind='source-manifest',
    )
    controller.submit_result(run_id, result, lease)
    controller.tick(run_id)
    source_id = ledger.list_artifacts(run_id)[0]['id']

    controller.dispatch_current_task(run_id)
    checkpoint_path = tmp_path / 'model.pth'
    checkpoint_path.write_bytes(b'model')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run_id,
        output=checkpoint_path,
        kind='checkpoint',
        lineage_input=source_id,
    )
    controller.submit_result(run_id, result, lease)
    controller.tick(run_id)
    checkpoint_id = next(
        item['id']
        for item in ledger.list_artifacts(run_id)
        if item['kind'] == 'checkpoint'
    )

    controller.dispatch_current_task(run_id)
    release_path = tmp_path / 'release.json'
    release_path.write_text('{"protocol":"release/v1"}')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run_id,
        output=release_path,
        kind='release-manifest',
        lineage_input=checkpoint_id,
    )
    controller.submit_result(run_id, result, lease)
    assert controller.tick(run_id) is RunState.COMPLETED
    artifact_ids = [item['id'] for item in ledger.list_artifacts(run_id)]
    release_manifest_id = next(
        item for item in artifact_ids if item.startswith('release-manifest:')
    )

    release = ledger.seal_release(
        run_id=run_id,
        manifest_artifact_id=release_manifest_id,
        artifact_ids=artifact_ids,
        root_kinds=frozenset({'source-manifest'}),
        expected_version=ledger.get_run(run_id)['version'],
        validation_id='validation:release',
        idempotency_key='seal:release-1',
    )
    deployment = ledger.request_deployment(
        run_id=run_id,
        release_id=release['id'],
        environment='rightarm-staging',
        initial_traffic_percent=1,
        expected_version=ledger.get_run(run_id)['version'],
        idempotency_key='deploy:release-1:staging',
    )
    duplicate = ledger.request_deployment(
        run_id=run_id,
        release_id=release['id'],
        environment='rightarm-staging',
        initial_traffic_percent=1,
        expected_version=0,
        idempotency_key='deploy:release-1:staging',
    )
    assert duplicate['id'] == deployment['id']

    before_observation = ledger.get_run(run_id)['state']
    ledger.record_deployment_observation(
        run_id=run_id,
        deployment_id=deployment['id'],
        observed_state='HEALTHY',
        revision='revision-1',
        metrics={'error_rate': 0.0},
        expected_version=ledger.get_run(run_id)['version'],
        idempotency_key='observe:revision-1',
    )
    assert ledger.get_run(run_id)['state'] == before_observation
    ledger.promote_deployment(
        run_id=run_id,
        deployment_id=deployment['id'],
        target_traffic_percent=100,
        validation_id='validation:canary',
        evidence_refs=['metrics:revision-1'],
        expected_version=ledger.get_run(run_id)['version'],
        idempotency_key='promote:revision-1',
    )
    assert ledger.list_deployments(run_id)[0]['desired_state'] == 'PROMOTED'
    ledger.assert_projection_matches_replay(run_id)


def test_release_without_provenance_closure_is_rejected(outer_loop, tmp_path: Path):
    controller, ledger, artifacts, _ = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-open-lineage',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[_task('build-release', 'package', 'release-manifest')],
    )
    controller.dispatch_current_task(run['id'])
    path = tmp_path / 'orphan-release.json'
    path.write_text('{}')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=run['id'],
        output=path,
        kind='release-manifest',
    )
    controller.submit_result(run['id'], result, lease)
    controller.tick(run['id'])
    manifest_id = ledger.list_artifacts(run['id'])[0]['id']
    with pytest.raises(InvariantViolation, match='no accepted root'):
        ledger.seal_release(
            run_id=run['id'],
            manifest_artifact_id=manifest_id,
            artifact_ids=[manifest_id],
            root_kinds=frozenset({'source-manifest'}),
            expected_version=ledger.get_run(run['id'])['version'],
            validation_id='validation:release',
            idempotency_key='seal:orphan',
        )


def test_data_inspection_can_finish_without_training_and_preserve_pending_data(
    outer_loop, tmp_path: Path
):
    controller, ledger, artifacts, _ = outer_loop
    inspect_task = _task(
        'inspect-data', 'inspect-data', 'data-validation-report'
    ).model_copy(
        update={
            'completion_contract': CompletionContract(
                required_artifact_kinds=['data-validation-report'],
                required_evidence=True,
                allow_skip_training=True,
            )
        }
    )
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-too-small',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[inspect_task, _task('train', 'train', 'checkpoint')],
    )
    controller.dispatch_current_task(run['id'])
    task, attempt, lease = _active_attempt(ledger, run['id'])
    report = tmp_path / 'too-small.json'
    report.write_text('{"valid_samples":3}')
    candidate = artifacts.stage(report, kind='data-validation-report')
    result = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='succeeded',
        summary='valid batch, but below the training threshold',
        artifact_candidates=[candidate],
        metrics={'skip_training': True, 'valid_samples': 3},
        evidence_refs=['report:too-small'],
    )
    controller.submit_result(run['id'], result, lease['lease_id'])
    assert controller.tick(run['id']) is RunState.COMPLETED_NO_TRAINING
    pending = ledger.list_pending_artifacts('gesture-loop')
    assert len(pending) == 1
    assert pending[0].startswith('data-validation-report:sha256:')
    assert ledger.list_tasks(run['id'])[1]['status'] == 'CANCELLED'
    ledger.assert_projection_matches_replay(run['id'])


def test_expired_conversation_is_fenced_and_recovered_as_new_attempt(outer_loop):
    controller, ledger, _, dispatcher = outer_loop
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-agent-crash',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[_task('train', 'train', 'checkpoint', max_attempts=2)],
    )
    old_conversation = controller.dispatch_current_task(run['id'])
    old_attempt = ledger.list_attempts(run['id'])[0]
    with ledger._connect() as connection:
        connection.execute(
            'UPDATE mlops_attempt_leases SET expires_at=0 WHERE attempt_id=?',
            (old_attempt['id'],),
        )

    assert controller.reconcile_expired_attempt(run['id']) is True
    attempts = ledger.list_attempts(run['id'])
    assert [item['status'] for item in attempts] == ['CRASHED', 'CREATED']
    assert any(
        event['event_type'] == 'RetryDecided'
        and event['payload']['failure_class'] == 'agent-conversation-lost'
        for event in ledger.list_events(run['id'])
    )
    new_conversation = controller.dispatch_current_task(run['id'])
    assert new_conversation != old_conversation
    assert len(dispatcher.conversations) == 2
    ledger.assert_projection_matches_replay(run['id'])


def test_human_confirmation_is_an_audited_dispatch_gate(outer_loop):
    controller, ledger, _, _ = outer_loop
    task = _task('approve-me', 'inspect-data', 'source-manifest').model_copy(
        update={'requires_human_confirmation': True}
    )
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-confirmation',
        run_spec={'source': {'uri': 'tos://bucket/complete'}},
        tasks=[task],
    )
    assert controller.dispatch_current_task(run['id']) is None
    assert ledger.list_attempts(run['id']) == []
    controller.approve_current_task(
        run['id'], approved_by='operator@example.com', reason='staging release reviewed'
    )
    assert controller.dispatch_current_task(run['id']) is not None
    assert any(
        event['event_type'] == 'HumanConfirmationRecorded'
        for event in ledger.list_events(run['id'])
    )


def test_release_canary_and_promotion_are_derived_from_validated_artifacts(
    outer_loop, tmp_path: Path
):
    controller, ledger, artifacts, _ = outer_loop
    prepare = TaskSpec(
        name='prepare-release-artifacts',
        kind='package',
        objective='prepare immutable release artifacts',
        agent_profile='test-profile',
        validator='test-validator',
        max_attempts=1,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=sorted(RIGHTARM_RELEASE_REQUIRED_KINDS),
            required_evidence=True,
        ),
    )
    verify = TaskSpec(
        name='verify-board',
        kind='verify-board',
        objective='verify and seal release',
        agent_profile='test-profile',
        validator='test-validator',
        max_attempts=1,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=['board-report', 'release-manifest'],
            required_evidence=True,
        ),
    )
    deploy = _task(
        'deploy-canary', 'deploy', 'deployment-observation', max_attempts=1
    ).model_copy(update={'requires_human_confirmation': True})
    monitor = _task(
        'monitor-canary', 'monitor', 'deployment-observation', max_attempts=1
    )
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='batch-domain-lifecycle',
        run_spec={
            'source': {'uri': 'tos://bucket/complete'},
            'spec': {
                'deployment': {
                    'environment': 'rightarm-production',
                    'initialCanaryPercent': 1,
                }
            },
        },
        tasks=[prepare, verify, deploy, monitor],
    )
    run_id = run['id']

    controller.dispatch_current_task(run_id)
    task, attempt, lease = _active_attempt(ledger, run_id)
    candidates = []
    for kind in sorted(RIGHTARM_RELEASE_REQUIRED_KINDS):
        path = tmp_path / f'{kind}.bin'
        path.write_bytes(f'validated-{kind}'.encode())
        candidates.append(artifacts.stage(path, kind=kind))
    raw = next(candidate for candidate in candidates if candidate.kind == 'raw-dataset')
    root_kinds = {
        'raw-dataset',
        'base-model',
        'code',
        'container-image',
        'training-config',
        'label-mapping',
        'tensor-contract',
    }
    lineage = [
        LineageProposal(
            input_artifact_id=raw.candidate_id,
            output_candidate_id=candidate.candidate_id,
            transform_kind=f'build-{candidate.kind}',
        )
        for candidate in candidates
        if candidate.kind not in root_kinds
    ]
    prepared = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='succeeded',
        summary='release inputs built',
        artifact_candidates=candidates,
        lineage_proposals=lineage,
        evidence_refs=['machine-report:prepare'],
    )
    controller.submit_result(run_id, prepared, lease['lease_id'])
    assert controller.tick(run_id) is RunState.WAITING_FOR_AGENT

    artifact_rows = ledger.list_artifacts(run_id)
    artifact_by_kind = {item['kind']: item for item in artifact_rows}
    release_document = {
        'schema_version': 'rightarm.gesture.release/v1',
        'sealed': False,
        'artifacts': [
            {
                'kind': kind,
                'sha256': artifact_by_kind[kind]['sha256'],
                'size_bytes': artifact_by_kind[kind]['size_bytes'],
            }
            for kind in sorted(RIGHTARM_RELEASE_REQUIRED_KINDS)
        ],
    }
    release_path = tmp_path / 'release-manifest.json'
    release_path.write_text(json.dumps(release_document))
    board_path = tmp_path / 'board-report.bin'
    board_path.write_bytes(b'validated-board-report')

    controller.dispatch_current_task(run_id)
    task, attempt, lease = _active_attempt(ledger, run_id)
    board = artifacts.stage(board_path, kind='board-report')
    manifest = artifacts.stage(release_path, kind='release-manifest')
    verified = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='succeeded',
        summary='board verified and release proposed',
        artifact_candidates=[board, manifest],
        lineage_proposals=[
            LineageProposal(
                input_artifact_id=artifact_by_kind['board-report']['id'],
                output_candidate_id=manifest.candidate_id,
                transform_kind='build-release-manifest',
            )
        ],
        evidence_refs=['machine-report:board'],
    )
    controller.submit_result(run_id, verified, lease['lease_id'])
    assert controller.tick(run_id) is RunState.WAITING_FOR_AGENT
    release = ledger.list_releases(run_id)[0]
    assert release['sealed'] == 1

    assert controller.dispatch_current_task(run_id) is None
    controller.approve_current_task(
        run_id, approved_by='release-owner', reason='production canary authorized'
    )
    assert controller.dispatch_current_task(run_id) is not None
    deployment = ledger.list_deployments(run_id)[0]
    context = controller.get_task_context(
        ledger.list_attempts(run_id)[-1]['id'],
        ledger.get_lease(ledger.list_attempts(run_id)[-1]['id'])['lease_id'],
    )
    assert context.domain_refs == {
        'release_id': release['id'],
        'deployment_id': deployment['id'],
    }
    _submit_observation(
        controller,
        ledger,
        artifacts,
        run_id,
        tmp_path / 'deploy-observation.json',
        deployment['id'],
        revision='revision-canary',
    )
    assert controller.tick(run_id) is RunState.WAITING_FOR_AGENT
    assert ledger.list_deployments(run_id)[0]['observed_state'] == 'HEALTHY'

    controller.dispatch_current_task(run_id)
    _submit_observation(
        controller,
        ledger,
        artifacts,
        run_id,
        tmp_path / 'monitor-observation.json',
        deployment['id'],
        revision='revision-canary',
    )
    assert controller.tick(run_id) is RunState.COMPLETED
    final_deployment = ledger.list_deployments(run_id)[0]
    assert final_deployment['desired_state'] == 'PROMOTED'
    assert final_deployment['traffic_percent'] == 100
    event_types = [item['event_type'] for item in ledger.list_events(run_id)]
    assert 'ReleaseSealed' in event_types
    assert 'HumanConfirmationRecorded' in event_types
    assert event_types.count('DeploymentObserved') == 2
    assert 'TrafficShiftDecided' in event_types
    ledger.assert_projection_matches_replay(run_id)


def _submit_observation(
    controller: LoopController,
    ledger: SQLiteLedger,
    artifacts: LocalArtifactStore,
    run_id: str,
    path: Path,
    deployment_id: str,
    *,
    revision: str,
    healthy: bool = True,
    active_release_id: str | None = None,
) -> None:
    task, attempt, lease = _active_attempt(ledger, run_id)
    document = {
        'schema_version': 'rightarm.gesture.deployment-observation/v1',
        'deployment_id': deployment_id,
        'observed_state': 'HEALTHY' if healthy else 'UNHEALTHY',
        'revision': revision,
        'metrics': {'error_rate': 0.0 if healthy else 0.5},
        'smoke_test': {'passed': healthy},
    }
    if active_release_id is not None:
        document['active_release_id'] = active_release_id
    path.write_text(json.dumps(document))
    candidate = artifacts.stage(path, kind='deployment-observation')
    result = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='succeeded',
        summary='deployment is healthy',
        artifact_candidates=[candidate],
        evidence_refs=[str(path)],
    )
    controller.submit_result(run_id, result, lease['lease_id'])


def test_failed_canary_schedules_and_verifies_rollback_to_proven_champion(
    outer_loop, tmp_path: Path
):
    controller, ledger, artifacts, _ = outer_loop
    champion_run = controller.create_run(
        loop_id='gesture-loop',
        watermark='champion-release',
        run_spec={'source': {'uri': 'tos://bucket/champion'}},
        tasks=[
            _task('champion-source', 'inspect-data', 'source-manifest'),
            _task('champion-manifest', 'package', 'release-manifest'),
        ],
    )
    controller.dispatch_current_task(champion_run['id'])
    champion_source_path = tmp_path / 'champion-source.json'
    champion_source_path.write_text('{}')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=champion_run['id'],
        output=champion_source_path,
        kind='source-manifest',
    )
    controller.submit_result(champion_run['id'], result, lease)
    controller.tick(champion_run['id'])
    champion_source_id = ledger.list_artifacts(champion_run['id'])[0]['id']
    controller.dispatch_current_task(champion_run['id'])
    champion_manifest_path = tmp_path / 'champion-release.json'
    champion_manifest_path.write_text('{}')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=champion_run['id'],
        output=champion_manifest_path,
        kind='release-manifest',
        lineage_input=champion_source_id,
    )
    controller.submit_result(champion_run['id'], result, lease)
    controller.tick(champion_run['id'])
    champion_artifacts = ledger.list_artifacts(champion_run['id'])
    champion_manifest_id = next(
        item['id'] for item in champion_artifacts if item['kind'] == 'release-manifest'
    )
    champion = ledger.seal_release(
        run_id=champion_run['id'],
        manifest_artifact_id=champion_manifest_id,
        artifact_ids=[item['id'] for item in champion_artifacts],
        root_kinds=frozenset({'source-manifest'}),
        expected_version=ledger.get_run(champion_run['id'])['version'],
        validation_id='validation:champion',
        idempotency_key='seal:champion',
    )
    champion_deployment = ledger.request_deployment(
        run_id=champion_run['id'],
        release_id=champion['id'],
        environment='rightarm-production',
        initial_traffic_percent=1,
        expected_version=ledger.get_run(champion_run['id'])['version'],
        idempotency_key='deploy:champion',
    )
    ledger.record_deployment_observation(
        run_id=champion_run['id'],
        deployment_id=champion_deployment['id'],
        observed_state='HEALTHY',
        revision='champion-revision',
        metrics={'error_rate': 0.0},
        expected_version=ledger.get_run(champion_run['id'])['version'],
        idempotency_key='observe:champion',
    )
    ledger.promote_deployment(
        run_id=champion_run['id'],
        deployment_id=champion_deployment['id'],
        target_traffic_percent=100,
        validation_id='validation:champion-canary',
        evidence_refs=['champion-canary'],
        expected_version=ledger.get_run(champion_run['id'])['version'],
        idempotency_key='promote:champion',
    )
    assert ledger.get_champion('gesture-loop')['release_id'] == champion['id']

    candidate_run = controller.create_run(
        loop_id='gesture-loop',
        watermark='failed-candidate',
        run_spec={
            'source': {'uri': 'tos://bucket/candidate'},
            'spec': {
                'deployment': {
                    'environment': 'rightarm-production',
                    'initialCanaryPercent': 1,
                }
            },
        },
        tasks=[
            _task('candidate-source', 'inspect-data', 'source-manifest'),
            _task('candidate-manifest', 'package', 'release-manifest'),
            _task('candidate-deploy', 'deploy', 'deployment-observation').model_copy(
                update={'validator': 'rightarm-deployment-v1'}
            ),
            _task('candidate-monitor', 'monitor', 'deployment-observation').model_copy(
                update={'validator': 'rightarm-canary-v1'}
            ),
        ],
    )
    candidate_run_id = candidate_run['id']
    controller.dispatch_current_task(candidate_run_id)
    source_path = tmp_path / 'candidate-source.json'
    source_path.write_text('{}')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=candidate_run_id,
        output=source_path,
        kind='source-manifest',
    )
    controller.submit_result(candidate_run_id, result, lease)
    controller.tick(candidate_run_id)
    source_id = ledger.list_artifacts(candidate_run_id)[0]['id']
    assert source_id == champion_source_id
    controller.dispatch_current_task(candidate_run_id)
    manifest_path = tmp_path / 'candidate-release.json'
    manifest_path.write_text('{"candidate":true}')
    result, lease = _success_result(
        ledger=ledger,
        artifacts=artifacts,
        run_id=candidate_run_id,
        output=manifest_path,
        kind='release-manifest',
        lineage_input=source_id,
    )
    controller.submit_result(candidate_run_id, result, lease)
    controller.tick(candidate_run_id)
    candidate_artifacts = ledger.list_artifacts(candidate_run_id)
    manifest_id = next(
        item['id'] for item in candidate_artifacts if item['kind'] == 'release-manifest'
    )
    candidate_release = ledger.seal_release(
        run_id=candidate_run_id,
        manifest_artifact_id=manifest_id,
        artifact_ids=[item['id'] for item in candidate_artifacts],
        root_kinds=frozenset({'source-manifest'}),
        expected_version=ledger.get_run(candidate_run_id)['version'],
        validation_id='validation:candidate',
        idempotency_key='seal:candidate',
    )
    assert candidate_release['id'] != champion['id']

    controller.validators = build_rightarm_validator_registry(artifacts)
    controller.dispatch_current_task(candidate_run_id)
    candidate_deployment = ledger.list_deployments(candidate_run_id)[0]
    _submit_observation(
        controller,
        ledger,
        artifacts,
        candidate_run_id,
        tmp_path / 'candidate-deploy-observation.json',
        candidate_deployment['id'],
        revision='candidate-revision',
    )
    assert controller.tick(candidate_run_id) is RunState.WAITING_FOR_AGENT

    controller.dispatch_current_task(candidate_run_id)
    _submit_observation(
        controller,
        ledger,
        artifacts,
        candidate_run_id,
        tmp_path / 'unhealthy-canary.json',
        candidate_deployment['id'],
        revision='candidate-revision',
        healthy=False,
    )
    assert controller.tick(candidate_run_id) is RunState.WAITING_FOR_AGENT
    rollback = ledger.get_active_rollback(candidate_run_id)
    assert rollback['target_release_id'] == champion['id']
    assert ledger.current_task(candidate_run_id)['spec']['kind'] == 'rollback'

    controller.dispatch_current_task(candidate_run_id)
    rollback_context = controller.get_task_context(
        ledger.list_attempts(candidate_run_id)[-1]['id'],
        ledger.get_lease(ledger.list_attempts(candidate_run_id)[-1]['id'])['lease_id'],
    )
    assert rollback_context.domain_refs['rollback_target_release_id'] == champion['id']
    _submit_observation(
        controller,
        ledger,
        artifacts,
        candidate_run_id,
        tmp_path / 'rollback-observation.json',
        candidate_deployment['id'],
        revision='champion-revision',
        active_release_id=champion['id'],
    )
    assert controller.tick(candidate_run_id) is RunState.COMPLETED
    assert ledger.get_active_rollback(candidate_run_id)['status'] == 'SUCCEEDED'
    assert (
        ledger.list_deployments(candidate_run_id)[0]['desired_state'] == 'ROLLED_BACK'
    )
    assert ledger.get_champion('gesture-loop')['release_id'] == champion['id']
    assert any(
        event['event_type'] == 'DeploymentObserved'
        and event['payload']['observed_state'] == 'UNHEALTHY'
        for event in ledger.list_events(candidate_run_id)
    )
    assert any(
        artifact['kind'] == 'deployment-observation'
        for artifact in ledger.list_artifacts(candidate_run_id)
    )
    ledger.assert_projection_matches_replay(candidate_run_id)
