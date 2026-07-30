import logging
from pathlib import Path

from openhands.mlops_loop.artifact_store import LocalArtifactStore
from openhands.mlops_loop.cli import _reconcile_inactive_jobs, _run_reconciler
from openhands.mlops_loop.controller import LoopController
from openhands.mlops_loop.dispatcher import InMemoryConversationDispatcher
from openhands.mlops_loop.ledger import SQLiteLedger
from openhands.mlops_loop.models import TaskResult, TaskSpec
from openhands.mlops_loop.retry_policy import RetryPolicy
from openhands.mlops_loop.validators import ContractValidator, ValidatorRegistry


class OneCycleStop:
    def __init__(self):
        self.finished = False

    def is_set(self) -> bool:
        return self.finished

    def wait(self, _: float) -> None:
        self.finished = True


def test_reconciler_dispatches_and_advances_manual_run(tmp_path: Path):
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=SQLiteLedger(tmp_path / 'ledger.db'),
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    run = controller.create_run(
        loop_id='rightarm-manual-v1',
        watermark='manual-reconcile-test',
        run_spec={'spec': {'manualTrigger': {'mode': 'train'}}},
        tasks=[
            TaskSpec(
                name='train-test',
                kind='train',
                objective='test automatic advancement',
                agent_profile='default',
                validator='default',
                max_attempts=1,
                timeout_seconds=60,
            )
        ],
    )

    _run_reconciler(controller, OneCycleStop(), 0.01, None)
    assert controller.ledger.get_run(run['id'])['state'] == 'WAITING_FOR_AGENT'

    attempt = controller.ledger.list_attempts(run['id'])[0]
    lease = controller.ledger.get_lease(attempt['id'])
    task = controller.ledger.current_task(run['id'])
    controller.submit_result(
        run['id'],
        TaskResult(
            task_id=task['id'],
            attempt_id=attempt['id'],
            status='succeeded',
            summary='done',
        ),
        lease['lease_id'],
    )

    _run_reconciler(controller, OneCycleStop(), 0.01, None)
    assert controller.ledger.get_run(run['id'])['state'] == 'COMPLETED'


def test_reconciler_isolates_one_broken_run_from_later_runs(tmp_path: Path):
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=SQLiteLedger(tmp_path / 'ledger.db'),
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    task = TaskSpec(
        name='train-test',
        kind='train',
        objective='test per-Run failure isolation',
        agent_profile='default',
        validator='default',
        max_attempts=1,
        timeout_seconds=60,
    )
    broken = controller.create_run(
        loop_id='rightarm-manual-v1',
        watermark='broken-reconcile-test',
        run_spec={'spec': {'manualTrigger': {'mode': 'train'}}},
        tasks=[task],
    )
    healthy = controller.create_run(
        loop_id='rightarm-manual-v1',
        watermark='healthy-reconcile-test',
        run_spec={'spec': {'manualTrigger': {'mode': 'train'}}},
        tasks=[task],
    )
    original_tick = controller.tick

    def tick_with_one_failure(run_id: str):
        if run_id == broken['id']:
            raise RuntimeError('stale Conversation status')
        return original_tick(run_id)

    controller.tick = tick_with_one_failure  # type: ignore[method-assign]

    _run_reconciler(controller, OneCycleStop(), 0.01, None)

    assert controller.ledger.get_run(broken['id'])['state'] == 'CREATED'
    assert controller.ledger.get_run(healthy['id'])['state'] == 'WAITING_FOR_AGENT'


def test_reconciler_does_not_cancel_jobs_while_attempt_waits_for_continuation(
    tmp_path: Path,
):
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=SQLiteLedger(tmp_path / 'ledger.db'),
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    run = controller.create_run(
        loop_id='rightarm-manual-v1',
        watermark='waiting-job-reconcile-test',
        run_spec={'spec': {'manualTrigger': {'mode': 'improve'}}},
        tasks=[
            TaskSpec(
                name='improve-model',
                kind='improve-model',
                objective='continue a long job',
                agent_profile='default',
                validator='default',
                max_attempts=1,
                timeout_seconds=60,
            )
        ],
    )
    controller.dispatch_current_task(run['id'])
    task = controller.ledger.current_task(run['id'])
    assert task is not None
    attempt = controller.ledger.latest_attempt(task['id'])
    assert attempt is not None
    lease = controller.ledger.get_lease(attempt['id'])
    controller.ledger.record_job_launch(
        attempt_id=attempt['id'],
        job_id='job-still-running',
        idempotency_key='trial-1',
        argv=['rightarm-mlops', 'train-eval-trial'],
        status='running',
    )
    controller.submit_result(
        run['id'],
        TaskResult(
            task_id=task['id'],
            attempt_id=attempt['id'],
            status='waiting_external',
            summary='keep the job alive',
            evidence_refs=['job:job-still-running'],
        ),
        lease['lease_id'],
    )

    class TrackingJobManager:
        cancelled_attempt_ids: list[str] = []

        def running_attempt_ids(self):
            return {attempt['id']}

        def cancel_attempt(self, attempt_id):
            self.cancelled_attempt_ids.append(attempt_id)
            return []

    manager = TrackingJobManager()
    _reconcile_inactive_jobs(
        controller,
        manager,
        logging.getLogger(__name__),
    )

    assert manager.cancelled_attempt_ids == []
