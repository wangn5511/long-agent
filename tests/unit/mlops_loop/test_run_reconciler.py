from pathlib import Path

from openhands.mlops_loop.artifact_store import LocalArtifactStore
from openhands.mlops_loop.cli import _run_reconciler
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
