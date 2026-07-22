import json
from pathlib import Path

import pytest

from openhands.mlops_loop.artifact_store import LocalArtifactStore
from openhands.mlops_loop.cli import _command_argv
from openhands.mlops_loop.controller import LoopController
from openhands.mlops_loop.dispatcher import InMemoryConversationDispatcher
from openhands.mlops_loop.errors import LeaseConflict
from openhands.mlops_loop.gesture_loop import GestureLoopService
from openhands.mlops_loop.ledger import SQLiteLedger
from openhands.mlops_loop.models import (
    CompletionContract,
    LineageProposal,
    RunState,
    TaskResult,
    TaskSpec,
)
from openhands.mlops_loop.retry_policy import RetryPolicy
from openhands.mlops_loop.source_watcher import LocalBatchWatcher
from openhands.mlops_loop.validators import ContractValidator, ValidatorRegistry


def test_three_watermarks_are_discovered_exactly_once(tmp_path: Path):
    incoming = tmp_path / 'incoming'
    incoming.mkdir()
    for number in range(3):
        batch = incoming / f'batch-{number}'
        batch.mkdir()
        (batch / 'batch.manifest.json').write_text(
            json.dumps({'batch_id': f'batch-{number}'})
        )
        (batch / '_SUCCESS').touch()

    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    validators = ValidatorRegistry(ContractValidator(artifacts))
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=validators,
        retry_policy=RetryPolicy(frozenset()),
    )
    loop_spec = {
        'metadata': {'name': 'rightarm-gesture-v1'},
        'spec': {
            'taskSequence': [
                {
                    'name': 'inspect-data',
                    'kind': 'inspect-data',
                    'agentProfile': 'profile',
                    'validator': 'validator',
                    'maxAttempts': 1,
                    'timeoutSeconds': 60,
                }
            ]
        },
    }
    run_template = {
        'metadata': {'name': 'rightarm-run'},
        'spec': {
            'source': {'uri': '<batch-source-uri>'},
            'training': {'image': 'image@sha256:<image-digest>'},
            'evaluation': {'gates': []},
        },
    }
    service = GestureLoopService(
        controller=controller,
        watcher=LocalBatchWatcher(incoming),
        loop_spec=loop_spec,
        run_template=run_template,
        replacements={'image-digest': 'a' * 64},
    )

    for expected in range(3):
        run = service.poll_once()
        assert run is not None
        assert run['watermark'] == f'batch-{expected}'
        controller.dispatch_current_task(run['id'])
        task = ledger.current_task(run['id'])
        attempt = ledger.latest_attempt(task['id'])
        lease = ledger.get_lease(attempt['id'])
        report = tmp_path / f'report-{expected}.json'
        report.write_text('{}')
        source = tmp_path / f'source-{expected}.json'
        source.write_text('{}')
        result = TaskResult(
            task_id=task['id'],
            attempt_id=attempt['id'],
            status='succeeded',
            summary='batch inspection completed',
            artifact_candidates=[
                artifacts.stage(source, kind='source-manifest'),
                artifacts.stage(report, kind='data-validation-report'),
            ],
            evidence_refs=[f'report:{expected}'],
        )
        controller.submit_result(run['id'], result, lease['lease_id'])
        controller.tick(run['id'])

    assert service.poll_once() is None
    assert [run['watermark'] for run in ledger.list_runs('rightarm-gesture-v1')] == [
        'batch-0',
        'batch-1',
        'batch-2',
    ]
    for run in ledger.list_runs('rightarm-gesture-v1'):
        resolved = ledger.get_run(run['id'])
        manifest_id = resolved['spec']['spec']['source']['manifestArtifactId']
        assert manifest_id in {
            artifact['id'] for artifact in ledger.list_artifacts(run['id'])
        }


def test_cloud_watcher_command_is_an_argv_not_a_shell_string():
    assert _command_argv('["tos-discover", "--prefix", "business/batches"]') == [
        'tos-discover',
        '--prefix',
        'business/batches',
    ]


def test_discovered_manifest_is_required_as_raw_lineage_root(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    manifest_path = tmp_path / 'batch.manifest.json'
    manifest_path.write_text('{"batch_id":"batch-1"}')
    source = controller.bootstrap_artifact(
        path=str(manifest_path), kind='source-manifest'
    )
    run = controller.create_run(
        loop_id='rightarm-gesture-v1',
        watermark='batch-1',
        run_spec={
            'spec': {
                'source': {
                    'manifestArtifactId': source['id'],
                    'manifestUri': manifest_path.as_uri(),
                }
            }
        },
        initial_artifact_ids=[source['id']],
        tasks=[
            TaskSpec(
                name='inspect-data',
                kind='inspect-data',
                objective='inspect and land',
                agent_profile='default',
                validator='default',
                max_attempts=1,
                timeout_seconds=60,
                completion_contract=CompletionContract(
                    required_artifact_kinds=[
                        'source-manifest',
                        'data-validation-report',
                        'raw-dataset',
                    ],
                    required_evidence=True,
                ),
            )
        ],
    )
    controller.dispatch_current_task(run['id'])
    task = ledger.current_task(run['id'])
    attempt = ledger.latest_attempt(task['id'])
    lease = ledger.get_lease(attempt['id'])
    report_path = tmp_path / 'report.json'
    report_path.write_text('{}')
    raw_path = tmp_path / 'raw.tar'
    raw_path.write_bytes(b'raw')
    source_candidate = artifacts.stage(manifest_path, kind='source-manifest')
    raw_candidate = artifacts.stage(raw_path, kind='raw-dataset')
    result = TaskResult(
        task_id=task['id'],
        attempt_id=attempt['id'],
        status='succeeded',
        summary='landed the discovered batch',
        artifact_candidates=[
            source_candidate,
            artifacts.stage(report_path, kind='data-validation-report'),
            raw_candidate,
        ],
        lineage_proposals=[
            LineageProposal(
                input_artifact_id=source['id'],
                output_candidate_id=raw_candidate.candidate_id,
                transform_kind='land-batch',
            )
        ],
        evidence_refs=['job:land'],
    )

    controller.submit_result(run['id'], result, lease['lease_id'])

    assert controller.tick(run['id']) is RunState.COMPLETED
    assert ledger.list_lineage(run['id'])[0]['input_artifact_id'] == source['id']


def test_consecutive_failed_runs_open_persistent_loop_circuit_breaker(
    tmp_path: Path,
):
    incoming = tmp_path / 'incoming'
    incoming.mkdir()
    for number in range(4):
        batch = incoming / f'batch-{number}'
        batch.mkdir()
        (batch / 'batch.manifest.json').write_text(
            json.dumps({'batch_id': f'batch-{number}'})
        )
        (batch / '_SUCCESS').touch()
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    service = GestureLoopService(
        controller=controller,
        watcher=LocalBatchWatcher(incoming),
        loop_spec={
            'metadata': {'name': 'rightarm-gesture-v1'},
            'spec': {
                'taskSequence': [
                    {
                        'name': 'inspect-data',
                        'kind': 'inspect-data',
                        'agentProfile': 'default',
                        'validator': 'validator',
                        'maxAttempts': 1,
                        'timeoutSeconds': 60,
                    }
                ],
                'stop': {'maxConsecutiveFailedRuns': 3},
            },
        },
        run_template={
            'metadata': {'name': 'rightarm-run'},
            'spec': {
                'source': {'uri': '<batch-source-uri>'},
                'evaluation': {'gates': []},
            },
        },
        replacements={},
    )
    for _ in range(3):
        run = service.poll_once()
        assert run is not None
        controller.dispatch_current_task(run['id'])
        task = ledger.current_task(run['id'])
        assert task is not None
        attempt = ledger.latest_attempt(task['id'])
        assert attempt is not None
        lease = ledger.get_lease(attempt['id'])
        controller.submit_result(
            run['id'],
            TaskResult(
                task_id=task['id'],
                attempt_id=attempt['id'],
                status='succeeded',
                summary='fake success without required artifacts',
                evidence_refs=['claim-only'],
            ),
            lease['lease_id'],
        )
        assert controller.tick(run['id']) is RunState.NEEDS_INTERVENTION

    assert service.poll_once() is None
    control = ledger.get_loop_control('rightarm-gesture-v1')
    assert control['state'] == 'STOPPED'
    assert len(ledger.list_runs('rightarm-gesture-v1')) == 3
    stopped_run = ledger.get_run(control['stopped_by_run_id'])
    assert stopped_run['watermark'] == 'batch-2'
    assert ledger.list_events(stopped_run['id'])[-1]['event_type'] == 'LoopStopped'


def test_scheduler_lease_has_single_holder_and_expired_takeover(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    first = ledger.acquire_scheduler_lease(
        loop_id='rightarm-gesture-v1', holder_id='controller-a', ttl_seconds=30
    )
    ledger.heartbeat_scheduler_lease(
        loop_id='rightarm-gesture-v1', lease_id=first, ttl_seconds=30
    )
    with pytest.raises(LeaseConflict, match='controller-a'):
        ledger.acquire_scheduler_lease(
            loop_id='rightarm-gesture-v1',
            holder_id='controller-b',
            ttl_seconds=30,
        )
    with ledger._connect() as connection:
        connection.execute(
            'UPDATE mlops_scheduler_leases SET expires_at=0 WHERE loop_id=?',
            ('rightarm-gesture-v1',),
        )

    second = ledger.acquire_scheduler_lease(
        loop_id='rightarm-gesture-v1', holder_id='controller-b', ttl_seconds=30
    )

    assert second != first
    with pytest.raises(LeaseConflict, match='fencing rejected'):
        ledger.heartbeat_scheduler_lease(
            loop_id='rightarm-gesture-v1', lease_id=first, ttl_seconds=30
        )
    assert ledger.release_scheduler_lease(
        loop_id='rightarm-gesture-v1', lease_id=second
    )
    assert ledger.get_scheduler_lease('rightarm-gesture-v1') is None
