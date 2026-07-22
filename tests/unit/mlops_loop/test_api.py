from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openhands.mlops_loop.api import create_app
from openhands.mlops_loop.artifact_store import LocalArtifactStore
from openhands.mlops_loop.controller import LoopController
from openhands.mlops_loop.dispatcher import InMemoryConversationDispatcher
from openhands.mlops_loop.ledger import SQLiteLedger
from openhands.mlops_loop.models import CompletionContract, TaskSpec
from openhands.mlops_loop.retry_policy import RetryPolicy
from openhands.mlops_loop.validators import ContractValidator, ValidatorRegistry


def test_api_exposes_fenced_task_context_and_audit(outer_loop_api):
    controller, client = outer_loop_api
    task = TaskSpec(
        name='inspect-data',
        kind='inspect-data',
        objective='inspect one sealed batch',
        agent_profile='profile-id',
        validator='data-v1',
        max_attempts=2,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=['source-manifest']
        ),
    )
    response = client.post(
        '/api/v1/mlops/runs',
        json={
            'loop_id': 'gesture-loop',
            'watermark': 'batch-api',
            'run_spec': {'source': {'uri': 'file:///batch-api'}},
            'tasks': [task.model_dump(mode='json')],
        },
    )
    assert response.status_code == 200
    run_id = response.json()['id']
    controller.dispatch_current_task(run_id)
    attempt = controller.ledger.list_attempts(run_id)[0]
    lease = controller.ledger.get_lease(attempt['id'])

    denied = client.get(f'/api/v1/mlops/attempts/{attempt["id"]}/context')
    assert denied.status_code == 422
    accepted = client.get(
        f'/api/v1/mlops/attempts/{attempt["id"]}/context',
        headers={'X-MLOps-Lease-ID': lease['lease_id']},
    )
    assert accepted.status_code == 200
    assert accepted.json()['attempt_id'] == attempt['id']

    audit = client.get(f'/api/v1/mlops/runs/{run_id}/events')
    assert [item['sequence'] for item in audit.json()] == [1, 2, 3, 4, 5]

    package = client.get(f'/api/v1/mlops/runs/{run_id}/audit')
    assert package.status_code == 200
    assert package.json()['protocol'] == 'mlops.openhands.dev/run-audit/v1'
    assert package.json()['run']['id'] == run_id
    assert package.json()['replayed_projection']['version'] == 5
    assert package.json()['outbox'][0]['status'] == 'DELIVERED'


def test_api_rejects_unknown_initial_artifact(outer_loop_api):
    _, client = outer_loop_api
    response = client.post(
        '/api/v1/mlops/runs',
        json={
            'loop_id': 'gesture-loop',
            'watermark': 'missing-input',
            'run_spec': {'source': {'uri': 'file:///batch'}},
            'tasks': [
                {
                    'name': 'train',
                    'kind': 'train',
                    'objective': 'train',
                    'agent_profile': 'profile',
                    'validator': 'validator',
                    'max_attempts': 1,
                    'timeout_seconds': 60,
                }
            ],
            'initial_artifact_ids': ['base-model:sha256:' + '0' * 64],
        },
    )
    assert response.status_code == 409


def test_api_cost_observation_enforces_hard_run_budget(outer_loop_api):
    _, client = outer_loop_api
    created = client.post(
        '/api/v1/mlops/runs',
        json={
            'loop_id': 'gesture-loop',
            'watermark': 'cost-api',
            'run_spec': {
                'spec': {'policy': {'maxRunCostUsd': 0.1}},
                'source': {'uri': 'file:///batch'},
            },
            'tasks': [
                {
                    'name': 'train',
                    'kind': 'train',
                    'objective': 'train',
                    'agent_profile': 'default',
                    'validator': 'validator',
                    'max_attempts': 1,
                    'timeout_seconds': 60,
                }
            ],
        },
    )
    run_id = created.json()['id']
    cost = client.post(
        f'/api/v1/mlops/runs/{run_id}/costs',
        json={
            'source_type': 'llm',
            'source_id': 'conversation-1',
            'amount_usd': 0.1,
            'metadata': {'tokens': 10},
            'idempotency_key': 'charge-1',
        },
    )

    assert cost.status_code == 200
    assert client.get(f'/api/v1/mlops/runs/{run_id}').json()['state'] == (
        'NEEDS_INTERVENTION'
    )
    audit = client.get(f'/api/v1/mlops/runs/{run_id}/audit').json()
    assert audit['total_cost_usd'] == 0.1
    assert [event['event_type'] for event in audit['events']] == [
        'RunCreated',
        'CostRecorded',
        'RunPolicyStopped',
    ]


def test_operator_mutations_require_configured_token(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    client = TestClient(
        create_app(
            controller,
            operator_token='s' * 32,
            operator_identity='release-owner',
        )
    )
    request = {
        'loop_id': 'gesture-loop',
        'watermark': 'operator-auth',
        'run_spec': {'source': {'uri': 'file:///batch'}},
        'tasks': [
            {
                'name': 'inspect',
                'kind': 'inspect-data',
                'objective': 'inspect',
                'agent_profile': 'default',
                'validator': 'default',
                'max_attempts': 1,
                'timeout_seconds': 60,
            }
        ],
    }

    assert client.post('/api/v1/mlops/runs', json=request).status_code == 401
    accepted = client.post(
        '/api/v1/mlops/runs',
        json=request,
        headers={'X-MLOps-Operator-Token': 's' * 32},
    )
    assert accepted.status_code == 200
    run_id = accepted.json()['id']
    assert client.get(f'/api/v1/mlops/runs/{run_id}').status_code == 401
    assert (
        client.get(
            f'/api/v1/mlops/runs/{run_id}',
            headers={'X-MLOps-Operator-Token': 's' * 32},
        ).status_code
        == 200
    )


def test_operator_can_import_artifacts_and_create_manual_train_run(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    import_root = tmp_path / 'imports'
    import_root.mkdir()
    dataset = import_root / 'dataset.pkl'
    dataset.write_bytes(b'dataset')
    base_model = import_root / 'base.pth'
    base_model.write_bytes(b'model')
    token = 'm' * 32
    client = TestClient(
        create_app(
            controller,
            operator_token=token,
            operator_identity='operator',
            operator_import_root=import_root,
        )
    )
    headers = {'X-MLOps-Operator-Token': token}

    artifact_ids = []
    for path, kind in (
        (dataset, 'mmaction-dataset'),
        (base_model, 'base-model'),
    ):
        response = client.post(
            '/api/v1/mlops/artifacts/bootstrap',
            headers=headers,
            json={'path': str(path), 'kind': kind},
        )
        assert response.status_code == 200
        artifact_ids.append(response.json()['id'])

    response = client.post(
        '/api/v1/mlops/manual-runs',
        headers=headers,
        json={
            'mode': 'train',
            'request_id': 'train-api-test',
            'initial_artifact_ids': artifact_ids,
        },
    )

    assert response.status_code == 200
    assert response.json()['state'] == 'CREATED'
    assert response.json()['queued'] is True
    assert response.json()['tasks'] == [
        'train-challenger',
        'evaluate-challenger',
        'package-rknn',
    ]
    listed = client.get('/api/v1/mlops/runs', headers=headers)
    assert listed.status_code == 200
    assert listed.json()['runs'][0]['mode'] == 'train'


def test_operator_import_rejects_path_outside_boundary(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    import_root = tmp_path / 'imports'
    import_root.mkdir()
    outside = tmp_path / 'outside.pkl'
    outside.write_bytes(b'outside')
    client = TestClient(create_app(controller, operator_import_root=import_root))

    response = client.post(
        '/api/v1/mlops/artifacts/bootstrap',
        json={'path': str(outside), 'kind': 'mmaction-dataset'},
    )

    assert response.status_code == 422
    assert 'outside' in response.json()['detail']


@pytest.fixture
def outer_loop_api(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    validators = ValidatorRegistry(ContractValidator(artifacts))
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=validators,
        retry_policy=RetryPolicy(frozenset({'transient-network'})),
    )
    return controller, TestClient(create_app(controller))
