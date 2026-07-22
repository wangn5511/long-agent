import json
import time
from pathlib import Path

import pytest
from fastmcp import Client

from openhands.app_server.mlops_task.mcp_server import (
    DockerJobManager,
    LocalJobManager,
    create_gateway_mcp_server,
)
from openhands.mlops_loop.artifact_store import LocalArtifactStore, sha256_file
from openhands.mlops_loop.controller import LoopController
from openhands.mlops_loop.dispatcher import InMemoryConversationDispatcher
from openhands.mlops_loop.errors import IdempotencyConflict, InvariantViolation
from openhands.mlops_loop.ledger import SQLiteLedger
from openhands.mlops_loop.models import (
    ArtifactCandidate,
    ArtifactRef,
    CompletionContract,
    TaskResult,
    TaskSpec,
)
from openhands.mlops_loop.retry_policy import RetryPolicy
from openhands.mlops_loop.validators import ContractValidator, ValidatorRegistry


def test_artifact_store_accepts_encoded_staging_uri(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / 'cas with spaces')
    source = tmp_path / 'source with spaces.json'
    source.write_text('{}')
    candidate = store.stage(source, kind='source-manifest')

    store.verify(candidate)
    sealed = store.seal(candidate)

    assert sealed.uri.startswith('file://')


def test_agent_cannot_register_an_arbitrary_host_file(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / 'cas')
    outside = tmp_path / 'outside-secret.txt'
    outside.write_text('not an artifact candidate')
    digest, size = sha256_file(outside)
    forged = ArtifactCandidate(
        candidate_id='forged',
        kind='job-log',
        uri=outside.as_uri(),
        sha256=digest,
        size_bytes=size,
    )

    with pytest.raises(InvariantViolation, match='outside the untrusted staging'):
        store.verify(forged)


def test_production_controller_requires_attempt_owned_candidates(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
        candidate_registry_required=True,
    )
    output = tmp_path / 'checkpoint.pth'
    output.write_bytes(b'checkpoint')

    def run_attempt(watermark: str):
        run = controller.create_run(
            loop_id='gesture-loop',
            watermark=watermark,
            run_spec={'source': {'uri': 'file:///batch'}},
            tasks=[
                TaskSpec(
                    name='train',
                    kind='train',
                    objective='train',
                    agent_profile='default',
                    validator='default',
                    max_attempts=1,
                    timeout_seconds=60,
                    completion_contract=CompletionContract(
                        required_artifact_kinds=['checkpoint'],
                        required_evidence=True,
                    ),
                )
            ],
        )
        controller.dispatch_current_task(run['id'])
        attempt = ledger.list_attempts(run['id'])[0]
        lease = ledger.get_lease(attempt['id'])
        candidate = artifacts.stage(output, kind='checkpoint')
        result = TaskResult(
            task_id=attempt['task_id'],
            attempt_id=attempt['id'],
            status='succeeded',
            summary='trained',
            artifact_candidates=[candidate],
            evidence_refs=['job:train'],
        )
        return run, attempt, lease, candidate, result

    unowned = run_attempt('unowned-candidate')
    controller.submit_result(unowned[0]['id'], unowned[4], unowned[2]['lease_id'])
    assert controller.tick(unowned[0]['id']) == 'NEEDS_INTERVENTION'

    owned = run_attempt('owned-candidate')
    ledger.record_artifact_candidate(
        attempt_id=owned[1]['id'],
        lease_id=owned[2]['lease_id'],
        candidate=owned[3],
    )
    controller.submit_result(owned[0]['id'], owned[4], owned[2]['lease_id'])
    assert controller.tick(owned[0]['id']) == 'COMPLETED'
    cross_attempt = run_attempt('cross-attempt-candidate')
    with pytest.raises(IdempotencyConflict, match='reused across attempts'):
        ledger.record_artifact_candidate(
            attempt_id=cross_attempt[1]['id'],
            lease_id=cross_attempt[2]['lease_id'],
            candidate=owned[3],
        )


def test_local_job_manager_rejects_executable_and_external_wrapper_bypasses(
    tmp_path: Path,
):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    manager = LocalJobManager(tmp_path / 'jobs', workspace)

    with pytest.raises(ValueError, match='logical rightarm-mlops'):
        manager.launch(
            ['/tmp/rightarm-mlops', 'train'], idempotency_key='forged-executable'
        )
    with pytest.raises(ValueError, match='not allowed'):
        manager.launch(
            ['rightarm-mlops', 'export-onnx'], idempotency_key='external-command'
        )


class _FakeContainer:
    def __init__(self, container_id: str) -> None:
        self.id = container_id
        self.status = 'running'
        self.attrs = {'State': {'ExitCode': None}}
        self.stopped = False

    def reload(self) -> None:
        return None

    def logs(self, **_kwargs):
        return b'container log\n'

    def stop(self, *, timeout: int) -> None:
        assert timeout == 10
        self.status = 'exited'
        self.attrs = {'State': {'ExitCode': 143}}
        self.stopped = True


class _FakeContainers:
    def __init__(self) -> None:
        self.values = {}
        self.run_requests = []

    def get(self, identity: str):
        import docker

        try:
            return self.values[identity]
        except KeyError as exc:
            raise docker.errors.NotFound('missing') from exc

    def run(self, **request):
        container = _FakeContainer(f'container-{len(self.run_requests) + 1}')
        self.run_requests.append(request)
        self.values[request['name']] = container
        self.values[container.id] = container
        return container


class _FakeDockerClient:
    def __init__(self) -> None:
        self.containers = _FakeContainers()


def test_docker_job_manager_requires_digest_and_enforces_sandbox(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    with pytest.raises(ValueError, match='pinned by sha256'):
        DockerJobManager(
            tmp_path / 'bad-jobs',
            workspace,
            image_catalog={'train': 'rightarm:latest'},
            client=_FakeDockerClient(),
        )

    source_root = tmp_path / 'cloud-materialization'
    source_root.mkdir()
    client = _FakeDockerClient()
    manager = DockerJobManager(
        tmp_path / 'jobs',
        workspace,
        image_catalog={'train': 'rightarm@sha256:' + 'a' * 64},
        read_only_source_roots=[source_root],
        gpu_counts={'train': 1},
        client=client,
    )
    launched = manager.launch(
        [
            'rightarm-mlops',
            'train',
            '--config',
            'config.py',
            '--dataset',
            '/mlops-inputs/dataset',
            '--base-model',
            '/mlops-inputs/base',
            '--gpus',
            '1',
            '--code-sha',
            'a' * 40,
            '--image-digest',
            'sha256:' + 'a' * 64,
        ],
        'train-v1',
        owner_attempt_id='attempt-1',
    )
    repeated = manager.launch(
        [
            'rightarm-mlops',
            'train',
            '--config',
            'config.py',
            '--dataset',
            '/mlops-inputs/dataset',
            '--base-model',
            '/mlops-inputs/base',
            '--gpus',
            '1',
            '--code-sha',
            'a' * 40,
            '--image-digest',
            'sha256:' + 'a' * 64,
        ],
        'train-v1',
        owner_attempt_id='attempt-1',
    )

    assert repeated['job_id'] == launched['job_id']
    assert len(client.containers.run_requests) == 1
    request = client.containers.run_requests[0]
    assert request['network_disabled'] is True
    assert request['read_only'] is True
    assert request['cap_drop'] == ['ALL']
    assert request['security_opt'] == ['no-new-privileges:true']
    assert request['volumes'][str(source_root.resolve())]['mode'] == 'ro'
    assert request['device_requests']
    assert request['device_requests'][0].count == 1
    artifact_ref = ArtifactRef(
        id=f'mmaction-dataset:sha256:{"b" * 64}',
        kind='mmaction-dataset',
        uri=(tmp_path / 'unused').as_uri(),
        sha256='b' * 64,
        size_bytes=1,
    )
    materialized = manager.managed_input_path('attempt-1', artifact_ref)
    assert manager.execution_input_path('attempt-1', artifact_ref, materialized) == str(
        Path('/mlops-inputs') / materialized.name
    )

    output_path = tmp_path / 'jobs' / launched['job_id'] / 'output' / 'model.bin'
    output_path.write_bytes(b'model')
    output_sha, output_size = sha256_file(output_path)
    result_path = tmp_path / 'jobs' / launched['job_id'] / 'output' / 'result.json'
    result_path.write_text(
        json.dumps(
            {
                'protocol': 'rightarm.mlops.job-result/v1',
                'status': 'succeeded',
                'outputs': [
                    {
                        'kind': 'checkpoint',
                        'path': '/mlops-output/model.bin',
                        'sha256': output_sha,
                        'size_bytes': output_size,
                    }
                ],
            }
        )
    )
    container = client.containers.values[request['name']]
    container.status = 'exited'
    container.attrs = {'State': {'ExitCode': 0}}
    observation = manager.observe(launched['job_id'], owner_attempt_id='attempt-1')
    assert observation['status'] == 'succeeded'
    assert observation['image'].endswith('a' * 64)
    assert observation['result']['outputs'][0]['path'] == str(output_path)
    assert (
        manager.assert_declared_output(
            observation['result']['outputs'][0]['path'], 'attempt-1', 'checkpoint'
        )
        == output_path
    )


def test_artifact_staging_requires_declared_job_output_kind_and_hash(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    manager = LocalJobManager(tmp_path / 'jobs', workspace)
    job_dir = tmp_path / 'jobs' / 'job_test'
    output_dir = job_dir / 'output'
    output_dir.mkdir(parents=True)
    output = output_dir / 'model.onnx'
    output.write_bytes(b'model')
    digest, size = sha256_file(output)
    (job_dir / 'request.json').write_text(json.dumps({'owner_attempt_id': 'attempt-1'}))
    (output_dir / 'result.json').write_text(
        json.dumps(
            {
                'protocol': 'rightarm.mlops.job-result/v1',
                'status': 'succeeded',
                'outputs': [
                    {
                        'kind': 'onnx-model',
                        'path': str(output),
                        'sha256': digest,
                        'size_bytes': size,
                    }
                ],
            }
        )
    )

    assert manager.assert_declared_output(output, 'attempt-1', 'onnx-model') == output
    with pytest.raises(PermissionError, match='did not declare'):
        manager.assert_declared_output(output, 'attempt-1', 'checkpoint')
    output.write_bytes(b'tampered')
    with pytest.raises(PermissionError, match='changed after'):
        manager.assert_declared_output(output, 'attempt-1', 'onnx-model')


def test_enabled_external_job_still_requires_checked_in_script(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    with pytest.raises(ValueError, match='external job catalog'):
        LocalJobManager(
            tmp_path / 'no-catalog-jobs', workspace, allow_external_jobs=True
        )
    script = workspace / 'deploy_adapter.py'
    script.write_text('print("adapter")')
    manager = LocalJobManager(
        tmp_path / 'jobs',
        workspace,
        allow_external_jobs=True,
        external_job_catalog={'deploy': ['python', str(script)]},
    )
    malicious = [
        'rightarm-mlops',
        'deploy',
        '--command-json',
        json.dumps(['/bin/sh', '-c', 'true']),
    ]
    with pytest.raises(ValueError, match='catalog argv prefix'):
        manager.launch(malicious, idempotency_key='malicious')

    manager._validate_external_command(
        [
            'rightarm-mlops',
            'deploy',
            '--command-json',
            json.dumps(['python', str(script)]),
        ]
    )


def test_job_output_directory_is_controller_managed(tmp_path: Path):
    requested = tmp_path / 'agent-selected'
    managed = tmp_path / 'controller-selected'
    normalized = LocalJobManager._with_managed_output_dir(
        [
            'rightarm-mlops',
            'evaluate',
            '--output-dir',
            str(requested),
            f'--output-dir={tmp_path / "second-agent-path"}',
        ],
        managed,
    )

    assert normalized[-1] == str(managed)
    assert normalized.count('--output-dir') == 1
    assert all('agent-path' not in value for value in normalized)


def test_job_launch_idempotency_survives_manager_restart(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    package = workspace / 'rightarm_mlops'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / '__main__.py').write_text(
        """
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
output.mkdir(parents=True, exist_ok=True)
with (Path.cwd() / 'launches.txt').open('a') as stream:
    stream.write('launch\\n')
(output / 'result.json').write_text(json.dumps({'status': 'succeeded'}))
"""
    )
    first = LocalJobManager(tmp_path / 'jobs', workspace)
    request = ['rightarm-mlops', 'inspect-batch']
    launched = first.launch(
        request,
        'stable-request',
        owner_attempt_id='attempt-1',
    )
    for _ in range(100):
        observation = first.observe(launched['job_id'], owner_attempt_id='attempt-1')
        if observation['status'] == 'succeeded':
            break
        time.sleep(0.01)
    assert observation['status'] == 'succeeded'

    restarted = LocalJobManager(tmp_path / 'jobs', workspace)
    recovered = restarted.launch(
        request,
        'stable-request',
        owner_attempt_id='attempt-1',
    )
    assert recovered['job_id'] == launched['job_id']
    assert recovered['status'] == 'succeeded'
    assert (workspace / 'launches.txt').read_text().splitlines() == ['launch']
    result_path = tmp_path / 'jobs' / launched['job_id'] / 'output' / 'result.json'
    assert first.assert_owned_output(result_path, 'attempt-1') == result_path
    with pytest.raises(PermissionError, match='different task attempt'):
        first.assert_owned_output(result_path, 'attempt-2')


def test_expired_attempt_jobs_are_scoped_and_cancelled(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    package = workspace / 'rightarm_mlops'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / '__main__.py').write_text('import time\ntime.sleep(30)\n')
    manager = LocalJobManager(tmp_path / 'jobs', workspace)
    first = manager.launch(
        ['rightarm-mlops', 'inspect-batch'],
        'first',
        owner_attempt_id='attempt-expired',
    )
    second = manager.launch(
        ['rightarm-mlops', 'inspect-batch'],
        'second',
        owner_attempt_id='attempt-active',
    )

    assert manager.cancel_attempt('attempt-expired') == [first['job_id']]
    assert (
        manager.observe(first['job_id'], owner_attempt_id='attempt-expired')['status']
        == 'cancelled'
    )
    assert (
        manager.observe(second['job_id'], owner_attempt_id='attempt-active')['status']
        == 'running'
    )
    manager.cancel_attempt('attempt-active')


@pytest.mark.asyncio
async def test_shared_mcp_gateway_returns_only_fenced_task_context(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='mcp-gateway',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[
            TaskSpec(
                name='inspect-data',
                kind='inspect-data',
                objective='inspect the batch',
                agent_profile='profile',
                validator='validator',
                max_attempts=1,
                timeout_seconds=60,
                completion_contract=CompletionContract(),
            )
        ],
    )
    controller.dispatch_current_task(run['id'])
    attempt = ledger.list_attempts(run['id'])[0]
    lease = ledger.get_lease(attempt['id'])
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    gateway = create_gateway_mcp_server(
        controller=controller,
        job_manager=LocalJobManager(tmp_path / 'jobs', workspace),
    )

    async with Client(gateway) as client:
        response = await client.call_tool(
            'mlops_get_task_context',
            {'attempt_id': attempt['id'], 'lease_id': lease['lease_id']},
        )
        unauthorized_artifact = await client.call_tool(
            'mlops_stage_artifact',
            {
                'attempt_id': attempt['id'],
                'lease_id': lease['lease_id'],
                'path': '/not/a/job/output',
                'kind': 'checkpoint',
            },
            raise_on_error=False,
        )

    assert response.data['run_id'] == run['id']
    assert response.data['attempt_id'] == attempt['id']
    assert response.data['objective'] == 'inspect the batch'
    assert unauthorized_artifact.is_error


@pytest.mark.asyncio
async def test_gateway_materializes_only_a_declared_input(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    source = tmp_path / 'source-batch.json'
    source.write_text('{"batch": "sealed"}')
    artifact = controller.bootstrap_artifact(
        path=str(source), kind='source-batch-manifest'
    )
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='materialize-input',
        run_spec={'source': {'uri': artifact['uri']}},
        initial_artifact_ids=[artifact['id']],
        tasks=[
            TaskSpec(
                name='inspect-data',
                kind='inspect-data',
                objective='inspect the batch',
                agent_profile='profile',
                validator='validator',
                max_attempts=1,
                timeout_seconds=60,
            )
        ],
    )
    controller.dispatch_current_task(run['id'])
    attempt = ledger.list_attempts(run['id'])[0]
    lease = ledger.get_lease(attempt['id'])
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    jobs = LocalJobManager(tmp_path / 'jobs', workspace)
    gateway = create_gateway_mcp_server(controller=controller, job_manager=jobs)

    async with Client(gateway) as client:
        response = await client.call_tool(
            'mlops_materialize_artifact',
            {
                'attempt_id': attempt['id'],
                'lease_id': lease['lease_id'],
                'artifact_id': artifact['id'],
            },
        )
        rejected = await client.call_tool(
            'mlops_materialize_artifact',
            {
                'attempt_id': attempt['id'],
                'lease_id': lease['lease_id'],
                'artifact_id': 'secret:sha256:' + '0' * 64,
            },
            raise_on_error=False,
        )

    materialized = Path(response.data['path'])
    assert materialized.read_text() == source.read_text()
    assert materialized.is_relative_to((tmp_path / 'jobs' / 'inputs').resolve())
    assert rejected.is_error


@pytest.mark.asyncio
async def test_gateway_persists_job_request_and_terminal_observation(tmp_path: Path):
    ledger = SQLiteLedger(tmp_path / 'ledger.db')
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    controller = LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=InMemoryConversationDispatcher(),
        validators=ValidatorRegistry(ContractValidator(artifacts)),
        retry_policy=RetryPolicy(frozenset()),
    )
    run = controller.create_run(
        loop_id='gesture-loop',
        watermark='job-audit',
        run_spec={'source': {'uri': 'file:///batch'}},
        tasks=[
            TaskSpec(
                name='inspect-data',
                kind='inspect-data',
                objective='inspect the batch',
                agent_profile='profile',
                validator='validator',
                max_attempts=1,
                timeout_seconds=60,
            )
        ],
    )
    controller.dispatch_current_task(run['id'])
    attempt = ledger.list_attempts(run['id'])[0]
    lease = ledger.get_lease(attempt['id'])
    workspace = tmp_path / 'workspace'
    package = workspace / 'rightarm_mlops'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / '__main__.py').write_text(
        """
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
output.mkdir(parents=True, exist_ok=True)
(output / 'result.json').write_text(json.dumps({'status': 'succeeded'}))
"""
    )
    gateway = create_gateway_mcp_server(
        controller=controller,
        job_manager=LocalJobManager(tmp_path / 'jobs', workspace),
    )
    request = {
        'attempt_id': attempt['id'],
        'lease_id': lease['lease_id'],
        'argv': ['rightarm-mlops', 'inspect-batch'],
        'idempotency_key': 'inspect-v1',
    }
    async with Client(gateway) as client:
        unauthorized = await client.call_tool(
            'mlops_launch_job',
            {
                **request,
                'argv': ['rightarm-mlops', 'deploy'],
                'idempotency_key': 'deploy-bypass',
            },
            raise_on_error=False,
        )
        launched = await client.call_tool('mlops_launch_job', request)
        for _ in range(100):
            observed = await client.call_tool(
                'mlops_observe_job',
                {
                    'attempt_id': attempt['id'],
                    'lease_id': lease['lease_id'],
                    'job_id': launched.data['job_id'],
                },
            )
            if observed.data['status'] == 'succeeded':
                break
            time.sleep(0.01)

    assert observed.data['status'] == 'succeeded'
    assert unauthorized.is_error
    jobs = ledger.list_job_attempts(run['id'])
    assert len(jobs) == 1
    assert jobs[0]['task_attempt_id'] == attempt['id']
    assert jobs[0]['status'] == 'succeeded'
    assert jobs[0]['log_sha256']
    audit = ledger.audit_run(run['id'])
    assert audit['job_attempts'][0]['id'] == launched.data['job_id']
    job_events = [
        event['event_type']
        for event in audit['events']
        if event['event_type'].startswith('JobAttempt')
    ]
    assert job_events[0] == 'JobAttemptRequested'
    assert job_events[-1] == 'JobAttemptObserved'
    assert set(job_events[1:]) == {'JobAttemptObserved'}
