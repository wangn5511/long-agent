import json
import time
from pathlib import Path

import pytest
from fastmcp import Client

from openhands.app_server.mlops_task.mcp_server import (
    TASK_JOB_KINDS,
    DockerJobManager,
    LocalJobManager,
    _fresh_poll_observation,
    _observe_job_until_terminal,
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


def test_job_poll_observations_are_distinct_without_changing_job_state():
    first = _fresh_poll_observation({'job_id': 'job-1', 'status': 'running'})
    second = _fresh_poll_observation({'job_id': 'job-1', 'status': 'running'})

    assert first['job_id'] == second['job_id'] == 'job-1'
    assert first['status'] == second['status'] == 'running'
    assert first['poll_observation_id'] != second['poll_observation_id']


def test_terminal_job_observation_remains_stable():
    terminal = {'job_id': 'job-1', 'status': 'succeeded'}

    assert _fresh_poll_observation(terminal) is terminal


def test_job_observation_long_poll_returns_as_soon_as_job_is_terminal():
    class SequencedJobManager:
        def __init__(self):
            self.calls = 0

        def observe(self, job_id, *, owner_attempt_id):
            assert job_id == 'job-1'
            assert owner_attempt_id == 'attempt-1'
            self.calls += 1
            return {
                'job_id': job_id,
                'status': 'running' if self.calls < 3 else 'succeeded',
            }

    manager = SequencedJobManager()
    observation = _observe_job_until_terminal(
        manager,
        'job-1',
        owner_attempt_id='attempt-1',
        max_wait_seconds=1,
        max_poll_interval_seconds=0.001,
    )

    assert observation['status'] == 'succeeded'
    assert manager.calls == 3


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


def test_distribution_review_task_has_only_profile_and_merge_capabilities():
    assert TASK_JOB_KINDS['review-dataset-distribution'] == frozenset(
        {'profile-dataset-distribution', 'merge-dataset'}
    )
    assert 'train' not in TASK_JOB_KINDS['review-dataset-distribution']
    assert 'deploy' not in TASK_JOB_KINDS['review-dataset-distribution']
    assert {
        'profile-dataset-distribution',
        'merge-dataset',
    }.issubset(LocalJobManager.SAFE_JOB_KINDS)


def test_model_improvement_task_has_only_bounded_experiment_capabilities():
    allowed = frozenset(
        {
            'benchmark-checkpoint',
            'train-eval-trial',
            'record-model-improvement',
        }
    )
    assert TASK_JOB_KINDS['improve-model'] == allowed
    assert {
        'train',
        'evaluate',
        'export-onnx',
        'deploy',
        'build-release',
    }.isdisjoint(TASK_JOB_KINDS['improve-model'])
    assert allowed.issubset(LocalJobManager.SAFE_JOB_KINDS)


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


@pytest.mark.parametrize(
    'kind',
    ['benchmark-checkpoint', 'train-eval-trial'],
)
def test_model_improvement_gpu_jobs_require_operator_owned_gpu_count(
    tmp_path: Path, kind: str
):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    client = _FakeDockerClient()
    image = 'rightarm@sha256:' + 'a' * 64
    manager = DockerJobManager(
        tmp_path / 'jobs',
        workspace,
        image_catalog={kind: image},
        gpu_counts={kind: 2},
        client=client,
    )
    argv = ['rightarm-mlops', kind, '--gpus', '1']
    with pytest.raises(ValueError, match='operator-owned Docker GPU count'):
        manager.launch(argv, f'{kind}-wrong-gpus', owner_attempt_id='attempt-1')

    argv[3] = '2'
    launched = manager.launch(argv, f'{kind}-right-gpus', owner_attempt_id='attempt-1')

    assert launched['status'] == 'running'
    request = client.containers.run_requests[0]
    assert request['network_disabled'] is True
    assert request['read_only'] is True
    assert request['device_requests'][0].count == 2


def test_model_improvement_record_job_does_not_require_or_receive_gpu(
    tmp_path: Path,
):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    client = _FakeDockerClient()
    manager = DockerJobManager(
        tmp_path / 'jobs',
        workspace,
        image_catalog={'record-model-improvement': 'rightarm@sha256:' + 'a' * 64},
        client=client,
    )

    launched = manager.launch(
        ['rightarm-mlops', 'record-model-improvement'],
        'record-improvement',
        owner_attempt_id='attempt-1',
    )

    assert launched['status'] == 'running'
    request = client.containers.run_requests[0]
    assert 'device_requests' not in request


def test_docker_job_manager_can_pin_operator_owned_gpu_devices(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    client = _FakeDockerClient()
    image = 'rightarm@sha256:' + 'a' * 64
    manager = DockerJobManager(
        tmp_path / 'jobs',
        workspace,
        image_catalog={'train-eval-trial': image},
        gpu_counts={'train-eval-trial': 3},
        gpu_device_ids={'train-eval-trial': ['4', '5', '6']},
        client=client,
    )

    manager.launch(
        ['rightarm-mlops', 'train-eval-trial', '--gpus', '3'],
        'trial-on-free-gpus',
        owner_attempt_id='attempt-1',
    )

    request = client.containers.run_requests[0]
    device_request = request['device_requests'][0]
    assert device_request.count == 0
    assert device_request.device_ids == ['4', '5', '6']

    with pytest.raises(ValueError, match='matching its GPU count'):
        DockerJobManager(
            tmp_path / 'bad-jobs',
            workspace,
            image_catalog={'train-eval-trial': image},
            gpu_counts={'train-eval-trial': 3},
            gpu_device_ids={'train-eval-trial': ['4', '5']},
            client=client,
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


def test_structured_job_outputs_are_bounded_fenced_and_grouped_by_kind(
    tmp_path: Path,
):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    manager = LocalJobManager(tmp_path / 'jobs', workspace)
    job_id = 'job_' + 'a' * 24
    job_dir = manager.root / job_id
    output_dir = job_dir / 'output'
    output_dir.mkdir(parents=True)
    (job_dir / 'request.json').write_text(json.dumps({'owner_attempt_id': 'attempt-1'}))

    def declared(kind: str, path: Path) -> dict[str, object]:
        digest, size = sha256_file(path)
        return {
            'kind': kind,
            'path': str(path),
            'sha256': digest,
            'size_bytes': size,
        }

    baseline = output_dir / 'baseline.json'
    baseline.write_text(json.dumps({'per_class': {'NONE': {'recall': 0.75}}}))
    evaluation = output_dir / 'evaluation.json'
    evaluation.write_text(json.dumps({'confusion_matrix': [[3, 1], [0, 4]]}))
    trial = output_dir / 'trial.json'
    trial.write_text(json.dumps({'trial_number': 1, 'decision': 'promote'}))
    tampered = output_dir / 'tampered.json'
    tampered.write_text(json.dumps({'trusted': True}))
    invalid = output_dir / 'invalid.json'
    invalid.write_text('{')
    json_in_text_file = output_dir / 'report.txt'
    json_in_text_file.write_text(json.dumps({'must_not': 'be inferred as JSON'}))
    oversized = output_dir / 'oversized.json'
    oversized.write_text(
        json.dumps({'padding': 'x' * manager.MAX_STRUCTURED_OUTPUT_BYTES})
    )
    outside = tmp_path / 'outside.json'
    outside.write_text(json.dumps({'secret': True}))

    other_job_dir = manager.root / ('job_' + 'b' * 24)
    other_output_dir = other_job_dir / 'output'
    other_output_dir.mkdir(parents=True)
    (other_job_dir / 'request.json').write_text(
        json.dumps({'owner_attempt_id': 'attempt-1'})
    )
    other_report = other_output_dir / 'other.json'
    other_report.write_text(json.dumps({'other_job': True}))

    outputs = [
        declared('evaluation-report', baseline),
        declared('evaluation-report', evaluation),
        declared('optimization-trial-report', trial),
        declared('tampered-report', tampered),
        declared('invalid-report', invalid),
        declared('text-report', json_in_text_file),
        declared('oversized-report', oversized),
        declared('outside-report', outside),
        declared('other-job-report', other_report),
    ]
    result = {
        'protocol': 'rightarm.mlops.job-result/v1',
        'status': 'succeeded',
        'outputs': outputs,
    }
    (output_dir / 'result.json').write_text(json.dumps(result))
    tampered.write_text(json.dumps({'trusted': False}))
    observation = {'status': 'succeeded', 'result': result}

    structured = manager.read_structured_outputs(
        job_id,
        observation,
        'attempt-1',
    )

    assert structured == {
        'evaluation-report': [
            {'per_class': {'NONE': {'recall': 0.75}}},
            {'confusion_matrix': [[3, 1], [0, 4]]},
        ],
        'optimization-trial-report': [{'trial_number': 1, 'decision': 'promote'}],
    }
    assert (
        manager.read_structured_outputs(
            job_id,
            {**observation, 'status': 'failed'},
            'attempt-1',
        )
        == {}
    )
    assert manager.read_structured_outputs(job_id, observation, 'attempt-2') == {}


def test_structured_job_output_count_limits_fail_closed(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    manager = LocalJobManager(tmp_path / 'jobs', workspace)
    job_id = 'job_' + 'c' * 24
    job_dir = manager.root / job_id
    output_dir = job_dir / 'output'
    output_dir.mkdir(parents=True)
    (job_dir / 'request.json').write_text(json.dumps({'owner_attempt_id': 'attempt-1'}))

    outputs = []
    for index in range(manager.MAX_STRUCTURED_OUTPUTS + 1):
        path = output_dir / f'report-{index}.json'
        path.write_text(json.dumps({'index': index}))
        digest, size = sha256_file(path)
        outputs.append(
            {
                'kind': 'evaluation-report',
                'path': str(path),
                'sha256': digest,
                'size_bytes': size,
            }
        )
    result = {
        'protocol': 'rightarm.mlops.job-result/v1',
        'status': 'succeeded',
        'outputs': outputs,
    }
    (output_dir / 'result.json').write_text(json.dumps(result))

    structured = manager.read_structured_outputs(
        job_id,
        {'status': 'succeeded', 'result': result},
        'attempt-1',
    )

    assert [item['index'] for item in structured['evaluation-report']] == list(
        range(manager.MAX_STRUCTURED_OUTPUTS)
    )

    manager.MAX_STRUCTURED_OUTPUT_TOTAL_BYTES = sum(
        item['size_bytes'] for item in outputs[:2]
    )
    structured = manager.read_structured_outputs(
        job_id,
        {'status': 'succeeded', 'result': result},
        'attempt-1',
    )
    assert [item['index'] for item in structured['evaluation-report']] == [0, 1]

    result['outputs'] = [
        {'kind': 'ignored', 'path': 'ignored', 'size_bytes': 0}
        for _ in range(manager.MAX_STRUCTURED_OUTPUT_DECLARATIONS + 1)
    ]
    (output_dir / 'result.json').write_text(json.dumps(result))
    assert (
        manager.read_structured_outputs(
            job_id,
            {'status': 'succeeded', 'result': result},
            'attempt-1',
        )
        == {}
    )


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


@pytest.mark.asyncio
async def test_shared_gateway_inlines_declared_machine_json_reports(tmp_path: Path):
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
        watermark='structured-job-observation',
        run_spec={'source': {'uri': 'file:///dataset.pkl'}},
        tasks=[
            TaskSpec(
                name='improve-model',
                kind='improve-model',
                objective='inspect machine-readable evaluation reports',
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
    package = workspace / 'rightarm_mlops'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / '__main__.py').write_text(
        """
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
output.mkdir(parents=True, exist_ok=True)
reports = [
    ('evaluation-report', 'baseline.json', {
        'per_class': {'NONE': {'recall': 0.8}},
        'confusion_matrix': [[8, 2], [1, 9]],
    }),
    ('optimization-trial-report', 'trial.json', {
        'trial_number': 1,
        'decision': 'promote',
    }),
]
declared = []
for kind, filename, value in reports:
    path = output / filename
    path.write_text(json.dumps(value))
    payload = path.read_bytes()
    declared.append({
        'kind': kind,
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(payload).hexdigest(),
        'size_bytes': len(payload),
    })
(output / 'result.json').write_text(json.dumps({
    'protocol': 'rightarm.mlops.job-result/v1',
    'status': 'succeeded',
    'outputs': declared,
}))
"""
    )
    gateway = create_gateway_mcp_server(
        controller=controller,
        job_manager=LocalJobManager(tmp_path / 'jobs', workspace),
    )

    async with Client(gateway) as client:
        launched = await client.call_tool(
            'mlops_launch_job',
            {
                'attempt_id': attempt['id'],
                'lease_id': lease['lease_id'],
                'argv': ['rightarm-mlops', 'benchmark-checkpoint'],
                'idempotency_key': 'benchmark-v1',
            },
        )
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

    assert observed.data['structured_outputs'] == {
        'evaluation-report': [
            {
                'per_class': {'NONE': {'recall': 0.8}},
                'confusion_matrix': [[8, 2], [1, 9]],
            }
        ],
        'optimization-trial-report': [{'trial_number': 1, 'decision': 'promote'}],
    }


@pytest.mark.asyncio
async def test_gateway_exposes_only_sanitized_controlled_launch_errors(tmp_path: Path):
    class ErrorJobManager(LocalJobManager):
        def launch(
            self,
            argv: list[str],
            idempotency_key: str,
            *,
            owner_attempt_id: str | None = None,
        ):
            del argv, owner_attempt_id
            if idempotency_key.endswith(':controlled'):
                raise ValueError(
                    'benchmark-checkpoint job requires an integer --gpus value; '
                    'token=sk-this-must-not-leak'
                )
            raise RuntimeError('unexpected executor failure sk-also-must-not-leak')

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
        watermark='gateway-error-masking',
        run_spec={'source': {'uri': 'file:///dataset.pkl'}},
        tasks=[
            TaskSpec(
                name='improve-model',
                kind='improve-model',
                objective='run a bounded model experiment',
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
        job_manager=ErrorJobManager(tmp_path / 'jobs', workspace),
    )
    base_request = {
        'attempt_id': attempt['id'],
        'lease_id': lease['lease_id'],
        'argv': ['rightarm-mlops', 'benchmark-checkpoint'],
    }

    async with Client(gateway) as client:
        controlled = await client.call_tool(
            'mlops_launch_job',
            {**base_request, 'idempotency_key': 'controlled'},
            raise_on_error=False,
        )
        unexpected = await client.call_tool(
            'mlops_launch_job',
            {**base_request, 'idempotency_key': 'unexpected'},
            raise_on_error=False,
        )

    controlled_text = str(controlled.content)
    unexpected_text = str(unexpected.content)
    assert controlled.is_error
    assert 'requires an integer --gpus value' in controlled_text
    assert '[REDACTED]' in controlled_text
    assert 'sk-this-must-not-leak' not in controlled_text
    assert unexpected.is_error
    assert "Error calling tool 'mlops_launch_job'" in unexpected_text
    assert 'sk-also-must-not-leak' not in unexpected_text
