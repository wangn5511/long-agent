import sys
from pathlib import Path

import docker

from openhands.app_server.mlops_task.hybrid_job_manager import HybridJobManager
from openhands.app_server.mlops_task.mcp_server import DockerJobManager
from openhands.mlops_loop.models import ArtifactRef


class _Container:
    def __init__(self, container_id: str) -> None:
        self.id = container_id
        self.status = 'running'
        self.attrs = {'State': {'ExitCode': None}}

    def reload(self) -> None:
        return None

    def logs(self, **_kwargs):
        return b''

    def stop(self, *, timeout: int) -> None:
        assert timeout == 10
        self.status = 'exited'
        self.attrs = {'State': {'ExitCode': 143}}


class _Containers:
    def __init__(self) -> None:
        self.values = {}
        self.run_requests = []

    def get(self, identity: str):
        try:
            return self.values[identity]
        except KeyError as exc:
            raise docker.errors.NotFound('missing') from exc

    def run(self, **request):
        container = _Container(f'container-{len(self.run_requests) + 1}')
        self.run_requests.append(request)
        self.values[request['name']] = container
        self.values[container.id] = container
        return container


class _DockerClient:
    def __init__(self) -> None:
        self.containers = _Containers()


def test_docker_executor_uses_operator_host_paths_and_preserves_source_target(
    tmp_path: Path,
):
    controller_root = tmp_path / 'controller-jobs'
    controller_workspace = tmp_path / 'controller-workspace'
    controller_workspace.mkdir()
    source_target = tmp_path / 'source-target'
    source_target.mkdir()
    source_link = tmp_path / 'source-link'
    source_link.symlink_to(source_target, target_is_directory=True)
    host_root = Path('/operator/state/mlops/jobs')
    host_workspace = Path('/operator/repository/rightArmPerception')
    image = 'rightarm@sha256:' + 'a' * 64
    client = _DockerClient()
    manager = DockerJobManager(
        controller_root,
        controller_workspace,
        image_catalog={'benchmark-checkpoint': image},
        host_root=host_root,
        host_workspace=host_workspace,
        read_only_source_roots=[source_link],
        gpu_counts={'benchmark-checkpoint': 1},
        client=client,
    )

    launched = manager.launch(
        ['rightarm-mlops', 'benchmark-checkpoint', '--gpus', '1'],
        'benchmark',
        owner_attempt_id='attempt-1',
    )

    request = client.containers.run_requests[0]
    job_id = launched['job_id']
    assert request['volumes'][str(host_root / job_id / 'output')] == {
        'bind': '/mlops-output',
        'mode': 'rw',
    }
    owner_key = manager.managed_input_path(
        'attempt-1',
        ArtifactRef(
            id='dataset',
            kind='mmaction-dataset',
            uri=(tmp_path / 'dataset').as_uri(),
            sha256='b' * 64,
            size_bytes=1,
        ),
    ).parent.name
    assert request['volumes'][str(host_root / 'inputs' / owner_key)] == {
        'bind': '/mlops-inputs',
        'mode': 'ro',
    }
    assert request['volumes'][str(host_workspace)]['bind'] == '/workspace'
    assert str(source_link) in request['volumes']
    assert str(source_target) not in request['volumes']
    assert request['environment']['RIGHTARM_MLOPS_EXECUTOR_IMAGE'] == image
    assert request['shm_size'] == '16g'


def test_hybrid_materialized_paths_follow_the_task_executor(tmp_path: Path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    manager = HybridJobManager(
        tmp_path / 'jobs',
        workspace,
        docker_job_kinds={'benchmark-checkpoint', 'train-eval-trial'},
        image_catalog={
            'benchmark-checkpoint': 'rightarm@sha256:' + 'a' * 64,
            'train-eval-trial': 'rightarm@sha256:' + 'a' * 64,
        },
        python_executable=Path(sys.executable),
        gpu_counts={'benchmark-checkpoint': 1, 'train-eval-trial': 1},
        client=_DockerClient(),
    )
    artifact = ArtifactRef(
        id='dataset',
        kind='mmaction-dataset',
        uri=(tmp_path / 'dataset').as_uri(),
        sha256='b' * 64,
        size_bytes=1,
    )
    materialized = manager.managed_input_path('attempt-1', artifact)

    docker_path = manager.execution_input_path_for_task(
        'improve-model',
        'attempt-1',
        artifact,
        materialized,
    )
    local_path = manager.execution_input_path_for_task(
        'review-dataset-distribution',
        'attempt-1',
        artifact,
        materialized,
    )

    assert docker_path == str(Path('/mlops-inputs') / materialized.name)
    assert local_path == str(materialized)
