"""Route selected GPU jobs to Docker while keeping other jobs local."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openhands.app_server.mlops_task.mcp_server import (
    TASK_JOB_KINDS,
    DockerJobManager,
    LocalJobManager,
)
from openhands.mlops_loop.models import ArtifactRef


class HybridJobManager(LocalJobManager):
    """Use one shared job root with an operator-owned per-kind executor route.

    Data landing and other controller-local operations keep their existing
    environment and secret mounts. Only explicitly listed jobs enter the
    digest-pinned Docker executor, which is where GPU access is granted.
    """

    def __init__(
        self,
        root: str | Path,
        workspace: str | Path,
        *,
        docker_job_kinds: set[str],
        image_catalog: dict[str, str],
        allow_external_jobs: bool = False,
        external_job_catalog: dict[str, list[str]] | None = None,
        python_executable: str | Path,
        host_root: str | Path | None = None,
        host_workspace: str | Path | None = None,
        read_only_source_roots: list[str | Path] | None = None,
        networked_job_kinds: set[str] | None = None,
        gpu_counts: dict[str, int] | None = None,
        container_python: str = 'python',
        memory_limit: str = '128g',
        nano_cpus: int = 32_000_000_000,
        client: Any | None = None,
    ):
        self.local = LocalJobManager(
            root,
            workspace,
            allow_external_jobs=allow_external_jobs,
            python_executable=python_executable,
            external_job_catalog=external_job_catalog,
        )
        self.docker = DockerJobManager(
            root,
            workspace,
            image_catalog=image_catalog,
            allow_external_jobs=allow_external_jobs,
            external_job_catalog=external_job_catalog,
            host_root=host_root,
            host_workspace=host_workspace,
            read_only_source_roots=read_only_source_roots,
            networked_job_kinds=networked_job_kinds,
            gpu_counts=gpu_counts,
            container_python=container_python,
            memory_limit=memory_limit,
            nano_cpus=nano_cpus,
            client=client,
        )
        invalid = set(docker_job_kinds) - self.local.allowed_job_kinds
        if invalid:
            raise ValueError(
                f'hybrid Docker routes contain unknown job kinds: {sorted(invalid)}'
            )
        missing_images = set(docker_job_kinds) - set(image_catalog)
        if missing_images:
            raise ValueError(
                f'hybrid Docker routes have no pinned image: {sorted(missing_images)}'
            )
        self.docker_job_kinds = frozenset(docker_job_kinds)
        self.root = self.local.root
        self.workspace = self.local.workspace
        self.allowed_job_kinds = self.local.allowed_job_kinds

    def launch(
        self,
        argv: list[str],
        idempotency_key: str,
        *,
        owner_attempt_id: str | None = None,
    ) -> dict[str, Any]:
        manager = self._manager_for_kind(argv[1] if len(argv) > 1 else '')
        return manager.launch(
            argv,
            idempotency_key,
            owner_attempt_id=owner_attempt_id,
        )

    def observe(
        self, job_id: str, *, owner_attempt_id: str | None = None
    ) -> dict[str, Any]:
        return self._manager_for_job(job_id).observe(
            job_id, owner_attempt_id=owner_attempt_id
        )

    def cancel_attempt(self, attempt_id: str) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *self.local.cancel_attempt(attempt_id),
                    *self.docker.cancel_attempt(attempt_id),
                ]
            )
        )

    def running_attempt_ids(self) -> set[str]:
        return self.local.running_attempt_ids() | self.docker.running_attempt_ids()

    def assert_owned_output(self, path: str | Path, attempt_id: str) -> Path:
        source = Path(path).resolve(strict=True)
        job_id = self._job_id_from_output(source)
        return self._manager_for_job(job_id).assert_owned_output(source, attempt_id)

    def assert_declared_output(
        self, path: str | Path, attempt_id: str, kind: str
    ) -> Path:
        source = Path(path).resolve(strict=True)
        job_id = self._job_id_from_output(source)
        return self._manager_for_job(job_id).assert_declared_output(
            source, attempt_id, kind
        )

    def managed_input_path(self, attempt_id: str, artifact: ArtifactRef) -> Path:
        return self.local.managed_input_path(attempt_id, artifact)

    def execution_input_path(
        self, attempt_id: str, artifact: ArtifactRef, materialized: Path
    ) -> str:
        return self.local.execution_input_path(attempt_id, artifact, materialized)

    def execution_input_path_for_task(
        self,
        task_kind: str,
        attempt_id: str,
        artifact: ArtifactRef,
        materialized: Path,
    ) -> str:
        allowed = TASK_JOB_KINDS.get(task_kind, frozenset())
        manager = self.docker if allowed & self.docker_job_kinds else self.local
        return manager.execution_input_path(attempt_id, artifact, materialized)

    def _manager_for_kind(self, kind: str) -> LocalJobManager:
        return self.docker if kind in self.docker_job_kinds else self.local

    def _manager_for_job(self, job_id: str) -> LocalJobManager:
        request = self._read_optional_json(self.root / job_id / 'request.json')
        if request is None:
            raise KeyError(job_id)
        return self.docker if request.get('executor') == 'docker' else self.local

    def _job_id_from_output(self, source: Path) -> str:
        if not source.is_relative_to(self.root):
            raise PermissionError('only managed job output files may be staged')
        relative = source.relative_to(self.root)
        if len(relative.parts) < 3 or relative.parts[1] != 'output':
            raise PermissionError('only managed job output files may be staged')
        return relative.parts[0]
