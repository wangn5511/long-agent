"""Fenced HTTP client exposed to task-local MCP tools."""

from __future__ import annotations

from pathlib import Path

import httpx

from openhands.mlops_loop.artifact_store import LocalArtifactStore
from openhands.mlops_loop.models import ArtifactCandidate, TaskContext, TaskResult


class MLOpsTaskClient:
    def __init__(
        self,
        *,
        base_url: str,
        run_id: str,
        attempt_id: str,
        lease_id: str,
        artifact_root: str | Path,
        timeout: float = 30,
    ):
        self.base_url = base_url.rstrip('/')
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.lease_id = lease_id
        self.timeout = timeout
        self.artifacts = LocalArtifactStore(artifact_root)

    @property
    def headers(self) -> dict[str, str]:
        return {'X-MLOps-Lease-ID': self.lease_id}

    def get_task_context(self) -> TaskContext:
        response = httpx.get(
            f'{self.base_url}/api/v1/mlops/attempts/{self.attempt_id}/context',
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return TaskContext.model_validate(response.json())

    def stage_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        media_type: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ArtifactCandidate:
        candidate = self.artifacts.stage(
            path,
            kind=kind,
            media_type=media_type,
            metadata=metadata,
            candidate_key=f'{self.attempt_id}:{kind}:{Path(path).resolve()}',
        )
        response = httpx.post(
            f'{self.base_url}/api/v1/mlops/attempts/{self.attempt_id}/candidates',
            headers=self.headers,
            json={'candidate': candidate.model_dump(mode='json')},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return candidate

    def submit_task_result(self, result: TaskResult) -> dict[str, object]:
        if result.attempt_id != self.attempt_id:
            raise ValueError('result belongs to a different task attempt')
        response = httpx.post(
            f'{self.base_url}/api/v1/mlops/attempts/{self.attempt_id}/result',
            headers=self.headers,
            json={
                'run_id': self.run_id,
                'result': result.model_dump(mode='json'),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
