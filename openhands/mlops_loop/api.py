"""REST boundary for the outer loop and task-local Agent tools."""

from __future__ import annotations

import secrets
import stat
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from openhands.app_server.mlops_task.mcp_server import (
    LocalJobManager,
    create_gateway_mcp_server,
)
from openhands.mlops_loop.artifact_store import LocalArtifactStore, S3ArtifactStore
from openhands.mlops_loop.controller import LoopController
from openhands.mlops_loop.dispatcher import (
    InMemoryConversationDispatcher,
    OpenHandsHTTPDispatcher,
)
from openhands.mlops_loop.errors import MLOpsLoopError
from openhands.mlops_loop.ledger import SQLiteLedger
from openhands.mlops_loop.manual_runs import ManualRunMode, build_manual_run
from openhands.mlops_loop.models import ArtifactCandidate, TaskResult, TaskSpec
from openhands.mlops_loop.retry_policy import RetryPolicy
from openhands.mlops_loop.rightarm_validators import (
    build_rightarm_validator_registry,
)


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    loop_id: str = Field(min_length=1)
    watermark: str = Field(min_length=1)
    run_spec: dict[str, Any]
    tasks: list[TaskSpec] = Field(min_length=1)
    initial_artifact_ids: list[str] = Field(default_factory=list)


class SubmitTaskResultRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    run_id: str
    result: TaskResult


class ArtifactCandidateRegisterRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    candidate: ArtifactCandidate


class TaskApprovalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    approved_by: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)


class CostRecordRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_type: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    source_id: str = Field(min_length=1)
    amount_usd: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=256)


class BootstrapArtifactRequest(BaseModel):
    """An operator-owned file copied into the controller import boundary."""

    model_config = ConfigDict(extra='forbid')

    path: str = Field(min_length=1)
    kind: str = Field(pattern=r'^[a-z][a-z0-9-]*$')
    media_type: str | None = None
    sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManualRunCreateRequest(BaseModel):
    """One public rightArm workflow entry point."""

    model_config = ConfigDict(extra='forbid')

    mode: ManualRunMode
    request_id: str = Field(min_length=1, max_length=256)
    source_uri: str | None = None
    initial_artifact_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


def build_controller(
    *,
    database: str | Path,
    artifact_root: str | Path,
    openhands_url: str | None = None,
    openhands_token: str | None = None,
    repository: str | None = None,
    branch: str | None = None,
    task_mcp_url: str | None = None,
    controller_id: str = 'loop-controller-1',
    artifact_backend: str = 'local',
    s3_bucket: str | None = None,
    s3_prefix: str = 'agentic-mlops',
    s3_endpoint_url: str | None = None,
    s3_region_name: str | None = None,
) -> LoopController:
    ledger = SQLiteLedger(database)
    if artifact_backend == 'local':
        if s3_bucket is not None:
            raise ValueError('--s3-bucket requires --artifact-backend=s3')
        artifacts = LocalArtifactStore(artifact_root)
    elif artifact_backend == 's3':
        if s3_bucket is None:
            raise ValueError('--artifact-backend=s3 requires --s3-bucket')
        artifacts = S3ArtifactStore(
            artifact_root,
            bucket=s3_bucket,
            prefix=s3_prefix,
            endpoint_url=s3_endpoint_url,
            region_name=s3_region_name,
        )
    else:
        raise ValueError(f'unsupported artifact backend: {artifact_backend!r}')
    dispatcher = (
        OpenHandsHTTPDispatcher(
            openhands_url,
            token=openhands_token,
            repository=repository,
            branch=branch,
            task_mcp_url=task_mcp_url,
        )
        if openhands_url
        else InMemoryConversationDispatcher()
    )
    validators = build_rightarm_validator_registry(artifacts)
    return LoopController(
        ledger=ledger,
        artifact_store=artifacts,
        dispatcher=dispatcher,
        validators=validators,
        retry_policy=RetryPolicy(
            frozenset(
                {
                    'transient-network',
                    'provider-rate-limit',
                    'gpu-node-lost',
                    'executor-timeout',
                    'cuda-oom-with-valid-proposal',
                }
            )
        ),
        controller_id=controller_id,
        candidate_registry_required=True,
    )


def create_app(
    controller: LoopController,
    *,
    job_manager: LocalJobManager | None = None,
    operator_token: str | None = None,
    operator_identity: str | None = None,
    operator_import_root: str | Path | None = None,
) -> FastAPI:
    if operator_token is not None and not operator_identity:
        raise ValueError('operator identity is required when operator auth is enabled')
    task_mcp_app = None
    if job_manager is not None:
        task_mcp = create_gateway_mcp_server(
            controller=controller, job_manager=job_manager
        )
        task_mcp_app = task_mcp.http_app(path='/', stateless_http=True)
    app = FastAPI(
        title='OpenHands Agentic MLOps Loop',
        version='0.1.0',
        # A mounted FastMCP application does not receive lifespan events from
        # Starlette automatically. Its session manager must be started by the
        # parent application before Agents connect to /mcp/.
        lifespan=task_mcp_app.lifespan if task_mcp_app is not None else None,
    )
    resolved_import_root = (
        Path(operator_import_root).resolve()
        if operator_import_root is not None
        else None
    )

    if task_mcp_app is not None:
        app.mount(
            '/mcp',
            task_mcp_app,
            name='mlops-task-mcp',
        )

    @app.exception_handler(MLOpsLoopError)
    async def domain_error_handler(_: Request, exc: MLOpsLoopError):
        return JSONResponse(status_code=409, content={'detail': str(exc)})

    def require_operator(token: str | None) -> None:
        if operator_token is not None and (
            token is None or not secrets.compare_digest(token, operator_token)
        ):
            raise HTTPException(status_code=401, detail='operator token is invalid')

    @app.post('/api/v1/mlops/runs')
    def create_run(
        request: RunCreateRequest,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        return controller.create_run(
            loop_id=request.loop_id,
            watermark=request.watermark,
            run_spec=request.run_spec,
            tasks=request.tasks,
            initial_artifact_ids=request.initial_artifact_ids,
        )

    @app.get('/api/v1/mlops/runs')
    def list_runs(
        limit: int = 20,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        if limit < 1 or limit > 200:
            raise HTTPException(
                status_code=422, detail='limit must be between 1 and 200'
            )
        rows = [
            row
            for row in controller.ledger.list_runs()
            if row['loop_id'] != 'artifact-bootstrap'
        ]
        result = []
        for row in reversed(rows[-limit:]):
            run = controller.ledger.get_run(row['id'])
            task = controller.ledger.current_task(row['id'])
            result.append(
                {
                    'id': run['id'],
                    'loop_id': run['loop_id'],
                    'watermark': run['watermark'],
                    'state': run['state'],
                    'version': run['version'],
                    'created_at': run['created_at'],
                    'updated_at': run['updated_at'],
                    'mode': run['spec']
                    .get('spec', {})
                    .get('manualTrigger', {})
                    .get('mode'),
                    'current_task': task['spec']['name'] if task is not None else None,
                }
            )
        return {'runs': result}

    @app.post('/api/v1/mlops/artifacts/bootstrap')
    def bootstrap_artifact(
        request: BootstrapArtifactRequest,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        if resolved_import_root is None:
            raise HTTPException(
                status_code=503,
                detail='operator artifact import is not configured',
            )
        unresolved = Path(request.path)
        try:
            if unresolved.is_symlink():
                raise HTTPException(
                    status_code=422, detail='artifact import cannot be a symlink'
                )
            source = unresolved.resolve(strict=True)
            source_stat = source.stat()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail='artifact import file was not found'
            ) from exc
        if not source.is_relative_to(resolved_import_root):
            raise HTTPException(
                status_code=422,
                detail='artifact import is outside the configured import root',
            )
        if not stat.S_ISREG(source_stat.st_mode):
            raise HTTPException(
                status_code=422, detail='artifact import must be a regular file'
            )
        from openhands.mlops_loop.artifact_store import sha256_file

        digest, size_bytes = sha256_file(source)
        if request.sha256 is not None and request.sha256 != digest:
            raise HTTPException(
                status_code=409, detail='artifact import SHA-256 does not match'
            )
        if request.size_bytes is not None and request.size_bytes != size_bytes:
            raise HTTPException(
                status_code=409, detail='artifact import size does not match'
            )
        return controller.bootstrap_artifact(
            path=str(source),
            kind=request.kind,
            media_type=request.media_type,
            metadata={
                **request.metadata,
                'operator_import_sha256': digest,
                'operator_import_size_bytes': size_bytes,
            },
        )

    @app.post('/api/v1/mlops/manual-runs')
    def create_manual_run(
        request: ManualRunCreateRequest,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        try:
            artifacts = controller.ledger.get_artifacts(request.initial_artifact_ids)
            run_spec, tasks = build_manual_run(
                request.mode,
                request_id=request.request_id,
                source_uri=request.source_uri,
                artifacts=artifacts,
                parameters=request.parameters,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run = controller.create_run(
            loop_id='rightarm-manual-v1',
            watermark=f'{request.mode}:{request.request_id}',
            run_spec=run_spec,
            tasks=tasks,
            initial_artifact_ids=request.initial_artifact_ids,
        )
        return {
            **controller.ledger.get_run(run['id']),
            'mode': request.mode,
            'queued': True,
            'tasks': [task.name for task in tasks],
        }

    @app.get('/api/v1/mlops/runs/{run_id}')
    def get_run(
        run_id: str,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        try:
            run = controller.ledger.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='Run not found') from exc
        return {
            **run,
            'tasks': controller.ledger.list_tasks(run_id),
            'attempts': controller.ledger.list_attempts(run_id),
            'artifacts': controller.ledger.list_artifacts(run_id),
        }

    @app.get('/api/v1/mlops/runs/{run_id}/events')
    def get_events(
        run_id: str,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        return controller.ledger.list_events(run_id)

    @app.post('/api/v1/mlops/runs/{run_id}/costs')
    def record_cost(
        run_id: str,
        request: CostRecordRequest,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        return controller.record_cost(
            run_id,
            source_type=request.source_type,
            source_id=request.source_id,
            amount_usd=request.amount_usd,
            metadata=request.metadata,
            idempotency_key=request.idempotency_key,
        )

    @app.get('/api/v1/mlops/runs/{run_id}/lineage')
    def get_lineage(
        run_id: str,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        return controller.ledger.list_lineage(run_id)

    @app.get('/api/v1/mlops/runs/{run_id}/audit')
    def get_audit(
        run_id: str,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        try:
            return controller.ledger.audit_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='Run not found') from exc

    @app.post('/api/v1/mlops/runs/{run_id}/tick')
    def tick_run(
        run_id: str,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        return {'state': controller.tick(run_id)}

    @app.post('/api/v1/mlops/runs/{run_id}/approve-current-task')
    def approve_current_task(
        run_id: str,
        request: TaskApprovalRequest,
        operator: Annotated[str | None, Header(alias='X-MLOps-Operator-Token')] = None,
    ):
        require_operator(operator)
        approved_by = operator_identity or request.approved_by
        if approved_by is None:
            raise HTTPException(status_code=400, detail='approver identity is required')
        controller.approve_current_task(
            run_id,
            approved_by=approved_by,
            reason=request.reason,
        )
        return {'accepted': True}

    @app.get('/api/v1/mlops/attempts/{attempt_id}/context')
    def get_task_context(
        attempt_id: str,
        lease_id: Annotated[str, Header(alias='X-MLOps-Lease-ID')],
    ):
        return controller.get_task_context(attempt_id, lease_id)

    @app.post('/api/v1/mlops/attempts/{attempt_id}/result')
    def submit_task_result(
        attempt_id: str,
        request: SubmitTaskResultRequest,
        lease_id: Annotated[str, Header(alias='X-MLOps-Lease-ID')],
    ):
        if request.result.attempt_id != attempt_id:
            raise HTTPException(status_code=400, detail='attempt id mismatch')
        run_state = controller.submit_result(request.run_id, request.result, lease_id)
        return {'accepted': True, 'run_state': run_state.value}

    @app.post('/api/v1/mlops/attempts/{attempt_id}/candidates')
    def register_artifact_candidate(
        attempt_id: str,
        request: ArtifactCandidateRegisterRequest,
        lease_id: Annotated[str, Header(alias='X-MLOps-Lease-ID')],
    ):
        return controller.ledger.record_artifact_candidate(
            attempt_id=attempt_id,
            lease_id=lease_id,
            candidate=request.candidate,
        )

    return app
