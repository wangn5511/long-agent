"""Runnable entry points for the staging Loop Controller service."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
from pathlib import Path

import uvicorn

from openhands.app_server.mlops_task.hybrid_job_manager import HybridJobManager
from openhands.app_server.mlops_task.mcp_server import (
    DockerJobManager,
    LocalJobManager,
)
from openhands.mlops_loop.api import build_controller, create_app
from openhands.mlops_loop.controller import TERMINAL_RUN_STATES, LoopController
from openhands.mlops_loop.errors import InvariantViolation, LeaseConflict
from openhands.mlops_loop.gesture_loop import GestureLoopService
from openhands.mlops_loop.source_watcher import CommandSourceWatcher, LocalBatchWatcher
from openhands.mlops_loop.spec import load_yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m openhands.mlops_loop')
    subparsers = parser.add_subparsers(dest='command', required=True)

    serve = subparsers.add_parser('serve')
    _storage_arguments(serve)
    serve.add_argument('--openhands-url')
    serve.add_argument('--openhands-token')
    serve.add_argument('--openhands-token-file', type=Path)
    serve.add_argument('--repository')
    serve.add_argument('--branch')
    serve.add_argument('--task-mcp-url')
    serve.add_argument('--job-workspace', type=Path)
    serve.add_argument('--job-root', type=Path)
    serve.add_argument('--job-python', type=Path)
    serve.add_argument(
        '--executor', choices=('local', 'docker', 'hybrid'), default='local'
    )
    serve.add_argument('--allow-external-jobs', action='store_true')
    serve.add_argument('--external-job-catalog', type=Path)
    serve.add_argument('--docker-image-catalog', type=Path)
    serve.add_argument(
        '--docker-job-kind',
        action='append',
        default=[],
        help='with --executor=hybrid, route this job kind to Docker',
    )
    serve.add_argument('--docker-container-python', default='python')
    serve.add_argument('--docker-host-job-root', type=Path)
    serve.add_argument('--docker-host-workspace', type=Path)
    serve.add_argument('--docker-source-root', type=Path, action='append', default=[])
    serve.add_argument('--docker-network-job-kind', action='append', default=[])
    serve.add_argument(
        '--docker-gpu-count',
        action='append',
        default=[],
        metavar='KIND=COUNT',
    )
    serve.add_argument('--docker-memory-limit', default='128g')
    serve.add_argument('--docker-nano-cpus', type=int, default=32_000_000_000)
    serve.add_argument('--incoming', type=Path)
    serve.add_argument(
        '--watch-command-json',
        help='operator-owned JSON argv that discovers sealed cloud batches',
    )
    serve.add_argument('--loop-spec', type=Path)
    serve.add_argument('--run-template', type=Path)
    serve.add_argument('--set', action='append', default=[], dest='replacements')
    serve.add_argument('--poll-interval-seconds', type=float, default=5.0)
    serve.add_argument('--reconcile-interval-seconds', type=float, default=5.0)
    serve.add_argument('--controller-id')
    serve.add_argument('--operator-token-file', type=Path)
    serve.add_argument('--operator-identity')
    serve.add_argument('--operator-import-root', type=Path)
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8765)

    poll = subparsers.add_parser('poll-local-once')
    _storage_arguments(poll)
    poll.add_argument('--incoming', required=True, type=Path)
    poll.add_argument('--loop-spec', required=True, type=Path)
    poll.add_argument('--run-template', required=True, type=Path)
    poll.add_argument('--set', action='append', default=[], dest='replacements')
    poll.add_argument('--openhands-url')
    poll.add_argument('--openhands-token')
    poll.add_argument('--openhands-token-file', type=Path)
    poll.add_argument('--repository')
    poll.add_argument('--branch')
    poll.add_argument('--task-mcp-url')
    poll.add_argument('--controller-id')

    bootstrap = subparsers.add_parser('bootstrap-artifact')
    _storage_arguments(bootstrap)
    bootstrap.add_argument('--path', required=True, type=Path)
    bootstrap.add_argument('--kind', required=True)
    bootstrap.add_argument('--media-type')
    bootstrap.add_argument('--metadata-json', default='{}')
    return parser


def _storage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--artifact-backend', choices=('local', 's3'), default='local')
    parser.add_argument('--s3-bucket')
    parser.add_argument('--s3-prefix', default='agentic-mlops')
    parser.add_argument('--s3-endpoint-url')
    parser.add_argument('--s3-region-name')


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    openhands_token = _resolve_openhands_token(
        getattr(args, 'openhands_token', None),
        getattr(args, 'openhands_token_file', None),
    )
    controller = build_controller(
        database=args.database,
        artifact_root=args.artifact_root,
        openhands_url=getattr(args, 'openhands_url', None),
        openhands_token=openhands_token,
        repository=getattr(args, 'repository', None),
        branch=getattr(args, 'branch', None),
        task_mcp_url=getattr(args, 'task_mcp_url', None),
        controller_id=getattr(args, 'controller_id', None)
        or f'{socket.gethostname()}:{os.getpid()}',
        artifact_backend=args.artifact_backend,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint_url=args.s3_endpoint_url,
        s3_region_name=args.s3_region_name,
    )
    if args.command == 'bootstrap-artifact':
        metadata = json.loads(args.metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError('--metadata-json must be a JSON object')
        artifact = controller.bootstrap_artifact(
            path=str(args.path),
            kind=args.kind,
            media_type=args.media_type,
            metadata=metadata,
        )
        print(json.dumps(artifact, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    if args.command == 'serve':
        if (args.job_workspace is None) != (args.job_root is None):
            raise ValueError('--job-workspace and --job-root must be provided together')
        job_manager: LocalJobManager | None
        if args.job_workspace is None:
            if args.executor != 'local':
                raise ValueError('Docker executor requires job workspace and root')
            job_manager = None
        elif args.executor in {'docker', 'hybrid'}:
            if (args.docker_host_job_root is None) != (
                args.docker_host_workspace is None
            ):
                raise ValueError(
                    '--docker-host-job-root and --docker-host-workspace '
                    'must be provided together'
                )
            image_catalog = _load_image_catalog(args.docker_image_catalog)
            if image_catalog is None:
                raise ValueError(
                    'Docker and hybrid executors require --docker-image-catalog'
                )
            common = {
                'image_catalog': image_catalog,
                'allow_external_jobs': args.allow_external_jobs,
                'external_job_catalog': _load_job_catalog(args.external_job_catalog),
                'host_root': args.docker_host_job_root,
                'host_workspace': args.docker_host_workspace,
                'read_only_source_roots': args.docker_source_root,
                'networked_job_kinds': set(args.docker_network_job_kind),
                'gpu_counts': _kind_int_map(args.docker_gpu_count),
                'container_python': args.docker_container_python,
                'memory_limit': args.docker_memory_limit,
                'nano_cpus': args.docker_nano_cpus,
            }
            if args.executor == 'docker':
                if args.docker_job_kind:
                    raise ValueError(
                        '--docker-job-kind is only valid with --executor=hybrid'
                    )
                job_manager = DockerJobManager(
                    args.job_root,
                    args.job_workspace,
                    **common,
                )
            else:
                docker_job_kinds = set(args.docker_job_kind)
                if not docker_job_kinds:
                    raise ValueError(
                        'Hybrid executor requires at least one --docker-job-kind'
                    )
                job_manager = HybridJobManager(
                    args.job_root,
                    args.job_workspace,
                    docker_job_kinds=docker_job_kinds,
                    python_executable=(
                        args.job_python
                        if args.job_python is not None
                        else sys.executable
                    ),
                    **common,
                )
        else:
            if args.docker_image_catalog is not None:
                raise ValueError(
                    '--docker-image-catalog requires --executor=docker or hybrid'
                )
            if (
                args.docker_host_job_root is not None
                or args.docker_host_workspace is not None
            ):
                raise ValueError(
                    '--docker-host-job-root/--docker-host-workspace require '
                    '--executor=docker or hybrid'
                )
            if args.docker_job_kind:
                raise ValueError('--docker-job-kind requires --executor=hybrid')
            job_manager = LocalJobManager(
                args.job_root,
                args.job_workspace,
                allow_external_jobs=args.allow_external_jobs,
                python_executable=args.job_python
                if args.job_python is not None
                else sys.executable,
                external_job_catalog=_load_job_catalog(args.external_job_catalog),
            )
        if args.incoming is not None and args.watch_command_json is not None:
            raise ValueError(
                '--incoming and --watch-command-json are mutually exclusive'
            )
        has_watcher = args.incoming is not None or args.watch_command_json is not None
        source_values = (args.loop_spec, args.run_template)
        if has_watcher != all(value is not None for value in source_values):
            raise ValueError(
                'a source watcher, --loop-spec and --run-template '
                'must be provided together'
            )
        operator_token = _load_operator_token(args.operator_token_file)
        if operator_token is not None and not args.operator_identity:
            raise ValueError(
                '--operator-identity is required with --operator-token-file'
            )
        if (
            args.host not in {'127.0.0.1', 'localhost', '::1'}
            and operator_token is None
        ):
            raise ValueError('non-loopback API binding requires --operator-token-file')
        stop = threading.Event()
        workers: list[threading.Thread] = []
        if args.reconcile_interval_seconds <= 0:
            raise ValueError('--reconcile-interval-seconds must be positive')
        reconciler = threading.Thread(
            target=_run_reconciler,
            args=(
                controller,
                stop,
                args.reconcile_interval_seconds,
                job_manager,
            ),
            name='rightarm-run-reconciler',
            daemon=True,
        )
        reconciler.start()
        workers.append(reconciler)
        if has_watcher:
            if args.poll_interval_seconds <= 0:
                raise ValueError('--poll-interval-seconds must be positive')
            assert args.loop_spec is not None
            assert args.run_template is not None
            watcher = (
                LocalBatchWatcher(args.incoming)
                if args.incoming is not None
                else CommandSourceWatcher(_command_argv(args.watch_command_json))
            )
            service = GestureLoopService(
                controller=controller,
                watcher=watcher,
                loop_spec=load_yaml(args.loop_spec),
                run_template=load_yaml(args.run_template),
                replacements=_replacements(args.replacements),
            )
            scheduler = threading.Thread(
                target=_run_scheduler,
                args=(service, stop, args.poll_interval_seconds),
                name='rightarm-gesture-loop',
                daemon=True,
            )
            scheduler.start()
            workers.append(scheduler)
        try:
            uvicorn.run(
                create_app(
                    controller,
                    job_manager=job_manager,
                    operator_token=operator_token,
                    operator_identity=args.operator_identity,
                    operator_import_root=args.operator_import_root,
                ),
                host=args.host,
                port=args.port,
            )
        finally:
            stop.set()
            for worker in workers:
                worker.join(timeout=10)
        return 0
    replacements = _replacements(args.replacements)
    service = GestureLoopService(
        controller=controller,
        watcher=LocalBatchWatcher(args.incoming),
        loop_spec=load_yaml(args.loop_spec),
        run_template=load_yaml(args.run_template),
        replacements=replacements,
    )
    run = service.poll_once()
    if run is not None:
        controller.tick(run['id'])
        run = controller.ledger.get_run(run['id'])
    print(json.dumps(run, ensure_ascii=False, sort_keys=True, default=str))
    return 0


def _replacements(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'--set requires NAME=VALUE, got {value!r}')
        name, replacement = value.split('=', 1)
        if not name or not replacement:
            raise ValueError(f'--set requires NAME=VALUE, got {value!r}')
        result[name] = replacement
    return result


def _command_argv(value: str | None) -> list[str]:
    if value is None:
        raise ValueError('--watch-command-json is required')
    parsed = json.loads(value)
    if (
        not isinstance(parsed, list)
        or not parsed
        or not all(isinstance(item, str) and item for item in parsed)
    ):
        raise ValueError('--watch-command-json must be a non-empty JSON string array')
    return parsed


def _load_job_catalog(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    value = json.loads(path.resolve(strict=True).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and isinstance(prefix, list)
        and prefix
        and all(isinstance(item, str) and item for item in prefix)
        for key, prefix in value.items()
    ):
        raise ValueError('external job catalog must map job kinds to argv prefixes')
    return value


def _load_image_catalog(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    value = json.loads(path.resolve(strict=True).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(image, str) and image
        for key, image in value.items()
    ):
        raise ValueError('Docker image catalog must map job kinds to image digests')
    return value


def _load_operator_token(path: Path | None) -> str | None:
    if path is None:
        return None
    token = path.resolve(strict=True).read_text(encoding='utf-8').strip()
    if len(token) < 32:
        raise ValueError('operator token must contain at least 32 characters')
    return token


def _kind_int_map(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        try:
            kind, raw_count = value.split('=', 1)
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                f'Docker GPU count requires KIND=COUNT, got {value!r}'
            ) from exc
        if not kind or count < 1 or kind in result:
            raise ValueError(f'invalid Docker GPU count: {value!r}')
        result[kind] = count
    return result


def _resolve_openhands_token(value: str | None, path: Path | None) -> str | None:
    if value is not None and path is not None:
        raise ValueError(
            '--openhands-token and --openhands-token-file are mutually exclusive'
        )
    if path is None:
        return value
    token = path.resolve(strict=True).read_text(encoding='utf-8').strip()
    if not token:
        raise ValueError('OpenHands token file is empty')
    return token


def _run_scheduler(
    service: GestureLoopService,
    stop: threading.Event,
    poll_interval_seconds: float,
) -> None:
    logger = logging.getLogger(__name__)
    lease_id = None
    lease_ttl = max(poll_interval_seconds * 3, 15.0)
    try:
        while not stop.is_set():
            try:
                if lease_id is None:
                    lease_id = service.controller.ledger.acquire_scheduler_lease(
                        loop_id=service.loop_id,
                        holder_id=service.controller.controller_id,
                        ttl_seconds=lease_ttl,
                    )
                else:
                    service.controller.ledger.heartbeat_scheduler_lease(
                        loop_id=service.loop_id,
                        lease_id=lease_id,
                        ttl_seconds=lease_ttl,
                    )
                service.poll_once()
            except LeaseConflict:
                lease_id = None
                logger.info('scheduler standby; another controller holds the loop')
            except Exception:  # noqa: BLE001 - keep the reconciler alive
                logger.exception('gesture loop reconciliation failed')
            stop.wait(poll_interval_seconds)
    finally:
        if lease_id is not None:
            service.controller.ledger.release_scheduler_lease(
                loop_id=service.loop_id, lease_id=lease_id
            )


def _cancel_attempt_jobs(
    controller: LoopController,
    job_manager: LocalJobManager,
    attempt_id: str,
    logger: logging.Logger,
) -> None:
    for job_id in job_manager.cancel_attempt(attempt_id):
        observation = job_manager.observe(job_id, owner_attempt_id=attempt_id)
        try:
            controller.ledger.record_job_observation(
                attempt_id=attempt_id,
                job_id=job_id,
                observation=observation,
            )
        except InvariantViolation:
            logger.warning(
                'cancelled orphan job without a persisted request: %s', job_id
            )


def _reconcile_inactive_jobs(
    controller: LoopController,
    job_manager: LocalJobManager,
    logger: logging.Logger,
) -> None:
    for attempt_id in job_manager.running_attempt_ids():
        try:
            attempt = controller.ledger.get_attempt(attempt_id)
        except KeyError:
            _cancel_attempt_jobs(controller, job_manager, attempt_id, logger)
            continue
        if attempt['status'] != 'RUNNING':
            _cancel_attempt_jobs(controller, job_manager, attempt_id, logger)


def _run_reconciler(
    controller: LoopController,
    stop: threading.Event,
    interval_seconds: float,
    job_manager: LocalJobManager | None,
) -> None:
    """Advance manual and watched Runs independently of source polling."""

    logger = logging.getLogger(__name__)
    while not stop.is_set():
        try:
            if job_manager is not None:
                _reconcile_inactive_jobs(controller, job_manager, logger)
            for row in controller.ledger.list_runs():
                if row['loop_id'] == 'artifact-bootstrap':
                    continue
                if row['state'] in {state.value for state in TERMINAL_RUN_STATES}:
                    continue
                task = controller.ledger.current_task(row['id'])
                expired_attempt = (
                    controller.ledger.latest_attempt(task['id'])
                    if task is not None
                    else None
                )
                recovered = controller.reconcile_expired_attempt(row['id'])
                if (
                    recovered
                    and job_manager is not None
                    and expired_attempt is not None
                ):
                    _cancel_attempt_jobs(
                        controller,
                        job_manager,
                        expired_attempt['id'],
                        logger,
                    )
                controller.tick(row['id'])
        except Exception:  # noqa: BLE001 - keep the reconciler alive
            logger.exception('manual Run reconciliation failed')
        stop.wait(interval_seconds)
