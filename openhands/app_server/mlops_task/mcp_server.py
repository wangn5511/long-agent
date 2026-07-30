"""Task-local MCP server for a single fenced MLOps attempt."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import docker
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from openhands.app_server.mlops_task.client import MLOpsTaskClient
from openhands.mlops_loop.models import ArtifactRef, LineageProposal, TaskResult

TASK_JOB_KINDS: dict[str, frozenset[str]] = {
    'inspect-data': frozenset({'materialize-cos', 'inspect-batch', 'land-batch'}),
    'build-dataset': frozenset({'build-dataset'}),
    'review-dataset-distribution': frozenset(
        {'profile-dataset-distribution', 'merge-dataset'}
    ),
    'improve-model': frozenset(
        {
            'benchmark-checkpoint',
            'train-eval-trial',
            'record-model-improvement',
        }
    ),
    'train': frozenset({'train'}),
    'evaluate': frozenset({'run-evaluation', 'evaluate'}),
    'package': frozenset(
        {'export-onnx', 'build-calibration', 'export-rknn', 'verify-conversion'}
    ),
    'verify-board': frozenset({'verify-board', 'build-release'}),
    'deploy': frozenset({'deploy'}),
    'monitor': frozenset({'observe-deployment'}),
    'rollback': frozenset({'rollback'}),
}

_SENSITIVE_ERROR_VALUE = re.compile(
    r'(?i)\b(?:sk-[a-z0-9_-]{8,}|bearer\s+\S+|'
    r'(?:api[_-]?key|token|authorization)\s*[:=]\s*\S+)'
)


def _controlled_gateway_tool_error(exc: ValueError | PermissionError) -> ToolError:
    """Expose an actionable bounded message without disabling global masking."""

    message = str(exc).strip() or type(exc).__name__
    message = _SENSITIVE_ERROR_VALUE.sub('[REDACTED]', message)
    return ToolError(message[:500])


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise ValueError(f'invalid JSON constant: {value}')


def _fresh_poll_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep legitimate long-running job polls distinct to the Agent runtime."""

    if observation.get('status') != 'running':
        return observation
    return {
        **observation,
        'poll_observation_id': f'poll:{uuid4().hex}',
    }


def _observe_job_until_terminal(
    job_manager: Any,
    job_id: str,
    *,
    owner_attempt_id: str,
    max_wait_seconds: float = 240.0,
    max_poll_interval_seconds: float = 15.0,
) -> dict[str, Any]:
    """Bound one Agent iteration around a long-running executor job."""

    deadline = time.monotonic() + max_wait_seconds
    delay = min(1.0, max_poll_interval_seconds)
    observation = job_manager.observe(job_id, owner_attempt_id=owner_attempt_id)
    while observation.get('status') == 'running':
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        observation = job_manager.observe(job_id, owner_attempt_id=owner_attempt_id)
        delay = min(max(delay * 2, 0.001), max_poll_interval_seconds)
    return observation


class LocalJobManager:
    """Run allowlisted rightarm jobs without a shell and observe them by id."""

    MAX_STRUCTURED_OUTPUT_DECLARATIONS = 32
    MAX_STRUCTURED_OUTPUTS = 16
    MAX_STRUCTURED_OUTPUT_BYTES = 256 * 1024
    MAX_STRUCTURED_OUTPUT_TOTAL_BYTES = 1024 * 1024
    SAFE_STRUCTURED_OUTPUT_KIND = re.compile(r'[a-z0-9][a-z0-9._-]{0,127}\Z')

    SAFE_JOB_KINDS = frozenset(
        {
            'inspect-batch',
            'materialize-cos',
            'land-batch',
            'build-dataset',
            'profile-dataset-distribution',
            'merge-dataset',
            'benchmark-checkpoint',
            'train-eval-trial',
            'record-model-improvement',
            'train',
            'evaluate',
            'build-release',
        }
    )
    EXTERNAL_JOB_KINDS = frozenset(
        {
            'run-evaluation',
            'export-onnx',
            'build-calibration',
            'export-rknn',
            'verify-conversion',
            'verify-board',
            'deploy',
            'observe-deployment',
            'rollback',
        }
    )

    def __init__(
        self,
        root: str | Path,
        workspace: str | Path,
        *,
        allow_external_jobs: bool = False,
        python_executable: str | Path = sys.executable,
        external_job_catalog: dict[str, list[str]] | None = None,
    ):
        self.root = Path(root).resolve()
        self.workspace = Path(workspace).resolve(strict=True)
        self.python_executable = str(Path(python_executable).resolve(strict=True))
        self.external_job_catalog = {
            key: list(value) for key, value in (external_job_catalog or {}).items()
        }
        invalid_catalog = set(self.external_job_catalog) - self.EXTERNAL_JOB_KINDS
        if invalid_catalog:
            raise ValueError(
                f'external job catalog has unknown kinds: {sorted(invalid_catalog)}'
            )
        if any(
            not prefix or not all(isinstance(item, str) and item for item in prefix)
            for prefix in self.external_job_catalog.values()
        ):
            raise ValueError('external job catalog prefixes must be string arrays')
        if allow_external_jobs and not self.external_job_catalog:
            raise ValueError(
                'external jobs require an operator-owned external job catalog'
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_job_kinds = self.SAFE_JOB_KINDS | (
            self.EXTERNAL_JOB_KINDS if allow_external_jobs else frozenset()
        )
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        for kind, prefix in self.external_job_catalog.items():
            self._validate_external_command(
                [
                    'rightarm-mlops',
                    kind,
                    '--command-json',
                    json.dumps(prefix),
                ]
            )

    def launch(
        self,
        argv: list[str],
        idempotency_key: str,
        *,
        owner_attempt_id: str | None = None,
    ) -> dict[str, Any]:
        if len(argv) < 2 or argv[0] != 'rightarm-mlops':
            raise ValueError('job argv must start with the logical rightarm-mlops name')
        if argv[1] not in self.allowed_job_kinds:
            raise ValueError(f'rightarm job kind is not allowed: {argv[1]!r}')
        if argv[1] in self.EXTERNAL_JOB_KINDS:
            self._validate_external_command(argv)
        with self._lock:
            job_id = self.job_id_for(owner_attempt_id, idempotency_key)
            job_dir = self.root / job_id
            output_dir = job_dir / 'output'
            normalized_argv = self._with_managed_output_dir(argv, output_dir)
            command = [
                self.python_executable,
                '-m',
                'rightarm_mlops',
                *normalized_argv[1:],
            ]
            request = {
                'protocol': 'mlops.openhands.dev/job-request/v1',
                'job_id': job_id,
                'owner_attempt_id': owner_attempt_id,
                'idempotency_key': idempotency_key,
                'argv': command,
            }
            request_path = job_dir / 'request.json'
            if job_dir.exists():
                existing = self._read_optional_json(request_path)
                if existing != request:
                    raise ValueError(
                        'job idempotency key was reused with a different request'
                    )
                return self.observe(job_id, owner_attempt_id=owner_attempt_id)
            job_dir.mkdir()
            self._write_json_exclusive(request_path, request)
            log_path = job_dir / 'job.log'
            log_stream = log_path.open('wb')
            process = subprocess.Popen(  # noqa: S603 - argv is allowlisted, no shell
                command,
                cwd=self.workspace,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            self._write_json_exclusive(
                job_dir / 'process.json',
                {
                    'pid': process.pid,
                    'start_time_ticks': self._process_start_time_ticks(process.pid),
                },
            )
            self._jobs[job_id] = {
                'process': process,
                'log_stream': log_stream,
                'log_path': log_path,
                'argv': command,
                'owner_attempt_id': owner_attempt_id,
            }
            return {'job_id': job_id, 'status': 'running', 'argv': command}

    @staticmethod
    def job_id_for(owner_attempt_id: str | None, idempotency_key: str) -> str:
        identity = json.dumps(
            [owner_attempt_id, idempotency_key],
            ensure_ascii=False,
            separators=(',', ':'),
        )
        return f'job_{hashlib.sha256(identity.encode()).hexdigest()[:24]}'

    def _validate_external_command(self, argv: list[str]) -> None:
        """Require external wrappers to invoke a checked-in workspace script."""

        try:
            index = argv.index('--command-json')
            raw_command = argv[index + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError('external job requires --command-json') from exc
        command = json.loads(raw_command)
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError('--command-json must be a non-empty JSON string array')
        expected_prefix = self.external_job_catalog.get(argv[1])
        if (
            expected_prefix is None
            or command[: len(expected_prefix)] != expected_prefix
        ):
            raise ValueError(
                f'external job {argv[1]!r} does not match its catalog argv prefix'
            )
        executable = Path(command[0])
        executable_name = executable.name
        if executable_name in {
            'python',
            'python3',
            Path(sys.executable).name,
            Path(self.python_executable).name,
            'bash',
        }:
            if len(command) < 2 or command[1].startswith('-'):
                raise ValueError(
                    'interpreter external jobs require a checked-in script path'
                )
            script = (self.workspace / command[1]).resolve()
            if not script.is_file() or not script.is_relative_to(self.workspace):
                raise ValueError('external job script must be inside the workspace')
            return
        resolved = (
            executable.resolve()
            if executable.is_absolute()
            else (self.workspace / executable).resolve()
        )
        if not resolved.is_file() or not resolved.is_relative_to(self.workspace):
            raise ValueError('external job executable must be inside the workspace')

    @staticmethod
    def _with_managed_output_dir(argv: list[str], output_dir: Path) -> list[str]:
        normalized = []
        index = 0
        while index < len(argv):
            value = argv[index]
            if value == '--output-dir':
                if index + 1 >= len(argv):
                    raise ValueError('--output-dir requires a value')
                index += 2
                continue
            if value.startswith('--output-dir='):
                index += 1
                continue
            normalized.append(value)
            index += 1
        normalized.extend(['--output-dir', str(output_dir)])
        return normalized

    def observe(
        self, job_id: str, *, owner_attempt_id: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job_dir = self.root / job_id
                request = self._read_optional_json(job_dir / 'request.json')
                if request is None:
                    raise KeyError(job_id)
                job = {
                    'process': None,
                    'log_stream': None,
                    'log_path': job_dir / 'job.log',
                    'argv': request['argv'],
                    'owner_attempt_id': request['owner_attempt_id'],
                }
            if job['owner_attempt_id'] != owner_attempt_id:
                raise PermissionError('job belongs to a different task attempt')
            process: subprocess.Popen | None = job['process']
            return_code = process.poll() if process is not None else None
            log_stream = job['log_stream']
            if (
                return_code is not None
                and log_stream is not None
                and not log_stream.closed
            ):
                log_stream.close()
            log_path: Path = job['log_path']
            tail = ''
            if log_path.exists():
                data = log_path.read_bytes()[-16_384:]
                tail = data.decode('utf-8', errors='replace')
                log_sha256, log_size_bytes = self._hash_file(log_path)
            else:
                log_sha256, log_size_bytes = None, None
            result_path = self._find_job_manifest(job['argv'], 'result.json')
            failure_path = self._find_job_manifest(job['argv'], 'failure.json')
            result = self._read_optional_json(result_path)
            failure = self._read_optional_json(failure_path)
            cancellation = self._read_optional_json(
                self.root / job_id / 'cancellation.json'
            )
            if result is not None:
                status = 'succeeded'
                return_code = 0
            elif failure is not None:
                status = 'failed'
                return_code = 1
            elif cancellation is not None:
                status = 'cancelled'
            elif process is not None and return_code is None:
                status = 'running'
            elif process is None and self._persisted_process_is_running(job_id):
                status = 'running'
            else:
                status = 'lost'
            return {
                'job_id': job_id,
                'status': status,
                'return_code': return_code,
                'log_path': str(log_path),
                'log_tail': tail,
                'log_sha256': log_sha256,
                'log_size_bytes': log_size_bytes,
                'result': result,
                'failure': failure,
                'cancellation': cancellation,
            }

    def cancel_attempt(self, attempt_id: str) -> list[str]:
        """Terminate only jobs fenced to an expired task attempt."""

        cancelled = []
        with self._lock:
            for job_dir in sorted(self.root.glob('job_*')):
                request = self._read_optional_json(job_dir / 'request.json')
                if (
                    request is None
                    or not self._owns_request(request)
                    or request.get('owner_attempt_id') != attempt_id
                ):
                    continue
                job_id = request['job_id']
                observation = self.observe(job_id, owner_attempt_id=attempt_id)
                if observation['status'] != 'running':
                    continue
                job = self._jobs.get(job_id)
                if job is not None and job['process'] is not None:
                    job['process'].terminate()
                else:
                    process = self._read_optional_json(job_dir / 'process.json')
                    if process is not None and self._persisted_process_is_running(
                        job_id
                    ):
                        os.kill(process['pid'], signal.SIGTERM)
                cancellation_path = job_dir / 'cancellation.json'
                if not cancellation_path.exists():
                    self._write_json_exclusive(
                        cancellation_path,
                        {
                            'attempt_id': attempt_id,
                            'reason': 'attempt lease expired',
                        },
                    )
                cancelled.append(job_id)
        return cancelled

    def running_attempt_ids(self) -> set[str]:
        """Return owners of executor processes that still have no terminal manifest."""

        owners: set[str] = set()
        with self._lock:
            for job_dir in sorted(self.root.glob('job_*')):
                request = self._read_optional_json(job_dir / 'request.json')
                if (
                    request is None
                    or not self._owns_request(request)
                    or not isinstance(request.get('owner_attempt_id'), str)
                ):
                    continue
                owner = request['owner_attempt_id']
                if (
                    self.observe(job_dir.name, owner_attempt_id=owner)['status']
                    == 'running'
                ):
                    owners.add(owner)
        return owners

    def assert_owned_output(self, path: str | Path, attempt_id: str) -> Path:
        """Resolve a file only inside this attempt's managed job output."""

        source = Path(path).resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(self.root):
            raise PermissionError('only managed job output files may be staged')
        relative = source.relative_to(self.root)
        if len(relative.parts) < 3 or relative.parts[1] != 'output':
            raise PermissionError('only managed job output files may be staged')
        job_id = relative.parts[0]
        request = self._read_optional_json(self.root / job_id / 'request.json')
        if request is None or request.get('owner_attempt_id') != attempt_id:
            raise PermissionError('job output belongs to a different task attempt')
        return source

    def assert_declared_output(
        self, path: str | Path, attempt_id: str, kind: str
    ) -> Path:
        """Require a successful deterministic job to declare this kind and hash."""

        source = self.assert_owned_output(path, attempt_id)
        job_id = source.relative_to(self.root).parts[0]
        job_dir = self.root / job_id
        result = self._read_optional_json(job_dir / 'output' / 'result.json')
        if (
            result is None
            or result.get('protocol') != 'rightarm.mlops.job-result/v1'
            or result.get('status') != 'succeeded'
            or not isinstance(result.get('outputs'), list)
        ):
            raise PermissionError(
                'artifact staging requires a successful structured job result'
            )
        matches = []
        for item in result['outputs']:
            if not isinstance(item, dict) or item.get('kind') != kind:
                continue
            declared = item.get('path')
            if not isinstance(declared, str):
                continue
            if self._declared_output_host_path(job_dir, declared) == source:
                matches.append(item)
        if len(matches) != 1:
            raise PermissionError(
                f'job result did not declare exactly one {kind!r} output at this path'
            )
        digest, size = self._hash_file(source)
        if matches[0].get('sha256') != digest or matches[0].get('size_bytes') != size:
            raise PermissionError('declared job output changed after job completion')
        return source

    def read_structured_outputs(
        self,
        job_id: str,
        observation: dict[str, Any],
        attempt_id: str,
    ) -> dict[str, list[Any]]:
        """Read bounded JSON outputs declared by this successful fenced job.

        Output paths remain untrusted. Every included file must be inside the
        observed job's managed output directory and pass the same declaration,
        ownership, size, and digest checks used for artifact staging.
        """

        result = observation.get('result')
        if (
            observation.get('status') != 'succeeded'
            or not isinstance(result, dict)
            or result.get('protocol') != 'rightarm.mlops.job-result/v1'
            or result.get('status') != 'succeeded'
        ):
            return {}
        outputs = result.get('outputs')
        if (
            not isinstance(outputs, list)
            or len(outputs) > self.MAX_STRUCTURED_OUTPUT_DECLARATIONS
        ):
            return {}
        job_dir = (self.root / job_id).resolve()
        if job_dir.parent != self.root or job_dir.name != job_id:
            return {}

        structured: dict[str, list[Any]] = {}
        output_count = 0
        total_size = 0
        for item in outputs:
            if output_count >= self.MAX_STRUCTURED_OUTPUTS:
                break
            if not isinstance(item, dict):
                continue
            kind = item.get('kind')
            path = item.get('path')
            declared_size = item.get('size_bytes')
            if (
                not isinstance(kind, str)
                or self.SAFE_STRUCTURED_OUTPUT_KIND.fullmatch(kind) is None
                or not isinstance(path, str)
                or Path(path).suffix.casefold() != '.json'
                or type(declared_size) is not int
                or declared_size < 0
                or declared_size > self.MAX_STRUCTURED_OUTPUT_BYTES
                or total_size + declared_size > self.MAX_STRUCTURED_OUTPUT_TOTAL_BYTES
            ):
                continue
            try:
                source = self.assert_owned_output(path, attempt_id)
                relative = source.relative_to(self.root)
                if relative.parts[0] != job_id:
                    continue
                if source.stat().st_size != declared_size:
                    continue
                source = self.assert_declared_output(source, attempt_id, kind)
                with source.open('rb') as stream:
                    payload = stream.read(self.MAX_STRUCTURED_OUTPUT_BYTES + 1)
                if len(payload) != declared_size:
                    continue
                parsed = json.loads(
                    payload.decode('utf-8'),
                    parse_constant=_reject_nonfinite_json_constant,
                )
            except (
                json.JSONDecodeError,
                OSError,
                PermissionError,
                RecursionError,
                RuntimeError,
                UnicodeDecodeError,
                ValueError,
            ):
                continue
            structured.setdefault(kind, []).append(parsed)
            output_count += 1
            total_size += declared_size
        return structured

    @staticmethod
    def _declared_output_host_path(job_dir: Path, declared: str) -> Path:
        return Path(declared).resolve()

    def managed_input_path(self, attempt_id: str, artifact: ArtifactRef) -> Path:
        """Allocate a stable, executor-owned path for one fenced input artifact."""

        owner_key = hashlib.sha256(attempt_id.encode()).hexdigest()[:24]
        filename = f'{artifact.kind}-{artifact.sha256}'
        destination = (self.root / 'inputs' / owner_key / filename).resolve()
        if not destination.is_relative_to(self.root):
            raise PermissionError('managed artifact input escaped the executor root')
        return destination

    def execution_input_path(
        self, attempt_id: str, artifact: ArtifactRef, materialized: Path
    ) -> str:
        """Return the path an Agent must pass to this executor."""

        expected = self.managed_input_path(attempt_id, artifact)
        if materialized.resolve() != expected:
            raise PermissionError('materialized input is outside its managed path')
        return str(materialized)

    def execution_input_path_for_task(
        self,
        task_kind: str,
        attempt_id: str,
        artifact: ArtifactRef,
        materialized: Path,
    ) -> str:
        """Return an input path for the executor selected for this task."""

        del task_kind
        return self.execution_input_path(attempt_id, artifact, materialized)

    @staticmethod
    def _owns_request(request: dict[str, Any]) -> bool:
        """Whether this manager owns a request in a shared hybrid job root."""

        return request.get('executor') != 'docker'

    @staticmethod
    def _hash_file(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                size += len(block)
        return digest.hexdigest(), size

    @staticmethod
    def _find_job_manifest(argv: list[str], name: str) -> Path | None:
        try:
            index = argv.index('--output-dir')
        except ValueError:
            return None
        if index + 1 >= len(argv):
            return None
        return Path(argv[index + 1]).resolve() / name

    @staticmethod
    def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        return json.loads(path.read_text(encoding='utf-8'))

    @staticmethod
    def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _persisted_process_is_running(self, job_id: str) -> bool:
        process = self._read_optional_json(self.root / job_id / 'process.json')
        if process is None or not isinstance(process.get('pid'), int):
            return False
        expected_start = process.get('start_time_ticks')
        actual_start = self._process_start_time_ticks(process['pid'])
        if expected_start is not None and actual_start != expected_start:
            return False
        try:
            os.kill(process['pid'], 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    @staticmethod
    def _process_start_time_ticks(pid: int) -> int | None:
        try:
            remainder = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1]
            return int(remainder.split()[19])
        except (IndexError, OSError, ValueError):
            return None


class DockerJobManager(LocalJobManager):
    """Run each allowlisted attempt job in a separately fenced container."""

    GPU_JOB_KINDS = frozenset({'train', 'benchmark-checkpoint', 'train-eval-trial'})

    def __init__(
        self,
        root: str | Path,
        workspace: str | Path,
        *,
        image_catalog: dict[str, str],
        allow_external_jobs: bool = False,
        external_job_catalog: dict[str, list[str]] | None = None,
        host_root: str | Path | None = None,
        host_workspace: str | Path | None = None,
        read_only_source_roots: list[str | Path] | None = None,
        networked_job_kinds: set[str] | None = None,
        gpu_counts: dict[str, int] | None = None,
        gpu_device_ids: dict[str, list[str]] | None = None,
        container_python: str = 'python',
        memory_limit: str = '128g',
        nano_cpus: int = 32_000_000_000,
        client: Any | None = None,
    ):
        super().__init__(
            root,
            workspace,
            allow_external_jobs=allow_external_jobs,
            external_job_catalog=external_job_catalog,
        )
        self.image_catalog = dict(image_catalog)
        invalid_images = set(self.image_catalog) - self.allowed_job_kinds
        if invalid_images:
            raise ValueError(
                f'Docker image catalog has unknown job kinds: {sorted(invalid_images)}'
            )
        for kind, image in self.image_catalog.items():
            digest = image.rpartition('@sha256:')[2]
            if len(digest) != 64 or any(
                character not in '0123456789abcdef' for character in digest
            ):
                raise ValueError(
                    f'Docker image for {kind!r} must be pinned by sha256 digest'
                )
        self.host_root = self._absolute_operator_path(
            host_root if host_root is not None else self.root,
            label='Docker host job root',
        )
        self.host_workspace = self._absolute_operator_path(
            host_workspace if host_workspace is not None else self.workspace,
            label='Docker host workspace',
        )
        self.read_only_source_roots = [
            self._absolute_operator_path(value, label='Docker source root')
            for value in read_only_source_roots or []
        ]
        self.networked_job_kinds = set(networked_job_kinds or set())
        self.gpu_counts = dict(gpu_counts or {})
        self.gpu_device_ids = {
            kind: list(device_ids)
            for kind, device_ids in (gpu_device_ids or {}).items()
        }
        for label, values in (
            ('network', self.networked_job_kinds),
            ('GPU', set(self.gpu_counts)),
            ('GPU device', set(self.gpu_device_ids)),
        ):
            invalid = values - self.allowed_job_kinds
            if invalid:
                raise ValueError(
                    f'Docker {label} policy has unknown job kinds: {sorted(invalid)}'
                )
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in self.gpu_counts.values()
        ):
            raise ValueError('Docker GPU counts must be positive integers')
        for kind, device_ids in self.gpu_device_ids.items():
            if kind not in self.gpu_counts:
                raise ValueError(
                    f'Docker GPU devices for {kind!r} require an operator-owned '
                    'GPU count'
                )
            if (
                not device_ids
                or len(device_ids) != self.gpu_counts[kind]
                or len(set(device_ids)) != len(device_ids)
                or any(not device_id.isdigit() for device_id in device_ids)
            ):
                raise ValueError(
                    f'Docker GPU devices for {kind!r} must be unique numeric '
                    'device ids matching its GPU count'
                )
        if not container_python:
            raise ValueError('container Python executable must not be empty')
        if nano_cpus <= 0:
            raise ValueError('Docker nano_cpus must be positive')
        self.container_python = container_python
        self.memory_limit = memory_limit
        self.nano_cpus = nano_cpus
        self.client = client or docker.from_env()

    @staticmethod
    def _absolute_operator_path(
        value: str | Path,
        *,
        label: str,
    ) -> Path:
        """Normalize an operator-owned host path without resolving symlinks.

        Docker resolves bind sources on the host, not inside the controller
        container. Keeping the requested absolute target also preserves paths
        embedded in dataset indexes, such as ``/home/results/...``.
        """

        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f'{label} must be absolute')
        return Path(os.path.abspath(os.path.normpath(str(path))))

    def _host_job_path(self, controller_path: Path) -> Path:
        resolved = controller_path.resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError('Docker job path escaped the managed job root')
        return self.host_root / resolved.relative_to(self.root)

    @staticmethod
    def _owns_request(request: dict[str, Any]) -> bool:
        return request.get('executor') == 'docker'

    def launch(
        self,
        argv: list[str],
        idempotency_key: str,
        *,
        owner_attempt_id: str | None = None,
    ) -> dict[str, Any]:
        if len(argv) < 2 or argv[0] != 'rightarm-mlops':
            raise ValueError('job argv must start with the logical rightarm-mlops name')
        kind = argv[1]
        if kind not in self.allowed_job_kinds:
            raise ValueError(f'rightarm job kind is not allowed: {kind!r}')
        if kind in self.EXTERNAL_JOB_KINDS:
            self._validate_external_command(argv)
        image = self.image_catalog.get(kind)
        if image is None:
            raise ValueError(f'Docker image catalog has no image for {kind!r}')
        gpu_count = self.gpu_counts.get(kind)
        gpu_device_ids = self.gpu_device_ids.get(kind)
        if kind in self.GPU_JOB_KINDS:
            try:
                requested_gpus = int(argv[argv.index('--gpus') + 1])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f'{kind} job requires an integer --gpus value'
                ) from exc
            if gpu_count is None or requested_gpus != gpu_count:
                raise ValueError(
                    f'{kind} --gpus must exactly match its operator-owned '
                    'Docker GPU count'
                )
        with self._lock:
            job_id = self.job_id_for(owner_attempt_id, idempotency_key)
            job_dir = self.root / job_id
            output_dir = job_dir / 'output'
            normalized_argv = self._with_managed_output_dir(argv, Path('/mlops-output'))
            command = [
                self.container_python,
                '-m',
                'rightarm_mlops',
                *normalized_argv[1:],
            ]
            request = {
                'protocol': 'mlops.openhands.dev/job-request/v1',
                'executor': 'docker',
                'job_id': job_id,
                'owner_attempt_id': owner_attempt_id,
                'idempotency_key': idempotency_key,
                'argv': command,
                'image': image,
                'gpu_device_ids': gpu_device_ids,
                'output_dir': str(output_dir),
            }
            request_path = job_dir / 'request.json'
            if job_dir.exists():
                existing = self._read_optional_json(request_path)
                if existing != request:
                    raise ValueError(
                        'job idempotency key was reused with a different request'
                    )
                recovered = self._find_container(job_id)
                if recovered is not None:
                    process_path = job_dir / 'process.json'
                    if not process_path.exists():
                        self._write_json_exclusive(
                            process_path, {'container_id': recovered.id}
                        )
                    return self.observe(job_id, owner_attempt_id=owner_attempt_id)
                if (job_dir / 'process.json').exists():
                    return self.observe(job_id, owner_attempt_id=owner_attempt_id)
            else:
                job_dir.mkdir()
                self._write_json_exclusive(request_path, request)
            output_dir.mkdir(mode=0o770, exist_ok=True)
            owner_key = hashlib.sha256((owner_attempt_id or '').encode()).hexdigest()[
                :24
            ]
            input_dir = self.root / 'inputs' / owner_key
            input_dir.mkdir(parents=True, exist_ok=True)
            volumes = {
                str(self._host_job_path(output_dir)): {
                    'bind': '/mlops-output',
                    'mode': 'rw',
                },
                str(self._host_job_path(input_dir)): {
                    'bind': '/mlops-inputs',
                    'mode': 'ro',
                },
                str(self.host_workspace): {'bind': '/workspace', 'mode': 'ro'},
                **{
                    str(source): {'bind': str(source), 'mode': 'ro'}
                    for source in self.read_only_source_roots
                },
            }
            run_options: dict[str, Any] = {
                'image': image,
                'command': command,
                'name': self._container_name(job_id),
                'detach': True,
                'working_dir': '/workspace',
                'user': f'{os.getuid()}:{os.getgid()}',
                'volumes': volumes,
                'network_disabled': kind not in self.networked_job_kinds,
                'read_only': True,
                'cap_drop': ['ALL'],
                'security_opt': ['no-new-privileges:true'],
                'tmpfs': {'/tmp': 'rw,noexec,nosuid,size=8g'},
                'shm_size': '16g',
                'mem_limit': self.memory_limit,
                'nano_cpus': self.nano_cpus,
                'labels': {
                    'mlops.job_id': job_id,
                    'mlops.attempt_id': owner_attempt_id or '',
                },
                'environment': {
                    'RIGHTARM_MLOPS_EXECUTOR_IMAGE': image,
                    'RIGHTARM_MLOPS_EXECUTOR_JOB_KIND': kind,
                },
            }
            if gpu_count is not None:
                request_options: dict[str, Any] = {'capabilities': [['gpu']]}
                if gpu_device_ids is None:
                    request_options['count'] = gpu_count
                else:
                    request_options['device_ids'] = gpu_device_ids
                run_options['device_requests'] = [
                    docker.types.DeviceRequest(**request_options)
                ]
            container = self.client.containers.run(**run_options)
            process_path = job_dir / 'process.json'
            process = {'container_id': container.id}
            if process_path.exists():
                if self._read_optional_json(process_path) != process:
                    raise RuntimeError('Docker job recovered a conflicting container')
            else:
                self._write_json_exclusive(process_path, process)
            return {'job_id': job_id, 'status': 'running', 'argv': command}

    def execution_input_path(
        self, attempt_id: str, artifact: ArtifactRef, materialized: Path
    ) -> str:
        expected = self.managed_input_path(attempt_id, artifact)
        if materialized.resolve() != expected:
            raise PermissionError('materialized input is outside its managed path')
        return str(Path('/mlops-inputs') / expected.name)

    @staticmethod
    def _declared_output_host_path(job_dir: Path, declared: str) -> Path:
        container_path = Path(declared)
        container_root = Path('/mlops-output')
        if container_path.is_absolute() and container_path.is_relative_to(
            container_root
        ):
            return (
                job_dir / 'output' / container_path.relative_to(container_root)
            ).resolve()
        return container_path.resolve()

    def observe(
        self, job_id: str, *, owner_attempt_id: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            job_dir = self.root / job_id
            request = self._read_optional_json(job_dir / 'request.json')
            if request is None:
                raise KeyError(job_id)
            if request['owner_attempt_id'] != owner_attempt_id:
                raise PermissionError('job belongs to a different task attempt')
            container = self._find_container(job_id)
            container_status = 'lost'
            return_code = None
            if container is not None:
                container.reload()
                container_status = container.status
                state = container.attrs.get('State', {})
                if isinstance(state.get('ExitCode'), int):
                    return_code = state['ExitCode']
                logs = container.logs(stdout=True, stderr=True, tail=2000)
                if isinstance(logs, str):
                    logs = logs.encode()
                (job_dir / 'job.log').write_bytes(logs)
            log_path = job_dir / 'job.log'
            if log_path.exists():
                data = log_path.read_bytes()
                tail = data[-16_384:].decode('utf-8', errors='replace')
                log_sha256, log_size_bytes = self._hash_file(log_path)
            else:
                tail = ''
                log_sha256, log_size_bytes = None, None
            output_dir = Path(request['output_dir'])
            result = self._read_optional_json(output_dir / 'result.json')
            if result is not None:
                result = self._translate_result_output_paths(job_dir, result)
            failure = self._read_optional_json(output_dir / 'failure.json')
            cancellation = self._read_optional_json(job_dir / 'cancellation.json')
            if result is not None:
                status = 'succeeded'
                return_code = 0
            elif failure is not None:
                status = 'failed'
            elif cancellation is not None:
                status = 'cancelled'
            elif container_status in {'created', 'restarting', 'running', 'paused'}:
                status = 'running'
            else:
                status = 'lost'
            return {
                'job_id': job_id,
                'status': status,
                'return_code': return_code,
                'container_id': container.id if container is not None else None,
                'container_status': container_status,
                'image': request['image'],
                'log_path': str(log_path),
                'log_tail': tail,
                'log_sha256': log_sha256,
                'log_size_bytes': log_size_bytes,
                'result': result,
                'failure': failure,
                'cancellation': cancellation,
            }

    @classmethod
    def _translate_result_output_paths(
        cls, job_dir: Path, result: dict[str, Any]
    ) -> dict[str, Any]:
        translated = dict(result)
        outputs = result.get('outputs')
        if not isinstance(outputs, list):
            return translated
        translated_outputs = []
        for item in outputs:
            if not isinstance(item, dict) or not isinstance(item.get('path'), str):
                translated_outputs.append(item)
                continue
            translated_outputs.append(
                {
                    **item,
                    'path': str(cls._declared_output_host_path(job_dir, item['path'])),
                }
            )
        translated['outputs'] = translated_outputs
        return translated

    def cancel_attempt(self, attempt_id: str) -> list[str]:
        cancelled = []
        with self._lock:
            for job_dir in sorted(self.root.glob('job_*')):
                request = self._read_optional_json(job_dir / 'request.json')
                if (
                    request is None
                    or not self._owns_request(request)
                    or request.get('owner_attempt_id') != attempt_id
                ):
                    continue
                job_id = request['job_id']
                observation = self.observe(job_id, owner_attempt_id=attempt_id)
                if observation['status'] != 'running':
                    continue
                container = self._find_container(job_id)
                if container is not None:
                    container.stop(timeout=10)
                cancellation_path = job_dir / 'cancellation.json'
                if not cancellation_path.exists():
                    self._write_json_exclusive(
                        cancellation_path,
                        {
                            'attempt_id': attempt_id,
                            'reason': 'attempt lease expired',
                        },
                    )
                cancelled.append(job_id)
        return cancelled

    def _find_container(self, job_id: str) -> Any | None:
        process = self._read_optional_json(self.root / job_id / 'process.json')
        identity = (
            process.get('container_id')
            if process is not None
            else self._container_name(job_id)
        )
        if not isinstance(identity, str):
            raise RuntimeError('Docker process record has no container id')
        try:
            return self.client.containers.get(identity)
        except docker.errors.NotFound:
            return None

    @staticmethod
    def _container_name(job_id: str) -> str:
        return f'openhands-mlops-{job_id}'


def create_mcp_server(client: MLOpsTaskClient, job_manager: LocalJobManager) -> FastMCP:
    server = FastMCP(
        'agentic-mlops-task-v1',
        instructions=(
            'Every tool call is fenced to one TaskAttempt. Use rightarm-mlops '
            'jobs for effects and submit a structured TaskResult when done.'
        ),
        mask_error_details=True,
    )
    observations: list[dict[str, Any]] = []

    def authorize_job(argv: list[str]) -> None:
        context = client.get_task_context()
        allowed = TASK_JOB_KINDS.get(context.task_kind, frozenset())
        if len(argv) < 2 or argv[1] not in allowed:
            raise PermissionError(
                f'job is outside the {context.task_kind!r} task capability'
            )

    def authorize_artifact_kind(kind: str) -> None:
        context = client.get_task_context()
        if kind not in context.completion_contract.required_artifact_kinds:
            raise PermissionError(
                f'artifact kind is outside the {context.task_kind!r} '
                'completion contract'
            )

    @server.tool()
    def mlops_get_task_context() -> dict[str, Any]:
        """Get this attempt's immutable objective, inputs, contract, and budget."""

        return client.get_task_context().model_dump(mode='json')

    @server.tool()
    def mlops_record_task_plan(
        steps: list[str], assumptions: list[str], risks: list[str]
    ) -> dict[str, Any]:
        """Record an attempt-local plan; this does not change Run state."""

        record = {
            'observation_id': f'observation:{uuid4().hex}',
            'kind': 'task-plan',
            'steps': steps,
            'assumptions': assumptions,
            'risks': risks,
        }
        observations.append(record)
        return record

    @server.tool()
    def mlops_launch_job(argv: list[str], idempotency_key: str) -> dict[str, Any]:
        """Launch an allowlisted rightarm-mlops argv without a shell."""

        authorize_job(argv)
        return job_manager.launch(
            argv, idempotency_key, owner_attempt_id=client.attempt_id
        )

    @server.tool()
    def mlops_observe_job(job_id: str) -> dict[str, Any]:
        """Read job state, log tail, and structured result/failure manifests."""

        return _fresh_poll_observation(
            _observe_job_until_terminal(
                job_manager,
                job_id,
                owner_attempt_id=client.attempt_id,
            )
        )

    @server.tool()
    def mlops_stage_artifact(
        path: str,
        kind: str,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Hash and stage an artifact candidate; it is not sealed yet."""

        authorize_artifact_kind(kind)
        source = job_manager.assert_declared_output(path, client.attempt_id, kind)
        return client.stage_artifact(
            source, kind=kind, media_type=media_type, metadata=metadata
        ).model_dump(mode='json')

    @server.tool()
    def mlops_materialize_artifact(artifact_id: str) -> dict[str, Any]:
        """Materialize one declared input without exposing storage credentials."""

        context = client.get_task_context()
        artifact = next(
            (item for item in context.input_artifacts if item.id == artifact_id),
            None,
        )
        if artifact is None:
            raise PermissionError('artifact is not an input of this task attempt')
        destination = job_manager.managed_input_path(client.attempt_id, artifact)
        path = client.artifacts.materialize(artifact, destination)
        return {
            'artifact_id': artifact.id,
            'path': job_manager.execution_input_path_for_task(
                context.task_kind,
                client.attempt_id,
                artifact,
                path,
            ),
            'sha256': artifact.sha256,
            'size_bytes': artifact.size_bytes,
        }

    @server.tool()
    def mlops_propose_lineage(
        input_artifact_id: str,
        output_candidate_id: str,
        transform_kind: str,
        code_git_sha: str | None = None,
        container_image_digest: str | None = None,
        config_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a lineage proposal for inclusion in the final TaskResult."""

        return LineageProposal(
            input_artifact_id=input_artifact_id,
            output_candidate_id=output_candidate_id,
            transform_kind=transform_kind,
            code_git_sha=code_git_sha,
            container_image_digest=container_image_digest,
            config_artifact_id=config_artifact_id,
        ).model_dump(mode='json')

    @server.tool()
    def mlops_record_observation(kind: str, detail: dict[str, Any]) -> dict[str, Any]:
        """Record task-local structured evidence for later result submission."""

        record = {
            'observation_id': f'observation:{uuid4().hex}',
            'kind': kind,
            'detail': detail,
        }
        observations.append(record)
        return record

    @server.tool()
    def mlops_submit_task_result(result: dict[str, Any]) -> dict[str, object]:
        """Submit the typed TaskResult; only the outer validator can accept it."""

        parsed = TaskResult.model_validate(result)
        return client.submit_task_result(parsed)

    return server


def create_gateway_mcp_server(
    *,
    controller: Any,
    job_manager: LocalJobManager,
) -> FastMCP:
    """Create one remote MCP gateway shared by all fenced task attempts."""

    server = FastMCP(
        'agentic-mlops-gateway-v1',
        instructions=(
            'Pass the attempt_id and lease_id from your immutable TaskContext to '
            'every tool. The gateway rejects stale or cross-attempt access.'
        ),
        mask_error_details=True,
    )

    def fence(attempt_id: str, lease_id: str) -> None:
        controller.ledger.assert_attempt_lease(attempt_id, lease_id)

    def task_spec_for_attempt(attempt_id: str) -> dict[str, Any]:
        attempt = controller.ledger.get_attempt(attempt_id)
        task = controller.ledger.get_task(attempt['task_id'])
        return task['spec']

    def authorize_job(attempt_id: str, argv: list[str]) -> None:
        task_spec = task_spec_for_attempt(attempt_id)
        allowed = TASK_JOB_KINDS.get(task_spec['kind'], frozenset())
        if len(argv) < 2 or argv[1] not in allowed:
            raise PermissionError(
                f'job is outside the {task_spec["kind"]!r} task capability'
            )

    def authorize_artifact_kind(attempt_id: str, kind: str) -> None:
        task_spec = task_spec_for_attempt(attempt_id)
        allowed = set(
            task_spec.get('completion_contract', {}).get('required_artifact_kinds', [])
        )
        if kind not in allowed:
            raise PermissionError(
                f'artifact kind is outside the {task_spec["kind"]!r} '
                'completion contract'
            )

    @server.tool()
    def mlops_get_task_context(attempt_id: str, lease_id: str) -> dict[str, Any]:
        """Fetch the immutable context for one currently leased attempt."""

        return controller.get_task_context(attempt_id, lease_id).model_dump(mode='json')

    @server.tool()
    def mlops_launch_job(
        attempt_id: str,
        lease_id: str,
        argv: list[str],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Launch an allowlisted job owned by this attempt."""

        fence(attempt_id, lease_id)
        try:
            authorize_job(attempt_id, argv)
        except (ValueError, PermissionError) as exc:
            raise _controlled_gateway_tool_error(exc) from exc
        qualified_key = f'{attempt_id}:{idempotency_key}'
        job_id = job_manager.job_id_for(attempt_id, qualified_key)
        controller.ledger.record_job_launch(
            attempt_id=attempt_id,
            job_id=job_id,
            idempotency_key=idempotency_key,
            argv=argv,
            status='requested',
        )
        try:
            launched = job_manager.launch(
                argv,
                qualified_key,
                owner_attempt_id=attempt_id,
            )
        except Exception as exc:
            controller.ledger.record_job_observation(
                attempt_id=attempt_id,
                job_id=job_id,
                observation={
                    'status': 'failed',
                    'return_code': None,
                    'failure': {
                        'failure_class': 'executor-launch-failed',
                        'error_type': type(exc).__name__,
                    },
                },
            )
            if isinstance(exc, (ValueError, PermissionError)):
                raise _controlled_gateway_tool_error(exc) from exc
            raise
        controller.ledger.record_job_observation(
            attempt_id=attempt_id,
            job_id=job_id,
            observation=launched,
        )
        return launched

    @server.tool()
    def mlops_observe_job(
        attempt_id: str, lease_id: str, job_id: str
    ) -> dict[str, Any]:
        """Observe a fenced job and inline its bounded, declared JSON outputs."""

        fence(attempt_id, lease_id)
        observation = _observe_job_until_terminal(
            job_manager,
            job_id,
            owner_attempt_id=attempt_id,
        )
        controller.ledger.record_job_observation(
            attempt_id=attempt_id,
            job_id=job_id,
            observation=observation,
        )
        return _fresh_poll_observation(
            {
                **observation,
                'structured_outputs': job_manager.read_structured_outputs(
                    job_id,
                    observation,
                    attempt_id,
                ),
            }
        )

    @server.tool()
    def mlops_stage_artifact(
        attempt_id: str,
        lease_id: str,
        path: str,
        kind: str,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stage one job output after enforcing attempt and job-root ownership."""

        fence(attempt_id, lease_id)
        authorize_artifact_kind(attempt_id, kind)
        source = job_manager.assert_declared_output(path, attempt_id, kind)
        candidate = controller.artifact_store.stage(
            source,
            kind=kind,
            media_type=media_type,
            metadata=metadata,
            candidate_key=f'{attempt_id}:{kind}:{source}',
        )
        controller.ledger.record_artifact_candidate(
            attempt_id=attempt_id,
            lease_id=lease_id,
            candidate=candidate,
        )
        return candidate.model_dump(mode='json')

    @server.tool()
    def mlops_materialize_artifact(
        attempt_id: str,
        lease_id: str,
        artifact_id: str,
    ) -> dict[str, Any]:
        """Materialize one declared input without exposing storage credentials."""

        context = controller.get_task_context(attempt_id, lease_id)
        artifact = next(
            (item for item in context.input_artifacts if item.id == artifact_id),
            None,
        )
        if artifact is None:
            raise PermissionError('artifact is not an input of this task attempt')
        destination = job_manager.managed_input_path(attempt_id, artifact)
        path = controller.artifact_store.materialize(artifact, destination)
        task_spec = task_spec_for_attempt(attempt_id)
        return {
            'artifact_id': artifact.id,
            'path': job_manager.execution_input_path_for_task(
                task_spec['kind'],
                attempt_id,
                artifact,
                path,
            ),
            'sha256': artifact.sha256,
            'size_bytes': artifact.size_bytes,
        }

    @server.tool()
    def mlops_propose_lineage(
        input_artifact_id: str,
        output_candidate_id: str,
        transform_kind: str,
        code_git_sha: str | None = None,
        container_image_digest: str | None = None,
        config_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a typed lineage proposal without changing the ledger."""

        return LineageProposal(
            input_artifact_id=input_artifact_id,
            output_candidate_id=output_candidate_id,
            transform_kind=transform_kind,
            code_git_sha=code_git_sha,
            container_image_digest=container_image_digest,
            config_artifact_id=config_artifact_id,
        ).model_dump(mode='json')

    @server.tool()
    def mlops_submit_task_result(
        attempt_id: str,
        lease_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit a typed result; only independent validation can accept it."""

        fence(attempt_id, lease_id)
        parsed = TaskResult.model_validate(result)
        if parsed.attempt_id != attempt_id:
            raise ValueError('result belongs to a different task attempt')
        attempt = controller.ledger.get_attempt(attempt_id)
        task = controller.ledger.get_task(attempt['task_id'])
        run_state = controller.submit_result(task['run_id'], parsed, lease_id)
        return {'accepted': True, 'run_state': run_state.value}

    return server


def main() -> None:
    required = {
        name: os.environ[name]
        for name in (
            'MLOPS_LOOP_URL',
            'MLOPS_RUN_ID',
            'MLOPS_ATTEMPT_ID',
            'MLOPS_LEASE_ID',
            'MLOPS_ARTIFACT_ROOT',
            'MLOPS_WORKSPACE',
            'MLOPS_JOB_ROOT',
        )
    }
    client = MLOpsTaskClient(
        base_url=required['MLOPS_LOOP_URL'],
        run_id=required['MLOPS_RUN_ID'],
        attempt_id=required['MLOPS_ATTEMPT_ID'],
        lease_id=required['MLOPS_LEASE_ID'],
        artifact_root=required['MLOPS_ARTIFACT_ROOT'],
    )
    jobs = LocalJobManager(
        required['MLOPS_JOB_ROOT'],
        required['MLOPS_WORKSPACE'],
        allow_external_jobs=os.environ.get('MLOPS_ALLOW_EXTERNAL_JOBS') == '1',
        python_executable=os.environ.get('MLOPS_JOB_PYTHON', sys.executable),
        external_job_catalog=(
            json.loads(Path(os.environ['MLOPS_EXTERNAL_JOB_CATALOG']).read_text())
            if os.environ.get('MLOPS_EXTERNAL_JOB_CATALOG')
            else None
        ),
    )
    create_mcp_server(client, jobs).run()


if __name__ == '__main__':
    main()
