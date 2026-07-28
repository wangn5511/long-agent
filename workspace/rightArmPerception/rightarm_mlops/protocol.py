"""Shared result/failure manifest protocol for every rightarm-mlops job."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

PROTOCOL = 'rightarm.mlops.job-result/v1'


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def output_record(kind: str, path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        'kind': kind,
        'path': str(path.resolve()),
        'sha256': digest,
        'size_bytes': size,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f'.{path.name}.', suffix='.tmp'
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def deterministic_tar(source_root: Path, output_path: Path) -> None:
    """Archive a directory with normalized metadata and stable member order."""

    source_root = source_root.resolve(strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(output_path.parent), prefix=f'.{output_path.name}.', suffix='.tmp'
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            for path in sorted(
                item for item in source_root.rglob('*') if item.is_file()
            ):
                relative = path.relative_to(source_root)
                info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = ''
                info.gname = ''
                info.mtime = 0
                info.mode = 0o644
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a regular-file tar without traversal, links, or device nodes."""

    archive_path = archive_path.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive_path, mode='r:*') as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'tar member escapes destination: {member.name!r}')
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'tar member type is not allowed: {member.name!r}')
            target = (destination_root / relative).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError:
                raise ValueError(f'tar member escapes destination: {member.name!r}')
        # Every member was prevalidated as an in-root regular file/directory;
        # avoid the Python 3.12-only ``filter=`` API because the pinned
        # training image intentionally still runs Python 3.8.
        archive.extractall(destination_root)  # noqa: S202


def success(
    *,
    output_dir: Path,
    job_kind: str,
    inputs: Iterable[dict[str, Any]] = (),
    outputs: Iterable[dict[str, Any]] = (),
    metrics: Optional[dict[str, Any]] = None,
    runtime: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    # main() publishes the terminal manifest exactly once after attaching the
    # complete monotonic runtime. Keeping construction side-effect free avoids
    # a controller observing a partial result between two atomic rewrites.
    del output_dir
    result = {
        'protocol': PROTOCOL,
        'job_kind': job_kind,
        'status': 'succeeded',
        'inputs': list(inputs),
        'outputs': list(outputs),
        'metrics': metrics or {},
        'runtime': {
            'python': platform.python_version(),
            **(runtime or {}),
        },
    }
    return result


def failure(
    *,
    output_dir: Path,
    job_kind: str,
    failure_class: str,
    message: str,
    evidence: Iterable[str] = (),
    retryable: bool = False,
) -> dict[str, Any]:
    # See success(): terminal files are published only after runtime is final.
    del output_dir
    result = {
        'protocol': PROTOCOL,
        'job_kind': job_kind,
        'status': 'failed',
        'failure_class': failure_class,
        'message': message,
        'retryable': retryable,
        'evidence': list(evidence),
    }
    return result


def parse_output_specs(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if '=' not in value:
            raise ValueError(f'output must use KIND=PATH syntax: {value!r}')
        kind, raw_path = value.split('=', 1)
        if not kind or not raw_path:
            raise ValueError(f'output must use KIND=PATH syntax: {value!r}')
        parsed.append((kind, Path(raw_path)))
    return parsed
