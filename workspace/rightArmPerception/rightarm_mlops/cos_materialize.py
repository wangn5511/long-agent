"""Credential-isolated Tencent COS batch materialization."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from rightarm_mlops.batch import inspect_batch
from rightarm_mlops.protocol import atomic_json, sha256_file


def materialize_cos_batch(source_uri: str, output_dir: Path) -> dict[str, Any]:
    source_uri = _validate_source_uri(source_uri)
    coscli, config = _trusted_cos_runtime()
    preflight = output_dir / 'preflight'
    preflight.mkdir(parents=True, exist_ok=True)
    manifest_path = preflight / 'batch.manifest.json'
    marker_path = preflight / '_SUCCESS'
    _copy(coscli, config, f'{source_uri}batch.manifest.json', manifest_path)
    _copy(coscli, config, f'{source_uri}_SUCCESS', marker_path)

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    declared_bytes = _declared_bytes(manifest)
    reserve = int(os.environ.get('RIGHTARM_COS_DISK_RESERVE_BYTES', 20 << 30))
    required_free = declared_bytes * 3 + reserve
    actual_free = shutil.disk_usage(output_dir).free
    if actual_free < required_free:
        raise ValueError(
            f'insufficient disk for sealed COS batch: need {required_free} free bytes, '
            f'have {actual_free}'
        )

    download = output_dir / 'download'
    download.mkdir()
    command = [
        str(coscli),
        'cp',
        source_uri,
        str(download),
        '--recursive',
        '--config-path',
        str(config),
        '--disable-log',
        '--process-log=false',
        '--fail-output=false',
        '--err-retry-num',
        '5',
    ]
    subprocess.run(command, check=True)  # noqa: S603 - executable is operator-owned

    candidates = [
        path.parent
        for path in download.rglob('batch.manifest.json')
        if (path.parent / '_SUCCESS').is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f'COS prefix must contain exactly one sealed batch, found {len(candidates)}'
        )
    batch_dir = candidates[0].resolve(strict=True)
    preflight_digest, _ = sha256_file(manifest_path)
    downloaded_digest, _ = sha256_file(batch_dir / 'batch.manifest.json')
    if downloaded_digest != preflight_digest:
        raise ValueError('COS batch manifest changed while it was downloaded')
    inspection = inspect_batch(batch_dir)
    report = {
        'schema_version': 'rightarm.gesture.cos-materialization/v1',
        'source_uri': source_uri,
        'materialized_source_uri': batch_dir.as_uri(),
        'batch_id': inspection['batch_id'],
        'declared_bytes': declared_bytes,
        'verified_object_count': inspection['object_count'],
        'manifest_sha256': downloaded_digest,
    }
    report_path = output_dir / 'cos-materialization-report.json'
    atomic_json(report_path, report)
    return report


def _validate_source_uri(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != 'cos'
        or not parsed.netloc
        or not parsed.path.strip('/')
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError('source URI must be cos://<bucket>/<non-empty-prefix>/')
    normalized = f'cos://{parsed.netloc}/{parsed.path.strip("/")}/'
    allowed = os.environ.get('RIGHTARM_COS_ALLOWED_PREFIX', '').strip()
    if allowed and not normalized.startswith(allowed.rstrip('/') + '/'):
        raise ValueError('COS source URI is outside RIGHTARM_COS_ALLOWED_PREFIX')
    return normalized


def _trusted_cos_runtime() -> tuple[Path, Path]:
    coscli = Path(
        os.environ.get('RIGHTARM_COSCLI_PATH', '/usr/local/bin/coscli')
    ).resolve(strict=True)
    config = Path(
        os.environ.get('RIGHTARM_COS_CONFIG_PATH', '/run/secrets/tencent-cos.yaml')
    )
    if config.is_symlink():
        raise ValueError('Tencent COS config cannot be a symlink')
    config = config.resolve(strict=True)
    config_stat = config.stat()
    if not stat.S_ISREG(config_stat.st_mode) or config_stat.st_mode & 0o077:
        raise ValueError(
            'Tencent COS config must be a private regular file (mode 0600)'
        )
    if not coscli.is_file() or not os.access(coscli, os.X_OK):
        raise ValueError('configured coscli is not executable')
    return coscli, config


def _copy(coscli: Path, config: Path, source: str, destination: Path) -> None:
    command = [
        str(coscli),
        'cp',
        source,
        str(destination),
        '--config-path',
        str(config),
        '--disable-log',
        '--process-log=false',
        '--fail-output=false',
        '--err-retry-num',
        '5',
    ]
    subprocess.run(command, check=True)  # noqa: S603 - executable is operator-owned


def _declared_bytes(manifest: dict[str, Any]) -> int:
    if manifest.get('schema_version') != 'rightarm.gesture.batch/v1':
        raise ValueError('manifest schema_version must be rightarm.gesture.batch/v1')
    items = manifest.get('objects')
    annotations = manifest.get('annotations')
    if not isinstance(items, list) or not isinstance(annotations, dict):
        raise ValueError('manifest must contain objects and annotations')
    total = 0
    for item in [*items, annotations]:
        if not isinstance(item, dict):
            raise ValueError('manifest entries must be objects')
        raw_path = item.get('path', '')
        relative = PurePosixPath(raw_path)
        if raw_path and (relative.is_absolute() or '..' in relative.parts):
            raise ValueError(f'manifest object path is unsafe: {raw_path!r}')
        digest = item.get('sha256')
        size = item.get('bytes', item.get('size_bytes'))
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError('manifest entry has invalid sha256 or size')
        total += size
    return total
