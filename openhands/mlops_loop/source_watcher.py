"""Deterministic source discovery for the long-lived Loop Engineering layer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiscoveredBatch(BaseModel):
    model_config = ConfigDict(extra='forbid')

    watermark: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    manifest_uri: str = Field(min_length=1)

    @field_validator('source_uri', 'manifest_uri')
    @classmethod
    def require_read_only_materialization_uri(cls, value: str) -> str:
        if urlparse(value).scheme != 'file':
            raise ValueError('V1 source adapters must return file:// materializations')
        return value


class SourceWatcher(Protocol):
    def discover(self) -> list[DiscoveredBatch]: ...


class LocalBatchWatcher:
    """Staging watcher with the same `_SUCCESS` + manifest contract as TOS."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        completion_marker: str = '_SUCCESS',
        manifest_name: str = 'batch.manifest.json',
    ):
        self.source_root = Path(source_root).resolve(strict=True)
        self.completion_marker = completion_marker
        self.manifest_name = manifest_name

    def discover(self) -> list[DiscoveredBatch]:
        discovered = []
        for marker in self.source_root.glob(f'*/{self.completion_marker}'):
            batch_dir = marker.parent.resolve()
            manifest_path = batch_dir / self.manifest_name
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            watermark = manifest.get('batch_id')
            if not isinstance(watermark, str) or not watermark:
                continue
            discovered.append(
                DiscoveredBatch(
                    watermark=watermark,
                    source_uri=batch_dir.as_uri(),
                    manifest_uri=manifest_path.as_uri(),
                )
            )
        return sorted(discovered, key=lambda item: item.watermark)


class CommandSourceWatcher:
    """Cloud adapter boundary: an operator-owned argv returns JSON batches.

    This keeps TOS/COS credentials and vendor CLI behavior outside the Agent.
    The command must print a JSON array matching :class:`DiscoveredBatch` and
    expose each sealed cloud batch as a read-only local ``file://`` URI.
    """

    def __init__(self, argv: list[str]):
        if not argv:
            raise ValueError('watcher command cannot be empty')
        self.argv = list(argv)

    def discover(self) -> list[DiscoveredBatch]:
        completed = subprocess.run(  # noqa: S603 - operator configuration, no shell
            self.argv,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, list):
            raise ValueError('source watcher command must return a JSON array')
        return sorted(
            (DiscoveredBatch.model_validate(item) for item in payload),
            key=lambda item: item.watermark,
        )
