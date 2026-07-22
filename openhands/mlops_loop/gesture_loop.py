"""Long-lived gesture-loop service that turns watermarks into immutable Runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from openhands.mlops_loop.controller import TERMINAL_RUN_STATES, LoopController
from openhands.mlops_loop.models import RunState
from openhands.mlops_loop.source_watcher import SourceWatcher
from openhands.mlops_loop.spec import resolve_template, task_specs_from_documents


class GestureLoopService:
    def __init__(
        self,
        *,
        controller: LoopController,
        watcher: SourceWatcher,
        loop_spec: dict[str, Any],
        run_template: dict[str, Any],
        replacements: dict[str, str],
    ):
        self.controller = controller
        self.watcher = watcher
        self.loop_spec = loop_spec
        self.run_template = run_template
        self.replacements = dict(replacements)
        self.loop_id = loop_spec['metadata']['name']

    def poll_once(self) -> dict[str, Any] | None:
        """Create at most one active Run; duplicate notifications are harmless."""

        if self.controller.ledger.get_loop_control(self.loop_id)['state'] == 'STOPPED':
            return None
        runs = self.controller.ledger.list_runs(self.loop_id)
        stop_spec = self.loop_spec.get('spec', {}).get('stop', {})
        max_failures = int(stop_spec.get('maxConsecutiveFailedRuns', 0))
        if max_failures > 0:
            consecutive_failures = 0
            for previous in reversed(runs):
                if previous['state'] in {
                    RunState.FAILED,
                    RunState.NEEDS_INTERVENTION,
                }:
                    consecutive_failures += 1
                else:
                    break
            if consecutive_failures >= max_failures:
                latest = runs[-1]
                self.controller.ledger.stop_loop(
                    loop_id=self.loop_id,
                    run_id=latest['id'],
                    reason=(
                        f'{consecutive_failures} consecutive runs required '
                        'intervention or failed'
                    ),
                    idempotency_key=f'loop-stop:consecutive-failures:{max_failures}',
                )
                return None
        active = [
            run for run in runs if RunState(run['state']) not in TERMINAL_RUN_STATES
        ]
        if active:
            return self.controller.ledger.get_run(active[0]['id'])
        known = {run['watermark']: run for run in runs}
        for batch in self.watcher.discover():
            if batch.watermark in known:
                continue
            manifest_path = self._file_uri_path(batch.manifest_uri)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if not isinstance(manifest, dict):
                raise ValueError('discovered batch manifest must be a JSON object')
            if manifest.get('batch_id') != batch.watermark:
                raise ValueError(
                    'discovered batch watermark does not match its sealed manifest'
                )
            source_manifest = self.controller.bootstrap_artifact(
                path=str(manifest_path),
                kind='source-manifest',
                media_type='application/json',
                metadata={
                    'watermark': batch.watermark,
                    'source_uri': batch.source_uri,
                },
            )
            replacements = {
                **self.replacements,
                'batch-id': batch.watermark,
                'batch-source-uri': batch.source_uri,
            }
            resolved_run = resolve_template(self.run_template, replacements)
            # The batch URI discovered from the sealed marker is authoritative.
            resolved_run['spec']['source']['uri'] = batch.source_uri
            resolved_run['spec']['source']['manifestUri'] = batch.manifest_uri
            resolved_run['spec']['source']['manifestArtifactId'] = source_manifest['id']
            resolved_run['spec']['pending_artifact_ids'] = (
                self.controller.ledger.list_pending_artifacts(self.loop_id)
            )
            initial_artifact_ids = list(
                dict.fromkeys(
                    [
                        *resolved_run['spec'].get('inputArtifactIds', []),
                        *resolved_run['spec']['pending_artifact_ids'],
                        source_manifest['id'],
                    ]
                )
            )
            tasks = task_specs_from_documents(self.loop_spec, resolved_run)
            return self.controller.create_run(
                loop_id=self.loop_id,
                watermark=batch.watermark,
                run_spec=resolved_run,
                tasks=tasks,
                initial_artifact_ids=initial_artifact_ids,
            )
        return None

    @staticmethod
    def _file_uri_path(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != 'file' or parsed.netloc not in {'', 'localhost'}:
            raise ValueError('source manifest must be a read-only file URI')
        return Path(unquote(parsed.path)).resolve(strict=True)
