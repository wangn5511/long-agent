"""Small domain facade for immutable releases and deployment state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openhands.mlops_loop.ledger import SQLiteLedger


class ReleaseService:
    def __init__(self, ledger: SQLiteLedger):
        self.ledger = ledger

    def seal(
        self,
        *,
        run_id: str,
        manifest_artifact_id: str,
        artifact_ids: Sequence[str],
        validation_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        run = self.ledger.get_run(run_id)
        return self.ledger.seal_release(
            run_id=run_id,
            manifest_artifact_id=manifest_artifact_id,
            artifact_ids=artifact_ids,
            root_kinds=frozenset(
                {'source-manifest', 'raw-dataset', 'base-model', 'code'}
            ),
            expected_version=run['version'],
            validation_id=validation_id,
            idempotency_key=idempotency_key,
        )
