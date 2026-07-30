"""Transactional SQLite event ledger for the staging outer loop.

SQLite is intentionally the local/staging backend.  The domain API avoids
SQLite-specific types so a PostgreSQL implementation can replace it without
changing task or controller protocols.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from openhands.mlops_loop.errors import (
    ConcurrencyConflict,
    IdempotencyConflict,
    InvalidTransition,
    InvariantViolation,
    LeaseConflict,
)
from openhands.mlops_loop.models import (
    ArtifactCandidate,
    AttemptStatus,
    CompletionContract,
    RunState,
    TaskResult,
    TaskSpec,
    TaskStatus,
    ValidationDisposition,
    ValidationResult,
)
from openhands.mlops_loop.retry_policy import RetryDecision
from openhands.mlops_loop.spec import RIGHTARM_RELEASE_REQUIRED_KINDS


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _payload_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256('\x00'.join(parts).encode()).hexdigest()[:24]
    return f'{prefix}_{digest}'


class SQLiteLedger:
    """Append-only events plus transactional query projections."""

    def __init__(self, database: str | Path):
        self.database = str(database)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('PRAGMA journal_mode = WAL')
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS mlops_runs (
                    id TEXT PRIMARY KEY,
                    loop_id TEXT NOT NULL,
                    watermark TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    active_task_ordinal INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(loop_id, watermark)
                );
                CREATE TABLE IF NOT EXISTS mlops_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(run_id, sequence),
                    UNIQUE(run_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS mlops_agent_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    ordinal INTEGER NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(run_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS mlops_task_attempts (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES mlops_agent_tasks(id),
                    attempt_number INTEGER NOT NULL,
                    attempt_spec_json TEXT NOT NULL,
                    conversation_id TEXT UNIQUE,
                    dispatch_key TEXT UNIQUE,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    validator_result_json TEXT,
                    failure_class TEXT,
                    started_at REAL,
                    finished_at REAL,
                    UNIQUE(task_id, attempt_number)
                );
                CREATE TABLE IF NOT EXISTS mlops_attempt_leases (
                    attempt_id TEXT PRIMARY KEY REFERENCES mlops_task_attempts(id),
                    lease_id TEXT NOT NULL,
                    agent_instance_id TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_job_attempts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    task_attempt_id TEXT NOT NULL REFERENCES mlops_task_attempts(id),
                    idempotency_key TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    argv_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    return_code INTEGER,
                    result_json TEXT,
                    failure_json TEXT,
                    log_path TEXT,
                    log_sha256 TEXT,
                    log_size_bytes INTEGER,
                    last_observation_digest TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(task_attempt_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS mlops_outbox (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    attempt_id TEXT NOT NULL UNIQUE REFERENCES mlops_task_attempts(id),
                    kind TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    payload_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    external_id TEXT,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_cost_entries (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    amount_usd REAL NOT NULL CHECK(amount_usd >= 0),
                    metadata_json TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(run_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS mlops_artifacts (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    media_type TEXT,
                    metadata_json TEXT NOT NULL,
                    sealed INTEGER NOT NULL,
                    created_by_run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    created_by_attempt_id TEXT REFERENCES mlops_task_attempts(id),
                    created_at REAL NOT NULL,
                    UNIQUE(kind, sha256)
                );
                CREATE TABLE IF NOT EXISTS mlops_artifact_candidates (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    task_attempt_id TEXT NOT NULL REFERENCES mlops_task_attempts(id),
                    candidate_json TEXT NOT NULL,
                    candidate_digest TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_lineage_edges (
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    input_artifact_id TEXT NOT NULL REFERENCES mlops_artifacts(id),
                    output_artifact_id TEXT NOT NULL REFERENCES mlops_artifacts(id),
                    transform_kind TEXT NOT NULL,
                    code_git_sha TEXT,
                    container_image_digest TEXT,
                    config_artifact_id TEXT,
                    created_by_event_id TEXT NOT NULL REFERENCES mlops_events(id),
                    PRIMARY KEY(
                        run_id,
                        input_artifact_id,
                        output_artifact_id,
                        transform_kind
                    ),
                    CHECK(input_artifact_id <> output_artifact_id)
                );
                CREATE TABLE IF NOT EXISTS mlops_run_artifacts (
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    artifact_id TEXT NOT NULL REFERENCES mlops_artifacts(id),
                    introduced_by_attempt_id TEXT REFERENCES mlops_task_attempts(id),
                    created_at REAL NOT NULL,
                    PRIMARY KEY(run_id, artifact_id)
                );
                CREATE TABLE IF NOT EXISTS mlops_releases (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    manifest_artifact_id TEXT NOT NULL REFERENCES mlops_artifacts(id),
                    manifest_json TEXT NOT NULL,
                    sealed INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_deployments (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    release_id TEXT NOT NULL REFERENCES mlops_releases(id),
                    environment TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    desired_state TEXT NOT NULL,
                    observed_state TEXT NOT NULL,
                    revision TEXT,
                    traffic_percent REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_loop_pending_artifacts (
                    loop_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL REFERENCES mlops_artifacts(id),
                    source_run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    consumed_by_run_id TEXT REFERENCES mlops_runs(id),
                    created_at REAL NOT NULL,
                    PRIMARY KEY(loop_id, artifact_id)
                );
                CREATE TABLE IF NOT EXISTS mlops_task_approvals (
                    task_id TEXT PRIMARY KEY REFERENCES mlops_agent_tasks(id),
                    approved_by TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_loop_champions (
                    loop_id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL REFERENCES mlops_releases(id),
                    deployment_id TEXT NOT NULL REFERENCES mlops_deployments(id),
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_rollbacks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                    deployment_id TEXT NOT NULL REFERENCES mlops_deployments(id),
                    target_release_id TEXT NOT NULL REFERENCES mlops_releases(id),
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_loop_controls (
                    loop_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    reason TEXT,
                    stopped_by_run_id TEXT REFERENCES mlops_runs(id),
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mlops_scheduler_leases (
                    loop_id TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL,
                    holder_id TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                """
            )
            self._ensure_run_scoped_lineage_key(connection)
            connection.execute(
                'INSERT OR IGNORE INTO mlops_run_artifacts '
                '(run_id, artifact_id, introduced_by_attempt_id, created_at) '
                'SELECT created_by_run_id, id, created_by_attempt_id, created_at '
                'FROM mlops_artifacts'
            )

    @staticmethod
    def _ensure_run_scoped_lineage_key(connection: sqlite3.Connection) -> None:
        columns = connection.execute(
            'PRAGMA table_info(mlops_lineage_edges)'
        ).fetchall()
        run_id = next(row for row in columns if row['name'] == 'run_id')
        if run_id['pk']:
            return
        connection.executescript(
            """
            ALTER TABLE mlops_lineage_edges RENAME TO mlops_lineage_edges_legacy;
            CREATE TABLE mlops_lineage_edges (
                run_id TEXT NOT NULL REFERENCES mlops_runs(id),
                input_artifact_id TEXT NOT NULL REFERENCES mlops_artifacts(id),
                output_artifact_id TEXT NOT NULL REFERENCES mlops_artifacts(id),
                transform_kind TEXT NOT NULL,
                code_git_sha TEXT,
                container_image_digest TEXT,
                config_artifact_id TEXT,
                created_by_event_id TEXT NOT NULL REFERENCES mlops_events(id),
                PRIMARY KEY(
                    run_id,
                    input_artifact_id,
                    output_artifact_id,
                    transform_kind
                ),
                CHECK(input_artifact_id <> output_artifact_id)
            );
            INSERT INTO mlops_lineage_edges
            SELECT * FROM mlops_lineage_edges_legacy;
            DROP TABLE mlops_lineage_edges_legacy;
            """
        )

    def create_run(
        self,
        *,
        loop_id: str,
        watermark: str,
        spec: dict[str, Any],
        tasks: Sequence[TaskSpec],
        idempotency_key: str,
        initial_artifact_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not tasks:
            raise ValueError('a run requires at least one task')
        unresolved = self._find_placeholders(spec)
        if unresolved:
            raise ValueError(f'run spec contains unresolved placeholders: {unresolved}')
        run_id = _stable_id('run', loop_id, watermark)
        initial_artifact_ids = tuple(dict.fromkeys(initial_artifact_ids))
        payload = {
            'loop_id': loop_id,
            'watermark': watermark,
            'spec': spec,
            'tasks': [task.model_dump(mode='json') for task in tasks],
            'initial_artifact_ids': list(initial_artifact_ids),
        }
        now = time.time()
        with self._transaction() as connection:
            existing = connection.execute(
                'SELECT * FROM mlops_runs WHERE loop_id = ? AND watermark = ?',
                (loop_id, watermark),
            ).fetchone()
            if existing is not None:
                event = connection.execute(
                    'SELECT payload_digest FROM mlops_events '
                    'WHERE run_id = ? AND idempotency_key = ?',
                    (existing['id'], idempotency_key),
                ).fetchone()
                if event is None or event['payload_digest'] != _payload_digest(payload):
                    raise IdempotencyConflict(
                        f'watermark {watermark!r} already created with different input'
                    )
                return dict(existing)
            if initial_artifact_ids:
                rows = connection.execute(
                    f'SELECT id FROM mlops_artifacts WHERE id IN '
                    f'({",".join("?" for _ in initial_artifact_ids)}) AND sealed=1',
                    initial_artifact_ids,
                ).fetchall()
                found = {row['id'] for row in rows}
                missing = sorted(set(initial_artifact_ids) - found)
                if missing:
                    raise InvariantViolation(
                        'run references missing or unsealed initial artifacts: '
                        f'{missing}'
                    )
            connection.execute(
                'INSERT INTO mlops_runs '
                '(id, loop_id, watermark, spec_json, state, version, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, 0, ?, ?)',
                (
                    run_id,
                    loop_id,
                    watermark,
                    _canonical_json(spec),
                    RunState.CREATED,
                    now,
                    now,
                ),
            )
            event_id, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=0,
                event_type='RunCreated',
                actor_type='loop-controller',
                actor_id=loop_id,
                idempotency_key=idempotency_key,
                payload=payload,
            )
            for ordinal, task in enumerate(tasks):
                task_id = _stable_id('task', run_id, str(ordinal), task.name)
                connection.execute(
                    'INSERT INTO mlops_agent_tasks '
                    '(id, run_id, ordinal, spec_json, status, created_at, updated_at) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        task_id,
                        run_id,
                        ordinal,
                        _canonical_json(task.model_dump(mode='json')),
                        TaskStatus.PENDING,
                        now,
                        now,
                    ),
                )
            for artifact_id in initial_artifact_ids:
                connection.execute(
                    'INSERT INTO mlops_run_artifacts '
                    '(run_id, artifact_id, introduced_by_attempt_id, created_at) '
                    'VALUES (?, ?, NULL, ?)',
                    (run_id, artifact_id, now),
                )
            connection.execute(
                'UPDATE mlops_runs SET version = ?, updated_at = ? WHERE id = ?',
                (version, now, run_id),
            )
            if not event_id:
                raise AssertionError('RunCreated event was not appended')
            return dict(
                connection.execute(
                    'SELECT * FROM mlops_runs WHERE id = ?', (run_id,)
                ).fetchone()
            )

    def create_first_attempt(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt_spec: dict[str, Any],
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            task = self._task_for_update(connection, run_id, task_id)
            if task['attempt_count'] != 0 or task['status'] != TaskStatus.PENDING:
                existing = connection.execute(
                    'SELECT a.* FROM mlops_task_attempts a '
                    'WHERE a.task_id = ? AND a.attempt_number = 1',
                    (task_id,),
                ).fetchone()
                if existing is not None:
                    return dict(existing)
                raise InvalidTransition(
                    'first attempt can only be created for a pending task'
                )
            attempt_id = _stable_id('attempt', task_id, '1')
            payload = {
                'task_id': task_id,
                'attempt_id': attempt_id,
                'attempt_number': 1,
                'attempt_spec': attempt_spec,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskAttemptCreated',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            now = time.time()
            connection.execute(
                'INSERT INTO mlops_task_attempts '
                '(id, task_id, attempt_number, attempt_spec_json, status) '
                'VALUES (?, ?, 1, ?, ?)',
                (
                    attempt_id,
                    task_id,
                    _canonical_json(attempt_spec),
                    AttemptStatus.CREATED,
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET attempt_count = 1, updated_at = ? '
                'WHERE id = ?',
                (now, task_id),
            )
            # ``_append_event`` advances the version while preserving the current
            # lifecycle state. A later task's first attempt must not rewind the Run.
            created = dict(
                connection.execute(
                    'SELECT * FROM mlops_task_attempts WHERE id = ?', (attempt_id,)
                ).fetchone()
            )
            created['attempt_spec'] = json.loads(created.pop('attempt_spec_json'))
            return created

    def acquire_attempt_lease(
        self,
        *,
        run_id: str,
        attempt_id: str,
        agent_instance_id: str,
        ttl_seconds: int,
        expected_version: int,
        idempotency_key: str,
    ) -> str:
        now = time.time()
        with self._transaction() as connection:
            self._attempt_for_run(connection, run_id, attempt_id)
            current = connection.execute(
                'SELECT * FROM mlops_attempt_leases WHERE attempt_id = ?',
                (attempt_id,),
            ).fetchone()
            if current is not None and current['expires_at'] > now:
                event = self._idempotent_event(connection, run_id, idempotency_key)
                if event is not None:
                    return json.loads(event['payload_json'])['lease_id']
                raise LeaseConflict(f'attempt {attempt_id} already has an active lease')
            lease_id = f'lease_{uuid4().hex}'
            payload = {
                'attempt_id': attempt_id,
                'lease_id': lease_id,
                'agent_instance_id': agent_instance_id,
                'ttl_seconds': ttl_seconds,
            }
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='AttemptLeaseAcquired',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            connection.execute(
                'INSERT INTO mlops_attempt_leases '
                '(attempt_id, lease_id, agent_instance_id, acquired_at, heartbeat_at, expires_at) '
                'VALUES (?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(attempt_id) DO UPDATE SET '
                'lease_id=excluded.lease_id, agent_instance_id=excluded.agent_instance_id, '
                'acquired_at=excluded.acquired_at, heartbeat_at=excluded.heartbeat_at, '
                'expires_at=excluded.expires_at',
                (attempt_id, lease_id, agent_instance_id, now, now, now + ttl_seconds),
            )
            # Lease acquisition is an audit fact, not a lifecycle transition.
            return lease_id

    def dispatch_attempt(
        self,
        *,
        run_id: str,
        attempt_id: str,
        conversation_id: str,
        dispatch_key: str,
        lease_id: str,
        expected_version: int,
        idempotency_key: str,
    ) -> None:
        with self._transaction() as connection:
            attempt = self._attempt_for_run(connection, run_id, attempt_id)
            self._assert_lease(connection, attempt_id, lease_id)
            if attempt['conversation_id'] is not None:
                if (
                    attempt['conversation_id'] == conversation_id
                    and attempt['dispatch_key'] == dispatch_key
                ):
                    return
                raise IdempotencyConflict(
                    'attempt is linked to a different conversation'
                )
            payload = {
                'task_id': attempt['task_id'],
                'attempt_id': attempt_id,
                'conversation_id': conversation_id,
                'dispatch_key': dispatch_key,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskAttemptDispatched',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            now = time.time()
            connection.execute(
                'UPDATE mlops_task_attempts SET conversation_id=?, dispatch_key=?, '
                'status=?, started_at=? WHERE id=?',
                (
                    conversation_id,
                    dispatch_key,
                    AttemptStatus.RUNNING,
                    now,
                    attempt_id,
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                (TaskStatus.RUNNING, now, attempt['task_id']),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.WAITING_FOR_AGENT
            )

    def enqueue_conversation_dispatch(
        self,
        *,
        run_id: str,
        attempt_id: str,
        conversation_id: str,
        dispatch_key: str,
        lease_id: str,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Atomically persist dispatch intent before the OpenHands side effect."""

        outbox_id = _stable_id('outbox', attempt_id, dispatch_key)
        payload = {
            'attempt_id': attempt_id,
            'conversation_id': conversation_id,
            'dispatch_key': dispatch_key,
        }
        digest = _payload_digest(payload)
        with self._transaction() as connection:
            attempt = self._attempt_for_run(connection, run_id, attempt_id)
            self._assert_lease(connection, attempt_id, lease_id)
            existing = connection.execute(
                'SELECT * FROM mlops_outbox WHERE attempt_id=?', (attempt_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing['payload_digest'] != digest
                    or existing['idempotency_key'] != idempotency_key
                ):
                    raise IdempotencyConflict(
                        'attempt dispatch outbox was reused with a different payload'
                    )
                return self._decode_outbox(existing)
            if attempt['status'] != AttemptStatus.CREATED:
                raise InvalidTransition(
                    'conversation dispatch can only be queued for a created attempt'
                )
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='ConversationDispatchRequested',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload={'outbox_id': outbox_id, **payload},
            )
            now = time.time()
            connection.execute(
                'INSERT INTO mlops_outbox '
                '(id, run_id, attempt_id, kind, idempotency_key, payload_digest, '
                'payload_json, status, available_at, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    outbox_id,
                    run_id,
                    attempt_id,
                    'conversation-dispatch',
                    idempotency_key,
                    digest,
                    _canonical_json(payload),
                    'PENDING',
                    now,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                'SELECT * FROM mlops_outbox WHERE id=?', (outbox_id,)
            ).fetchone()
            assert row is not None
            return self._decode_outbox(row)

    def complete_conversation_dispatch(
        self,
        *,
        run_id: str,
        outbox_id: str,
        conversation_id: str,
        lease_id: str,
        expected_version: int,
    ) -> None:
        """Atomically link the Conversation and mark the outbox delivered."""

        with self._transaction() as connection:
            outbox = connection.execute(
                'SELECT * FROM mlops_outbox WHERE id=? AND run_id=?',
                (outbox_id, run_id),
            ).fetchone()
            if outbox is None:
                raise KeyError(outbox_id)
            payload = json.loads(outbox['payload_json'])
            attempt = self._attempt_for_run(connection, run_id, outbox['attempt_id'])
            self._assert_lease(connection, attempt['id'], lease_id)
            if outbox['status'] == 'DELIVERED':
                if outbox['external_id'] != conversation_id:
                    raise IdempotencyConflict(
                        'outbox was delivered to a different Conversation'
                    )
                return
            if conversation_id != payload['conversation_id']:
                raise InvariantViolation(
                    'dispatcher returned a Conversation other than the reserved id'
                )
            if attempt['status'] != AttemptStatus.CREATED:
                raise InvalidTransition('queued attempt is no longer dispatchable')
            event_payload = {
                'task_id': attempt['task_id'],
                'attempt_id': attempt['id'],
                'conversation_id': conversation_id,
                'dispatch_key': payload['dispatch_key'],
                'outbox_id': outbox_id,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskAttemptDispatched',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=f'outbox-delivered:{outbox_id}',
                payload=event_payload,
            )
            now = time.time()
            connection.execute(
                'UPDATE mlops_outbox SET status=?, external_id=?, '
                'delivery_attempts=delivery_attempts+1, last_error=NULL, updated_at=? '
                'WHERE id=?',
                ('DELIVERED', conversation_id, now, outbox_id),
            )
            connection.execute(
                'UPDATE mlops_task_attempts SET conversation_id=?, dispatch_key=?, '
                'status=?, started_at=? WHERE id=?',
                (
                    conversation_id,
                    payload['dispatch_key'],
                    AttemptStatus.RUNNING,
                    now,
                    attempt['id'],
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                (TaskStatus.RUNNING, now, attempt['task_id']),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.WAITING_FOR_AGENT
            )

    def record_conversation_dispatch_failure(
        self,
        *,
        run_id: str,
        outbox_id: str,
        error: str,
        expected_version: int,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            outbox = connection.execute(
                'SELECT * FROM mlops_outbox WHERE id=? AND run_id=?',
                (outbox_id, run_id),
            ).fetchone()
            if outbox is None:
                raise KeyError(outbox_id)
            if outbox['status'] == 'DELIVERED':
                return self._decode_outbox(outbox)
            delivery_attempts = outbox['delivery_attempts'] + 1
            backoff_seconds = min(2 ** (delivery_attempts - 1), 60)
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='ConversationDispatchFailed',
                actor_type='conversation-dispatcher',
                actor_id=outbox_id,
                idempotency_key=(f'outbox-failure:{outbox_id}:{delivery_attempts}'),
                payload={
                    'outbox_id': outbox_id,
                    'attempt_id': outbox['attempt_id'],
                    'delivery_attempt': delivery_attempts,
                    'backoff_seconds': backoff_seconds,
                    'error': error[:2000],
                },
            )
            now = time.time()
            connection.execute(
                'UPDATE mlops_outbox SET delivery_attempts=?, available_at=?, '
                'last_error=?, updated_at=? WHERE id=?',
                (
                    delivery_attempts,
                    now + backoff_seconds,
                    error[:2000],
                    now,
                    outbox_id,
                ),
            )
            row = connection.execute(
                'SELECT * FROM mlops_outbox WHERE id=?', (outbox_id,)
            ).fetchone()
            assert row is not None
            return self._decode_outbox(row)

    def submit_task_result(
        self,
        *,
        run_id: str,
        result: TaskResult,
        lease_id: str,
        expected_version: int,
        idempotency_key: str,
    ) -> None:
        with self._transaction() as connection:
            attempt = self._attempt_for_run(connection, run_id, result.attempt_id)
            self._assert_lease(connection, result.attempt_id, lease_id)
            if attempt['task_id'] != result.task_id:
                raise InvariantViolation('task result ids do not match the attempt')
            if attempt['status'] == AttemptStatus.RESULT_SUBMITTED:
                existing = TaskResult.model_validate_json(attempt['result_json'])
                if existing == result:
                    return
                raise IdempotencyConflict('attempt already has a different result')
            if attempt['status'] != AttemptStatus.RUNNING:
                raise InvalidTransition(
                    f'cannot submit result for attempt in {attempt["status"]}'
                )
            payload = result.model_dump(mode='json')
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskResultSubmitted',
                actor_type='task-agent',
                actor_id=attempt['conversation_id'],
                idempotency_key=idempotency_key,
                payload=payload,
            )
            now = time.time()
            connection.execute(
                'UPDATE mlops_task_attempts SET status=?, result_json=? WHERE id=?',
                (
                    AttemptStatus.RESULT_SUBMITTED,
                    result.model_dump_json(),
                    result.attempt_id,
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                (TaskStatus.VALIDATING, now, result.task_id),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.VALIDATING_RESULT
            )

    def yield_task_attempt(
        self,
        *,
        run_id: str,
        result: TaskResult,
        lease_id: str,
        running_job_ids: Sequence[str],
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Fence one Conversation while preserving its Attempt and running Jobs."""

        normalized_job_ids = sorted(set(running_job_ids))
        if not normalized_job_ids:
            raise InvariantViolation(
                'a task attempt can wait externally only for a running job'
            )
        with self._transaction() as connection:
            attempt = self._attempt_for_run(connection, run_id, result.attempt_id)
            self._assert_lease(connection, result.attempt_id, lease_id)
            if attempt['task_id'] != result.task_id:
                raise InvariantViolation('task result ids do not match the attempt')
            existing_event = self._idempotent_event(connection, run_id, idempotency_key)
            if existing_event is not None:
                payload = json.loads(existing_event['payload_json'])
                if (
                    existing_event['event_type'] != 'TaskAttemptYielded'
                    or payload.get('job_ids') != normalized_job_ids
                    or payload.get('source_status') != result.status
                    or payload.get('reason') != result.summary
                ):
                    raise IdempotencyConflict(
                        'task attempt yield was reused with a different payload'
                    )
                return payload
            if attempt['status'] != AttemptStatus.RUNNING:
                raise InvalidTransition(f'cannot yield attempt in {attempt["status"]}')
            rows = connection.execute(
                'SELECT id, status FROM mlops_job_attempts '
                'WHERE task_attempt_id=? ORDER BY id',
                (attempt['id'],),
            ).fetchall()
            persisted_running_job_ids = sorted(
                row['id'] for row in rows if row['status'] == 'running'
            )
            if persisted_running_job_ids != normalized_job_ids:
                raise InvariantViolation(
                    'waiting job ids do not match the running jobs owned by the attempt'
                )
            attempt_spec = json.loads(attempt['attempt_spec_json'])
            previous = attempt_spec.get('_continuation', {})
            previous_yield_count = (
                previous.get('yield_count', 0) if isinstance(previous, dict) else 0
            )
            if (
                not isinstance(previous_yield_count, int)
                or isinstance(previous_yield_count, bool)
                or previous_yield_count < 0
            ):
                raise InvariantViolation('attempt continuation count is invalid')
            yielded_at = time.time()
            continuation = {
                'yield_count': previous_yield_count + 1,
                'job_ids': normalized_job_ids,
                'reason': result.summary,
                'source_status': result.status,
                'previous_conversation_id': attempt['conversation_id'],
                'yielded_at': yielded_at,
            }
            attempt_spec['_continuation'] = continuation
            payload = {
                'task_id': attempt['task_id'],
                'attempt_id': attempt['id'],
                **continuation,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskAttemptYielded',
                actor_type='task-agent',
                actor_id=attempt['conversation_id'],
                idempotency_key=idempotency_key,
                payload=payload,
            )
            connection.execute(
                'UPDATE mlops_task_attempts SET status=?, attempt_spec_json=? '
                'WHERE id=?',
                (
                    AttemptStatus.WAITING_EXTERNAL,
                    _canonical_json(attempt_spec),
                    attempt['id'],
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                (TaskStatus.WAITING_EXTERNAL, yielded_at, attempt['task_id']),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.WAITING_EXTERNAL
            )
            return payload

    def rotate_attempt_lease_for_continuation(
        self,
        *,
        run_id: str,
        attempt_id: str,
        agent_instance_id: str,
        ttl_seconds: int,
        expected_version: int,
        idempotency_key: str,
    ) -> tuple[str, int]:
        """Issue a new fence token for the next Conversation of one Attempt."""

        if ttl_seconds < 1:
            raise ValueError('continuation lease TTL must be positive')
        now = time.time()
        with self._transaction() as connection:
            attempt = self._attempt_for_run(connection, run_id, attempt_id)
            if attempt['status'] != AttemptStatus.WAITING_EXTERNAL:
                raise InvalidTransition(
                    'a continuation lease requires a waiting task attempt'
                )
            attempt_spec = json.loads(attempt['attempt_spec_json'])
            continuation = attempt_spec.get('_continuation')
            if not isinstance(continuation, dict):
                raise InvariantViolation('waiting attempt has no continuation state')
            yield_count = continuation.get('yield_count')
            if not isinstance(yield_count, int) or isinstance(yield_count, bool):
                raise InvariantViolation('attempt continuation count is invalid')
            rotation_events = connection.execute(
                'SELECT * FROM mlops_events WHERE run_id=? '
                'AND event_type=? ORDER BY sequence',
                (run_id, 'AttemptContinuationLeaseRotated'),
            ).fetchall()
            matching_rotations = []
            for event in rotation_events:
                payload = json.loads(event['payload_json'])
                if (
                    payload.get('attempt_id') == attempt_id
                    and payload.get('yield_count') == yield_count
                ):
                    matching_rotations.append((event, payload))
            current_lease = connection.execute(
                'SELECT * FROM mlops_attempt_leases WHERE attempt_id=?',
                (attempt_id,),
            ).fetchone()
            if current_lease is not None and current_lease['expires_at'] > now:
                for generation, (_, payload) in enumerate(matching_rotations):
                    if payload.get('lease_id') == current_lease['lease_id']:
                        return current_lease['lease_id'], generation
            generation = len(matching_rotations)
            effective_idempotency_key = idempotency_key
            if matching_rotations:
                effective_idempotency_key = f'{idempotency_key}:renew:{generation}'
            existing_event = self._idempotent_event(
                connection, run_id, effective_idempotency_key
            )
            if existing_event is not None:
                if existing_event['event_type'] != 'AttemptContinuationLeaseRotated':
                    raise IdempotencyConflict(
                        'continuation lease key was reused by another event'
                    )
                payload = json.loads(existing_event['payload_json'])
                return payload['lease_id'], payload.get('generation', generation)
            lease_id = f'lease_{uuid4().hex}'
            payload = {
                'attempt_id': attempt_id,
                'lease_id': lease_id,
                'agent_instance_id': agent_instance_id,
                'ttl_seconds': ttl_seconds,
                'yield_count': yield_count,
                'generation': generation,
            }
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='AttemptContinuationLeaseRotated',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=effective_idempotency_key,
                payload=payload,
            )
            connection.execute(
                'INSERT INTO mlops_attempt_leases '
                '(attempt_id, lease_id, agent_instance_id, acquired_at, heartbeat_at, expires_at) '
                'VALUES (?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(attempt_id) DO UPDATE SET '
                'lease_id=excluded.lease_id, agent_instance_id=excluded.agent_instance_id, '
                'acquired_at=excluded.acquired_at, heartbeat_at=excluded.heartbeat_at, '
                'expires_at=excluded.expires_at',
                (attempt_id, lease_id, agent_instance_id, now, now, now + ttl_seconds),
            )
            return lease_id, generation

    def continue_task_attempt(
        self,
        *,
        run_id: str,
        attempt_id: str,
        conversation_id: str,
        dispatch_key: str,
        lease_id: str,
        expected_version: int,
        idempotency_key: str,
    ) -> None:
        """Attach a new Conversation episode without allocating a new Attempt."""

        with self._transaction() as connection:
            attempt = self._attempt_for_run(connection, run_id, attempt_id)
            self._assert_lease(connection, attempt_id, lease_id)
            existing_event = self._idempotent_event(connection, run_id, idempotency_key)
            if existing_event is not None:
                payload = json.loads(existing_event['payload_json'])
                if (
                    existing_event['event_type'] != 'TaskAttemptContinued'
                    or payload.get('conversation_id') != conversation_id
                    or payload.get('dispatch_key') != dispatch_key
                    or payload.get('lease_id') != lease_id
                ):
                    raise IdempotencyConflict(
                        'task continuation was reused with a different payload'
                    )
                return
            if attempt['status'] != AttemptStatus.WAITING_EXTERNAL:
                raise InvalidTransition(
                    'only a waiting task attempt can continue in a new Conversation'
                )
            attempt_spec = json.loads(attempt['attempt_spec_json'])
            continuation = attempt_spec.get('_continuation')
            if not isinstance(continuation, dict):
                raise InvariantViolation('waiting attempt has no continuation state')
            yield_count = continuation.get('yield_count')
            if not isinstance(yield_count, int) or isinstance(yield_count, bool):
                raise InvariantViolation('attempt continuation count is invalid')
            payload = {
                'task_id': attempt['task_id'],
                'attempt_id': attempt_id,
                'previous_conversation_id': attempt['conversation_id'],
                'conversation_id': conversation_id,
                'dispatch_key': dispatch_key,
                'lease_id': lease_id,
                'yield_count': yield_count,
                'job_ids': continuation.get('job_ids', []),
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskAttemptContinued',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            now = time.time()
            connection.execute(
                'UPDATE mlops_task_attempts SET conversation_id=?, dispatch_key=?, '
                'status=? WHERE id=?',
                (
                    conversation_id,
                    dispatch_key,
                    AttemptStatus.RUNNING,
                    attempt_id,
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                (TaskStatus.RUNNING, now, attempt['task_id']),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.WAITING_FOR_AGENT
            )

    def record_cost(
        self,
        *,
        run_id: str,
        source_type: str,
        source_id: str,
        amount_usd: float,
        metadata: dict[str, Any],
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not math.isfinite(amount_usd) or amount_usd < 0:
            raise ValueError('cost amount must be finite and non-negative')
        cost_id = _stable_id('cost', run_id, idempotency_key)
        payload = {
            'cost_id': cost_id,
            'source_type': source_type,
            'source_id': source_id,
            'amount_usd': amount_usd,
            'metadata': metadata,
        }
        with self._transaction() as connection:
            existing = connection.execute(
                'SELECT * FROM mlops_cost_entries WHERE run_id=? AND idempotency_key=?',
                (run_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                existing_payload = {
                    'cost_id': existing['id'],
                    'source_type': existing['source_type'],
                    'source_id': existing['source_id'],
                    'amount_usd': existing['amount_usd'],
                    'metadata': json.loads(existing['metadata_json']),
                }
                if _payload_digest(existing_payload) != _payload_digest(payload):
                    raise IdempotencyConflict(
                        'cost idempotency key was reused with different input'
                    )
                return self._decode_cost_entry(existing)
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='CostRecorded',
                actor_type='cost-adapter',
                actor_id=source_id,
                idempotency_key=f'cost:{idempotency_key}',
                payload=payload,
            )
            connection.execute(
                'INSERT INTO mlops_cost_entries '
                '(id, run_id, source_type, source_id, amount_usd, metadata_json, '
                'idempotency_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    cost_id,
                    run_id,
                    source_type,
                    source_id,
                    amount_usd,
                    _canonical_json(metadata),
                    idempotency_key,
                    time.time(),
                ),
            )
            row = connection.execute(
                'SELECT * FROM mlops_cost_entries WHERE id=?', (cost_id,)
            ).fetchone()
            assert row is not None
            return self._decode_cost_entry(row)

    def stop_run_for_policy(
        self,
        *,
        run_id: str,
        reason: str,
        policy: str,
        expected_version: int,
        idempotency_key: str,
    ) -> bool:
        """Fence active attempts and stop a Run at a hard outer-loop budget."""

        with self._transaction() as connection:
            run = connection.execute(
                'SELECT * FROM mlops_runs WHERE id=?', (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            if RunState(run['state']) in {
                RunState.COMPLETED,
                RunState.COMPLETED_NO_TRAINING,
                RunState.NEEDS_INTERVENTION,
                RunState.FAILED,
                RunState.CANCELLED,
            }:
                return False
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='RunPolicyStopped',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload={
                    'policy': policy,
                    'reason': reason,
                    'run_state_after': RunState.NEEDS_INTERVENTION,
                },
            )
            now = time.time()
            active_attempt_ids = [
                row['id']
                for row in connection.execute(
                    'SELECT a.id FROM mlops_task_attempts a '
                    'JOIN mlops_agent_tasks t ON t.id=a.task_id '
                    'WHERE t.run_id=? AND a.status IN (?, ?, ?, ?)',
                    (
                        run_id,
                        AttemptStatus.CREATED,
                        AttemptStatus.DISPATCHED,
                        AttemptStatus.RUNNING,
                        AttemptStatus.WAITING_EXTERNAL,
                    ),
                ).fetchall()
            ]
            if active_attempt_ids:
                placeholders = ','.join('?' for _ in active_attempt_ids)
                connection.execute(
                    f'UPDATE mlops_task_attempts SET status=?, finished_at=? '
                    f'WHERE id IN ({placeholders})',
                    (AttemptStatus.CANCELLED, now, *active_attempt_ids),
                )
                connection.execute(
                    f'DELETE FROM mlops_attempt_leases '
                    f'WHERE attempt_id IN ({placeholders})',
                    active_attempt_ids,
                )
                connection.execute(
                    f'UPDATE mlops_outbox SET status=?, last_error=?, updated_at=? '
                    f'WHERE attempt_id IN ({placeholders}) AND status=?',
                    (
                        'CANCELLED',
                        reason[:2000],
                        now,
                        *active_attempt_ids,
                        'PENDING',
                    ),
                )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, updated_at=? '
                'WHERE run_id=? AND status NOT IN (?, ?, ?)',
                (
                    TaskStatus.CANCELLED,
                    now,
                    run_id,
                    TaskStatus.SUCCEEDED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                ),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.NEEDS_INTERVENTION
            )
            return True

    def accept_task_result(
        self,
        *,
        run_id: str,
        result: TaskResult,
        validation: ValidationResult,
        sealed_candidates: Sequence[ArtifactCandidate],
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, str]:
        if validation.disposition is not ValidationDisposition.ACCEPTED:
            raise ValueError('accept_task_result requires an accepted validation')
        with self._transaction() as connection:
            self._submitted_attempt(connection, run_id, result)
            task = self._task_for_update(connection, run_id, result.task_id)
            accepted = set(validation.accepted_candidate_ids)
            supplied = {item.candidate_id for item in sealed_candidates}
            if accepted != supplied:
                raise InvariantViolation(
                    'sealed candidates must exactly match validator acceptance'
                )
            task_spec = TaskSpec.model_validate_json(task['spec_json'])
            if task_spec.kind == 'inspect-data':
                run_row = connection.execute(
                    'SELECT spec_json FROM mlops_runs WHERE id=?', (run_id,)
                ).fetchone()
                if run_row is None:
                    raise KeyError(run_id)
                run_spec = json.loads(run_row['spec_json'])
                source_spec = run_spec.get('spec', {}).get('source', {})
                expected_source_id = source_spec.get('manifestArtifactId')
                if expected_source_id is not None:
                    sources = [
                        item
                        for item in sealed_candidates
                        if item.kind == 'source-manifest'
                    ]
                    if len(sources) != 1:
                        raise InvariantViolation(
                            'inspect-data must retain exactly one discovered '
                            'source manifest'
                        )
                    actual_source_id = f'source-manifest:sha256:{sources[0].sha256}'
                    if actual_source_id != expected_source_id:
                        raise InvariantViolation(
                            'inspect-data source manifest differs from the '
                            'outer-loop discovery snapshot'
                        )
                    raw_candidate_ids = {
                        item.candidate_id
                        for item in sealed_candidates
                        if item.kind == 'raw-dataset'
                    }
                    if not raw_candidate_ids or not any(
                        proposal.input_artifact_id == expected_source_id
                        and proposal.output_candidate_id in raw_candidate_ids
                        for proposal in result.lineage_proposals
                    ):
                        raise InvariantViolation(
                            'raw dataset lineage must start at the discovered '
                            'source manifest artifact'
                        )
            skip_training = result.metrics.get('skip_training') is True
            if skip_training and not task_spec.completion_contract.allow_skip_training:
                raise InvariantViolation('task is not authorized to skip training')
            is_last_task = self._is_last_task(connection, run_id, task['ordinal'])
            if skip_training:
                run_state_after = RunState.COMPLETED_NO_TRAINING
                active_task_ordinal_after = (
                    self._max_task_ordinal(connection, run_id) + 1
                )
            else:
                run_state_after = (
                    RunState.COMPLETED if is_last_task else RunState.WAITING_FOR_AGENT
                )
                active_task_ordinal_after = task['ordinal'] + 1
            payload = {
                'task_id': result.task_id,
                'attempt_id': result.attempt_id,
                'validation': validation.model_dump(mode='json'),
                'run_state_after': run_state_after,
                'active_task_ordinal_after': active_task_ordinal_after,
                'skip_training': skip_training,
            }
            event_id, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskResultValidated',
                actor_type='result-validator',
                actor_id=task['id'],
                idempotency_key=idempotency_key,
                payload=payload,
            )
            candidate_to_artifact: dict[str, str] = {}
            now = time.time()
            for candidate in sealed_candidates:
                artifact_id = f'{candidate.kind}:sha256:{candidate.sha256}'
                existing = connection.execute(
                    'SELECT * FROM mlops_artifacts WHERE kind=? AND sha256=?',
                    (candidate.kind, candidate.sha256),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        'INSERT INTO mlops_artifacts '
                        '(id, kind, uri, sha256, size_bytes, media_type, metadata_json, '
                        'sealed, created_by_run_id, created_by_attempt_id, created_at) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)',
                        (
                            artifact_id,
                            candidate.kind,
                            candidate.uri,
                            candidate.sha256,
                            candidate.size_bytes,
                            candidate.media_type,
                            _canonical_json(candidate.metadata),
                            run_id,
                            result.attempt_id,
                            now,
                        ),
                    )
                elif not existing['sealed']:
                    raise InvariantViolation(
                        f'artifact {artifact_id} is unexpectedly unsealed'
                    )
                candidate_to_artifact[candidate.candidate_id] = artifact_id
                connection.execute(
                    'INSERT OR IGNORE INTO mlops_run_artifacts '
                    '(run_id, artifact_id, introduced_by_attempt_id, created_at) '
                    'VALUES (?, ?, ?, ?)',
                    (run_id, artifact_id, result.attempt_id, now),
                )
            for proposal in result.lineage_proposals:
                output_id = candidate_to_artifact.get(proposal.output_candidate_id)
                if output_id is None:
                    raise InvariantViolation(
                        f'lineage output was not accepted: {proposal.output_candidate_id}'
                    )
                input_id = candidate_to_artifact.get(
                    proposal.input_artifact_id, proposal.input_artifact_id
                )
                if input_id == output_id:
                    raise InvariantViolation('lineage self-edges are not allowed')
                if (
                    connection.execute(
                        'SELECT 1 FROM mlops_run_artifacts '
                        'WHERE run_id=? AND artifact_id=?',
                        (run_id, input_id),
                    ).fetchone()
                    is None
                ):
                    raise InvariantViolation(
                        f'lineage input is not sealed into this Run: {input_id}'
                    )
                if (
                    proposal.config_artifact_id is not None
                    and connection.execute(
                        'SELECT 1 FROM mlops_run_artifacts '
                        'WHERE run_id=? AND artifact_id=?',
                        (run_id, proposal.config_artifact_id),
                    ).fetchone()
                    is None
                ):
                    raise InvariantViolation(
                        'lineage config artifact is not sealed into this Run: '
                        f'{proposal.config_artifact_id}'
                    )
                connection.execute(
                    'INSERT OR IGNORE INTO mlops_lineage_edges '
                    '(run_id, input_artifact_id, output_artifact_id, transform_kind, '
                    'code_git_sha, container_image_digest, config_artifact_id, '
                    'created_by_event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        run_id,
                        input_id,
                        output_id,
                        proposal.transform_kind,
                        proposal.code_git_sha,
                        proposal.container_image_digest,
                        proposal.config_artifact_id,
                        event_id,
                    ),
                )
            if task_spec.kind == 'verify-board':
                release_candidates = [
                    candidate
                    for candidate in sealed_candidates
                    if candidate.kind == 'release-manifest'
                ]
                if len(release_candidates) != 1:
                    raise InvariantViolation(
                        'verify-board must produce exactly one release-manifest'
                    )
                release_candidate = release_candidates[0]
                release_manifest_id = candidate_to_artifact[
                    release_candidate.candidate_id
                ]
                release_document = self._read_local_json(
                    release_candidate.validation_uri or release_candidate.uri
                )
                release_kinds = [
                    item['kind'] for item in release_document.get('artifacts', [])
                ]
                if len(release_kinds) != len(set(release_kinds)):
                    raise InvariantViolation(
                        'release manifest contains duplicate artifact kinds'
                    )
                missing_kinds = sorted(
                    RIGHTARM_RELEASE_REQUIRED_KINDS - set(release_kinds)
                )
                if missing_kinds:
                    raise InvariantViolation(
                        'release manifest is missing production artifacts: '
                        f'{missing_kinds}'
                    )
                release_artifact_ids = [
                    f'{item["kind"]}:sha256:{item["sha256"]}'
                    for item in release_document.get('artifacts', [])
                ]
                declared = list(
                    dict.fromkeys([*release_artifact_ids, release_manifest_id])
                )
                rows = connection.execute(
                    f'SELECT * FROM mlops_artifacts WHERE id IN '
                    f'({",".join("?" for _ in declared)}) AND sealed=1',
                    declared,
                ).fetchall()
                release_artifacts = {row['id']: row for row in rows}
                missing = sorted(set(declared) - set(release_artifacts))
                if missing:
                    raise InvariantViolation(
                        f'release manifest references missing artifacts: {missing}'
                    )
                self._assert_provenance_closure(
                    connection,
                    run_id=run_id,
                    artifact_ids=declared,
                    artifacts=release_artifacts,
                    root_kinds=frozenset(
                        {
                            'source-manifest',
                            'raw-dataset',
                            'base-model',
                            'code',
                            'container-image',
                            'training-config',
                            'label-mapping',
                            'tensor-contract',
                        }
                    ),
                )
                release_id = _stable_id('release', release_candidate.sha256)
                domain_manifest = {
                    'protocol': 'mlops.openhands.dev/release/v1',
                    'manifest_artifact_id': release_manifest_id,
                    'artifact_ids': sorted(declared),
                    'validation_id': validation.validation_id,
                }
                _, version = self._append_event(
                    connection,
                    run_id=run_id,
                    expected_version=version,
                    event_type='ReleaseSealed',
                    actor_type='loop-controller',
                    actor_id='loop-controller',
                    idempotency_key=f'{idempotency_key}:seal-release',
                    payload={'release_id': release_id, **domain_manifest},
                )
                connection.execute(
                    'INSERT INTO mlops_releases '
                    '(id, run_id, manifest_artifact_id, manifest_json, sealed, created_at) '
                    'VALUES (?, ?, ?, ?, 1, ?)',
                    (
                        release_id,
                        run_id,
                        release_manifest_id,
                        _canonical_json(domain_manifest),
                        now,
                    ),
                )
            if task_spec.kind in {'deploy', 'monitor', 'rollback'}:
                observation_candidates = [
                    candidate
                    for candidate in sealed_candidates
                    if candidate.kind == 'deployment-observation'
                ]
                if len(observation_candidates) != 1:
                    raise InvariantViolation(
                        f'{task_spec.kind} must produce exactly one '
                        'deployment-observation'
                    )
                observation = self._read_deployment_observation(
                    observation_candidates[0].validation_uri
                    or observation_candidates[0].uri
                )
                deployment = connection.execute(
                    'SELECT * FROM mlops_deployments WHERE run_id=? '
                    'ORDER BY created_at DESC LIMIT 1',
                    (run_id,),
                ).fetchone()
                if deployment is None:
                    raise InvariantViolation(
                        f'{task_spec.kind} requires an outer-loop deployment record'
                    )
                if observation['deployment_id'] != deployment['id']:
                    raise InvariantViolation(
                        'deployment observation belongs to a different deployment'
                    )
                observation_payload = {
                    'deployment_id': deployment['id'],
                    'observed_state': observation['observed_state'],
                    'revision': observation['revision'],
                    'metrics': observation['metrics'],
                    'artifact_id': candidate_to_artifact[
                        observation_candidates[0].candidate_id
                    ],
                    'domain_state_advanced': False,
                }
                _, version = self._append_event(
                    connection,
                    run_id=run_id,
                    expected_version=version,
                    event_type='DeploymentObserved',
                    actor_type='result-validator',
                    actor_id=task['id'],
                    idempotency_key=f'{idempotency_key}:deployment-observed',
                    payload=observation_payload,
                )
                connection.execute(
                    'UPDATE mlops_deployments SET observed_state=?, revision=?, '
                    'updated_at=? WHERE id=?',
                    (
                        observation['observed_state'],
                        observation['revision'],
                        now,
                        deployment['id'],
                    ),
                )
                if (
                    task_spec.kind == 'monitor'
                    and observation['observed_state'] == 'HEALTHY'
                ):
                    promotion_payload = {
                        'deployment_id': deployment['id'],
                        'target_traffic_percent': 100.0,
                        'validation_id': validation.validation_id,
                        'evidence_refs': result.evidence_refs,
                    }
                    _, version = self._append_event(
                        connection,
                        run_id=run_id,
                        expected_version=version,
                        event_type='TrafficShiftDecided',
                        actor_type='loop-controller',
                        actor_id='loop-controller',
                        idempotency_key=f'{idempotency_key}:promote',
                        payload=promotion_payload,
                    )
                    connection.execute(
                        'UPDATE mlops_deployments SET desired_state=?, '
                        'traffic_percent=?, updated_at=? WHERE id=?',
                        ('PROMOTED', 100.0, now, deployment['id']),
                    )
                    loop_id = connection.execute(
                        'SELECT loop_id FROM mlops_runs WHERE id=?', (run_id,)
                    ).fetchone()['loop_id']
                    connection.execute(
                        'INSERT INTO mlops_loop_champions '
                        '(loop_id, release_id, deployment_id, updated_at) '
                        'VALUES (?, ?, ?, ?) ON CONFLICT(loop_id) DO UPDATE SET '
                        'release_id=excluded.release_id, '
                        'deployment_id=excluded.deployment_id, '
                        'updated_at=excluded.updated_at',
                        (loop_id, deployment['release_id'], deployment['id'], now),
                    )
                elif task_spec.kind == 'rollback':
                    rollback = connection.execute(
                        'SELECT * FROM mlops_rollbacks WHERE run_id=? AND '
                        'deployment_id=? AND status=? ORDER BY created_at DESC LIMIT 1',
                        (run_id, deployment['id'], 'REQUESTED'),
                    ).fetchone()
                    if rollback is None:
                        raise InvariantViolation(
                            'rollback task has no outer-loop rollback intention'
                        )
                    if (
                        observation.get('active_release_id')
                        != rollback['target_release_id']
                    ):
                        raise InvariantViolation(
                            'rollback observation does not prove the champion is active'
                        )
                    _, version = self._append_event(
                        connection,
                        run_id=run_id,
                        expected_version=version,
                        event_type='RollbackObserved',
                        actor_type='result-validator',
                        actor_id=task['id'],
                        idempotency_key=f'{idempotency_key}:rollback-observed',
                        payload={
                            'rollback_id': rollback['id'],
                            'deployment_id': deployment['id'],
                            'active_release_id': rollback['target_release_id'],
                            'validation_id': validation.validation_id,
                            'artifact_id': candidate_to_artifact[
                                observation_candidates[0].candidate_id
                            ],
                        },
                    )
                    connection.execute(
                        'UPDATE mlops_rollbacks SET status=?, updated_at=? WHERE id=?',
                        ('SUCCEEDED', now, rollback['id']),
                    )
                    connection.execute(
                        'UPDATE mlops_deployments SET desired_state=?, '
                        'traffic_percent=?, updated_at=? WHERE id=?',
                        ('ROLLED_BACK', 0.0, now, deployment['id']),
                    )
            connection.execute(
                'UPDATE mlops_task_attempts SET status=?, validator_result_json=?, '
                'finished_at=? WHERE id=?',
                (
                    AttemptStatus.ACCEPTED,
                    validation.model_dump_json(),
                    now,
                    result.attempt_id,
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                (TaskStatus.SUCCEEDED, now, result.task_id),
            )
            if skip_training:
                connection.execute(
                    'UPDATE mlops_agent_tasks SET status=?, updated_at=? '
                    'WHERE run_id=? AND ordinal>? AND status=?',
                    (
                        TaskStatus.CANCELLED,
                        now,
                        run_id,
                        task['ordinal'],
                        TaskStatus.PENDING,
                    ),
                )
                loop_id = connection.execute(
                    'SELECT loop_id FROM mlops_runs WHERE id=?', (run_id,)
                ).fetchone()['loop_id']
                for artifact_id in candidate_to_artifact.values():
                    connection.execute(
                        'INSERT OR IGNORE INTO mlops_loop_pending_artifacts '
                        '(loop_id, artifact_id, source_run_id, created_at) '
                        'VALUES (?, ?, ?, ?)',
                        (loop_id, artifact_id, run_id, now),
                    )
            elif task_spec.kind == 'train':
                run_spec = json.loads(
                    connection.execute(
                        'SELECT spec_json FROM mlops_runs WHERE id=?', (run_id,)
                    ).fetchone()['spec_json']
                )
                pending_ids = run_spec.get('spec', {}).get('pending_artifact_ids', [])
                if pending_ids:
                    connection.execute(
                        f'UPDATE mlops_loop_pending_artifacts SET consumed_by_run_id=? '
                        f'WHERE artifact_id IN '
                        f'({",".join("?" for _ in pending_ids)}) '
                        f'AND consumed_by_run_id IS NULL',
                        (run_id, *pending_ids),
                    )
            self._set_run_projection(
                connection,
                run_id,
                version,
                run_state_after,
                active_task_ordinal=active_task_ordinal_after,
            )
            return candidate_to_artifact

    def reject_task_result(
        self,
        *,
        run_id: str,
        result: TaskResult,
        validation: ValidationResult,
        retry_decision: RetryDecision,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        if validation.disposition is ValidationDisposition.ACCEPTED:
            raise ValueError('accepted results use accept_task_result')
        with self._transaction() as connection:
            attempt = self._submitted_attempt(connection, run_id, result)
            task = self._task_for_update(connection, run_id, result.task_id)
            after_rejection = (
                RunState.RETRY_SCHEDULED
                if retry_decision.retry
                else RunState.NEEDS_INTERVENTION
            )
            payload = {
                'task_id': result.task_id,
                'attempt_id': result.attempt_id,
                'validation': validation.model_dump(mode='json'),
                'retry': retry_decision.retry,
                'run_state_after': after_rejection,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskResultRejected',
                actor_type='result-validator',
                actor_id=task['id'],
                idempotency_key=f'{idempotency_key}:validation',
                payload=payload,
            )
            now = time.time()
            connection.execute(
                'UPDATE mlops_task_attempts SET status=?, validator_result_json=?, '
                'failure_class=?, finished_at=? WHERE id=?',
                (
                    AttemptStatus.REJECTED,
                    validation.model_dump_json(),
                    retry_decision.failure_class,
                    now,
                    result.attempt_id,
                ),
            )
            if not retry_decision.retry:
                terminal_status = (
                    TaskStatus.NEEDS_DECISION
                    if validation.disposition is ValidationDisposition.NEEDS_DECISION
                    else TaskStatus.FAILED
                )
                connection.execute(
                    'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                    (terminal_status, now, task['id']),
                )
                self._set_run_projection(
                    connection, run_id, version, RunState.NEEDS_INTERVENTION
                )
                return None
            next_number = retry_decision.next_attempt_number
            if next_number is None or next_number != task['attempt_count'] + 1:
                raise InvariantViolation(
                    'retry attempt number is not strictly increasing'
                )
            attempt_id = _stable_id('attempt', task['id'], str(next_number))
            retry_payload = {
                'failed_attempt_id': result.attempt_id,
                'validation_id': validation.validation_id,
                'reason': retry_decision.reason,
                'failure_class': retry_decision.failure_class,
                'spec_diff': retry_decision.spec_diff,
                'next_attempt_id': attempt_id,
                'next_attempt_number': next_number,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=version,
                event_type='RetryDecided',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=f'{idempotency_key}:retry',
                payload=retry_payload,
            )
            old_spec = json.loads(attempt['attempt_spec_json'])
            next_spec = self._deep_merge(old_spec, retry_decision.spec_diff)
            connection.execute(
                'INSERT INTO mlops_task_attempts '
                '(id, task_id, attempt_number, attempt_spec_json, status) '
                'VALUES (?, ?, ?, ?, ?)',
                (
                    attempt_id,
                    task['id'],
                    next_number,
                    _canonical_json(next_spec),
                    AttemptStatus.CREATED,
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, attempt_count=?, updated_at=? '
                'WHERE id=?',
                (TaskStatus.RETRY_SCHEDULED, next_number, now, task['id']),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.RETRY_SCHEDULED
            )
            return dict(
                connection.execute(
                    'SELECT * FROM mlops_task_attempts WHERE id=?', (attempt_id,)
                ).fetchone()
            )

    def cancel_run(
        self,
        run_id: str,
        *,
        expected_version: int,
        reason: str,
        idempotency_key: str,
    ) -> None:
        with self._transaction() as connection:
            payload = {'reason': reason}
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='RunCancelled',
                actor_type='user',
                actor_id='user',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=? WHERE run_id=? AND status NOT IN (?, ?)',
                (TaskStatus.CANCELLED, run_id, TaskStatus.SUCCEEDED, TaskStatus.FAILED),
            )
            active_attempt_ids = [
                row['id']
                for row in connection.execute(
                    'SELECT a.id FROM mlops_task_attempts a '
                    'JOIN mlops_agent_tasks t ON t.id=a.task_id '
                    'WHERE t.run_id=? AND a.status IN (?, ?, ?, ?)',
                    (
                        run_id,
                        AttemptStatus.CREATED,
                        AttemptStatus.DISPATCHED,
                        AttemptStatus.RUNNING,
                        AttemptStatus.WAITING_EXTERNAL,
                    ),
                ).fetchall()
            ]
            if active_attempt_ids:
                placeholders = ','.join('?' for _ in active_attempt_ids)
                now = time.time()
                connection.execute(
                    f'UPDATE mlops_task_attempts SET status=?, finished_at=? '
                    f'WHERE id IN ({placeholders})',
                    (AttemptStatus.CANCELLED, now, *active_attempt_ids),
                )
                connection.execute(
                    f'DELETE FROM mlops_attempt_leases '
                    f'WHERE attempt_id IN ({placeholders})',
                    active_attempt_ids,
                )
            self._set_run_projection(connection, run_id, version, RunState.CANCELLED)

    def recover_expired_attempt(
        self,
        *,
        run_id: str,
        attempt_id: str,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Fence a lost Conversation and, within budget, create a new attempt."""

        with self._transaction() as connection:
            attempt = self._attempt_for_run(connection, run_id, attempt_id)
            task = self._task_for_update(connection, run_id, attempt['task_id'])
            lease = connection.execute(
                'SELECT * FROM mlops_attempt_leases WHERE attempt_id=?',
                (attempt_id,),
            ).fetchone()
            if lease is None or lease['expires_at'] > time.time():
                return None
            if attempt['status'] != AttemptStatus.RUNNING:
                return None
            task_spec = TaskSpec.model_validate_json(task['spec_json'])
            retry = attempt['attempt_number'] < task_spec.max_attempts
            after_crash = (
                RunState.RETRY_SCHEDULED if retry else RunState.NEEDS_INTERVENTION
            )
            crash_payload = {
                'task_id': task['id'],
                'attempt_id': attempt_id,
                'conversation_id': attempt['conversation_id'],
                'failure_class': 'agent-conversation-lost',
                'evidence_refs': [
                    f'lease:{lease["lease_id"]}:expired:{lease["expires_at"]}'
                ],
                'run_state_after': after_crash,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TaskAttemptCrashed',
                actor_type='loop-reconciler',
                actor_id='loop-reconciler',
                idempotency_key=f'{idempotency_key}:crashed',
                payload=crash_payload,
            )
            now = time.time()
            connection.execute(
                'UPDATE mlops_task_attempts SET status=?, failure_class=?, finished_at=? '
                'WHERE id=?',
                (
                    AttemptStatus.CRASHED,
                    'agent-conversation-lost',
                    now,
                    attempt_id,
                ),
            )
            if not retry:
                connection.execute(
                    'UPDATE mlops_agent_tasks SET status=?, updated_at=? WHERE id=?',
                    (TaskStatus.FAILED, now, task['id']),
                )
                self._set_run_projection(
                    connection, run_id, version, RunState.NEEDS_INTERVENTION
                )
                return None
            next_number = attempt['attempt_number'] + 1
            next_attempt_id = _stable_id('attempt', task['id'], str(next_number))
            retry_payload = {
                'failed_attempt_id': attempt_id,
                'validation_id': 'reconciler:expired-attempt-lease',
                'reason': 'Conversation heartbeat lease expired',
                'failure_class': 'agent-conversation-lost',
                'spec_diff': {},
                'next_attempt_id': next_attempt_id,
                'next_attempt_number': next_number,
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=version,
                event_type='RetryDecided',
                actor_type='loop-reconciler',
                actor_id='loop-reconciler',
                idempotency_key=f'{idempotency_key}:retry',
                payload=retry_payload,
            )
            connection.execute(
                'INSERT INTO mlops_task_attempts '
                '(id, task_id, attempt_number, attempt_spec_json, status) '
                'VALUES (?, ?, ?, ?, ?)',
                (
                    next_attempt_id,
                    task['id'],
                    next_number,
                    attempt['attempt_spec_json'],
                    AttemptStatus.CREATED,
                ),
            )
            connection.execute(
                'UPDATE mlops_agent_tasks SET status=?, attempt_count=?, updated_at=? '
                'WHERE id=?',
                (TaskStatus.RETRY_SCHEDULED, next_number, now, task['id']),
            )
            self._set_run_projection(
                connection, run_id, version, RunState.RETRY_SCHEDULED
            )
            created = dict(
                connection.execute(
                    'SELECT * FROM mlops_task_attempts WHERE id=?',
                    (next_attempt_id,),
                ).fetchone()
            )
            created['attempt_spec'] = json.loads(created.pop('attempt_spec_json'))
            return created

    def approve_task(
        self,
        *,
        run_id: str,
        task_id: str,
        approved_by: str,
        reason: str,
        expected_version: int,
        idempotency_key: str,
    ) -> None:
        if not approved_by.strip() or not reason.strip():
            raise ValueError('approval actor and reason are required')
        with self._transaction() as connection:
            task = self._task_for_update(connection, run_id, task_id)
            task_spec = TaskSpec.model_validate_json(task['spec_json'])
            if not task_spec.requires_human_confirmation:
                raise InvalidTransition('task does not require human confirmation')
            existing = connection.execute(
                'SELECT * FROM mlops_task_approvals WHERE task_id=?', (task_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing['approved_by'] != approved_by
                    or existing['reason'] != reason
                ):
                    raise IdempotencyConflict('task already has a different approval')
                return
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='HumanConfirmationRecorded',
                actor_type='human',
                actor_id=approved_by,
                idempotency_key=idempotency_key,
                payload={'task_id': task_id, 'reason': reason},
            )
            connection.execute(
                'INSERT INTO mlops_task_approvals '
                '(task_id, approved_by, reason, created_at) VALUES (?, ?, ?, ?)',
                (task_id, approved_by, reason, time.time()),
            )

    def is_task_approved(self, task_id: str) -> bool:
        with self._connect() as connection:
            return (
                connection.execute(
                    'SELECT 1 FROM mlops_task_approvals WHERE task_id=?', (task_id,)
                ).fetchone()
                is not None
            )

    def seal_release(
        self,
        *,
        run_id: str,
        manifest_artifact_id: str,
        artifact_ids: Sequence[str],
        root_kinds: frozenset[str],
        expected_version: int,
        validation_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Seal a release only when every declared artifact has closed provenance."""

        with self._transaction() as connection:
            manifest_artifact = connection.execute(
                'SELECT * FROM mlops_artifacts WHERE id=? AND sealed=1',
                (manifest_artifact_id,),
            ).fetchone()
            if manifest_artifact is None:
                raise InvariantViolation(
                    'release manifest artifact is missing or unsealed'
                )
            if manifest_artifact['kind'] != 'release-manifest':
                raise InvariantViolation(
                    'release must be sealed from a release-manifest artifact'
                )
            declared = list(dict.fromkeys([*artifact_ids, manifest_artifact_id]))
            if len(declared) != len(artifact_ids) + (
                0 if manifest_artifact_id in artifact_ids else 1
            ):
                raise InvariantViolation(
                    'release manifest contains duplicate artifacts'
                )
            rows = connection.execute(
                f'SELECT * FROM mlops_artifacts WHERE id IN '
                f'({",".join("?" for _ in declared)}) AND sealed=1',
                declared,
            ).fetchall()
            artifacts = {row['id']: row for row in rows}
            missing = sorted(set(declared) - set(artifacts))
            if missing:
                raise InvariantViolation(
                    f'release artifacts are missing or unsealed: {missing}'
                )
            self._assert_provenance_closure(
                connection,
                run_id=run_id,
                artifact_ids=declared,
                artifacts=artifacts,
                root_kinds=root_kinds,
            )
            release_id = _stable_id('release', manifest_artifact['sha256'])
            manifest = {
                'protocol': 'mlops.openhands.dev/release/v1',
                'manifest_artifact_id': manifest_artifact_id,
                'artifact_ids': sorted(declared),
                'validation_id': validation_id,
            }
            existing = connection.execute(
                'SELECT * FROM mlops_releases WHERE id=?', (release_id,)
            ).fetchone()
            if existing is not None:
                if json.loads(existing['manifest_json']) != manifest:
                    raise InvariantViolation('sealed release cannot be modified')
                return dict(existing)
            payload = {'release_id': release_id, **manifest}
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='ReleaseSealed',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            connection.execute(
                'INSERT INTO mlops_releases '
                '(id, run_id, manifest_artifact_id, manifest_json, sealed, created_at) '
                'VALUES (?, ?, ?, ?, 1, ?)',
                (
                    release_id,
                    run_id,
                    manifest_artifact_id,
                    _canonical_json(manifest),
                    time.time(),
                ),
            )
            # Release state is orthogonal to the Run lifecycle projection.
            return dict(
                connection.execute(
                    'SELECT * FROM mlops_releases WHERE id=?', (release_id,)
                ).fetchone()
            )

    def request_deployment(
        self,
        *,
        run_id: str,
        release_id: str,
        environment: str,
        initial_traffic_percent: float,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not 0 <= initial_traffic_percent <= 100:
            raise ValueError('traffic percent must be between 0 and 100')
        with self._transaction() as connection:
            release = connection.execute(
                'SELECT * FROM mlops_releases WHERE id=? AND sealed=1', (release_id,)
            ).fetchone()
            if release is None:
                raise InvariantViolation('deployment requires a sealed release')
            existing = connection.execute(
                'SELECT * FROM mlops_deployments WHERE idempotency_key=?',
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    existing['release_id'] != release_id
                    or existing['environment'] != environment
                    or existing['traffic_percent'] != initial_traffic_percent
                ):
                    raise IdempotencyConflict(
                        'deployment idempotency key was reused with different input'
                    )
                return dict(existing)
            deployment_id = _stable_id(
                'deployment', release_id, environment, idempotency_key
            )
            payload = {
                'deployment_id': deployment_id,
                'release_id': release_id,
                'environment': environment,
                'initial_traffic_percent': initial_traffic_percent,
            }
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='DeploymentRequested',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=f'deployment-request:{idempotency_key}',
                payload=payload,
            )
            now = time.time()
            connection.execute(
                'INSERT INTO mlops_deployments '
                '(id, run_id, release_id, environment, idempotency_key, desired_state, '
                'observed_state, traffic_percent, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    deployment_id,
                    run_id,
                    release_id,
                    environment,
                    idempotency_key,
                    'CANARY',
                    'PENDING',
                    initial_traffic_percent,
                    now,
                    now,
                ),
            )
            return dict(
                connection.execute(
                    'SELECT * FROM mlops_deployments WHERE id=?', (deployment_id,)
                ).fetchone()
            )

    def record_deployment_observation(
        self,
        *,
        run_id: str,
        deployment_id: str,
        observed_state: str,
        revision: str,
        metrics: dict[str, Any],
        expected_version: int,
        idempotency_key: str,
    ) -> None:
        """Record adapter reality without implicitly accepting or promoting it."""

        with self._transaction() as connection:
            deployment = connection.execute(
                'SELECT * FROM mlops_deployments WHERE id=? AND run_id=?',
                (deployment_id, run_id),
            ).fetchone()
            if deployment is None:
                raise KeyError(deployment_id)
            payload = {
                'deployment_id': deployment_id,
                'observed_state': observed_state,
                'revision': revision,
                'metrics': metrics,
                'domain_state_advanced': False,
            }
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='DeploymentObserved',
                actor_type='deployment-adapter',
                actor_id=deployment_id,
                idempotency_key=idempotency_key,
                payload=payload,
            )
            connection.execute(
                'UPDATE mlops_deployments SET observed_state=?, revision=?, updated_at=? '
                'WHERE id=?',
                (observed_state, revision, time.time(), deployment_id),
            )

    def promote_deployment(
        self,
        *,
        run_id: str,
        deployment_id: str,
        target_traffic_percent: float,
        validation_id: str,
        evidence_refs: Sequence[str],
        expected_version: int,
        idempotency_key: str,
    ) -> None:
        if not 0 <= target_traffic_percent <= 100:
            raise ValueError('traffic percent must be between 0 and 100')
        with self._transaction() as connection:
            deployment = connection.execute(
                'SELECT * FROM mlops_deployments WHERE id=? AND run_id=?',
                (deployment_id, run_id),
            ).fetchone()
            if deployment is None:
                raise KeyError(deployment_id)
            if deployment['observed_state'] != 'HEALTHY':
                raise InvariantViolation(
                    'deployment cannot be promoted before a HEALTHY observation'
                )
            payload = {
                'deployment_id': deployment_id,
                'target_traffic_percent': target_traffic_percent,
                'validation_id': validation_id,
                'evidence_refs': list(evidence_refs),
            }
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='TrafficShiftDecided',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            desired = 'PROMOTED' if target_traffic_percent == 100 else 'CANARY'
            connection.execute(
                'UPDATE mlops_deployments SET desired_state=?, traffic_percent=?, '
                'updated_at=? WHERE id=?',
                (desired, target_traffic_percent, time.time(), deployment_id),
            )
            if target_traffic_percent == 100:
                loop_id = connection.execute(
                    'SELECT loop_id FROM mlops_runs WHERE id=?', (run_id,)
                ).fetchone()['loop_id']
                connection.execute(
                    'INSERT INTO mlops_loop_champions '
                    '(loop_id, release_id, deployment_id, updated_at) '
                    'VALUES (?, ?, ?, ?) ON CONFLICT(loop_id) DO UPDATE SET '
                    'release_id=excluded.release_id, '
                    'deployment_id=excluded.deployment_id, '
                    'updated_at=excluded.updated_at',
                    (
                        loop_id,
                        deployment['release_id'],
                        deployment_id,
                        time.time(),
                    ),
                )

    def schedule_rollback_to_champion(
        self,
        *,
        run_id: str,
        expected_version: int,
        reason: str,
        evidence_refs: Sequence[str],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Create an audited rollback task targeting a proven prior champion."""

        with self._transaction() as connection:
            existing = connection.execute(
                'SELECT * FROM mlops_rollbacks WHERE idempotency_key=?',
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            run = connection.execute(
                'SELECT * FROM mlops_runs WHERE id=?', (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            deployment = connection.execute(
                'SELECT * FROM mlops_deployments WHERE run_id=? '
                'ORDER BY created_at DESC LIMIT 1',
                (run_id,),
            ).fetchone()
            if deployment is None:
                raise InvariantViolation('rollback requires an active deployment')
            champion = connection.execute(
                'SELECT c.*, d.desired_state, d.observed_state, d.traffic_percent, '
                'r.sealed FROM mlops_loop_champions c '
                'JOIN mlops_deployments d ON d.id=c.deployment_id '
                'JOIN mlops_releases r ON r.id=c.release_id WHERE c.loop_id=?',
                (run['loop_id'],),
            ).fetchone()
            if champion is None:
                raise InvariantViolation('rollback requires a proven champion')
            if (
                champion['sealed'] != 1
                or champion['desired_state'] != 'PROMOTED'
                or champion['observed_state'] != 'HEALTHY'
                or champion['traffic_percent'] != 100
            ):
                raise InvariantViolation(
                    'rollback target was not successfully deployed and verified'
                )
            if champion['release_id'] == deployment['release_id']:
                raise InvariantViolation('candidate release is already the champion')
            rollback_id = _stable_id(
                'rollback', deployment['id'], champion['release_id']
            )
            payload = {
                'rollback_id': rollback_id,
                'deployment_id': deployment['id'],
                'failed_release_id': deployment['release_id'],
                'target_release_id': champion['release_id'],
                'reason': reason,
                'evidence_refs': list(evidence_refs),
            }
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=expected_version,
                event_type='RollbackRequested',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=idempotency_key,
                payload=payload,
            )
            now = time.time()
            connection.execute(
                'INSERT INTO mlops_rollbacks '
                '(id, run_id, deployment_id, target_release_id, idempotency_key, '
                'status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    rollback_id,
                    run_id,
                    deployment['id'],
                    champion['release_id'],
                    idempotency_key,
                    'REQUESTED',
                    now,
                    now,
                ),
            )
            task_spec = TaskSpec(
                name='rollback-to-champion',
                kind='rollback',
                objective=(
                    'Roll back the failed canary to sealed champion release '
                    f'{champion["release_id"]} and prove the active revision.'
                ),
                agent_profile='default',
                validator='rightarm-deployment-v1',
                max_attempts=2,
                timeout_seconds=7200,
                completion_contract=CompletionContract(
                    required_artifact_kinds=['deployment-observation'],
                    required_evidence=True,
                ),
            )
            ordinal = self._max_task_ordinal(connection, run_id) + 1
            task_id = _stable_id('task', run_id, str(ordinal), task_spec.name)
            _, version = self._append_event(
                connection,
                run_id=run_id,
                expected_version=version,
                event_type='RollbackTaskScheduled',
                actor_type='loop-controller',
                actor_id='loop-controller',
                idempotency_key=f'{idempotency_key}:task',
                payload={
                    'rollback_id': rollback_id,
                    'task_id': task_id,
                    'active_task_ordinal_after': ordinal,
                    'run_state_after': RunState.WAITING_FOR_AGENT,
                },
            )
            connection.execute(
                'INSERT INTO mlops_agent_tasks '
                '(id, run_id, ordinal, spec_json, status, attempt_count, '
                'created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)',
                (
                    task_id,
                    run_id,
                    ordinal,
                    task_spec.model_dump_json(),
                    TaskStatus.PENDING,
                    now,
                    now,
                ),
            )
            connection.execute(
                'UPDATE mlops_deployments SET desired_state=?, updated_at=? WHERE id=?',
                ('ROLLBACK_REQUESTED', now, deployment['id']),
            )
            self._set_run_projection(
                connection,
                run_id,
                version,
                RunState.WAITING_FOR_AGENT,
                active_task_ordinal=ordinal,
            )
            return dict(
                connection.execute(
                    'SELECT * FROM mlops_rollbacks WHERE id=?', (rollback_id,)
                ).fetchone()
            )

    def get_champion(self, loop_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_loop_champions WHERE loop_id=?', (loop_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_active_rollback(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_rollbacks WHERE run_id=? '
                'ORDER BY created_at DESC LIMIT 1',
                (run_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_loop_control(self, loop_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_loop_controls WHERE loop_id=?', (loop_id,)
            ).fetchone()
        if row is None:
            return {
                'loop_id': loop_id,
                'state': 'RUNNING',
                'reason': None,
                'stopped_by_run_id': None,
            }
        return dict(row)

    def acquire_scheduler_lease(
        self, *, loop_id: str, holder_id: str, ttl_seconds: float
    ) -> str:
        if ttl_seconds <= 0:
            raise ValueError('scheduler lease TTL must be positive')
        now = time.time()
        with self._transaction() as connection:
            current = connection.execute(
                'SELECT * FROM mlops_scheduler_leases WHERE loop_id=?', (loop_id,)
            ).fetchone()
            if current is not None and current['expires_at'] > now:
                if current['holder_id'] != holder_id:
                    raise LeaseConflict(
                        f'loop {loop_id} scheduler is held by {current["holder_id"]}'
                    )
                connection.execute(
                    'UPDATE mlops_scheduler_leases SET heartbeat_at=?, expires_at=? '
                    'WHERE loop_id=?',
                    (now, now + ttl_seconds, loop_id),
                )
                return current['lease_id']
            lease_id = f'scheduler_lease_{uuid4().hex}'
            connection.execute(
                'INSERT INTO mlops_scheduler_leases '
                '(loop_id, lease_id, holder_id, acquired_at, heartbeat_at, expires_at) '
                'VALUES (?, ?, ?, ?, ?, ?) '
                'ON CONFLICT(loop_id) DO UPDATE SET '
                'lease_id=excluded.lease_id, holder_id=excluded.holder_id, '
                'acquired_at=excluded.acquired_at, heartbeat_at=excluded.heartbeat_at, '
                'expires_at=excluded.expires_at',
                (
                    loop_id,
                    lease_id,
                    holder_id,
                    now,
                    now,
                    now + ttl_seconds,
                ),
            )
            return lease_id

    def heartbeat_scheduler_lease(
        self, *, loop_id: str, lease_id: str, ttl_seconds: float
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError('scheduler lease TTL must be positive')
        now = time.time()
        with self._transaction() as connection:
            current = connection.execute(
                'SELECT * FROM mlops_scheduler_leases WHERE loop_id=?', (loop_id,)
            ).fetchone()
            if (
                current is None
                or current['lease_id'] != lease_id
                or current['expires_at'] <= now
            ):
                raise LeaseConflict(f'scheduler lease fencing rejected loop {loop_id}')
            connection.execute(
                'UPDATE mlops_scheduler_leases SET heartbeat_at=?, expires_at=? '
                'WHERE loop_id=?',
                (now, now + ttl_seconds, loop_id),
            )

    def release_scheduler_lease(self, *, loop_id: str, lease_id: str) -> bool:
        with self._transaction() as connection:
            deleted = connection.execute(
                'DELETE FROM mlops_scheduler_leases WHERE loop_id=? AND lease_id=?',
                (loop_id, lease_id),
            ).rowcount
        return deleted == 1

    def get_scheduler_lease(self, loop_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_scheduler_leases WHERE loop_id=?', (loop_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def stop_loop(
        self,
        *,
        loop_id: str,
        run_id: str,
        reason: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Persist a loop-level circuit breaker after deterministic failures."""

        with self._transaction() as connection:
            existing = connection.execute(
                'SELECT * FROM mlops_loop_controls WHERE loop_id=?', (loop_id,)
            ).fetchone()
            if existing is not None and existing['state'] == 'STOPPED':
                return dict(existing)
            run = connection.execute(
                'SELECT version FROM mlops_runs WHERE id=? AND loop_id=?',
                (run_id, loop_id),
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=run['version'],
                event_type='LoopStopped',
                actor_type='loop-controller',
                actor_id=loop_id,
                idempotency_key=idempotency_key,
                payload={'loop_id': loop_id, 'reason': reason},
            )
            now = time.time()
            connection.execute(
                'INSERT INTO mlops_loop_controls '
                '(loop_id, state, reason, stopped_by_run_id, updated_at) '
                'VALUES (?, ?, ?, ?, ?) '
                'ON CONFLICT(loop_id) DO UPDATE SET state=excluded.state, '
                'reason=excluded.reason, stopped_by_run_id=excluded.stopped_by_run_id, '
                'updated_at=excluded.updated_at',
                (loop_id, 'STOPPED', reason, run_id, now),
            )
            row = connection.execute(
                'SELECT * FROM mlops_loop_controls WHERE loop_id=?', (loop_id,)
            ).fetchone()
            assert row is not None
            return dict(row)

    def record_job_launch(
        self,
        *,
        attempt_id: str,
        job_id: str,
        idempotency_key: str,
        argv: Sequence[str],
        status: str,
    ) -> dict[str, Any]:
        """Persist the deterministic executor request behind an Agent tool call."""

        request = {
            'attempt_id': attempt_id,
            'job_id': job_id,
            'idempotency_key': idempotency_key,
            'argv': list(argv),
        }
        request_digest = _payload_digest(request)
        with self._transaction() as connection:
            owner = connection.execute(
                'SELECT t.run_id FROM mlops_task_attempts a '
                'JOIN mlops_agent_tasks t ON t.id=a.task_id WHERE a.id=?',
                (attempt_id,),
            ).fetchone()
            if owner is None:
                raise KeyError(attempt_id)
            run_id = owner['run_id']
            existing = connection.execute(
                'SELECT * FROM mlops_job_attempts WHERE id=?', (job_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing['task_attempt_id'] != attempt_id
                    or existing['idempotency_key'] != idempotency_key
                    or existing['request_digest'] != request_digest
                ):
                    raise IdempotencyConflict(
                        f'job identity {job_id!r} was reused with a different request'
                    )
                return self._decode_job_attempt(existing)
            version = connection.execute(
                'SELECT version FROM mlops_runs WHERE id=?', (run_id,)
            ).fetchone()['version']
            self._append_event(
                connection,
                run_id=run_id,
                expected_version=version,
                event_type='JobAttemptRequested',
                actor_type='task-tool-gateway',
                actor_id=attempt_id,
                idempotency_key=f'job-request:{job_id}',
                payload={**request, 'request_digest': request_digest, 'status': status},
            )
            now = time.time()
            connection.execute(
                'INSERT INTO mlops_job_attempts '
                '(id, run_id, task_attempt_id, idempotency_key, request_digest, '
                'argv_json, status, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    job_id,
                    run_id,
                    attempt_id,
                    idempotency_key,
                    request_digest,
                    _canonical_json(list(argv)),
                    status,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                'SELECT * FROM mlops_job_attempts WHERE id=?', (job_id,)
            ).fetchone()
            assert row is not None
            return self._decode_job_attempt(row)

    def record_job_observation(
        self,
        *,
        attempt_id: str,
        job_id: str,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one job state change; identical polls do not grow the ledger."""

        persisted = {
            key: observation.get(key)
            for key in (
                'status',
                'return_code',
                'result',
                'failure',
                'log_path',
                'log_sha256',
                'log_size_bytes',
            )
        }
        if persisted['failure'] is None and observation.get('cancellation') is not None:
            persisted['failure'] = {
                'failure_class': 'attempt-cancelled',
                **observation['cancellation'],
            }
        digest_input = dict(persisted)
        if persisted['status'] == 'running':
            digest_input['log_sha256'] = None
            digest_input['log_size_bytes'] = None
        observation_digest = _payload_digest(digest_input)
        with self._transaction() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_job_attempts WHERE id=? AND task_attempt_id=?',
                (job_id, attempt_id),
            ).fetchone()
            if row is None:
                raise InvariantViolation(
                    f'job {job_id!r} was observed before its request was persisted'
                )
            if row['last_observation_digest'] == observation_digest:
                return self._decode_job_attempt(row)
            version = connection.execute(
                'SELECT version FROM mlops_runs WHERE id=?', (row['run_id'],)
            ).fetchone()['version']
            self._append_event(
                connection,
                run_id=row['run_id'],
                expected_version=version,
                event_type='JobAttemptObserved',
                actor_type='task-tool-gateway',
                actor_id=attempt_id,
                idempotency_key=(f'job-observation:{job_id}:{observation_digest}'),
                payload={
                    'attempt_id': attempt_id,
                    'job_id': job_id,
                    'observation_digest': observation_digest,
                    **persisted,
                },
            )
            connection.execute(
                'UPDATE mlops_job_attempts SET status=?, return_code=?, '
                'result_json=?, failure_json=?, log_path=?, log_sha256=?, '
                'log_size_bytes=?, last_observation_digest=?, updated_at=? WHERE id=?',
                (
                    persisted['status'],
                    persisted['return_code'],
                    _canonical_json(persisted['result'])
                    if persisted['result'] is not None
                    else None,
                    _canonical_json(persisted['failure'])
                    if persisted['failure'] is not None
                    else None,
                    persisted['log_path'],
                    persisted['log_sha256'],
                    persisted['log_size_bytes'],
                    observation_digest,
                    time.time(),
                    job_id,
                ),
            )
            updated = connection.execute(
                'SELECT * FROM mlops_job_attempts WHERE id=?', (job_id,)
            ).fetchone()
            assert updated is not None
            return self._decode_job_attempt(updated)

    def record_artifact_candidate(
        self,
        *,
        attempt_id: str,
        lease_id: str,
        candidate: ArtifactCandidate,
    ) -> dict[str, Any]:
        """Bind an untrusted staged candidate to one currently leased attempt."""

        payload = candidate.model_dump(mode='json')
        digest = _payload_digest(payload)
        with self._transaction() as connection:
            self._assert_lease(connection, attempt_id, lease_id)
            attempt = connection.execute(
                'SELECT a.*, t.run_id FROM mlops_task_attempts a '
                'JOIN mlops_agent_tasks t ON t.id=a.task_id WHERE a.id=?',
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt['status'] != AttemptStatus.RUNNING:
                raise InvariantViolation(
                    'artifact candidates require a RUNNING task attempt'
                )
            existing = connection.execute(
                'SELECT * FROM mlops_artifact_candidates WHERE id=?',
                (candidate.candidate_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing['task_attempt_id'] != attempt_id
                    or existing['candidate_digest'] != digest
                ):
                    raise IdempotencyConflict(
                        'artifact candidate id was reused across attempts or payloads'
                    )
                return self._decode_artifact_candidate(existing)
            connection.execute(
                'INSERT INTO mlops_artifact_candidates '
                '(id, run_id, task_attempt_id, candidate_json, candidate_digest, '
                'created_at) VALUES (?, ?, ?, ?, ?, ?)',
                (
                    candidate.candidate_id,
                    attempt['run_id'],
                    attempt_id,
                    _canonical_json(payload),
                    digest,
                    time.time(),
                ),
            )
            row = connection.execute(
                'SELECT * FROM mlops_artifact_candidates WHERE id=?',
                (candidate.candidate_id,),
            ).fetchone()
            assert row is not None
            return self._decode_artifact_candidate(row)

    def assert_attempt_candidates(
        self, attempt_id: str, candidates: Sequence[ArtifactCandidate]
    ) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_artifact_candidates WHERE task_attempt_id=?',
                (attempt_id,),
            ).fetchall()
        registered = {row['id']: row for row in rows}
        for candidate in candidates:
            row = registered.get(candidate.candidate_id)
            if row is None or row['candidate_digest'] != _payload_digest(
                candidate.model_dump(mode='json')
            ):
                raise InvariantViolation(
                    f'artifact candidate {candidate.candidate_id} is not registered '
                    'to this task attempt'
                )

    def list_job_attempts(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_job_attempts WHERE run_id=? '
                'ORDER BY created_at, id',
                (run_id,),
            ).fetchall()
        return [self._decode_job_attempt(row) for row in rows]

    def list_artifact_candidates(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_artifact_candidates WHERE run_id=? '
                'ORDER BY created_at, id',
                (run_id,),
            ).fetchall()
        return [self._decode_artifact_candidate(row) for row in rows]

    def list_outbox(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_outbox WHERE run_id=? ORDER BY created_at, id',
                (run_id,),
            ).fetchall()
        return [self._decode_outbox(row) for row in rows]

    def list_cost_entries(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_cost_entries WHERE run_id=? '
                'ORDER BY created_at, id',
                (run_id,),
            ).fetchall()
        return [self._decode_cost_entry(row) for row in rows]

    def total_cost_usd(self, run_id: str) -> float:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT COALESCE(SUM(amount_usd), 0) AS total '
                'FROM mlops_cost_entries WHERE run_id=?',
                (run_id,),
            ).fetchone()
        return float(row['total'])

    def audit_run(self, run_id: str) -> dict[str, Any]:
        """Return one self-contained, deterministic domain audit projection."""

        run = self.get_run(run_id)
        return {
            'protocol': 'mlops.openhands.dev/run-audit/v1',
            'run': run,
            'events': self.list_events(run_id),
            'tasks': self.list_tasks(run_id),
            'attempts': self.list_attempts(run_id),
            'job_attempts': self.list_job_attempts(run_id),
            'artifact_candidates': self.list_artifact_candidates(run_id),
            'outbox': self.list_outbox(run_id),
            'cost_entries': self.list_cost_entries(run_id),
            'total_cost_usd': self.total_cost_usd(run_id),
            'artifacts': self.list_artifacts(run_id),
            'lineage': self.list_lineage(run_id),
            'releases': self.list_releases(run_id),
            'deployments': self.list_deployments(run_id),
            'rollback': self.get_active_rollback(run_id),
            'champion': self.get_champion(run['loop_id']),
            'loop_control': self.get_loop_control(run['loop_id']),
            'scheduler_lease': self.get_scheduler_lease(run['loop_id']),
            'replayed_projection': self.replay_run(run_id),
        }

    def list_releases(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_releases WHERE run_id=? ORDER BY created_at',
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_deployments(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_deployments WHERE run_id=? ORDER BY created_at',
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_pending_artifacts(self, loop_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT artifact_id FROM mlops_loop_pending_artifacts '
                'WHERE loop_id=? AND consumed_by_run_id IS NULL '
                'ORDER BY created_at, artifact_id',
                (loop_id,),
            ).fetchall()
        return [row['artifact_id'] for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_runs WHERE id=?', (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        result = dict(row)
        result['spec'] = json.loads(result.pop('spec_json'))
        return result

    def list_runs(self, loop_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if loop_id is None:
                rows = connection.execute(
                    'SELECT * FROM mlops_runs ORDER BY created_at, id'
                ).fetchall()
            else:
                rows = connection.execute(
                    'SELECT * FROM mlops_runs WHERE loop_id=? ORDER BY created_at, id',
                    (loop_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_agent_tasks WHERE id=?', (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        result = dict(row)
        result['spec'] = json.loads(result.pop('spec_json'))
        return result

    def get_attempt(self, attempt_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_task_attempts WHERE id=?', (attempt_id,)
            ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        result = dict(row)
        result['attempt_spec'] = json.loads(result.pop('attempt_spec_json'))
        return result

    @staticmethod
    def _decode_job_attempt(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result['argv'] = json.loads(result.pop('argv_json'))
        result['result'] = (
            json.loads(result.pop('result_json'))
            if result['result_json'] is not None
            else None
        )
        result['failure'] = (
            json.loads(result.pop('failure_json'))
            if result['failure_json'] is not None
            else None
        )
        return result

    @staticmethod
    def _decode_artifact_candidate(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result['candidate'] = json.loads(result.pop('candidate_json'))
        return result

    @staticmethod
    def _decode_outbox(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result['payload'] = json.loads(result.pop('payload_json'))
        return result

    @staticmethod
    def _decode_cost_entry(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result['metadata'] = json.loads(result.pop('metadata_json'))
        return result

    def current_task(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_agent_tasks WHERE run_id=? AND ordinal=?',
                (run_id, run['active_task_ordinal']),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['spec'] = json.loads(result.pop('spec_json'))
        return result

    def latest_attempt(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_task_attempts WHERE task_id=? '
                'ORDER BY attempt_number DESC LIMIT 1',
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['attempt_spec'] = json.loads(result.pop('attempt_spec_json'))
        return result

    def get_lease(self, attempt_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM mlops_attempt_leases WHERE attempt_id=?', (attempt_id,)
            ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return dict(row)

    def assert_attempt_lease(self, attempt_id: str, lease_id: str) -> None:
        with self._connect() as connection:
            self._assert_lease(connection, attempt_id, lease_id)
            attempt = connection.execute(
                'SELECT status FROM mlops_task_attempts WHERE id=?', (attempt_id,)
            ).fetchone()
            if attempt is None or attempt['status'] != AttemptStatus.RUNNING:
                raise LeaseConflict(
                    f'attempt {attempt_id} is no longer active for task tools'
                )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_events WHERE run_id=? ORDER BY sequence',
                (run_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item['payload'] = json.loads(item.pop('payload_json'))
            result.append(item)
        return result

    def list_tasks(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_agent_tasks WHERE run_id=? ORDER BY ordinal',
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_attempts(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT a.* FROM mlops_task_attempts a '
                'JOIN mlops_agent_tasks t ON t.id=a.task_id '
                'WHERE t.run_id=? ORDER BY t.ordinal, a.attempt_number',
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT a.* FROM mlops_artifacts a '
                'JOIN mlops_run_artifacts ra ON ra.artifact_id=a.id '
                'WHERE ra.run_id=? ORDER BY a.id',
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_artifacts(self, artifact_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not artifact_ids:
            return []
        unique_ids = list(dict.fromkeys(artifact_ids))
        with self._connect() as connection:
            rows = connection.execute(
                f'SELECT * FROM mlops_artifacts WHERE id IN '
                f'({",".join("?" for _ in unique_ids)}) AND sealed=1',
                unique_ids,
            ).fetchall()
        found = {row['id']: dict(row) for row in rows}
        missing = sorted(set(unique_ids) - set(found))
        if missing:
            raise InvariantViolation(
                f'task context references missing sealed artifacts: {missing}'
            )
        return [found[artifact_id] for artifact_id in unique_ids]

    def list_lineage(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT * FROM mlops_lineage_edges WHERE run_id=? '
                'ORDER BY input_artifact_id, output_artifact_id',
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def replay_run(self, run_id: str) -> dict[str, Any]:
        events = self.list_events(run_id)
        if not events or events[0]['event_type'] != 'RunCreated':
            raise InvariantViolation('run event stream does not start with RunCreated')
        state = RunState.CREATED
        active_task_ordinal = 0
        for event in events:
            event_type = event['event_type']
            payload = event['payload']
            if event_type == 'TaskAttemptDispatched':
                state = RunState.WAITING_FOR_AGENT
            elif event_type == 'TaskAttemptYielded':
                state = RunState.WAITING_EXTERNAL
            elif event_type == 'TaskAttemptContinued':
                state = RunState.WAITING_FOR_AGENT
            elif event_type == 'TaskResultSubmitted':
                state = RunState.VALIDATING_RESULT
            elif event_type == 'RetryDecided':
                state = RunState.RETRY_SCHEDULED
            elif event_type == 'TaskAttemptCrashed':
                state = RunState(payload['run_state_after'])
            elif event_type in {'TaskResultValidated', 'TaskResultRejected'}:
                state = RunState(payload['run_state_after'])
                if event_type == 'TaskResultValidated':
                    active_task_ordinal = payload.get(
                        'active_task_ordinal_after', active_task_ordinal + 1
                    )
            elif event_type == 'RunCancelled':
                state = RunState.CANCELLED
            elif event_type == 'RunPolicyStopped':
                state = RunState(payload['run_state_after'])
            elif event_type == 'RollbackTaskScheduled':
                state = RunState(payload['run_state_after'])
                active_task_ordinal = payload['active_task_ordinal_after']
        return {
            'id': run_id,
            'state': state,
            'version': len(events),
            'active_task_ordinal': active_task_ordinal,
        }

    def assert_projection_matches_replay(self, run_id: str) -> None:
        run = self.get_run(run_id)
        replayed = self.replay_run(run_id)
        for field in ('state', 'version', 'active_task_ordinal'):
            if run[field] != replayed[field]:
                raise InvariantViolation(
                    f'run projection mismatch for {field}: '
                    f'{run[field]!r} != {replayed[field]!r}'
                )

    def _append_event(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        expected_version: int,
        event_type: str,
        actor_type: str,
        actor_id: str,
        idempotency_key: str,
        payload: dict[str, Any],
    ) -> tuple[str, int]:
        existing = self._idempotent_event(connection, run_id, idempotency_key)
        digest = _payload_digest(payload)
        if existing is not None:
            if (
                existing['payload_digest'] != digest
                or existing['event_type'] != event_type
            ):
                raise IdempotencyConflict(
                    f'idempotency key {idempotency_key!r} was reused'
                )
            return existing['id'], existing['sequence']
        run = connection.execute(
            'SELECT version FROM mlops_runs WHERE id=?', (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        if run['version'] != expected_version:
            raise ConcurrencyConflict(
                f'expected run version {expected_version}, current is {run["version"]}'
            )
        sequence = expected_version + 1
        event_id = str(uuid4())
        connection.execute(
            'INSERT INTO mlops_events '
            '(id, run_id, sequence, event_type, actor_type, actor_id, '
            'idempotency_key, payload_digest, payload_json, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                event_id,
                run_id,
                sequence,
                event_type,
                actor_type,
                actor_id,
                idempotency_key,
                digest,
                _canonical_json(payload),
                time.time(),
            ),
        )
        connection.execute(
            'UPDATE mlops_runs SET version=?, updated_at=? WHERE id=?',
            (sequence, time.time(), run_id),
        )
        return event_id, sequence

    @staticmethod
    def _idempotent_event(
        connection: sqlite3.Connection, run_id: str, idempotency_key: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            'SELECT * FROM mlops_events WHERE run_id=? AND idempotency_key=?',
            (run_id, idempotency_key),
        ).fetchone()

    @staticmethod
    def _set_run_projection(
        connection: sqlite3.Connection,
        run_id: str,
        version: int,
        state: RunState,
        *,
        active_task_ordinal: int | None = None,
    ) -> None:
        if active_task_ordinal is None:
            connection.execute(
                'UPDATE mlops_runs SET version=?, state=?, updated_at=? WHERE id=?',
                (version, state, time.time(), run_id),
            )
        else:
            connection.execute(
                'UPDATE mlops_runs SET version=?, state=?, active_task_ordinal=?, '
                'updated_at=? WHERE id=?',
                (version, state, active_task_ordinal, time.time(), run_id),
            )

    @staticmethod
    def _task_for_update(
        connection: sqlite3.Connection, run_id: str, task_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            'SELECT * FROM mlops_agent_tasks WHERE id=? AND run_id=?',
            (task_id, run_id),
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return row

    @staticmethod
    def _attempt_for_run(
        connection: sqlite3.Connection, run_id: str, attempt_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            'SELECT a.* FROM mlops_task_attempts a '
            'JOIN mlops_agent_tasks t ON t.id=a.task_id '
            'WHERE a.id=? AND t.run_id=?',
            (attempt_id, run_id),
        ).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return row

    def _submitted_attempt(
        self, connection: sqlite3.Connection, run_id: str, result: TaskResult
    ) -> sqlite3.Row:
        attempt = self._attempt_for_run(connection, run_id, result.attempt_id)
        if attempt['task_id'] != result.task_id:
            raise InvariantViolation('result task does not own its attempt')
        if attempt['status'] != AttemptStatus.RESULT_SUBMITTED:
            raise InvalidTransition('only a submitted result can be validated')
        stored = TaskResult.model_validate_json(attempt['result_json'])
        if stored != result:
            raise InvariantViolation('validator result does not match submitted result')
        return attempt

    @staticmethod
    def _assert_lease(
        connection: sqlite3.Connection, attempt_id: str, lease_id: str
    ) -> None:
        lease = connection.execute(
            'SELECT * FROM mlops_attempt_leases WHERE attempt_id=?', (attempt_id,)
        ).fetchone()
        if (
            lease is None
            or lease['lease_id'] != lease_id
            or lease['expires_at'] <= time.time()
        ):
            raise LeaseConflict(f'lease fencing rejected attempt {attempt_id}')

    @staticmethod
    def _is_last_task(
        connection: sqlite3.Connection, run_id: str, ordinal: int
    ) -> bool:
        return ordinal == SQLiteLedger._max_task_ordinal(connection, run_id)

    @staticmethod
    def _max_task_ordinal(connection: sqlite3.Connection, run_id: str) -> int:
        row = connection.execute(
            'SELECT MAX(ordinal) AS max_ordinal FROM mlops_agent_tasks WHERE run_id=?',
            (run_id,),
        ).fetchone()
        return int(row['max_ordinal'])

    @staticmethod
    def _assert_provenance_closure(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        artifact_ids: Sequence[str],
        artifacts: dict[str, sqlite3.Row],
        root_kinds: frozenset[str],
    ) -> None:
        edges = connection.execute(
            'SELECT input_artifact_id, output_artifact_id '
            'FROM mlops_lineage_edges WHERE run_id=?',
            (run_id,),
        ).fetchall()
        parents: dict[str, set[str]] = {}
        for edge in edges:
            parents.setdefault(edge['output_artifact_id'], set()).add(
                edge['input_artifact_id']
            )
        root_ids = {
            artifact_id
            for artifact_id, artifact in artifacts.items()
            if artifact['kind'] in root_kinds
        }
        if not root_ids:
            raise InvariantViolation('release provenance has no accepted root artifact')

        def reaches_root(artifact_id: str, visiting: set[str]) -> bool:
            if artifact_id in root_ids:
                return True
            if artifact_id in visiting:
                raise InvariantViolation('release provenance graph contains a cycle')
            return any(
                reaches_root(parent, {*visiting, artifact_id})
                for parent in parents.get(artifact_id, set())
            )

        unclosed = [
            artifact_id
            for artifact_id in artifact_ids
            if not reaches_root(artifact_id, set())
        ]
        if unclosed:
            raise InvariantViolation(
                f'release provenance graph is not closed for artifacts: {unclosed}'
            )

    @classmethod
    def _find_placeholders(cls, value: object) -> list[str]:
        found: list[str] = []
        if isinstance(value, str) and '<' in value and '>' in value:
            found.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                found.extend(cls._find_placeholders(item))
        elif isinstance(value, list):
            for item in value:
                found.extend(cls._find_placeholders(item))
        return found

    @classmethod
    def _deep_merge(
        cls, base: dict[str, Any], changes: dict[str, object]
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _read_local_json(uri: str) -> dict[str, Any]:
        path = SQLiteLedger._file_uri_path(uri, 'release manifest')
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise InvariantViolation('release manifest must be a JSON object')
        if value.get('schema_version') != 'rightarm.gesture.release/v1':
            raise InvariantViolation('release manifest schema_version is invalid')
        artifacts = value.get('artifacts')
        if not isinstance(artifacts, list) or not artifacts:
            raise InvariantViolation('release manifest must declare artifacts')
        for item in artifacts:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get('kind'), str)
                or not isinstance(item.get('sha256'), str)
            ):
                raise InvariantViolation('release manifest artifact entry is invalid')
            digest = item['sha256']
            if len(digest) != 64 or any(
                character not in '0123456789abcdef' for character in digest
            ):
                raise InvariantViolation('release manifest artifact sha256 is invalid')
        return value

    @staticmethod
    def _read_deployment_observation(uri: str) -> dict[str, Any]:
        path = SQLiteLedger._file_uri_path(uri, 'deployment observation')
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise InvariantViolation('deployment observation must be a JSON object')
        if value.get('schema_version') != 'rightarm.gesture.deployment-observation/v1':
            raise InvariantViolation('deployment observation schema_version is invalid')
        if (
            not isinstance(value.get('deployment_id'), str)
            or not value['deployment_id']
        ):
            raise InvariantViolation('deployment observation has no deployment_id')
        if value.get('observed_state') not in {'HEALTHY', 'UNHEALTHY'}:
            raise InvariantViolation('deployment observation state is invalid')
        if not isinstance(value.get('revision'), str) or not value['revision']:
            raise InvariantViolation('deployment observation has no revision')
        if not isinstance(value.get('metrics'), dict):
            raise InvariantViolation('deployment observation metrics must be an object')
        return value

    @staticmethod
    def _file_uri_path(uri: str, artifact_name: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != 'file' or parsed.netloc not in {'', 'localhost'}:
            raise InvariantViolation(
                f'staging {artifact_name} must use a local file URI'
            )
        return Path(unquote(parsed.path)).resolve(strict=True)
