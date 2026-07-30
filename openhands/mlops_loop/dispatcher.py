"""Conversation dispatch adapters for fenced task-attempt episodes."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx

from openhands.mlops_loop.models import TaskContext, TaskSpec

DEFAULT_ACTION_NORM_PATH = (
    Path(__file__).resolve().parents[2]
    / 'skills'
    / 'agentic-mlops-task-agent'
    / 'SKILL.md'
)
MAX_ACTION_NORM_BYTES = 64 * 1024


class ConversationDispatcher(Protocol):
    def dispatch(
        self, *, context: TaskContext, task: TaskSpec, dispatch_key: str
    ) -> str:
        """Create, but do not run, a Conversation for ``dispatch_key``."""

    def start(self, *, conversation_id: str, dispatch_key: str) -> None:
        """Idempotently start a previously created Conversation."""


def conversation_id_for_dispatch(dispatch_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, dispatch_key))


def _same_conversation_id(actual: object, expected: object) -> bool:
    """Compare UUIDs independently of dashed versus hex serialization."""

    try:
        return UUID(str(actual)) == UUID(str(expected))
    except (ValueError, AttributeError, TypeError):
        return False


class InMemoryConversationDispatcher:
    """Deterministic fake used for outer-loop and recovery tests."""

    def __init__(self):
        self.conversations: dict[str, tuple[str, TaskContext]] = {}
        self.started_dispatches: set[str] = set()

    def dispatch(
        self, *, context: TaskContext, task: TaskSpec, dispatch_key: str
    ) -> str:
        existing = self.conversations.get(dispatch_key)
        if existing is not None:
            return existing[0]
        conversation_id = conversation_id_for_dispatch(dispatch_key)
        self.conversations[dispatch_key] = (conversation_id, context)
        return conversation_id

    def start(self, *, conversation_id: str, dispatch_key: str) -> None:
        conversation = self.conversations.get(dispatch_key)
        if conversation is None or conversation[0] != conversation_id:
            raise RuntimeError('cannot start an unknown dispatched Conversation')
        self.started_dispatches.add(dispatch_key)


class OpenHandsHTTPDispatcher:
    """Create and explicitly start actual OpenHands Conversations in two phases."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        repository: str | None = None,
        branch: str | None = None,
        plugin_source: str | None = None,
        task_mcp_url: str | None = None,
        action_norm_path: str | Path | None = DEFAULT_ACTION_NORM_PATH,
        timeout_seconds: float = 120,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ):
        self.base_url = base_url.rstrip('/')
        self.repository = repository
        self.branch = branch
        self.plugin_source = plugin_source
        self.task_mcp_url = task_mcp_url
        self.action_norm_path = (
            Path(action_norm_path) if action_norm_path is not None else None
        )
        self.timeout_seconds = timeout_seconds
        self._headers = {'Authorization': f'Bearer {token}'} if token else {}
        self._client_factory = client_factory
        self._dispatches: dict[str, str] = {}
        self._started_dispatches: set[str] = set()

    def _read_action_norm(self) -> str:
        if self.action_norm_path is None:
            return ''
        value = self.action_norm_path.read_text(encoding='utf-8').strip()
        if not value:
            raise ValueError('MLOps action norm must not be empty')
        if len(value.encode('utf-8')) > MAX_ACTION_NORM_BYTES:
            raise ValueError(f'MLOps action norm exceeds {MAX_ACTION_NORM_BYTES} bytes')
        return value

    @staticmethod
    def _lookup_conversation(
        client: httpx.Client, requested_conversation_id: object
    ) -> dict[str, object] | None:
        # OpenHands exposes conversation lookup through its batch endpoint.
        # The SPA fallback also returns HTTP 200 for unknown singular paths,
        # so checking /app-conversations/{id} by status code alone can
        # incorrectly treat an HTML page as an existing Conversation.
        existing_response = client.get(
            '/api/v1/app-conversations',
            params=[('ids', str(requested_conversation_id))],
        )
        existing_response.raise_for_status()
        existing_values = existing_response.json()
        if not isinstance(existing_values, list) or len(existing_values) != 1:
            raise RuntimeError(
                'OpenHands returned an invalid conversation lookup response'
            )
        existing_conversation = existing_values[0]
        if existing_conversation is None:
            return None
        if not isinstance(existing_conversation, dict) or not _same_conversation_id(
            existing_conversation.get('id'), requested_conversation_id
        ):
            raise RuntimeError(
                'OpenHands returned a mismatched conversation lookup response'
            )
        return existing_conversation

    def dispatch(
        self, *, context: TaskContext, task: TaskSpec, dispatch_key: str
    ) -> str:
        cached_conversation_id = self._dispatches.get(dispatch_key)
        if cached_conversation_id is not None:
            return cached_conversation_id
        tags = {
            'mlopsrunid': context.run_id,
            'mlopstaskid': context.task_id,
            'mlopsattempt': str(context.attempt_number),
            'mlopsprotocol': 'v1',
        }
        continuation = context.attempt_spec.get('_continuation')
        yield_count = (
            continuation.get('yield_count') if isinstance(continuation, dict) else None
        )
        if isinstance(yield_count, int) and not isinstance(yield_count, bool):
            tags['mlopscontinuation'] = str(yield_count)
        action_norm = self._read_action_norm()
        initial_text = (
            'You are one task Agent inside an MLOps outer loop. Work only on '
            'this attempt. Call the mlopstask MCP tools with the attempt_id and '
            'lease_id below: get the fenced context, launch/observe only the '
            'allowlisted rightarm-mlops job for this task, stage every required '
            'output, propose its lineage, then submit a structured TaskResult. '
            'When an inspect-data source is cos://, first launch the allowlisted '
            'materialize-cos job, then use its materialized_source_uri for '
            'inspect-batch and land-batch; never request cloud credentials. '
            'A prose success claim has no effect. You cannot advance Run, '
            'Release, Deployment, retry, or rollback state.\n\n'
            'The operator-maintained action norm below is authoritative for '
            'this attempt. Follow it exactly. New Conversations load the '
            'current checked-in SKILL.md, so this norm may be updated without '
            'changing dispatcher code.\n\n'
            f'<action_norm>\n{action_norm}\n</action_norm>\n\n'
            f'```json\n{context.model_dump_json(indent=2)}\n```'
        )
        body: dict[str, object] = {
            # A deterministic requested UUID makes a retried dispatch safe even
            # when the first HTTP response was lost after OpenHands created it.
            'conversation_id': conversation_id_for_dispatch(dispatch_key),
            'initial_message': {
                'role': 'user',
                'content': [{'type': 'text', 'text': initial_text}],
                # The controller must durably mark the attempt RUNNING before
                # the Agent can call any lease-fenced MCP tool.
                'run': False,
            },
            'selected_repository': self.repository,
            'selected_branch': self.branch,
            'title': (
                f'MLOps {task.name} attempt {context.attempt_number}'
                + (
                    f' continuation {yield_count}'
                    if isinstance(yield_count, int)
                    and not isinstance(yield_count, bool)
                    else ''
                )
            ),
            'trigger': 'automation',
            'tags': tags,
        }
        if task.agent_profile != 'default':
            body['agent_profile_id'] = task.agent_profile
        if self.plugin_source:
            body['plugins'] = [{'source': self.plugin_source}]
        if self.task_mcp_url:
            body['mcp_urls'] = {'mlopstask': self.task_mcp_url}
        deadline = time.monotonic() + self.timeout_seconds
        with self._client_factory(
            base_url=self.base_url, headers=self._headers, timeout=30
        ) as client:
            requested_conversation_id = body['conversation_id']
            existing_conversation = self._lookup_conversation(
                client, requested_conversation_id
            )
            if existing_conversation is not None:
                self._dispatches[dispatch_key] = str(requested_conversation_id)
                return str(requested_conversation_id)
            response = client.post('/api/v1/app-conversations', json=body)
            response.raise_for_status()
            start_task = response.json()
            start_task_id = start_task['id']
            while time.monotonic() < deadline:
                if start_task.get('status') == 'READY':
                    conversation_id = start_task.get('app_conversation_id')
                    if not conversation_id:
                        raise RuntimeError(
                            'OpenHands READY response has no conversation id'
                        )
                    if not _same_conversation_id(
                        conversation_id, requested_conversation_id
                    ):
                        raise RuntimeError(
                            'OpenHands returned a mismatched READY conversation id'
                        )
                    self._dispatches[dispatch_key] = str(requested_conversation_id)
                    return str(requested_conversation_id)
                if start_task.get('status') == 'ERROR':
                    raise RuntimeError(
                        f'OpenHands conversation failed to start: '
                        f'{start_task.get("detail")}'
                    )
                time.sleep(0.5)
                response = client.get(
                    '/api/v1/app-conversations/start-tasks',
                    params=[('ids', start_task_id)],
                )
                response.raise_for_status()
                values = response.json()
                if not values or values[0] is None:
                    raise RuntimeError(
                        f'OpenHands start task disappeared: {start_task_id}'
                    )
                start_task = values[0]
        raise TimeoutError(
            f'OpenHands conversation start timed out: {json.dumps(start_task)}'
        )

    def start(self, *, conversation_id: str, dispatch_key: str) -> None:
        """Start only after the outer loop has linked the Conversation.

        A retry first consults OpenHands' durable execution status. Any status
        other than ``idle`` proves that a prior start was accepted (or that the
        Conversation is no longer safe to start), so it must not append another
        start message.
        """

        if dispatch_key in self._started_dispatches:
            return
        expected_conversation_id = conversation_id_for_dispatch(dispatch_key)
        if not _same_conversation_id(conversation_id, expected_conversation_id):
            raise RuntimeError('cannot start a mismatched dispatched Conversation')
        with self._client_factory(
            base_url=self.base_url, headers=self._headers, timeout=30
        ) as client:
            conversation = self._lookup_conversation(client, conversation_id)
            if conversation is None:
                raise RuntimeError('cannot start an unknown dispatched Conversation')
            execution_status = conversation.get('execution_status')
            if not isinstance(execution_status, str):
                raise RuntimeError(
                    'OpenHands Conversation has no durable execution status'
                )
            if execution_status != 'idle':
                self._started_dispatches.add(dispatch_key)
                return
            response = client.post(
                f'/api/v1/app-conversations/{conversation_id}/send-message',
                json={
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': (
                                'Start this task now. Follow the immediately '
                                'preceding action norm and immutable TaskContext; '
                                'first call mlops_get_task_context.'
                            ),
                        }
                    ],
                    'run': True,
                },
            )
            response.raise_for_status()
        self._started_dispatches.add(dispatch_key)
