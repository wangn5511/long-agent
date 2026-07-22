"""Conversation dispatch adapters for one-attempt/one-conversation execution."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

import httpx

from openhands.mlops_loop.models import TaskContext, TaskSpec


class ConversationDispatcher(Protocol):
    def dispatch(
        self, *, context: TaskContext, task: TaskSpec, dispatch_key: str
    ) -> str:
        """Return a stable conversation id for ``dispatch_key``."""


def conversation_id_for_dispatch(dispatch_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, dispatch_key))


class InMemoryConversationDispatcher:
    """Deterministic fake used for outer-loop and recovery tests."""

    def __init__(self):
        self.conversations: dict[str, tuple[str, TaskContext]] = {}

    def dispatch(
        self, *, context: TaskContext, task: TaskSpec, dispatch_key: str
    ) -> str:
        existing = self.conversations.get(dispatch_key)
        if existing is not None:
            return existing[0]
        conversation_id = conversation_id_for_dispatch(dispatch_key)
        self.conversations[dispatch_key] = (conversation_id, context)
        return conversation_id


class OpenHandsHTTPDispatcher:
    """Start an actual OpenHands conversation and wait for its stable id."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        repository: str | None = None,
        branch: str | None = None,
        plugin_source: str | None = None,
        task_mcp_url: str | None = None,
        timeout_seconds: float = 120,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ):
        self.base_url = base_url.rstrip('/')
        self.repository = repository
        self.branch = branch
        self.plugin_source = plugin_source
        self.task_mcp_url = task_mcp_url
        self.timeout_seconds = timeout_seconds
        self._headers = {'Authorization': f'Bearer {token}'} if token else {}
        self._client_factory = client_factory
        self._dispatches: dict[str, str] = {}

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
            f'```json\n{context.model_dump_json(indent=2)}\n```'
        )
        body: dict[str, object] = {
            # A deterministic requested UUID makes a retried dispatch safe even
            # when the first HTTP response was lost after OpenHands created it.
            'conversation_id': conversation_id_for_dispatch(dispatch_key),
            'initial_message': {
                'role': 'user',
                'content': [{'type': 'text', 'text': initial_text}],
            },
            'selected_repository': self.repository,
            'selected_branch': self.branch,
            'title': f'MLOps {task.name} attempt {context.attempt_number}',
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
            existing_response = client.get(
                f'/api/v1/app-conversations/{requested_conversation_id}'
            )
            if existing_response.status_code == 200:
                self._dispatches[dispatch_key] = str(requested_conversation_id)
                return str(requested_conversation_id)
            if existing_response.status_code != 404:
                existing_response.raise_for_status()
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
                    self._dispatches[dispatch_key] = conversation_id
                    return conversation_id
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
