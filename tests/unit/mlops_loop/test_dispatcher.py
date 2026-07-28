from typing import Any

import pytest

from openhands.mlops_loop.dispatcher import (
    OpenHandsHTTPDispatcher,
    conversation_id_for_dispatch,
)
from openhands.mlops_loop.models import CompletionContract, TaskContext, TaskSpec


class _Response:
    def __init__(self, value: Any, status_code: int = 200):
        self.value = value
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.value


class _Client:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]], **_: Any):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def post(self, path: str, *, json: dict[str, Any]) -> _Response:
        self.calls.append((path, json))
        return _Response(
            {
                'id': 'start-task-1',
                'status': 'READY',
                # OpenHands currently serializes this UUID as 32-char hex.
                'app_conversation_id': json['conversation_id'].replace('-', ''),
            }
        )

    def get(self, path: str, **_: Any) -> _Response:
        return _Response([None])


class _ExistingConversationClient(_Client):
    def get(self, path: str, **kwargs: Any) -> _Response:
        if path == '/api/v1/app-conversations':
            requested_id = kwargs['params'][0][1]
            return _Response([{'id': requested_id.replace('-', '')}])
        return super().get(path)

    def post(self, path: str, *, json: dict[str, Any]) -> _Response:
        raise AssertionError('an existing deterministic Conversation must be reused')


class _TwoPhaseClient(_Client):
    def __init__(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        state: dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(calls, **kwargs)
        self.state = state

    def get(self, path: str, **kwargs: Any) -> _Response:
        if path == '/api/v1/app-conversations':
            conversation = self.state.get('conversation')
            return _Response([conversation])
        return super().get(path, **kwargs)

    def post(self, path: str, *, json: dict[str, Any]) -> _Response:
        self.calls.append((path, json))
        if path == '/api/v1/app-conversations':
            conversation_id = json['conversation_id']
            self.state['conversation'] = {
                'id': conversation_id.replace('-', ''),
                'execution_status': 'idle',
            }
            return _Response(
                {
                    'id': 'start-task-1',
                    'status': 'READY',
                    'app_conversation_id': conversation_id.replace('-', ''),
                }
            )
        self.state['conversation']['execution_status'] = 'running'
        return _Response({'success': True})


def test_dispatcher_injects_task_mcp_and_stable_domain_tags():
    calls: list[tuple[str, dict[str, Any]]] = []
    dispatcher = OpenHandsHTTPDispatcher(
        'http://openhands.test',
        repository='https://example.test/rightArmPerception.git',
        task_mcp_url='http://loop-controller.test:8765/mcp/',
        client_factory=lambda **kwargs: _Client(calls, **kwargs),
    )
    context = TaskContext(
        run_id='run-1',
        run_version=3,
        task_id='task-1',
        task_kind='inspect-data',
        attempt_id='attempt-1',
        attempt_number=1,
        lease_id='lease-1',
        objective='inspect one sealed batch',
        completion_contract=CompletionContract(),
    )
    task = TaskSpec(
        name='inspect-data',
        kind='inspect-data',
        objective=context.objective,
        agent_profile='rightarm-data-agent',
        validator='rightarm-data-v1',
        max_attempts=2,
        timeout_seconds=60,
    )

    first = dispatcher.dispatch(context=context, task=task, dispatch_key='dispatch-1')
    second = dispatcher.dispatch(context=context, task=task, dispatch_key='dispatch-1')

    assert first == second
    assert '-' in first
    assert len(calls) == 1
    path, body = calls[0]
    assert path == '/api/v1/app-conversations'
    assert body['initial_message']['run'] is False
    assert body['mcp_urls'] == {'mlopstask': 'http://loop-controller.test:8765/mcp/'}
    assert body['tags'] == {
        'mlopsrunid': 'run-1',
        'mlopstaskid': 'task-1',
        'mlopsattempt': '1',
        'mlopsprotocol': 'v1',
    }


def test_dispatcher_uses_active_openhands_profile_for_default_task_profile():
    calls: list[tuple[str, dict[str, Any]]] = []
    dispatcher = OpenHandsHTTPDispatcher(
        'http://openhands.test',
        client_factory=lambda **kwargs: _Client(calls, **kwargs),
    )
    context = TaskContext(
        run_id='run-1',
        run_version=1,
        task_id='task-1',
        task_kind='inspect-data',
        attempt_id='attempt-1',
        attempt_number=1,
        lease_id='lease-1',
        objective='inspect',
        completion_contract=CompletionContract(),
    )
    task = TaskSpec(
        name='inspect-data',
        kind='inspect-data',
        objective='inspect',
        agent_profile='default',
        validator='validator',
        max_attempts=1,
        timeout_seconds=60,
    )

    dispatcher.dispatch(context=context, task=task, dispatch_key='default-profile')

    assert 'agent_profile_id' not in calls[0][1]


def test_dispatcher_recovers_when_create_response_was_lost():
    calls: list[tuple[str, dict[str, Any]]] = []
    dispatcher = OpenHandsHTTPDispatcher(
        'http://openhands.test',
        client_factory=lambda **kwargs: _ExistingConversationClient(calls, **kwargs),
    )
    context = TaskContext(
        run_id='run-1',
        run_version=1,
        task_id='task-1',
        task_kind='inspect-data',
        attempt_id='attempt-1',
        attempt_number=1,
        lease_id='lease-1',
        objective='recover',
        completion_contract=CompletionContract(),
    )
    task = TaskSpec(
        name='recover-task',
        kind='inspect-data',
        objective='recover',
        agent_profile='default',
        validator='validator',
        max_attempts=1,
        timeout_seconds=60,
    )

    recovered = dispatcher.dispatch(
        context=context, task=task, dispatch_key='stable-dispatch'
    )

    assert recovered
    assert calls == []


def test_dispatcher_rejects_mismatched_existing_conversation():
    class _MismatchedConversationClient(_Client):
        def get(self, path: str, **_: Any) -> _Response:
            return _Response([{'id': 'different-conversation'}])

    dispatcher = OpenHandsHTTPDispatcher(
        'http://openhands.test',
        client_factory=lambda **kwargs: _MismatchedConversationClient([], **kwargs),
    )
    context = TaskContext(
        run_id='run-1',
        run_version=1,
        task_id='task-1',
        task_kind='inspect-data',
        attempt_id='attempt-1',
        attempt_number=1,
        lease_id='lease-1',
        objective='reject mismatched lookup',
        completion_contract=CompletionContract(),
    )
    task = TaskSpec(
        name='recover-task',
        kind='inspect-data',
        objective=context.objective,
        agent_profile='default',
        validator='validator',
        max_attempts=1,
        timeout_seconds=60,
    )

    with pytest.raises(RuntimeError, match='mismatched conversation lookup'):
        dispatcher.dispatch(
            context=context, task=task, dispatch_key='mismatched-dispatch'
        )


def test_dispatcher_creates_then_explicitly_starts_once():
    calls: list[tuple[str, dict[str, Any]]] = []
    state: dict[str, Any] = {}
    dispatcher = OpenHandsHTTPDispatcher(
        'http://openhands.test',
        action_norm_path=None,
        client_factory=lambda **kwargs: _TwoPhaseClient(calls, state, **kwargs),
    )
    context = TaskContext(
        run_id='run-1',
        run_version=1,
        task_id='task-1',
        task_kind='inspect-data',
        attempt_id='attempt-1',
        attempt_number=1,
        lease_id='lease-1',
        objective='create before starting',
        completion_contract=CompletionContract(),
    )
    task = TaskSpec(
        name='inspect-data',
        kind='inspect-data',
        objective=context.objective,
        agent_profile='default',
        validator='validator',
        max_attempts=1,
        timeout_seconds=60,
    )

    conversation_id = dispatcher.dispatch(
        context=context, task=task, dispatch_key='two-phase'
    )
    assert calls[0][1]['initial_message']['run'] is False

    dispatcher.start(
        conversation_id=conversation_id,
        dispatch_key='two-phase',
    )
    dispatcher.start(
        conversation_id=conversation_id,
        dispatch_key='two-phase',
    )

    assert [path for path, _ in calls] == [
        '/api/v1/app-conversations',
        f'/api/v1/app-conversations/{conversation_id}/send-message',
    ]
    assert calls[1][1]['run'] is True
    assert state['conversation']['execution_status'] == 'running'


def test_dispatcher_restart_does_not_restart_non_idle_conversation():
    dispatch_key = 'already-started'
    conversation_id = conversation_id_for_dispatch(dispatch_key)
    calls: list[tuple[str, dict[str, Any]]] = []
    state = {
        'conversation': {
            'id': conversation_id.replace('-', ''),
            'execution_status': 'finished',
        }
    }
    dispatcher = OpenHandsHTTPDispatcher(
        'http://openhands.test',
        action_norm_path=None,
        client_factory=lambda **kwargs: _TwoPhaseClient(calls, state, **kwargs),
    )

    dispatcher.start(
        conversation_id=conversation_id,
        dispatch_key=dispatch_key,
    )

    assert calls == []


def test_improve_model_prompt_loads_configured_action_norm(tmp_path):
    action_norm = tmp_path / 'SKILL.md'
    action_norm.write_text(
        '# Test MLOps norm\n\n'
        'For improve-model, benchmark first and pass the policy-owned --gpus value.',
        encoding='utf-8',
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    dispatcher = OpenHandsHTTPDispatcher(
        'http://openhands.test',
        action_norm_path=action_norm,
        client_factory=lambda **kwargs: _Client(calls, **kwargs),
    )
    context = TaskContext(
        run_id='run-1',
        run_version=1,
        task_id='task-1',
        task_kind='improve-model',
        attempt_id='attempt-1',
        attempt_number=1,
        lease_id='lease-1',
        objective='improve the gesture model',
        completion_contract=CompletionContract(),
    )
    task = TaskSpec(
        name='improve-model',
        kind='improve-model',
        objective=context.objective,
        agent_profile='default',
        validator='validator',
        max_attempts=1,
        timeout_seconds=60,
    )

    dispatcher.dispatch(context=context, task=task, dispatch_key='improve-model')

    prompt = calls[0][1]['initial_message']['content'][0]['text']
    assert '<action_norm>' in prompt
    assert action_norm.read_text(encoding='utf-8') in prompt
    assert context.model_dump_json(indent=2) in prompt
