from typing import Any

from openhands.mlops_loop.dispatcher import OpenHandsHTTPDispatcher
from openhands.mlops_loop.models import CompletionContract, TaskContext, TaskSpec


class _Response:
    def __init__(self, value: dict[str, Any], status_code: int = 200):
        self.value = value
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
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
                'app_conversation_id': json['conversation_id'],
            }
        )

    def get(self, path: str, **_: Any) -> _Response:
        return _Response({}, status_code=404)


class _ExistingConversationClient(_Client):
    def get(self, path: str, **_: Any) -> _Response:
        if path.startswith('/api/v1/app-conversations/'):
            return _Response({'id': path.rsplit('/', 1)[-1]})
        return super().get(path)

    def post(self, path: str, *, json: dict[str, Any]) -> _Response:
        raise AssertionError('an existing deterministic Conversation must be reused')


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
    assert len(calls) == 1
    path, body = calls[0]
    assert path == '/api/v1/app-conversations'
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
