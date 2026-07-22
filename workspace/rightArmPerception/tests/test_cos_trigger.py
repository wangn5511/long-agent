from pathlib import Path

from rightarm_mlops.cos_trigger import (
    COSSnapshot,
    OpenHandsClient,
    StateStore,
    TriggerConfig,
    TriggerEngine,
)


def _snapshot(version: str) -> COSSnapshot:
    return COSSnapshot.build(
        "cos://bucket/prefix/",
        [
            {
                "key": f"prefix/data-{version}.tar.gz",
                "size_bytes": 100,
                "etag": version,
                "last_modified": "2026-07-22T00:00:00Z",
                "storage_class": "STANDARD",
            }
        ],
    )


class _Inventory:
    def __init__(self, snapshot: COSSnapshot):
        self.snapshot = snapshot

    def fetch(self) -> COSSnapshot:
        return self.snapshot


class _OpenHands:
    def __init__(self):
        self.dispatches = []
        self.statuses = {}

    def dispatch(self, snapshot, inventory_path):
        self.dispatches.append((snapshot.digest, inventory_path))
        conversation_id = f"conversation-{len(self.dispatches)}"
        self.statuses[conversation_id] = "running"
        return conversation_id

    def get_status(self, conversation_id):
        return self.statuses.get(conversation_id)


def _config(tmp_path: Path, *, bootstrap_existing: bool = True) -> TriggerConfig:
    task = tmp_path / "task.md"
    task.write_text("task", encoding="utf-8")
    cos_config = tmp_path / "cos.yaml"
    cos_config.write_text("private", encoding="utf-8")
    cos_config.chmod(0o600)
    return TriggerConfig(
        source_uri="cos://bucket/prefix/",
        cos_config_path=cos_config,
        task_spec_path=task,
        state_dir=tmp_path / "state",
        openhands_url="http://127.0.0.1:3000",
        poll_interval_seconds=30,
        settle_seconds=60,
        active_timeout_seconds=3600,
        bootstrap_existing=bootstrap_existing,
    )


def test_snapshot_digest_is_stable_across_listing_order():
    objects = [
        {"key": "prefix/b", "size_bytes": 2, "etag": "b"},
        {"key": "prefix/a", "size_bytes": 1, "etag": "a"},
    ]
    first = COSSnapshot.build("cos://bucket/prefix/", objects)
    second = COSSnapshot.build("cos://bucket/prefix/", reversed(objects))
    assert first.digest == second.digest
    assert [item["key"] for item in first.objects] == ["prefix/a", "prefix/b"]


def test_first_snapshot_is_only_recorded_as_baseline(tmp_path: Path):
    current_time = [1000.0]
    inventory = _Inventory(_snapshot("v1"))
    openhands = _OpenHands()
    store = StateStore(tmp_path / "state")
    store.prepare()
    engine = TriggerEngine(
        _config(tmp_path), inventory, openhands, store, now=lambda: current_time[0]
    )

    result = engine.tick()

    assert result["event"] == "baseline-recorded"
    assert openhands.dispatches == []
    assert store.load()["baseline_snapshot_digest"] == inventory.snapshot.digest


def test_changed_snapshot_settles_then_dispatches_once(tmp_path: Path):
    current_time = [1000.0]
    inventory = _Inventory(_snapshot("v1"))
    openhands = _OpenHands()
    store = StateStore(tmp_path / "state")
    store.prepare()
    engine = TriggerEngine(
        _config(tmp_path), inventory, openhands, store, now=lambda: current_time[0]
    )
    engine.tick()
    inventory.snapshot = _snapshot("v2")

    current_time[0] = 1010.0
    assert engine.tick()["event"] == "settling"
    current_time[0] = 1071.0
    dispatched = engine.tick()
    assert dispatched["event"] == "conversation-dispatched"
    assert len(openhands.dispatches) == 1

    current_time[0] = 1080.0
    assert engine.tick()["event"] == "conversation-active"
    assert len(openhands.dispatches) == 1


def test_finished_conversation_is_not_redispatched_after_restart(tmp_path: Path):
    current_time = [1000.0]
    inventory = _Inventory(_snapshot("v1"))
    openhands = _OpenHands()
    store = StateStore(tmp_path / "state")
    store.prepare()
    config = _config(tmp_path, bootstrap_existing=False)
    engine = TriggerEngine(
        config, inventory, openhands, store, now=lambda: current_time[0]
    )
    assert engine.tick()["event"] == "settling"
    current_time[0] = 1061.0
    dispatched = engine.tick()
    conversation_id = dispatched["conversation_id"]
    openhands.statuses[conversation_id] = "finished"

    restarted = TriggerEngine(
        config, inventory, openhands, store, now=lambda: current_time[0]
    )
    result = restarted.tick()

    assert result["event"] == "unchanged"
    assert store.load()["last_finished_digest"] == inventory.snapshot.digest
    assert len(openhands.dispatches) == 1


def test_error_blocks_same_snapshot(tmp_path: Path):
    current_time = [1000.0]
    inventory = _Inventory(_snapshot("v1"))
    openhands = _OpenHands()
    store = StateStore(tmp_path / "state")
    store.prepare()
    config = _config(tmp_path, bootstrap_existing=False)
    engine = TriggerEngine(
        config, inventory, openhands, store, now=lambda: current_time[0]
    )
    assert engine.tick()["event"] == "settling"
    current_time[0] = 1061.0
    dispatched = engine.tick()
    openhands.statuses[dispatched["conversation_id"]] = "error"

    assert engine.tick()["event"] == "conversation-blocked"
    assert engine.tick()["event"] == "snapshot-blocked"
    assert len(openhands.dispatches) == 1


def test_openhands_dispatch_payload_references_spec_without_credentials(tmp_path: Path):
    calls = []
    status_reads = 0

    def request_json(method, path, body):
        nonlocal status_reads
        calls.append((method, path, body))
        if method == "GET" and path.startswith("/api/v1/app-conversations?"):
            status_reads += 1
            return [None]
        if method == "POST":
            return {
                "id": "start-1",
                "status": "READY",
                "app_conversation_id": body["conversation_id"],
            }
        raise AssertionError(path)

    config = _config(tmp_path)
    client = OpenHandsClient(config, request_json=request_json)
    snapshot = _snapshot("v1")
    inventory_path = tmp_path / "inventory.json"

    conversation_id = client.dispatch(snapshot, inventory_path)

    assert conversation_id
    assert status_reads == 1
    payload = calls[-1][2]
    message = payload["initial_message"]["content"][0]["text"]
    assert str(config.task_spec_path) in message
    assert str(inventory_path) in message
    assert "secretid" not in message.lower()
    assert "secretkey" not in message.lower()
    assert payload["tags"]["snapshotdigest"] == snapshot.digest
