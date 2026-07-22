"""Deterministic COS watcher that submits one OpenHands task per stable snapshot.

The watcher deliberately contains no planning logic.  It owns only discovery,
debouncing, idempotency, single-flight dispatch, and durable status tracking.
OpenHands reads the checked-in task specification and performs the actual work.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol
from uuid import NAMESPACE_URL, uuid5


STATE_VERSION = 1
TERMINAL_FINISHED = {"finished"}
TERMINAL_BLOCKED = {
    "error",
    "stuck",
    "paused",
    "waiting_for_confirmation",
    "deleting",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class TriggerConfig:
    source_uri: str
    cos_config_path: Path
    task_spec_path: Path
    state_dir: Path
    openhands_url: str
    poll_interval_seconds: int = 300
    settle_seconds: int = 1800
    active_timeout_seconds: int = 604800
    bootstrap_existing: bool = True
    openhands_token_file: Optional[Path] = None

    @property
    def bucket(self) -> str:
        return urllib.parse.urlparse(self.source_uri).netloc

    @property
    def prefix(self) -> str:
        return urllib.parse.urlparse(self.source_uri).path.lstrip("/")

    @classmethod
    def load(cls, path: Path) -> "TriggerConfig":
        config_path = path.resolve(strict=True)
        _require_private_file(config_path, "trigger configuration")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        allowed = {
            "source_uri",
            "cos_config_path",
            "task_spec_path",
            "state_dir",
            "openhands_url",
            "poll_interval_seconds",
            "settle_seconds",
            "active_timeout_seconds",
            "bootstrap_existing",
            "openhands_token_file",
        }
        unexpected = sorted(set(raw) - allowed)
        if unexpected:
            raise ValueError(f"unknown trigger configuration keys: {unexpected}")
        source_uri = str(raw["source_uri"])
        parsed_source = urllib.parse.urlparse(source_uri)
        if parsed_source.scheme != "cos" or not parsed_source.netloc:
            raise ValueError("source_uri must be cos://<bucket>/<non-empty-prefix>/")
        if not parsed_source.path.strip("/"):
            raise ValueError("refusing to watch an entire COS bucket")
        openhands_url = str(raw.get("openhands_url", "http://127.0.0.1:3000"))
        parsed_openhands = urllib.parse.urlparse(openhands_url)
        if parsed_openhands.scheme != "http" or parsed_openhands.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("OpenHands URL must be a loopback HTTP endpoint")
        poll_interval = int(raw.get("poll_interval_seconds", 300))
        settle_seconds = int(raw.get("settle_seconds", 1800))
        active_timeout = int(raw.get("active_timeout_seconds", 604800))
        if poll_interval < 30:
            raise ValueError("poll_interval_seconds must be at least 30")
        if settle_seconds < poll_interval:
            raise ValueError("settle_seconds must be at least one poll interval")
        if active_timeout < 3600:
            raise ValueError("active_timeout_seconds must be at least one hour")
        token_value = raw.get("openhands_token_file")
        token_file = Path(token_value).resolve(strict=True) if token_value else None
        if token_file is not None:
            _require_private_file(token_file, "OpenHands token")
        config = cls(
            source_uri=source_uri,
            cos_config_path=Path(raw["cos_config_path"]).resolve(strict=True),
            task_spec_path=Path(raw["task_spec_path"]).resolve(strict=True),
            state_dir=Path(raw["state_dir"]).resolve(),
            openhands_url=openhands_url.rstrip("/"),
            poll_interval_seconds=poll_interval,
            settle_seconds=settle_seconds,
            active_timeout_seconds=active_timeout,
            bootstrap_existing=bool(raw.get("bootstrap_existing", True)),
            openhands_token_file=token_file,
        )
        _require_private_file(config.cos_config_path, "COS configuration")
        if not config.task_spec_path.is_file():
            raise ValueError(
                f"task specification does not exist: {config.task_spec_path}"
            )
        return config


def _require_private_file(path: Path, label: str) -> None:
    metadata = path.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"{label} must be owned by uid {os.getuid()}: {path}")
    if mode & 0o077:
        raise PermissionError(f"{label} must not grant group/other access: {path}")


@dataclass(frozen=True)
class COSSnapshot:
    source_uri: str
    digest: str
    object_count: int
    total_bytes: int
    objects: List[Dict[str, Any]]

    @classmethod
    def build(cls, source_uri: str, objects: Iterable[Dict[str, Any]]) -> "COSSnapshot":
        normalized = sorted(
            (
                {
                    "key": str(item["key"]),
                    "size_bytes": int(item["size_bytes"]),
                    "etag": str(item.get("etag", "")),
                    "last_modified": str(item.get("last_modified", "")),
                    "storage_class": str(item.get("storage_class", "")),
                }
                for item in objects
            ),
            key=lambda item: item["key"],
        )
        identity = {"source_uri": source_uri, "objects": normalized}
        digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
        return cls(
            source_uri=source_uri,
            digest=digest,
            object_count=len(normalized),
            total_bytes=sum(item["size_bytes"] for item in normalized),
            objects=normalized,
        )

    def as_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "rightarm.cos-snapshot/v1",
            "captured_at": utc_now(),
            "source_uri": self.source_uri,
            "digest": self.digest,
            "object_count": self.object_count,
            "total_bytes": self.total_bytes,
            "objects": self.objects,
        }


class InventoryProvider(Protocol):
    def fetch(self) -> COSSnapshot: ...


class TencentCOSInventory:
    """List COS through the official SDK without exposing credentials to OpenHands."""

    def __init__(self, config: TriggerConfig):
        self.config = config

    def fetch(self) -> COSSnapshot:
        # These imports are deliberately lazy.  The long-lived service uses the
        # host Python where the Tencent SDK and PyYAML are already installed;
        # unit tests can exercise the state machine without cloud dependencies.
        import yaml  # type: ignore
        from qcloud_cos import CosConfig, CosS3Client  # type: ignore

        raw = yaml.safe_load(self.config.cos_config_path.read_text(encoding="utf-8"))
        cos = raw.get("cos", {}) if isinstance(raw, dict) else {}
        base = cos.get("base", {}) if isinstance(cos, dict) else {}
        buckets = cos.get("buckets", []) if isinstance(cos, dict) else []
        bucket_config = next(
            (
                item
                for item in buckets
                if isinstance(item, dict) and item.get("name") == self.config.bucket
            ),
            None,
        )
        if not isinstance(bucket_config, dict):
            raise ValueError("watched bucket is absent from the private COS config")
        secret_id = base.get("secretid")
        secret_key = base.get("secretkey")
        region = bucket_config.get("region")
        if not all(
            isinstance(value, str) and value
            for value in (secret_id, secret_key, region)
        ):
            raise ValueError(
                "private COS config is missing credentials or bucket region"
            )
        token = base.get("sessiontoken") or None
        protocol = str(base.get("protocol", "https"))
        if protocol != "https":
            raise ValueError("COS inventory requires HTTPS")
        client = CosS3Client(
            CosConfig(
                Region=region,
                SecretId=secret_id,
                SecretKey=secret_key,
                Token=token,
                Scheme="https",
            )
        )
        marker = ""
        objects: List[Dict[str, Any]] = []
        while True:
            response = client.list_objects(
                Bucket=self.config.bucket,
                Prefix=self.config.prefix,
                Marker=marker,
                MaxKeys=1000,
            )
            for item in response.get("Contents", []) or []:
                key = str(item["Key"])
                size = int(item.get("Size", 0))
                if key.rstrip("/") == self.config.prefix.rstrip("/") and size == 0:
                    continue
                objects.append(
                    {
                        "key": key,
                        "size_bytes": size,
                        "etag": str(item.get("ETag", "")).strip('"'),
                        "last_modified": str(item.get("LastModified", "")),
                        "storage_class": str(item.get("StorageClass", "")),
                    }
                )
            truncated = str(response.get("IsTruncated", "false")).lower() == "true"
            if not truncated:
                break
            next_marker = response.get("NextMarker")
            if (
                not isinstance(next_marker, str)
                or not next_marker
                or next_marker == marker
            ):
                raise RuntimeError("COS pagination returned no advancing marker")
            marker = next_marker
        if not objects:
            raise RuntimeError("authorized COS prefix contains no data objects")
        return COSSnapshot.build(self.config.source_uri, objects)


class OpenHandsClient:
    def __init__(
        self,
        config: TriggerConfig,
        *,
        request_json: Optional[
            Callable[[str, str, Optional[Dict[str, Any]]], Any]
        ] = None,
    ):
        self.config = config
        self._request_json_override = request_json

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.openhands_token_file is not None:
            token = self.config.openhands_token_file.read_text(encoding="utf-8").strip()
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request_json(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Any:
        if self._request_json_override is not None:
            return self._request_json_override(method, path, body)
        data = _canonical_json(body) if body is not None else None
        request = urllib.request.Request(
            f"{self.config.openhands_url}{path}",
            data=data,
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenHands HTTP {exc.code}: {detail}") from exc

    def get_status(self, conversation_id: str) -> Optional[str]:
        query = urllib.parse.urlencode({"ids": conversation_id})
        values = self._request_json("GET", f"/api/v1/app-conversations?{query}")
        if not isinstance(values, list) or not values or values[0] is None:
            return None
        status_value = values[0].get("execution_status")
        # The conversation already exists while its sandbox is still starting,
        # even though no execution status has been assigned yet.  Keep this
        # distinct from a missing conversation so a lost POST response cannot
        # cause a duplicate create attempt.
        return str(status_value) if status_value is not None else "unknown"

    def dispatch(self, snapshot: COSSnapshot, inventory_path: Path) -> str:
        dispatch_key = f"rightarm-cos:{snapshot.source_uri}:{snapshot.digest}"
        conversation_id = str(uuid5(NAMESPACE_URL, dispatch_key))
        if self.get_status(conversation_id) is not None:
            return conversation_id
        prompt = (
            "This task was started by the deterministic right-arm COS trigger. "
            "Read and strictly execute the complete task specification at "
            f"{self.config.task_spec_path}. Continue autonomously through every "
            "stage until its Done when contract passes or one evidenced external "
            "blocker is reached. Do not wait for routine human confirmation.\n\n"
            f"Trigger snapshot digest: {snapshot.digest}\n"
            f"Verified trigger inventory: {inventory_path}\n"
            f"Authorized COS source: {snapshot.source_uri}\n"
            "Treat the inventory as trigger evidence, then independently rerun the "
            "task specification's inventory, integrity, disk, data, and training gates."
        )
        body: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "initial_message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
            "title": f"RightArm COS {snapshot.digest[:12]}",
            "trigger": "automation",
            "tags": {
                "rightarmtrigger": "cosv1",
                "snapshotdigest": snapshot.digest,
            },
        }
        start_task = self._request_json("POST", "/api/v1/app-conversations", body)
        if not isinstance(start_task, dict) or "id" not in start_task:
            raise RuntimeError("OpenHands returned an invalid start-task response")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status_value = str(start_task.get("status", ""))
            if status_value == "READY":
                actual = start_task.get("app_conversation_id")
                if str(actual) != conversation_id:
                    raise RuntimeError("OpenHands returned a different conversation id")
                return conversation_id
            if status_value == "ERROR":
                raise RuntimeError(
                    f"OpenHands conversation failed to start: {start_task.get('detail')}"
                )
            time.sleep(0.5)
            query = urllib.parse.urlencode({"ids": str(start_task["id"])})
            values = self._request_json(
                "GET", f"/api/v1/app-conversations/start-tasks?{query}"
            )
            if not isinstance(values, list) or not values or values[0] is None:
                raise RuntimeError("OpenHands start task disappeared")
            start_task = values[0]
        raise TimeoutError("OpenHands conversation start timed out")


class StateStore:
    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / "state.json"
        self.lock_path = root / "trigger.lock"
        self.force_path = root / "trigger-current.request"
        self.inventory_dir = root / "inventories"

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.inventory_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.inventory_dir, 0o700)

    def load(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": STATE_VERSION}
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != STATE_VERSION:
            raise ValueError("unsupported trigger state version")
        return value

    def save(self, state: Dict[str, Any]) -> None:
        state["schema_version"] = STATE_VERSION
        _atomic_private_json(self.state_path, state)

    def save_inventory(self, snapshot: COSSnapshot) -> Path:
        path = self.inventory_dir / f"{snapshot.digest}.json"
        if not path.exists():
            _atomic_private_json(path, snapshot.as_record())
        return path

    def force_requested(self) -> bool:
        return self.force_path.is_file()

    def request_force(self) -> None:
        _atomic_private_json(
            self.force_path, {"requested_at": utc_now(), "reason": "operator-request"}
        )

    def consume_force(self) -> None:
        self.force_path.unlink(missing_ok=True)


class TriggerEngine:
    def __init__(
        self,
        config: TriggerConfig,
        inventory: InventoryProvider,
        openhands: OpenHandsClient,
        store: StateStore,
        *,
        now: Callable[[], float] = time.time,
    ):
        self.config = config
        self.inventory = inventory
        self.openhands = openhands
        self.store = store
        self.now = now

    def tick(self) -> Dict[str, Any]:
        snapshot = self.inventory.fetch()
        inventory_path = self.store.save_inventory(snapshot)
        state = self.store.load()
        current_time = self.now()
        state["last_poll_at"] = utc_now()
        state["current_snapshot"] = {
            "digest": snapshot.digest,
            "object_count": snapshot.object_count,
            "total_bytes": snapshot.total_bytes,
            "inventory_path": str(inventory_path),
        }
        force = self.store.force_requested()
        previous_digest = state.get("last_observed_digest")
        if previous_digest is None:
            state["last_observed_digest"] = snapshot.digest
            state["last_change_epoch"] = current_time
            state["last_change_at"] = utc_now()
            if self.config.bootstrap_existing and not force:
                state["baseline_snapshot_digest"] = snapshot.digest
                state["last_dispatched_digest"] = snapshot.digest
                state["last_event"] = "baseline-recorded"
                state["last_error"] = None
                self.store.save(state)
                return {"event": "baseline-recorded", "digest": snapshot.digest}
        elif previous_digest != snapshot.digest:
            state["last_observed_digest"] = snapshot.digest
            state["last_change_epoch"] = current_time
            state["last_change_at"] = utc_now()
            state["last_event"] = "snapshot-changed"

        active = state.get("active")
        if isinstance(active, dict):
            conversation_id = str(active["conversation_id"])
            status_value = self.openhands.get_status(conversation_id)
            if status_value is None:
                state["last_error"] = "active OpenHands conversation is not visible"
                self.store.save(state)
                return {
                    "event": "monitor-unavailable",
                    "conversation_id": conversation_id,
                }
            active["status"] = status_value
            active["last_checked_at"] = utc_now()
            state["active"] = active
            if status_value in TERMINAL_FINISHED:
                # "finished" is an OpenHands execution state, not proof that
                # every business gate passed.  The task report remains the
                # source of truth for success versus an evidenced blocker.
                state["last_finished_digest"] = active["snapshot_digest"]
                state["last_finished_at"] = utc_now()
                state["last_finished_conversation_id"] = conversation_id
                state["active"] = None
                state["last_event"] = "conversation-finished"
            elif status_value in TERMINAL_BLOCKED:
                state["blocked"] = {
                    "snapshot_digest": active["snapshot_digest"],
                    "conversation_id": conversation_id,
                    "status": status_value,
                    "blocked_at": utc_now(),
                }
                state["active"] = None
                state["last_event"] = "conversation-blocked"
                state["last_error"] = f"OpenHands conversation is {status_value}"
                self.store.save(state)
                return {
                    "event": "conversation-blocked",
                    "conversation_id": conversation_id,
                    "status": status_value,
                }
            else:
                started_epoch = float(active.get("started_epoch", current_time))
                if current_time - started_epoch > self.config.active_timeout_seconds:
                    state["blocked"] = {
                        "snapshot_digest": active["snapshot_digest"],
                        "conversation_id": conversation_id,
                        "status": "timeout",
                        "blocked_at": utc_now(),
                    }
                    state["active"] = None
                    state["last_event"] = "conversation-timeout"
                    state["last_error"] = (
                        "OpenHands conversation exceeded active timeout"
                    )
                    self.store.save(state)
                    return {
                        "event": "conversation-timeout",
                        "conversation_id": conversation_id,
                    }
                state["last_error"] = None
                self.store.save(state)
                return {
                    "event": "conversation-active",
                    "conversation_id": conversation_id,
                    "status": status_value,
                }

        blocked = state.get("blocked")
        if (
            isinstance(blocked, dict)
            and blocked.get("snapshot_digest") == snapshot.digest
            and not force
        ):
            self.store.save(state)
            return {"event": "snapshot-blocked", "digest": snapshot.digest}
        if state.get("last_dispatched_digest") == snapshot.digest and not force:
            state["last_event"] = "unchanged"
            state["last_error"] = None
            self.store.save(state)
            return {"event": "unchanged", "digest": snapshot.digest}
        last_change = float(state.get("last_change_epoch", current_time))
        stable_for = current_time - last_change
        if stable_for < self.config.settle_seconds and not force:
            state["last_event"] = "settling"
            state["last_error"] = None
            self.store.save(state)
            return {
                "event": "settling",
                "digest": snapshot.digest,
                "remaining_seconds": int(self.config.settle_seconds - stable_for),
            }

        conversation_id = self.openhands.dispatch(snapshot, inventory_path)
        state["last_dispatched_digest"] = snapshot.digest
        state["last_dispatched_at"] = utc_now()
        state["active"] = {
            "snapshot_digest": snapshot.digest,
            "conversation_id": conversation_id,
            "status": "starting",
            "started_at": utc_now(),
            "started_epoch": current_time,
            "inventory_path": str(inventory_path),
        }
        state["blocked"] = None
        state["last_event"] = "conversation-dispatched"
        state["last_error"] = None
        self.store.consume_force()
        self.store.save(state)
        return {
            "event": "conversation-dispatched",
            "digest": snapshot.digest,
            "conversation_id": conversation_id,
        }


class ProcessLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream: Optional[Any] = None

    def __enter__(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stream = self.path.open("a+", encoding="utf-8")
        os.chmod(self.path, 0o600)
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError(
                "another trigger process already holds the lock"
            ) from exc
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"pid={os.getpid()} started={utc_now()}\n")
        self.stream.flush()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


def _emit(value: Dict[str, Any]) -> None:
    print(json.dumps({"time": utc_now(), **value}, ensure_ascii=False), flush=True)


def _record_daemon_error(store: StateStore, exc: BaseException) -> None:
    try:
        state = store.load()
        state["last_poll_at"] = utc_now()
        state["last_event"] = "poll-error"
        state["last_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        store.save(state)
    except Exception:
        pass


def _build_runtime(
    config_path: Path,
) -> tuple[TriggerConfig, StateStore, TriggerEngine]:
    config = TriggerConfig.load(config_path)
    store = StateStore(config.state_dir)
    store.prepare()
    engine = TriggerEngine(
        config,
        TencentCOSInventory(config),
        OpenHandsClient(config),
        store,
    )
    return config, store, engine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rightarm-cos-trigger")
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("daemon")
    subparsers.add_parser("once")
    subparsers.add_parser("status")
    subparsers.add_parser("trigger-current")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config, store, engine = _build_runtime(args.config)
    if args.command == "status":
        print(json.dumps(store.load(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.command == "trigger-current":
        store.request_force()
        _emit({"event": "trigger-current-requested"})
        return 0
    try:
        with ProcessLock(store.lock_path):
            if args.command == "once":
                _emit(engine.tick())
                return 0
            _emit(
                {
                    "event": "daemon-started",
                    "source_uri": config.source_uri,
                    "poll_interval_seconds": config.poll_interval_seconds,
                }
            )
            while True:
                try:
                    event = engine.tick()
                    if event.get("event") not in {"unchanged", "conversation-active"}:
                        _emit(event)
                except Exception as exc:  # noqa: BLE001 - daemon must survive poll errors
                    _record_daemon_error(store, exc)
                    _emit(
                        {
                            "event": "poll-error",
                            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                        }
                    )
                time.sleep(config.poll_interval_seconds)
    except RuntimeError as exc:
        if "already holds the lock" in str(exc):
            return 0
        raise


if __name__ == "__main__":
    sys.exit(main())
