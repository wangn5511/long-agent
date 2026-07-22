"""Shared result/failure manifest protocol for every rightarm-mlops job."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROTOCOL = "rightarm.mlops.job-result/v1"


def sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def output_record(kind: str, path: Path) -> Dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size,
    }


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def deterministic_tar(source_root: Path, output_path: Path) -> None:
    """Archive a directory with normalized metadata and stable member order."""

    source_root = source_root.resolve(strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(output_path.parent), prefix=f".{output_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for path in sorted(
                item for item in source_root.rglob("*") if item.is_file()
            ):
                relative = path.relative_to(source_root)
                info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a regular-file tar without traversal, links, or device nodes."""

    archive_path = archive_path.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"tar member escapes destination: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"tar member type is not allowed: {member.name!r}")
            target = (destination_root / relative).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"tar member escapes destination: {member.name!r}")
        archive.extractall(destination_root, filter="data")


def success(
    *,
    output_dir: Path,
    job_kind: str,
    inputs: Iterable[Dict[str, Any]] = (),
    outputs: Iterable[Dict[str, Any]] = (),
    metrics: Optional[Dict[str, Any]] = None,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "protocol": PROTOCOL,
        "job_kind": job_kind,
        "status": "succeeded",
        "inputs": list(inputs),
        "outputs": list(outputs),
        "metrics": metrics or {},
        "runtime": {
            "python": platform.python_version(),
            **(runtime or {}),
        },
    }
    atomic_json(output_dir / "result.json", result)
    return result


def failure(
    *,
    output_dir: Path,
    job_kind: str,
    failure_class: str,
    message: str,
    evidence: Iterable[str] = (),
    retryable: bool = False,
) -> Dict[str, Any]:
    result = {
        "protocol": PROTOCOL,
        "job_kind": job_kind,
        "status": "failed",
        "failure_class": failure_class,
        "message": message,
        "retryable": retryable,
        "evidence": list(evidence),
    }
    atomic_json(output_dir / "failure.json", result)
    return result


def parse_output_specs(values: List[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"output must use KIND=PATH syntax: {value!r}")
        kind, raw_path = value.split("=", 1)
        if not kind or not raw_path:
            raise ValueError(f"output must use KIND=PATH syntax: {value!r}")
        parsed.append((kind, Path(raw_path)))
    return parsed
