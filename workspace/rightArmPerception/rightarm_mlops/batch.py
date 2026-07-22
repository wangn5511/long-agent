"""Sealed business-batch inspection and deterministic local landing."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Set, Tuple
from urllib.parse import unquote, urlparse

from rightarm_mlops.protocol import atomic_json, sha256_file

LABELS = {
    "WAVE_IN",
    "WAVE_OUT",
    "FIVE",
    "CLICK",
    "ROTATE",
    "SLIDE_DOWN",
    "SLIDE_UP",
    "OK",
    "CURSOR",
    "GUN",
    "NONE",
}
CSV_COLUMNS = ("视频名", "起始帧号", "终止帧号", "类别名")


def local_source_path(source_uri: str) -> Path:
    parsed = urlparse(source_uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve(strict=True)
    if not parsed.scheme:
        return Path(source_uri).resolve(strict=True)
    raise ValueError(
        f"{parsed.scheme}:// data must first be materialized by the configured "
        "TOS/COS adapter; this job never receives cloud credentials directly"
    )


def inspect_batch(
    batch_dir: Path, *, require_marker: str = "_SUCCESS"
) -> Dict[str, Any]:
    batch_dir = batch_dir.resolve(strict=True)
    marker = batch_dir / require_marker
    if not marker.is_file():
        raise ValueError(f"batch is not sealed: missing {marker}")
    manifest_path = batch_dir / "batch.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "rightarm.gesture.batch/v1":
        raise ValueError("manifest schema_version must be rightarm.gesture.batch/v1")
    for field in ("batch_id", "created_at", "producer", "label_ontology", "objects"):
        if field not in manifest:
            raise ValueError(f"manifest is missing {field!r}")
    if manifest["label_ontology"] != "rightarm-gesture-11/v1":
        raise ValueError(f"unsupported label ontology: {manifest['label_ontology']!r}")

    verified_objects = []
    seen_hashes: Set[str] = set()
    for item in manifest["objects"]:
        path = _resolve_manifest_object(batch_dir, manifest["batch_id"], item)
        digest, size = sha256_file(path)
        _assert_hash_and_size(item, digest, size, str(path))
        duplicate = digest in seen_hashes
        seen_hashes.add(digest)
        verified_objects.append(
            {
                "path": str(path.relative_to(batch_dir)),
                "sha256": digest,
                "size_bytes": size,
                "duplicate_content": duplicate,
                "device_id": item.get("device_id"),
                "session_id": item.get("session_id"),
                "subject_group_id": item.get("subject_group_id"),
            }
        )

    annotations = manifest.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError("manifest annotations must be an object")
    annotation_path = _resolve_manifest_object(
        batch_dir, manifest["batch_id"], annotations, default="annotations.csv"
    )
    digest, size = sha256_file(annotation_path)
    _assert_hash_and_size(annotations, digest, size, str(annotation_path))
    annotation_report = inspect_annotations(annotation_path)
    video_names = {Path(item["path"]).name for item in verified_objects}
    missing_videos = sorted(set(annotation_report["video_names"]) - video_names)
    if missing_videos:
        raise ValueError(f"annotations reference missing videos: {missing_videos[:20]}")
    return {
        "schema_version": "rightarm.gesture.data-validation/v1",
        "batch_id": manifest["batch_id"],
        "batch_dir": str(batch_dir),
        "object_count": len(verified_objects),
        "unique_object_count": len(seen_hashes),
        "duplicate_object_count": len(verified_objects) - len(seen_hashes),
        "objects": verified_objects,
        "annotations": {
            "path": str(annotation_path.relative_to(batch_dir)),
            "sha256": digest,
            "size_bytes": size,
            **annotation_report,
        },
        "source_manifest": manifest,
    }


def inspect_annotations(path: Path) -> Dict[str, Any]:
    counts = {label: 0 for label in sorted(LABELS)}
    video_names: Set[str] = set()
    spans: Dict[Tuple[str, int, int], str] = {}
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in CSV_COLUMNS
        ):
            raise ValueError(f"annotations CSV must contain columns {CSV_COLUMNS!r}")
        for line_number, row in enumerate(reader, 2):
            rows += 1
            video = (row.get("视频名") or "").strip()
            label = (row.get("类别名") or "").strip()
            if not video:
                raise ValueError(f"annotations row {line_number} has no video name")
            if label not in LABELS:
                raise ValueError(
                    f"annotations row {line_number} has unknown label {label!r}"
                )
            try:
                start = int(row["起始帧号"])
                end = int(row["终止帧号"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"annotations row {line_number} has non-integer frame bounds"
                ) from exc
            if start < 0 or end < start:
                raise ValueError(
                    f"annotations row {line_number} has invalid frame span {start}:{end}"
                )
            key = (video, start, end)
            previous = spans.get(key)
            if previous is not None and previous != label:
                raise ValueError(
                    f"conflicting labels for {video} frames {start}:{end}: "
                    f"{previous!r} and {label!r}"
                )
            spans[key] = label
            counts[label] += 1
            video_names.add(video)
    if rows == 0:
        raise ValueError("annotations CSV is empty")
    return {
        "annotation_count": rows,
        "label_counts": counts,
        "video_names": sorted(video_names),
    }


def land_batch(batch_dir: Path, output_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    report = inspect_batch(batch_dir)
    snapshot = output_dir / "raw-dataset"
    objects_dir = snapshot / "objects" / "sha256"
    objects_dir.mkdir(parents=True, exist_ok=True)
    canonical_objects = []
    for item in report["objects"]:
        source = Path(report["batch_dir"]) / item["path"]
        target = objects_dir / item["sha256"][:2] / item["sha256"]
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_verified(source, target, item["sha256"], item["size_bytes"])
        canonical_objects.append(
            {
                **item,
                "source_relative_path": item["path"],
                "path": str(target.relative_to(snapshot)),
            }
        )
    annotation_source = Path(report["batch_dir"]) / report["annotations"]["path"]
    annotation_target = snapshot / "annotations.csv"
    _copy_verified(
        annotation_source,
        annotation_target,
        report["annotations"]["sha256"],
        report["annotations"]["size_bytes"],
    )
    canonical = {
        "schema_version": "rightarm.gesture.raw-dataset/v1",
        "batch_id": report["batch_id"],
        "objects": canonical_objects,
        "annotations": {
            **report["annotations"],
            "path": str(annotation_target.relative_to(snapshot)),
        },
        "source_manifest": report["source_manifest"],
    }
    manifest_path = snapshot / "raw-dataset.manifest.json"
    atomic_json(manifest_path, canonical)
    return manifest_path, report


def _resolve_manifest_object(
    batch_dir: Path,
    batch_id: str,
    item: Dict[str, Any],
    *,
    default: str = "",
) -> Path:
    raw_path = item.get("path")
    if raw_path:
        relative = PurePosixPath(raw_path)
    else:
        uri = item.get("uri", "")
        parsed = urlparse(uri)
        marker = f"/batches/{batch_id}/"
        if marker in parsed.path:
            relative = PurePosixPath(parsed.path.split(marker, 1)[1])
        elif default:
            relative = PurePosixPath(default)
        else:
            candidates = list(batch_dir.rglob(Path(parsed.path).name))
            if len(candidates) != 1:
                raise ValueError(
                    f"cannot map manifest URI to one local object: {uri!r}"
                )
            return candidates[0].resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"manifest object escapes the batch directory: {relative}")
    path = (batch_dir / Path(*relative.parts)).resolve(strict=True)
    if batch_dir not in path.parents and path != batch_dir:
        raise ValueError(f"manifest object escapes the batch directory: {relative}")
    return path


def _assert_hash_and_size(
    manifest: Dict[str, Any], digest: str, size: int, display: str
) -> None:
    if manifest.get("sha256") != digest:
        raise ValueError(f"sha256 mismatch for {display}")
    if int(manifest.get("bytes", manifest.get("size_bytes", -1))) != size:
        raise ValueError(f"size mismatch for {display}")


def _copy_verified(source: Path, target: Path, digest: str, size: int) -> None:
    if target.exists():
        target_digest, target_size = sha256_file(target)
        if (target_digest, target_size) != (digest, size):
            raise ValueError(f"existing immutable object is corrupt: {target}")
        return
    temporary = target.with_suffix(".tmp")
    with source.open("rb") as input_, temporary.open("xb") as output:
        shutil.copyfileobj(input_, output, length=1024 * 1024)
    target_digest, target_size = sha256_file(temporary)
    if (target_digest, target_size) != (digest, size):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"copy verification failed for {source}")
    temporary.replace(target)
