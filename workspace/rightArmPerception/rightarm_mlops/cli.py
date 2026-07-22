"""Command-line entry point for deterministic right-arm MLOps jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from rightarm_mlops.batch import inspect_batch, land_batch, local_source_path
from rightarm_mlops.cos_materialize import materialize_cos_batch
from rightarm_mlops.dataset import (
    build_portable_dataset_bundle,
    compile_rtm_dataset,
    materialize_dataset_bundle,
)
from rightarm_mlops.evaluation import evaluate_predictions
from rightarm_mlops.pipeline import prepare_raw_rtm_splits
from rightarm_mlops.protocol import (
    deterministic_tar,
    failure,
    output_record,
    parse_output_specs,
    safe_extract_tar,
    sha256_file,
    success,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rightarm-mlops")
    subparsers = parser.add_subparsers(dest="job_kind", required=True)

    inspect = subparsers.add_parser("inspect-batch")
    inspect.add_argument("--source-uri", required=True)
    inspect.add_argument("--require-marker", default="_SUCCESS")
    inspect.add_argument("--min-training-annotations", type=int, default=1)
    inspect.add_argument("--output-dir", required=True, type=Path)

    materialize = subparsers.add_parser("materialize-cos")
    materialize.add_argument("--source-uri", required=True)
    materialize.add_argument("--output-dir", required=True, type=Path)

    land = subparsers.add_parser("land-batch")
    land.add_argument("--source-uri", required=True)
    land.add_argument("--require-marker", default="_SUCCESS")
    land.add_argument("--output-dir", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--champion-evaluation", type=Path)
    evaluate.add_argument("--output-dir", required=True, type=Path)

    release = subparsers.add_parser("build-release")
    release.add_argument("--artifact", action="append", default=[])
    release.add_argument("--output-dir", required=True, type=Path)

    dataset = subparsers.add_parser("build-dataset")
    dataset.add_argument("--train-rtm-root", type=Path)
    dataset.add_argument("--val-rtm-root", type=Path)
    dataset.add_argument("--raw-dataset", type=Path)
    dataset.add_argument("--repo-root", type=Path, default=Path.cwd())
    dataset.add_argument("--det-onnx", type=Path)
    dataset.add_argument("--person-pose-onnx", type=Path)
    dataset.add_argument("--hand-pose-onnx", type=Path)
    dataset.add_argument("--frame-base", choices=(0, 1), type=int, default=0)
    dataset.add_argument("--val-ratio", type=float, default=0.2)
    dataset.add_argument("--seed", type=int, default=42)
    dataset.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    dataset.add_argument("--workers", type=int, default=1)
    dataset.add_argument("--frozen-test-set", type=Path)
    dataset.add_argument("--min-primary-valid-ratio", type=float, default=0.5)
    dataset.add_argument("--output-dir", required=True, type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--base-model", required=True, type=Path)
    train.add_argument("--gpus", required=True, type=int)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--workers", type=int, default=12)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--repo-root", type=Path, default=Path.cwd())
    train.add_argument("--code-sha", required=True)
    train.add_argument("--image-digest", required=True)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--output-dir", required=True, type=Path)

    for name in (
        "run-evaluation",
        "export-onnx",
        "build-calibration",
        "export-rknn",
        "verify-conversion",
        "verify-board",
        "deploy",
        "observe-deployment",
        "rollback",
    ):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--command-json",
            required=True,
            help="JSON array argv; executed without a shell",
        )
        command.add_argument(
            "--output",
            action="append",
            default=[],
            help="Expected immutable output as KIND=PATH",
        )
        command.add_argument("--input-artifact", action="append", default=[])
        command.add_argument("--code-sha")
        command.add_argument("--image-digest")
        command.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _run(args, output_dir)
    except subprocess.CalledProcessError as exc:
        failure_class, retryable = _classify_process_failure(
            exc, output_dir / "job.log"
        )
        result = failure(
            output_dir=output_dir,
            job_kind=args.job_kind,
            failure_class=failure_class,
            message=f"command exited with status {exc.returncode}",
            evidence=[str(output_dir / "job.log")],
            retryable=retryable,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 1
    except Exception as exc:  # noqa: BLE001 - the CLI must always emit failure.json
        result = failure(
            output_dir=output_dir,
            job_kind=args.job_kind,
            failure_class=_classify_exception(exc),
            message=str(exc),
            evidence=[],
            retryable=False,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


def _run(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    if args.job_kind == "materialize-cos":
        report = materialize_cos_batch(args.source_uri, output_dir)
        report_path = output_dir / "cos-materialization-report.json"
        return success(
            output_dir=output_dir,
            job_kind=args.job_kind,
            inputs=[{"kind": "cos-prefix", "uri": report["source_uri"]}],
            outputs=[output_record("cos-materialization-report", report_path)],
            metrics={
                "batch_id": report["batch_id"],
                "declared_bytes": report["declared_bytes"],
                "materialized_source_uri": report["materialized_source_uri"],
                "verified_object_count": report["verified_object_count"],
            },
        )
    if args.job_kind == "inspect-batch":
        batch_dir = local_source_path(args.source_uri)
        report = inspect_batch(batch_dir, require_marker=args.require_marker)
        if args.min_training_annotations < 1:
            raise ValueError("--min-training-annotations must be positive")
        report["skip_training"] = (
            report["annotations"]["annotation_count"] < args.min_training_annotations
        )
        report_path = output_dir / "data-validation-report.json"
        from rightarm_mlops.protocol import atomic_json

        atomic_json(report_path, report)
        manifest_path = batch_dir / "batch.manifest.json"
        return success(
            output_dir=output_dir,
            job_kind=args.job_kind,
            inputs=[_input_record("source-manifest", manifest_path)],
            outputs=[output_record("data-validation-report", report_path)],
            metrics={
                "object_count": report["object_count"],
                "unique_object_count": report["unique_object_count"],
                "annotation_count": report["annotations"]["annotation_count"],
            },
        )
    if args.job_kind == "land-batch":
        batch_dir = local_source_path(args.source_uri)
        if not (batch_dir / args.require_marker).is_file():
            raise ValueError(f"batch is not sealed with {args.require_marker}")
        manifest_path, report = land_batch(batch_dir, output_dir)
        source_manifest_path = output_dir / "source-batch.manifest.json"
        from rightarm_mlops.protocol import atomic_json

        atomic_json(source_manifest_path, report["source_manifest"])
        snapshot_path = manifest_path.parent
        archive_path = output_dir / "raw-dataset.tar"
        deterministic_tar(snapshot_path, archive_path)
        return success(
            output_dir=output_dir,
            job_kind=args.job_kind,
            inputs=[
                _input_record("source-manifest", batch_dir / "batch.manifest.json")
            ],
            outputs=[
                output_record("source-manifest", source_manifest_path),
                output_record("raw-dataset", archive_path),
                output_record("raw-dataset-manifest", manifest_path),
            ],
            metrics={
                "object_count": report["object_count"],
                "unique_object_count": report["unique_object_count"],
            },
        )
    if args.job_kind == "evaluate":
        evaluation_path = output_dir / "evaluation.json"
        evaluation = evaluate_predictions(
            args.predictions,
            evaluation_path,
            champion_evaluation=args.champion_evaluation,
        )
        return success(
            output_dir=output_dir,
            job_kind=args.job_kind,
            inputs=[_input_record("predictions", args.predictions)],
            outputs=[output_record("evaluation-report", evaluation_path)],
            metrics=evaluation["metrics"],
        )
    if args.job_kind == "build-release":
        return _build_release(args.artifact, output_dir)
    if args.job_kind == "train":
        return _run_train(args, output_dir)
    if args.job_kind == "build-dataset":
        return _run_build_dataset(args, output_dir)
    return _run_external(args, output_dir)


def _run_build_dataset(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    raw_mode = args.raw_dataset is not None
    roots_mode = args.train_rtm_root is not None or args.val_rtm_root is not None
    if raw_mode == roots_mode:
        raise ValueError(
            "provide either --raw-dataset or both --train-rtm-root/--val-rtm-root"
        )
    pipeline_report = None
    if raw_mode:
        model_values = (args.det_onnx, args.person_pose_onnx, args.hand_pose_onnx)
        if any(value is None for value in model_values):
            raise ValueError(
                "raw mode requires --det-onnx, --person-pose-onnx and --hand-pose-onnx"
            )
        train_root, val_root, pipeline_report = prepare_raw_rtm_splits(
            raw_dataset=args.raw_dataset,
            repo_root=args.repo_root,
            output_root=output_dir / "data-pipeline",
            det_onnx=args.det_onnx,
            person_pose_onnx=args.person_pose_onnx,
            hand_pose_onnx=args.hand_pose_onnx,
            frame_base=args.frame_base,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=args.device,
            workers=args.workers,
        )
    else:
        if args.train_rtm_root is None or args.val_rtm_root is None:
            raise ValueError("both RTM roots are required")
        train_root, val_root = args.train_rtm_root, args.val_rtm_root
    dataset_path = output_dir / "rightarm-gesture-11.pkl"
    dataset_bundle_path = output_dir / "rightarm-gesture-11.tar"
    report_path = output_dir / "dataset-report.json"
    report = compile_rtm_dataset(
        train_root=train_root,
        val_root=val_root,
        output_pkl=dataset_path,
        report_path=report_path,
        frozen_test_set=args.frozen_test_set,
        min_primary_valid_ratio=args.min_primary_valid_ratio,
    )
    build_portable_dataset_bundle(dataset_path, dataset_bundle_path)
    inputs = [
        {"kind": "rtm-train-root", "path": str(train_root.resolve())},
        {"kind": "rtm-val-root", "path": str(val_root.resolve())},
    ]
    if args.raw_dataset:
        inputs.append(_input_record("raw-dataset", args.raw_dataset))
    if args.frozen_test_set:
        inputs.append(_input_record("frozen-test-set", args.frozen_test_set))
    return success(
        output_dir=output_dir,
        job_kind="build-dataset",
        inputs=inputs,
        outputs=[
            output_record("mmaction-dataset", dataset_bundle_path),
            output_record("mmaction-dataset-index", dataset_path),
            output_record("dataset-report", report_path),
            *(
                [
                    output_record(
                        "data-pipeline-report",
                        output_dir / "data-pipeline" / "data-pipeline-report.json",
                    )
                ]
                if pipeline_report is not None
                else []
            ),
        ],
        metrics={
            "train_samples": report["train_samples"],
            "val_samples": report["val_samples"],
            "low_quality_samples": report["low_quality_samples"],
        },
    )


def _run_train(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    if args.gpus < 1 or args.batch_size < 1 or args.workers < 0 or args.max_epochs < 1:
        raise ValueError("invalid training resource or schedule parameter")
    repo_root = args.repo_root.resolve(strict=True)
    mmaction_root = repo_root / "training" / "mmaction2"
    if not mmaction_root.is_dir():
        raise ValueError(f"MMAction2 checkout does not exist: {mmaction_root}")
    config = args.config.resolve(strict=True)
    dataset_artifact = args.dataset.resolve(strict=True)
    if dataset_artifact.suffix == ".tar":
        dataset_root = output_dir / "dataset-input"
        safe_extract_tar(dataset_artifact, dataset_root)
        dataset = materialize_dataset_bundle(
            dataset_root,
            dataset_root / "rightarm-gesture-11.pkl",
            output_dir / "materialized-rightarm-gesture-11.pkl",
        )
    else:
        dataset = dataset_artifact
    base_model = args.base_model.resolve(strict=True)
    work_dir = output_dir / "training"
    work_dir.mkdir(parents=True, exist_ok=True)
    overrides = {
        "load_from": str(base_model),
        "train_dataloader.dataset.dataset.ann_file": str(dataset),
        "val_dataloader.dataset.ann_file": str(dataset),
        "test_dataloader.dataset.ann_file": str(dataset),
        "train_dataloader.batch_size": args.batch_size,
        "val_dataloader.batch_size": args.batch_size,
        "test_dataloader.batch_size": args.batch_size,
        "train_dataloader.num_workers": args.workers,
        "val_dataloader.num_workers": args.workers,
        "test_dataloader.num_workers": args.workers,
        "randomness.seed": args.seed,
        "train_cfg.max_epochs": args.max_epochs,
        "param_scheduler.1.T_max": max(args.max_epochs - 3, 1),
        "param_scheduler.1.end": args.max_epochs,
    }
    resolved = {
        "schema_version": "rightarm.gesture.training-command/v1",
        "config": str(config),
        "dataset": str(dataset),
        "base_model": str(base_model),
        "work_dir": str(work_dir),
        "gpus": args.gpus,
        "overrides": overrides,
        "code_sha": args.code_sha,
        "image_digest": args.image_digest,
    }
    from rightarm_mlops.protocol import atomic_json

    resolved_path = output_dir / "resolved-training-command.json"
    atomic_json(resolved_path, resolved)
    command = [
        "bash",
        "tools/dist_train.sh",
        str(config),
        str(args.gpus),
        "--work-dir",
        str(work_dir),
        "--cfg-options",
        *[f"{key}={_mmengine_value(value)}" for key, value in overrides.items()],
    ]
    if args.dry_run:
        return success(
            output_dir=output_dir,
            job_kind="train-plan",
            inputs=[
                _input_record("mmaction-dataset", dataset_artifact),
                _input_record("base-model", base_model),
                _input_record("training-config-source", config),
            ],
            outputs=[output_record("training-command", resolved_path)],
            runtime={
                "code_sha": args.code_sha,
                "image_digest": args.image_digest,
                "argv": command,
                "dry_run": True,
            },
        )
    log_path = output_dir / "job.log"
    with log_path.open("wb") as log:
        subprocess.run(
            command,
            cwd=mmaction_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    checkpoints = sorted(
        work_dir.glob("best_*.pth"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        raise ValueError(
            f"training succeeded but emitted no best checkpoint in {work_dir}"
        )
    best = checkpoints[-1]
    last_checkpoint = _resolve_last_checkpoint(work_dir)
    metrics_path = output_dir / "training-metrics.json"
    _collect_training_metrics(work_dir, metrics_path)
    outputs = [
        output_record("checkpoint", best),
        output_record("last-checkpoint", last_checkpoint),
        output_record("training-metrics", metrics_path),
        output_record("training-config", resolved_path),
        output_record("training-log", log_path),
    ]
    return success(
        output_dir=output_dir,
        job_kind="train",
        inputs=[
            _input_record("mmaction-dataset", dataset_artifact),
            _input_record("base-model", base_model),
            _input_record("training-config-source", config),
        ],
        outputs=outputs,
        runtime={
            "code_sha": args.code_sha,
            "image_digest": args.image_digest,
            "argv": command,
        },
    )


def _resolve_last_checkpoint(work_dir: Path) -> Path:
    pointers = sorted(
        work_dir.rglob("last_checkpoint"), key=lambda path: path.stat().st_mtime
    )
    for pointer in reversed(pointers):
        value = pointer.read_text(encoding="utf-8").strip()
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            local = (pointer.parent / candidate).resolve()
            candidate = local if local.is_file() else (work_dir / candidate).resolve()
        if candidate.is_file() and candidate.is_relative_to(work_dir):
            return candidate
    epochs = sorted(
        work_dir.rglob("epoch_*.pth"), key=lambda path: path.stat().st_mtime
    )
    if epochs:
        return epochs[-1]
    raise ValueError(f"training succeeded but emitted no last checkpoint in {work_dir}")


def _collect_training_metrics(work_dir: Path, output_path: Path) -> Dict[str, Any]:
    scalar_paths = sorted(work_dir.rglob("scalars.json"))
    if not scalar_paths:
        raise ValueError(
            f"training succeeded but emitted no scalars.json in {work_dir}"
        )
    latest: Dict[str, Any] = {}
    source_records = []
    record_count = 0
    for scalar_path in scalar_paths:
        digest, size = sha256_file(scalar_path)
        source_records.append(
            {
                "path": str(scalar_path.relative_to(work_dir)),
                "sha256": digest,
                "size_bytes": size,
            }
        )
        for line_number, line in enumerate(
            scalar_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"invalid scalar record in {scalar_path}:{line_number}"
                )
            latest.update(record)
            record_count += 1
    if not record_count:
        raise ValueError("training scalars contain no metric records")
    document = {
        "schema_version": "rightarm.gesture.training-metrics/v1",
        "record_count": record_count,
        "latest": latest,
        "sources": source_records,
    }
    from rightarm_mlops.protocol import atomic_json

    atomic_json(output_path, document)
    return document


def _mmengine_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _run_external(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    command = json.loads(args.command_json)
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise ValueError("--command-json must be a non-empty JSON string array")
    log_path = output_dir / "job.log"
    with log_path.open("wb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    outputs = []
    for kind, path in parse_output_specs(args.output):
        path = path.resolve(strict=True)
        outputs.append(output_record(kind, path))
    if not outputs:
        raise ValueError("external jobs must declare at least one --output KIND=PATH")
    runtime = {
        "code_sha": args.code_sha,
        "image_digest": args.image_digest,
        "argv": command,
    }
    return success(
        output_dir=output_dir,
        job_kind=args.job_kind,
        inputs=[{"artifact_id": value} for value in args.input_artifact],
        outputs=outputs + [output_record("job-log", log_path)],
        runtime=runtime,
    )


def _build_release(values: List[str], output_dir: Path) -> Dict[str, Any]:
    records = []
    for kind, path in parse_output_specs(values):
        records.append(output_record(kind, path.resolve(strict=True)))
    present = {item["kind"] for item in records}
    required = {
        "raw-dataset",
        "mmaction-dataset",
        "base-model",
        "code",
        "container-image",
        "training-config",
        "checkpoint",
        "evaluation-report",
        "onnx-model",
        "rknn-model",
        "label-mapping",
        "tensor-contract",
        "parity-report",
        "board-report",
    }
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"release provenance is incomplete; missing {missing}")
    manifest = {
        "schema_version": "rightarm.gesture.release/v1",
        "sealed": False,
        "artifacts": sorted(records, key=lambda item: item["kind"]),
    }
    from rightarm_mlops.protocol import atomic_json

    manifest_path = output_dir / "release.manifest.json"
    atomic_json(manifest_path, manifest)
    return success(
        output_dir=output_dir,
        job_kind="build-release",
        inputs=records,
        outputs=[output_record("release-manifest", manifest_path)],
        metrics={"provenance_artifact_count": len(records)},
    )


def _input_record(kind: str, path: Path) -> Dict[str, Any]:
    digest, size = sha256_file(path)
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": size,
    }


def _classify_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if "sha256 mismatch" in message or "size mismatch" in message:
        return "source-integrity-failed"
    if "label ontology" in message or "unknown label" in message:
        return "invalid-label-ontology"
    if "not sealed" in message:
        return "source-not-sealed"
    if "provenance is incomplete" in message:
        return "release-lineage-incomplete"
    return "invalid-job-input"


def _classify_process_failure(
    exc: subprocess.CalledProcessError, log_path: Optional[Path] = None
):
    log_tail = ""
    if log_path is not None and log_path.is_file():
        log_tail = (
            log_path.read_bytes()[-65_536:].decode("utf-8", errors="replace").lower()
        )
    if "cuda out of memory" in log_tail or "cuda error: out of memory" in log_tail:
        return "cuda-oom-with-valid-proposal", True
    if any(
        marker in log_tail
        for marker in (
            "connection reset",
            "temporary failure in name resolution",
            "timed out while connecting",
        )
    ):
        return "transient-network", True
    if exc.returncode in (137, -9):
        return "executor-resource-exhausted", True
    return "job-command-failed", False
