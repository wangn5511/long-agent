import csv
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

from rightarm_mlops.batch import inspect_batch, land_batch
from rightarm_mlops.cli import (
    _classify_process_failure,
    _collect_training_metrics,
    _resolve_last_checkpoint,
    main,
)
from rightarm_mlops.evaluation import evaluate_predictions
from rightarm_mlops.pipeline import build_grouped_split
from rightarm_mlops.protocol import safe_extract_tar


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(tmp_path: Path) -> Path:
    batch = tmp_path / "batch"
    video = batch / "videos" / "device-1" / "session-1" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not-a-real-video-but-content-addressed")
    annotations = batch / "annotations.csv"
    with annotations.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["视频名", "起始帧号", "终止帧号", "类别名"]
        )
        writer.writeheader()
        writer.writerow(
            {"视频名": "clip.mp4", "起始帧号": 0, "终止帧号": 15, "类别名": "WAVE_IN"}
        )
    manifest = {
        "schema_version": "rightarm.gesture.batch/v1",
        "batch_id": "batch-001",
        "created_at": "2026-07-20T12:00:00Z",
        "producer": "test",
        "label_ontology": "rightarm-gesture-11/v1",
        "objects": [
            {
                "path": "videos/device-1/session-1/clip.mp4",
                "bytes": video.stat().st_size,
                "sha256": _sha(video),
                "device_id": "device-1",
                "session_id": "session-1",
            }
        ],
        "annotations": {
            "path": "annotations.csv",
            "bytes": annotations.stat().st_size,
            "sha256": _sha(annotations),
        },
    }
    (batch / "batch.manifest.json").write_text(json.dumps(manifest))
    (batch / "_SUCCESS").touch()
    return batch


def test_batch_inspection_and_landing_are_reproducible(tmp_path: Path):
    batch = _batch(tmp_path)
    report = inspect_batch(batch)
    assert report["object_count"] == 1
    assert report["annotations"]["label_counts"]["WAVE_IN"] == 1

    first, _ = land_batch(batch, tmp_path / "first")
    second, _ = land_batch(batch, tmp_path / "second")
    first_value = json.loads(first.read_text())
    second_value = json.loads(second.read_text())
    assert first_value == second_value

    first_output = tmp_path / "land-cli-first"
    second_output = tmp_path / "land-cli-second"
    assert (
        main(
            [
                "land-batch",
                "--source-uri",
                str(batch),
                "--output-dir",
                str(first_output),
            ]
        )
        == 0
    )
    landed_result = json.loads((first_output / "result.json").read_text())
    assert {item["kind"] for item in landed_result["outputs"]} == {
        "source-manifest",
        "raw-dataset",
        "raw-dataset-manifest",
    }
    assert (
        json.loads((first_output / "source-batch.manifest.json").read_text())[
            "batch_id"
        ]
        == "batch-001"
    )
    assert (
        main(
            [
                "land-batch",
                "--source-uri",
                batch.as_uri(),
                "--output-dir",
                str(second_output),
            ]
        )
        == 0
    )
    assert _sha(first_output / "raw-dataset.tar") == _sha(
        second_output / "raw-dataset.tar"
    )


def test_integrity_mismatch_writes_terminal_failure(tmp_path: Path):
    batch = _batch(tmp_path)
    video = next((batch / "videos").rglob("*.mp4"))
    video.write_bytes(b"corrupted")
    output = tmp_path / "output"
    assert (
        main(
            [
                "inspect-batch",
                "--source-uri",
                str(batch),
                "--output-dir",
                str(output),
            ]
        )
        == 1
    )
    failure = json.loads((output / "failure.json").read_text())
    assert failure["failure_class"] == "source-integrity-failed"
    assert failure["retryable"] is False


def test_evaluation_emits_product_metrics(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"truth": "WAVE_IN", "prediction": "WAVE_IN"},
        {"truth": "NONE", "prediction": "NONE"},
        {"truth": "NONE", "prediction": "CLICK"},
    ]
    predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    evaluation = evaluate_predictions(predictions, tmp_path / "evaluation.json")
    assert evaluation["metrics"]["offline.sample_count"] == 3
    assert evaluation["metrics"]["offline.none_false_accept_rate"] == 0.5
    assert "offline.macro_f1" in evaluation["metrics"]


def test_train_plan_resolves_every_machine_specific_input(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "training" / "mmaction2").mkdir(parents=True)
    config = repo / "config.py"
    dataset = repo / "dataset.pkl"
    base_model = repo / "base.pth"
    config.write_text("train_cfg = dict(max_epochs=50)")
    dataset.write_bytes(b"dataset")
    base_model.write_bytes(b"base")
    output = tmp_path / "train-output"
    assert (
        main(
            [
                "train",
                "--config",
                str(config),
                "--dataset",
                str(dataset),
                "--base-model",
                str(base_model),
                "--gpus",
                "4",
                "--batch-size",
                "16",
                "--workers",
                "8",
                "--seed",
                "7",
                "--max-epochs",
                "12",
                "--repo-root",
                str(repo),
                "--code-sha",
                "a" * 40,
                "--image-digest",
                "sha256:" + "b" * 64,
                "--dry-run",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    command = json.loads((output / "resolved-training-command.json").read_text())
    assert command["overrides"]["load_from"] == str(base_model)
    assert command["overrides"]["train_dataloader.dataset.dataset.ann_file"] == str(
        dataset
    )
    assert command["overrides"]["train_dataloader.batch_size"] == 16
    assert command["overrides"]["randomness.seed"] == 7
    assert command["overrides"]["train_cfg.max_epochs"] == 12


def test_build_dataset_compiles_rtm_tracks_into_v3_pose_contract(tmp_path: Path):
    train_root = tmp_path / "rtm-train"
    val_root = tmp_path / "rtm-val"
    train_root.mkdir()
    val_root.mkdir()
    _rtm_sample(train_root, "WAVE_IN_sample", handedness=1)
    _rtm_sample(val_root, "NONE_sample", handedness=0)
    output = tmp_path / "dataset-output"
    assert (
        main(
            [
                "build-dataset",
                "--train-rtm-root",
                str(train_root),
                "--val-rtm-root",
                str(val_root),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    with (output / "rightarm-gesture-11.pkl").open("rb") as stream:
        dataset = pickle.load(stream)
    assert dataset["categories"][-2:] == ["GUN", "NONE"]
    assert dataset["split"]["train"] == ["WAVE_IN_sample"]
    sample = dataset["annotations"][0]
    assert sample["keypoint"].shape == (1, 2, 21, 3)
    assert sample["keypoint_score"].shape == (1, 2, 21)
    assert sample["track_keypoint"].shape == (1, 2, 21, 3)
    assert sample["global_traj_source"].shape == (2, 3)
    assert sample["handedness_label_id"] == 1
    assert Path(sample["video_path"]).is_file()

    bundle_root = tmp_path / "bundle"
    safe_extract_tar(output / "rightarm-gesture-11.tar", bundle_root)
    with (bundle_root / "rightarm-gesture-11.pkl").open("rb") as stream:
        portable = pickle.load(stream)
    portable_video = Path(portable["annotations"][0]["video_path"])
    assert not portable_video.is_absolute()
    assert (bundle_root / portable_video).is_file()


def test_grouped_split_keeps_one_session_on_one_side(tmp_path: Path):
    annotations = tmp_path / "annotations.csv"
    with annotations.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["视频名", "起始帧号", "终止帧号", "类别名"]
        )
        writer.writeheader()
        for name in ("first.mp4", "second.mp4", "third.mp4"):
            writer.writerow(
                {"视频名": name, "起始帧号": 0, "终止帧号": 15, "类别名": "OK"}
            )
    canonical = {
        "objects": [
            {
                "source_relative_path": "videos/device-a/session-a/first.mp4",
                "device_id": "device-a",
                "session_id": "session-a",
                "subject_group_id": "subject-a",
            },
            {
                "source_relative_path": "videos/device-a/session-a/second.mp4",
                "device_id": "device-a",
                "session_id": "session-a",
                "subject_group_id": "subject-a",
            },
            {
                "source_relative_path": "videos/device-b/session-b/third.mp4",
                "device_id": "device-b",
                "session_id": "session-b",
                "subject_group_id": "subject-b",
            },
        ]
    }
    split = build_grouped_split(
        canonical,
        annotation_path=annotations,
        materialized_root=tmp_path,
        val_ratio=0.5,
        seed=42,
    )
    videos = {
        Path(path).name: record["split"] for path, record in split["videos"].items()
    }
    assert videos["first.mp4"] == videos["second.mp4"]
    assert videos["third.mp4"] != videos["first.mp4"]


def test_training_evidence_requires_last_checkpoint_and_scalars(tmp_path: Path):
    work_dir = tmp_path / "training"
    visual = work_dir / "20260720" / "vis_data"
    visual.mkdir(parents=True)
    epoch = work_dir / "epoch_12.pth"
    epoch.write_bytes(b"last")
    (work_dir / "last_checkpoint").write_text(str(epoch))
    (visual / "scalars.json").write_text(
        json.dumps({"epoch": 1, "loss": 0.8})
        + "\n"
        + json.dumps({"epoch": 12, "loss": 0.1, "acc/top1": 0.9})
        + "\n"
    )

    assert _resolve_last_checkpoint(work_dir) == epoch
    metrics_path = tmp_path / "training-metrics.json"
    metrics = _collect_training_metrics(work_dir, metrics_path)
    assert metrics["record_count"] == 2
    assert metrics["latest"]["epoch"] == 12
    assert metrics["latest"]["acc/top1"] == 0.9
    assert metrics["sources"][0]["path"] == "20260720/vis_data/scalars.json"
    assert metrics_path.is_file()


def test_cuda_oom_is_machine_classified_as_retryable(tmp_path: Path):
    log = tmp_path / "job.log"
    log.write_text("RuntimeError: CUDA out of memory. Tried to allocate 2 GiB")

    failure_class, retryable = _classify_process_failure(
        subprocess.CalledProcessError(1, ["train"]), log
    )

    assert failure_class == "cuda-oom-with-valid-proposal"
    assert retryable is True


def _rtm_sample(root: Path, clip_id: str, *, handedness: int) -> None:
    video = root / f"{clip_id}.mp4"
    video.write_bytes(b"video-placeholder")
    points = [[float(index), float(index + 1), float(index + 2)] for index in range(21)]
    frames = []
    for frame_index in range(2):
        frames.append(
            {
                "frame_idx": frame_index,
                "num_hands": 1,
                "hands": [
                    {
                        "track_id": 7,
                        "handedness_label": handedness,
                        "score": 0.9,
                        "hand21_kpts_xyz": points,
                        "hand21_kpt_scores": [0.8] * 21,
                        "upperbody_wrist_xyz": [100.0 + frame_index, 50.0, 0.1],
                    }
                ],
            }
        )
    payload = {
        "clip_id": clip_id,
        "video": str(video),
        "clip_path": str(video),
        "total_frames": 2,
        "resolution": [640, 480],
        "frames": frames,
    }
    (root / f"{clip_id}.json").write_text(json.dumps(payload))
