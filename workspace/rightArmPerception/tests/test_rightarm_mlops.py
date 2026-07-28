import csv
import hashlib
import json
import os
import pickle
import runpy
import shutil
import subprocess
from pathlib import Path

import numpy as np
from rightarm_mlops.batch import inspect_batch, land_batch
from rightarm_mlops.cli import (
    _classify_process_failure,
    _collect_training_metrics,
    _resolve_last_checkpoint,
    main,
)
from rightarm_mlops.dataset_review import CANONICAL_LABELS
from rightarm_mlops.evaluation import evaluate_predictions
from rightarm_mlops.model_improvement import (
    EVALUATION_SCHEMA,
    IMPROVEMENT_SCHEMA,
    LABELS,
    TRIAL_SCHEMA,
    normalize_experiment_parameters,
)
from rightarm_mlops.pipeline import build_grouped_split
from rightarm_mlops.protocol import safe_extract_tar


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(tmp_path: Path) -> Path:
    batch = tmp_path / 'batch'
    video = batch / 'videos' / 'device-1' / 'session-1' / 'clip.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'not-a-real-video-but-content-addressed')
    annotations = batch / 'annotations.csv'
    with annotations.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=['视频名', '起始帧号', '终止帧号', '类别名']
        )
        writer.writeheader()
        writer.writerow(
            {'视频名': 'clip.mp4', '起始帧号': 0, '终止帧号': 15, '类别名': 'WAVE_IN'}
        )
    manifest = {
        'schema_version': 'rightarm.gesture.batch/v1',
        'batch_id': 'batch-001',
        'created_at': '2026-07-20T12:00:00Z',
        'producer': 'test',
        'label_ontology': 'rightarm-gesture-11/v1',
        'objects': [
            {
                'path': 'videos/device-1/session-1/clip.mp4',
                'bytes': video.stat().st_size,
                'sha256': _sha(video),
                'device_id': 'device-1',
                'session_id': 'session-1',
            }
        ],
        'annotations': {
            'path': 'annotations.csv',
            'bytes': annotations.stat().st_size,
            'sha256': _sha(annotations),
        },
    }
    (batch / 'batch.manifest.json').write_text(json.dumps(manifest))
    (batch / '_SUCCESS').touch()
    return batch


def test_batch_inspection_and_landing_are_reproducible(tmp_path: Path):
    batch = _batch(tmp_path)
    report = inspect_batch(batch)
    assert report['object_count'] == 1
    assert report['annotations']['label_counts']['WAVE_IN'] == 1

    first, _ = land_batch(batch, tmp_path / 'first')
    second, _ = land_batch(batch, tmp_path / 'second')
    first_value = json.loads(first.read_text())
    second_value = json.loads(second.read_text())
    assert first_value == second_value

    first_output = tmp_path / 'land-cli-first'
    second_output = tmp_path / 'land-cli-second'
    assert (
        main(
            [
                'land-batch',
                '--source-uri',
                str(batch),
                '--output-dir',
                str(first_output),
            ]
        )
        == 0
    )
    landed_result = json.loads((first_output / 'result.json').read_text())
    assert {item['kind'] for item in landed_result['outputs']} == {
        'source-manifest',
        'raw-dataset',
        'raw-dataset-manifest',
    }
    assert (
        json.loads((first_output / 'source-batch.manifest.json').read_text())[
            'batch_id'
        ]
        == 'batch-001'
    )
    assert (
        main(
            [
                'land-batch',
                '--source-uri',
                batch.as_uri(),
                '--output-dir',
                str(second_output),
            ]
        )
        == 0
    )
    assert _sha(first_output / 'raw-dataset.tar') == _sha(
        second_output / 'raw-dataset.tar'
    )


def test_integrity_mismatch_writes_terminal_failure(tmp_path: Path):
    batch = _batch(tmp_path)
    video = next((batch / 'videos').rglob('*.mp4'))
    video.write_bytes(b'corrupted')
    output = tmp_path / 'output'
    assert (
        main(
            [
                'inspect-batch',
                '--source-uri',
                str(batch),
                '--output-dir',
                str(output),
            ]
        )
        == 1
    )
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['failure_class'] == 'source-integrity-failed'
    assert failure['retryable'] is False


def test_evaluation_emits_product_metrics(tmp_path: Path):
    predictions = tmp_path / 'predictions.jsonl'
    rows = [
        {'truth': 'WAVE_IN', 'prediction': 'WAVE_IN'},
        {'truth': 'NONE', 'prediction': 'NONE'},
        {'truth': 'NONE', 'prediction': 'CLICK'},
    ]
    predictions.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    evaluation = evaluate_predictions(predictions, tmp_path / 'evaluation.json')
    assert evaluation['metrics']['offline.sample_count'] == 3
    assert evaluation['metrics']['offline.none_false_accept_rate'] == 0.5
    assert 'offline.macro_f1' in evaluation['metrics']


def test_train_plan_resolves_every_machine_specific_input(tmp_path: Path):
    repo = tmp_path / 'repo'
    (repo / 'training' / 'mmaction2').mkdir(parents=True)
    config = repo / 'config.py'
    dataset = repo / 'dataset.pkl'
    base_model = repo / 'base.pth'
    config.write_text('train_cfg = dict(max_epochs=50)')
    dataset.write_bytes(b'dataset')
    base_model.write_bytes(b'base')
    output = tmp_path / 'train-output'
    assert (
        main(
            [
                'train',
                '--config',
                str(config),
                '--dataset',
                str(dataset),
                '--base-model',
                str(base_model),
                '--gpus',
                '4',
                '--batch-size',
                '16',
                '--workers',
                '8',
                '--seed',
                '7',
                '--max-epochs',
                '12',
                '--repo-root',
                str(repo),
                '--code-sha',
                'a' * 40,
                '--image-digest',
                'sha256:' + 'b' * 64,
                '--dry-run',
                '--output-dir',
                str(output),
            ]
        )
        == 0
    )
    command = json.loads((output / 'resolved-training-command.json').read_text())
    assert command['overrides']['load_from'] == str(base_model)
    assert command['overrides']['train_dataloader.dataset.dataset.ann_file'] == str(
        dataset
    )
    assert command['overrides']['train_dataloader.batch_size'] == 16
    assert command['overrides']['randomness.seed'] == 7
    assert command['overrides']['train_cfg.max_epochs'] == 12


def test_build_dataset_compiles_rtm_tracks_into_v3_pose_contract(tmp_path: Path):
    train_root = tmp_path / 'rtm-train'
    val_root = tmp_path / 'rtm-val'
    train_root.mkdir()
    val_root.mkdir()
    _rtm_sample(train_root, 'WAVE_IN_sample', handedness=1)
    _rtm_sample(val_root, 'NONE_sample', handedness=0)
    output = tmp_path / 'dataset-output'
    assert (
        main(
            [
                'build-dataset',
                '--train-rtm-root',
                str(train_root),
                '--val-rtm-root',
                str(val_root),
                '--output-dir',
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / 'result.json').read_text())
    assert 'incremental-mmaction-dataset' in {
        item['kind'] for item in result['outputs']
    }
    with (output / 'rightarm-gesture-11.pkl').open('rb') as stream:
        dataset = pickle.load(stream)
    assert dataset['categories'][-2:] == ['GUN', 'NONE']
    assert dataset['split']['train'] == ['WAVE_IN_sample']
    sample = dataset['annotations'][0]
    assert sample['keypoint'].shape == (1, 2, 21, 3)
    assert sample['keypoint_score'].shape == (1, 2, 21)
    assert sample['track_keypoint'].shape == (1, 2, 21, 3)
    assert sample['global_traj_source'].shape == (2, 3)
    assert sample['handedness_label_id'] == 1
    assert sample['source_group_id'] == 'group-1'
    assert sample['source_frame_range'] == [10, 11]
    assert Path(sample['video_path']).is_file()

    bundle_root = tmp_path / 'bundle'
    safe_extract_tar(output / 'rightarm-gesture-11.tar', bundle_root)
    with (bundle_root / 'rightarm-gesture-11.pkl').open('rb') as stream:
        portable = pickle.load(stream)
    portable_video = Path(portable['annotations'][0]['video_path'])
    assert not portable_video.is_absolute()
    assert (bundle_root / portable_video).is_file()


def test_grouped_split_keeps_one_session_on_one_side(tmp_path: Path):
    annotations = tmp_path / 'annotations.csv'
    with annotations.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=['视频名', '起始帧号', '终止帧号', '类别名']
        )
        writer.writeheader()
        for name in ('first.mp4', 'second.mp4', 'third.mp4'):
            writer.writerow(
                {'视频名': name, '起始帧号': 0, '终止帧号': 15, '类别名': 'OK'}
            )
    canonical = {
        'objects': [
            {
                'source_relative_path': 'videos/device-a/session-a/first.mp4',
                'device_id': 'device-a',
                'session_id': 'session-a',
                'subject_group_id': 'subject-a',
            },
            {
                'source_relative_path': 'videos/device-a/session-a/second.mp4',
                'device_id': 'device-a',
                'session_id': 'session-a',
                'subject_group_id': 'subject-a',
            },
            {
                'source_relative_path': 'videos/device-b/session-b/third.mp4',
                'device_id': 'device-b',
                'session_id': 'session-b',
                'subject_group_id': 'subject-b',
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
        Path(path).name: record['split'] for path, record in split['videos'].items()
    }
    assert videos['first.mp4'] == videos['second.mp4']
    assert videos['third.mp4'] != videos['first.mp4']
    first_record = next(
        record
        for path, record in split['videos'].items()
        if Path(path).name == 'first.mp4'
    )
    assert first_record['subject_group_id'] == 'subject-a'
    assert first_record['device_id'] == 'device-a'
    assert first_record['session_id'] == 'session-a'


def test_distribution_review_profiles_and_merges_a_green_increment(
    tmp_path: Path,
) -> None:
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    baseline = tmp_path / 'baseline.pkl'
    incremental = tmp_path / 'incremental.pkl'
    _pose_dataset(
        baseline,
        video,
        samples_per_label=4,
        prefix='baseline',
        include_val=True,
    )
    _pose_dataset(
        incremental,
        video,
        samples_per_label=1,
        prefix='new',
        include_val=False,
    )

    profile_output = tmp_path / 'profile'
    assert (
        main(
            [
                'profile-dataset-distribution',
                '--baseline-dataset',
                str(baseline),
                '--incremental-dataset',
                str(incremental),
                '--baseline-artifact-id',
                'baseline-artifact',
                '--incremental-artifact-id',
                'incremental-artifact',
                '--output-dir',
                str(profile_output),
            ]
        )
        == 0
    )
    profile_result = json.loads((profile_output / 'result.json').read_text())
    assert profile_result['metrics']['recommended_action'] == 'merge-all-unique'
    report_sha256 = profile_result['metrics']['distribution_report_sha256']

    merge_output = tmp_path / 'merge'
    assert (
        main(
            [
                'merge-dataset',
                '--baseline-dataset',
                str(baseline),
                '--incremental-dataset',
                str(incremental),
                '--baseline-artifact-id',
                'baseline-artifact',
                '--incremental-artifact-id',
                'incremental-artifact',
                '--distribution-report-sha256',
                report_sha256,
                '--decision',
                'merge-all-unique',
                '--reason-code',
                'agent-reviewed-green-distribution',
                '--output-dir',
                str(merge_output),
            ]
        )
        == 0
    )
    merge_result = json.loads((merge_output / 'result.json').read_text())
    assert {item['kind'] for item in merge_result['outputs']} >= {
        'dataset-merge-plan',
        'dataset-merge-report',
        'mmaction-dataset',
    }
    merge_report = json.loads((merge_output / 'dataset-merge-report.json').read_text())
    assert merge_report['output']['sample_count'] == 55
    assert merge_report['distribution_report_sha256'] == report_sha256
    assert (merge_output / 'merged-rightarm-gesture-11.tar').is_file()


def test_distribution_profiler_rejects_pickle_globals_without_executing_them(
    tmp_path: Path,
) -> None:
    marker = tmp_path / 'unsafe-code-ran'
    malicious = tmp_path / 'malicious.pkl'
    with malicious.open('wb') as stream:
        pickle.dump(_PickleExploit(marker), stream)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'video')
    incremental = tmp_path / 'incremental.pkl'
    _pose_dataset(
        incremental,
        video,
        samples_per_label=1,
        prefix='new',
        include_val=False,
    )
    output = tmp_path / 'profile'

    assert (
        main(
            [
                'profile-dataset-distribution',
                '--baseline-dataset',
                str(malicious),
                '--incremental-dataset',
                str(incremental),
                '--baseline-artifact-id',
                'baseline-artifact',
                '--incremental-artifact-id',
                'incremental-artifact',
                '--output-dir',
                str(output),
            ]
        )
        == 1
    )
    assert not marker.exists()
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['failure_class'] == 'invalid-job-input'


def test_training_evidence_requires_last_checkpoint_and_scalars(tmp_path: Path):
    work_dir = tmp_path / 'training'
    visual = work_dir / '20260720' / 'vis_data'
    visual.mkdir(parents=True)
    epoch = work_dir / 'epoch_12.pth'
    epoch.write_bytes(b'last')
    (work_dir / 'last_checkpoint').write_text(str(epoch))
    (visual / 'scalars.json').write_text(
        json.dumps({'epoch': 1, 'loss': 0.8})
        + '\n'
        + json.dumps({'epoch': 12, 'loss': 0.1, 'acc/top1': 0.9})
        + '\n'
    )

    assert _resolve_last_checkpoint(work_dir) == epoch
    metrics_path = tmp_path / 'training-metrics.json'
    metrics = _collect_training_metrics(work_dir, metrics_path)
    assert metrics['record_count'] == 2
    assert metrics['latest']['epoch'] == 12
    assert metrics['latest']['acc/top1'] == 0.9
    assert metrics['sources'][0]['path'] == '20260720/vis_data/scalars.json'
    assert metrics_path.is_file()


def test_cuda_oom_is_machine_classified_as_retryable(tmp_path: Path):
    log = tmp_path / 'job.log'
    log.write_text('RuntimeError: CUDA out of memory. Tried to allocate 2 GiB')

    failure_class, retryable = _classify_process_failure(
        subprocess.CalledProcessError(1, ['train']), log
    )

    assert failure_class == 'cuda-oom-with-valid-proposal'
    assert retryable is True


def test_model_improvement_jobs_benchmark_train_compare_and_record(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / 'repo'
    mmaction_root = repo / 'training' / 'mmaction2'
    config_dir = mmaction_root / 'handoff' / 'stgcn_hand21_3d' / 'configs'
    config_dir.mkdir(parents=True)
    source_config = (
        Path(__file__).parents[1]
        / 'training'
        / 'mmaction2'
        / 'handoff'
        / 'stgcn_hand21_3d'
        / 'configs'
        / 'stgcn_hand21-rgb-bs64-gated-v3-ft-rtm_add-11cls.py'
    )
    config = config_dir / source_config.name
    shutil.copyfile(source_config, config)
    (mmaction_root / 'tools').mkdir()
    dataset = tmp_path / 'dataset.pkl'
    dataset.write_bytes(b'immutable-dataset')
    base_checkpoint = tmp_path / 'base.pth'
    base_checkpoint.write_bytes(b'incumbent')

    def fake_run(command, cwd, env, stdout, stderr, check):
        assert check is True
        assert str(mmaction_root) in env['PYTHONPATH'].split(os.pathsep)
        assert str(repo) in env['PYTHONPATH'].split(os.pathsep)
        assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        if command[0] == 'bash':
            assert command[1] == str(mmaction_root / 'tools' / 'dist_train.sh')
            work_dir = Path(command[command.index('--work-dir') + 1])
            assert Path(cwd) == work_dir.parent
            visual = work_dir / '20260723' / 'vis_data'
            visual.mkdir(parents=True)
            best = work_dir / 'best_acc_top1_epoch_1.pth'
            last = work_dir / 'epoch_2.pth'
            best.write_bytes(b'best-challenger')
            last.write_bytes(b'last-challenger')
            (work_dir / 'last_checkpoint').write_text(str(last))
            (visual / 'scalars.json').write_text(
                json.dumps({'epoch': 1, 'acc/top1': 0.95, 'loss': 0.2})
                + '\n'
                + json.dumps({'epoch': 2, 'acc/top1': 0.90, 'loss': 0.3})
                + '\n'
            )
            stdout.write(b'training complete\n')
            return subprocess.CompletedProcess(command, 0)

        assert command[1] == str(mmaction_root / 'tools' / 'test.py')
        checkpoint = Path(command[3])
        work_dir = Path(command[command.index('--work-dir') + 1])
        assert Path(cwd) == work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        confusion = np.eye(len(LABELS), dtype=np.int64) * 2
        if checkpoint == base_checkpoint:
            confusion[0, 0] -= 1
            confusion[0, -1] += 1
        elif checkpoint.name == 'epoch_2.pth':
            confusion[-1, -1] -= 1
            confusion[-1, 3] += 1
        np.save(work_dir / 'confusion_matrix_test.npy', confusion)
        stdout.write(b'evaluation complete\n')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', fake_run)

    baseline_output = tmp_path / 'baseline'
    assert (
        main(
            [
                'benchmark-checkpoint',
                '--config',
                str(config),
                '--dataset',
                str(dataset),
                '--checkpoint',
                str(base_checkpoint),
                '--gpus',
                '1',
                '--batch-size',
                '4',
                '--workers',
                '0',
                '--repo-root',
                str(repo),
                '--output-dir',
                str(baseline_output),
            ]
        )
        == 0
    )
    baseline_path = baseline_output / 'evaluation' / 'checkpoint-evaluation.json'
    baseline = json.loads(baseline_path.read_text())
    baseline_result = json.loads((baseline_output / 'result.json').read_text())
    baseline_config = runpy.run_path(
        str(baseline_output / 'evaluation' / 'resolved-evaluation-config.py')
    )
    assert baseline_config['model']['rgb_branch']['pretrained'] is False
    assert {
        'checkpoint',
        'baseline-evaluation-report',
        'evaluation-report',
    } <= {item['kind'] for item in baseline_result['outputs']}
    assert baseline['schema_version'] == EVALUATION_SCHEMA
    assert baseline['labels'] == list(LABELS)
    assert baseline['identity']['dataset_sha256'] == _sha(dataset)
    assert baseline['identity']['checkpoint_sha256'] == _sha(base_checkpoint)
    assert baseline['metrics']['offline.sample_count'] == 22
    assert baseline['confusion_matrix']['WAVE_IN']['NONE'] == 1

    trial_output = tmp_path / 'trial'
    assert (
        main(
            [
                'train-eval-trial',
                '--trial-number',
                '1',
                '--strategy',
                'motion-ablation',
                '--parameters-json',
                json.dumps(
                    {
                        'motion_preset': 'no-delta-z',
                        'batch_size': 4,
                        'epochs': 2,
                    }
                ),
                '--config',
                str(config),
                '--dataset',
                str(dataset),
                '--base-checkpoint',
                str(base_checkpoint),
                '--gpus',
                '1',
                '--workers',
                '0',
                '--repo-root',
                str(repo),
                '--output-dir',
                str(trial_output),
            ]
        )
        == 0
    )
    trial_path = trial_output / 'optimization-trial.json'
    trial = json.loads(trial_path.read_text())
    trial_config = runpy.run_path(str(trial_output / 'resolved-training-config.py'))
    assert trial_config['model']['rgb_branch']['pretrained'] is False
    assert trial['schema_version'] == TRIAL_SCHEMA
    assert trial['parameters']['motion_preset'] == 'no-delta-z'
    assert trial['selection']['winner_role'] == 'best'
    assert trial['evaluation']['metrics']['offline.accuracy'] == 1.0
    assert (
        trial['input']['preprocessing_fingerprint']
        == baseline['identity']['preprocessing_fingerprint']
    )
    result = json.loads((trial_output / 'result.json').read_text())
    assert {
        'checkpoint',
        'training-config',
        'training-log',
        'training-metrics',
        'evaluation-report',
        'optimization-trial-report',
    } <= {item['kind'] for item in result['outputs']}

    final_output = tmp_path / 'final'
    history = {
        'baseline_evaluation': str(baseline_path),
        'trial_reports': [str(trial_path)],
        'policy': {
            'max_trials': 1,
            'patience': 1,
            'target_macro_f1_delta': 0.001,
            'min_improvement': 0.0005,
            'max_epochs': 2,
            'batch_size': 4,
            'workers': 0,
            'gpus': 1,
        },
    }
    assert (
        main(
            [
                'record-model-improvement',
                '--history-json',
                json.dumps(history),
                '--analysis-json',
                json.dumps(
                    {
                        'summary': 'no-delta-z removed the observed error',
                        'next_hypothesis': 'gesture-gate',
                    }
                ),
                '--output-dir',
                str(final_output),
            ]
        )
        == 0
    )
    final = json.loads((final_output / 'model-improvement.json').read_text())
    assert final['schema_version'] == IMPROVEMENT_SCHEMA
    assert final['decision'] == 'promote-challenger'
    assert final['stop_reason'] == 'target-reached'
    assert final['winner']['source'] == 'trial'
    assert final['winner']['checkpoint_sha256'] == trial['outputs']['checkpoint_sha256']


def test_model_improvement_aliases_extensionless_pickle_without_copying(
    tmp_path: Path,
) -> None:
    from rightarm_mlops.cli import _materialize_model_dataset

    dataset = tmp_path / 'mmaction-dataset-content-hash'
    dataset.write_bytes(b'not-a-tar')
    output = tmp_path / 'output'
    output.mkdir()

    materialized = _materialize_model_dataset(dataset.resolve(), output)

    assert materialized.name == 'materialized-model-dataset.pkl'
    assert materialized.is_symlink()
    assert materialized.resolve(strict=True) == dataset.resolve()
    assert materialized.read_bytes() == b'not-a-tar'


def test_model_improvement_parameter_allowlist_is_closed() -> None:
    normalized = normalize_experiment_parameters(
        {'lr': 0.001, 'batch_size': 8, 'epochs': 10}
    )
    assert normalized['lr'] == 0.001
    assert normalized['gate_threshold'] == 0.45
    try:
        normalize_experiment_parameters({'arbitrary_cfg_key': 'owned'})
    except ValueError as exc:
        assert 'unsupported experiment parameter' in str(exc)
    else:
        raise AssertionError('arbitrary experiment key was accepted')


def test_model_improvement_dry_runs_do_not_import_torch(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / 'repo'
    config_dir = (
        repo / 'training' / 'mmaction2' / 'handoff' / 'stgcn_hand21_3d' / 'configs'
    )
    config_dir.mkdir(parents=True)
    config = config_dir / 'stgcn_hand21-rgb-bs64-gated-v3-ft-rtm_add-11cls.py'
    config.write_text('this file is not executed during a dry run\n')
    dataset = tmp_path / 'dataset.pkl'
    checkpoint = tmp_path / 'checkpoint.pth'
    dataset.write_bytes(b'dataset')
    checkpoint.write_bytes(b'checkpoint')

    def fail_run(*args, **kwargs):
        raise AssertionError('dry run attempted to execute MMAction')

    monkeypatch.setattr(subprocess, 'run', fail_run)
    assert (
        main(
            [
                'benchmark-checkpoint',
                '--config',
                str(config),
                '--dataset',
                str(dataset),
                '--checkpoint',
                str(checkpoint),
                '--gpus',
                '1',
                '--batch-size',
                '4',
                '--workers',
                '0',
                '--repo-root',
                str(repo),
                '--dry-run',
                '--output-dir',
                str(tmp_path / 'benchmark-plan'),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                'train-eval-trial',
                '--trial-number',
                '1',
                '--strategy',
                'fine-tune',
                '--parameters-json',
                json.dumps({'batch_size': 4, 'epochs': 1}),
                '--config',
                str(config),
                '--dataset',
                str(dataset),
                '--base-checkpoint',
                str(checkpoint),
                '--gpus',
                '1',
                '--workers',
                '0',
                '--repo-root',
                str(repo),
                '--dry-run',
                '--output-dir',
                str(tmp_path / 'trial-plan'),
            ]
        )
        == 0
    )


def _rtm_sample(root: Path, clip_id: str, *, handedness: int) -> None:
    video = root / f'{clip_id}.mp4'
    video.write_bytes(b'video-placeholder')
    points = [[float(index), float(index + 1), float(index + 2)] for index in range(21)]
    frames = []
    for frame_index in range(2):
        frames.append(
            {
                'frame_idx': frame_index,
                'num_hands': 1,
                'hands': [
                    {
                        'track_id': 7,
                        'handedness_label': handedness,
                        'score': 0.9,
                        'hand21_kpts_xyz': points,
                        'hand21_kpt_scores': [0.8] * 21,
                        'upperbody_wrist_xyz': [100.0 + frame_index, 50.0, 0.1],
                    }
                ],
            }
        )
    payload = {
        'clip_id': clip_id,
        'video': str(video),
        'clip_path': str(video),
        'total_frames': 2,
        'resolution': [640, 480],
        'source_group_id': 'group-1',
        'source_video_sha256': 'a' * 64,
        'source_frame_range': [10, 11],
        'frames': frames,
    }
    (root / f'{clip_id}.json').write_text(json.dumps(payload))


def _pose_dataset(
    path: Path,
    video: Path,
    *,
    samples_per_label: int,
    prefix: str,
    include_val: bool,
) -> None:
    split = {'train': [], 'val': []}
    annotations = []
    for label_id, label in enumerate(CANONICAL_LABELS):
        for index in range(samples_per_label):
            sample_id = f'{prefix}-{label}-{index}'
            split_name = (
                'val' if include_val and index == samples_per_label - 1 else 'train'
            )
            split[split_name].append(sample_id)
            annotations.append(
                {
                    'frame_dir': sample_id,
                    'label': label_id,
                    'total_frames': 2,
                    'keypoint': np.ones((1, 2, 21, 3), dtype=np.float32),
                    'keypoint_score': np.ones((1, 2, 21), dtype=np.float32),
                    'video_path': str(video),
                    'clip_video_path': str(video),
                    'source_group_id': f'{prefix}-group-{label}-{index}',
                }
            )
    with path.open('wb') as stream:
        pickle.dump(
            {
                'split': split,
                'annotations': annotations,
                'categories': list(CANONICAL_LABELS),
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


class _PickleExploit:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f'touch {self.marker}',)
