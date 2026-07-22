import hashlib
import json
import tarfile
from pathlib import Path

from openhands.mlops_loop.artifact_store import LocalArtifactStore
from openhands.mlops_loop.models import (
    CompletionContract,
    MetricGate,
    TaskResult,
    TaskSpec,
    ValidationDisposition,
)
from openhands.mlops_loop.rightarm_validators import (
    RIGHTARM_LABELS,
    build_rightarm_validator_registry,
)


def test_data_validator_recomputes_landed_archive_hashes(tmp_path: Path):
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    video = b'video-content'
    annotations = '视频名,起始帧号,终止帧号,类别名\nclip.mp4,0,15,WAVE_IN\n'.encode()
    video_sha = hashlib.sha256(video).hexdigest()
    annotation_sha = hashlib.sha256(annotations).hexdigest()
    source = {
        'schema_version': 'rightarm.gesture.batch/v1',
        'batch_id': 'batch-1',
        'created_at': '2026-07-20T00:00:00Z',
        'producer': 'test',
        'label_ontology': 'rightarm-gesture-11/v1',
        'objects': [
            {
                'path': 'videos/clip.mp4',
                'bytes': len(video),
                'sha256': video_sha,
            }
        ],
        'annotations': {
            'path': 'annotations.csv',
            'bytes': len(annotations),
            'sha256': annotation_sha,
        },
    }
    report = {
        'schema_version': 'rightarm.gesture.data-validation/v1',
        'object_count': 1,
        'duplicate_object_count': 0,
        'annotations': {'annotation_count': 1},
        'source_manifest': source,
    }
    raw_manifest = {
        'schema_version': 'rightarm.gesture.raw-dataset/v1',
        'batch_id': 'batch-1',
        'objects': [
            {
                'path': f'objects/sha256/{video_sha[:2]}/{video_sha}',
                'size_bytes': len(video),
                'sha256': video_sha,
            }
        ],
        'annotations': {
            'path': 'annotations.csv',
            'size_bytes': len(annotations),
            'sha256': annotation_sha,
        },
        'source_manifest': source,
    }
    source_path = tmp_path / 'source.json'
    source_path.write_text(json.dumps(source))
    report_path = tmp_path / 'report.json'
    report_path.write_text(json.dumps(report))
    raw_root = tmp_path / 'raw'
    object_path = raw_root / raw_manifest['objects'][0]['path']
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b'tampered-after-inspection')
    (raw_root / 'annotations.csv').write_bytes(annotations)
    (raw_root / 'raw-dataset.manifest.json').write_text(json.dumps(raw_manifest))
    archive_path = tmp_path / 'raw-dataset.tar'
    with tarfile.open(archive_path, 'w') as archive:
        for path in sorted(raw_root.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(raw_root))

    candidates = [
        artifacts.stage(source_path, kind='source-manifest'),
        artifacts.stage(report_path, kind='data-validation-report'),
        artifacts.stage(archive_path, kind='raw-dataset'),
    ]
    task = TaskSpec(
        name='inspect',
        kind='inspect-data',
        objective='inspect and land data',
        agent_profile='default',
        validator='rightarm-data-inspection-v1',
        max_attempts=1,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=[
                'source-manifest',
                'data-validation-report',
                'raw-dataset',
            ],
            required_evidence=True,
        ),
    )
    result = TaskResult(
        task_id='task_data',
        attempt_id='attempt_data',
        status='succeeded',
        summary='Agent claims landing is valid',
        artifact_candidates=candidates,
        evidence_refs=['job:land'],
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'hash mismatch' in validation.reasons[0]


def test_evaluation_gate_uses_report_not_agent_claim(tmp_path: Path):
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    report_path = tmp_path / 'evaluation.json'
    report_path.write_text(
        json.dumps(
            {
                'schema_version': 'rightarm.gesture.evaluation/v1',
                'labels': sorted(RIGHTARM_LABELS),
                'metrics': {
                    'offline.none_false_accept_rate': 0.5,
                    'offline.sample_count': 11,
                },
                'per_class': {
                    label: {'precision': 1, 'recall': 1, 'f1': 1, 'support': 1}
                    for label in RIGHTARM_LABELS
                },
                'confusion_matrix': {
                    truth: {
                        prediction: int(truth == prediction)
                        for prediction in RIGHTARM_LABELS
                    }
                    for truth in RIGHTARM_LABELS
                },
            }
        )
    )
    candidate = artifacts.stage(report_path, kind='evaluation-report')
    task = _task(
        kind='evaluate',
        validator='rightarm-evaluation-v1',
        artifact_kind='evaluation-report',
        gate=MetricGate(metric='offline.none_false_accept_rate', op='lte', value=0.01),
    )
    result = TaskResult(
        task_id='task_eval',
        attempt_id='attempt_eval',
        status='succeeded',
        summary='Agent claims the false accept rate is zero',
        artifact_candidates=[candidate],
        metrics={'offline.none_false_accept_rate': 0.0},
        evidence_refs=['evaluation.json'],
    )
    validation = build_rightarm_validator_registry(artifacts).validate(task, result)
    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'metric gate failed' in validation.reasons[0]
    assert validation.verified_metrics['offline.none_false_accept_rate'] == 0.5


def test_board_validator_requires_physical_keypoint_transpose(tmp_path: Path):
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    report_path = tmp_path / 'board.json'
    report_path.write_text(
        json.dumps(
            {
                'schema_version': 'rightarm.gesture.board-report/v1',
                'hardware': {'model': 'RK3588'},
                'runtime': {'version': '2.3.0'},
                'deployment_binary_sha': 'a' * 40,
                'preprocessing_version': 'rightarm-preprocess/v1',
                'input_layout': {'keypoint_physical': '(1,T,V,C)'},
                'metrics': {'edge.e2e_p95_latency_ratio_vs_champion': 1.0},
            }
        )
    )
    candidate = artifacts.stage(report_path, kind='board-report')
    task = _task(
        kind='verify-board',
        validator='rightarm-rk3588-v1',
        artifact_kind='board-report',
        gate=MetricGate(
            metric='edge.e2e_p95_latency_ratio_vs_champion', op='lte', value=1.05
        ),
    )
    result = TaskResult(
        task_id='task_board',
        attempt_id='attempt_board',
        status='succeeded',
        summary='board command exited zero',
        artifact_candidates=[candidate],
        evidence_refs=['board.json'],
    )
    validation = build_rightarm_validator_registry(artifacts).validate(task, result)
    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'does not prove keypoint' in validation.reasons[0]


def test_training_validator_reads_metrics_artifact_not_agent_claim(tmp_path: Path):
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    metrics_path = tmp_path / 'training-metrics.json'
    metrics_path.write_text(
        json.dumps(
            {
                'schema_version': 'rightarm.gesture.training-metrics/v1',
                'record_count': 2,
                'latest': {'loss': 0.25, 'acc/top1': 0.9},
                'sources': [
                    {
                        'path': 'vis_data/scalars.json',
                        'sha256': 'a' * 64,
                        'size_bytes': 10,
                    }
                ],
            }
        )
    )
    candidate = artifacts.stage(metrics_path, kind='training-metrics')
    task = TaskSpec(
        name='train-task',
        kind='train',
        objective='train',
        agent_profile='default',
        validator='rightarm-training-v1',
        max_attempts=1,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=['training-metrics'],
            required_evidence=True,
            gates=[MetricGate(metric='training.loss', op='lte', value=0.3)],
        ),
    )
    result = TaskResult(
        task_id='task_train',
        attempt_id='attempt_train',
        status='succeeded',
        summary='training complete',
        artifact_candidates=[candidate],
        metrics={'training.loss': 0.0},
        evidence_refs=['training-metrics.json'],
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.ACCEPTED
    assert validation.verified_metrics['training.loss'] == 0.25


def _task(
    *, kind: str, validator: str, artifact_kind: str, gate: MetricGate
) -> TaskSpec:
    return TaskSpec(
        name=f'{kind}-task',
        kind=kind,
        objective=kind,
        agent_profile='profile',
        validator=validator,
        max_attempts=1,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=[artifact_kind],
            required_evidence=True,
            gates=[gate],
        ),
    )
