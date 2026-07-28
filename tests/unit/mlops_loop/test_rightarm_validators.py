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


def test_dataset_distribution_review_recomputes_metrics_and_accepts_bound_outputs(
    tmp_path: Path,
):
    artifacts, task, result, expected_samples = _dataset_distribution_review(tmp_path)
    result.metrics = {
        'distribution.sample_count': 999,
        'merge.plan_actionable': False,
    }

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.ACCEPTED
    assert validation.verified_metrics['distribution.sample_count'] == expected_samples
    assert validation.verified_metrics['distribution.baseline_sample_count'] == 11
    assert validation.verified_metrics['distribution.new_batch_sample_count'] == 1
    assert validation.verified_metrics['distribution.hard_failure_count'] == 0
    assert validation.verified_metrics['distribution.missing_train_class_count'] == 0
    assert validation.verified_metrics['merge.plan_actionable'] is True
    assert validation.verified_metrics['merge.output_sample_count'] == expected_samples


def test_dataset_distribution_review_rejects_summary_not_derived_from_samples(
    tmp_path: Path,
):
    artifacts, task, result, _ = _dataset_distribution_review(
        tmp_path, wrong_summary=True
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'summary counts do not match' in validation.reasons[0]


def test_dataset_distribution_review_rejects_unbound_plan(tmp_path: Path):
    artifacts, task, result, _ = _dataset_distribution_review(
        tmp_path, wrong_plan_hash=True
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'plan does not bind the distribution report hash' in validation.reasons[0]


def test_dataset_distribution_review_rejects_final_dataset_identity_mismatch(
    tmp_path: Path,
):
    artifacts, task, result, _ = _dataset_distribution_review(
        tmp_path, wrong_output_hash=True
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'does not match the final mmaction dataset artifact' in validation.reasons[0]


def test_dataset_distribution_review_requires_exactly_one_completion_artifact(
    tmp_path: Path,
):
    artifacts, task, result, _ = _dataset_distribution_review(tmp_path)
    result.artifact_candidates = [
        candidate
        for candidate in result.artifact_candidates
        if candidate.kind != 'dataset-merge-plan'
    ]

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'requires exactly one artifact' in validation.reasons[0]


def _dataset_distribution_review(
    tmp_path: Path,
    *,
    wrong_summary: bool = False,
    wrong_plan_hash: bool = False,
    wrong_output_hash: bool = False,
) -> tuple[LocalArtifactStore, TaskSpec, TaskResult, int]:
    labels = [
        'WAVE_IN',
        'WAVE_OUT',
        'FIVE',
        'CLICK',
        'ROTATE',
        'SLIDE_DOWN',
        'SLIDE_UP',
        'OK',
        'CURSOR',
        'GUN',
        'NONE',
    ]
    inputs = [
        {
            'role': 'baseline',
            'artifact_id': 'artifact:baseline',
            'sha256': 'a' * 64,
            'size_bytes': 100,
        },
        {
            'role': 'new-batch',
            'artifact_id': 'artifact:new',
            'sha256': 'b' * 64,
            'size_bytes': 200,
        },
    ]
    samples = [
        {
            'id': f'baseline-{label.lower()}',
            'origin': 'baseline',
            'split': 'train',
            'label': label,
        }
        for label in labels
    ]
    samples.append(
        {
            'id': 'new-wave-in-val',
            'origin': 'new-batch',
            'split': 'val',
            'label': 'WAVE_IN',
        }
    )
    counts = {
        'baseline': {'train': {label: 1 for label in labels}},
        'new-batch': {'val': {'WAVE_IN': 1}},
    }
    if wrong_summary:
        counts['new-batch']['val']['WAVE_IN'] = 2
    distribution = {
        'schema_version': 'rightarm.gesture.dataset-distribution-report/v1',
        'canonical_categories': labels,
        'inputs': inputs,
        'samples': samples,
        'summary': {'counts': counts},
        'hard_failures': [],
        'recommendation': {'action': 'merge-all-unique'},
    }
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    distribution_path = tmp_path / 'distribution.json'
    distribution_path.write_text(json.dumps(distribution))
    distribution_candidate = artifacts.stage(
        distribution_path,
        kind='dataset-distribution-report',
        media_type='application/json',
    )

    plan = {
        'schema_version': 'rightarm.gesture.dataset-merge-plan/v1',
        'distribution_report_sha256': (
            'f' * 64 if wrong_plan_hash else distribution_candidate.sha256
        ),
        'inputs': inputs,
        'decision': 'merge-all-unique',
        'requires_human_decision': False,
    }
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    plan_candidate = artifacts.stage(
        plan_path,
        kind='dataset-merge-plan',
        media_type='application/json',
    )

    dataset_path = tmp_path / 'merged-dataset.tar'
    dataset_path.write_bytes(b'merged-mmaction-dataset')
    dataset_candidate = artifacts.stage(dataset_path, kind='mmaction-dataset')
    merged_counts = {
        'train': {label: 1 for label in labels},
        'val': {'WAVE_IN': 1},
    }
    merge = {
        'schema_version': 'rightarm.gesture.dataset-merge-report/v1',
        'distribution_report_sha256': distribution_candidate.sha256,
        'dataset_merge_plan_sha256': plan_candidate.sha256,
        'inputs': inputs,
        'output': {
            'sha256': 'e' * 64 if wrong_output_hash else dataset_candidate.sha256,
            'size_bytes': dataset_candidate.size_bytes,
            'sample_count': len(samples),
            'counts': merged_counts,
        },
    }
    merge_path = tmp_path / 'merge.json'
    merge_path.write_text(json.dumps(merge))
    merge_candidate = artifacts.stage(
        merge_path,
        kind='dataset-merge-report',
        media_type='application/json',
    )
    task = TaskSpec(
        name='review-dataset-distribution',
        kind='decision',
        objective='review the profiled dataset distribution and merged output',
        agent_profile='default',
        validator='rightarm-dataset-distribution-review-v1',
        max_attempts=1,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=[
                'dataset-distribution-report',
                'dataset-merge-plan',
                'dataset-merge-report',
                'mmaction-dataset',
            ],
            required_evidence=True,
            gates=[
                MetricGate(metric='distribution.hard_failure_count', op='eq', value=0),
                MetricGate(metric='merge.plan_actionable', op='eq', value=True),
            ],
        ),
    )
    result = TaskResult(
        task_id='task_dataset_review',
        attempt_id='attempt_dataset_review',
        status='succeeded',
        summary='distribution reviewed and dataset merged',
        artifact_candidates=[
            distribution_candidate,
            plan_candidate,
            merge_candidate,
            dataset_candidate,
        ],
        metrics={'distribution.hard_failure_count': 99},
        evidence_refs=['job:profile', 'job:merge'],
    )
    return artifacts, task, result, len(samples)


def test_model_improvement_recomputes_metrics_and_accepts_promoted_winner(
    tmp_path: Path,
):
    artifacts, task, result, expected_delta = _model_improvement_result(tmp_path)
    result.metrics = {
        'optimization.trial_count': 999,
        'optimization.macro_f1_delta_vs_baseline': -1.0,
        'optimization.promotion_eligible': False,
    }

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.ACCEPTED
    assert validation.verified_metrics['optimization.trial_count'] == 1
    assert (
        validation.verified_metrics['optimization.macro_f1_delta_vs_baseline']
        == expected_delta
    )
    assert validation.verified_metrics['optimization.promotion_eligible'] is True
    assert validation.verified_metrics['optimization.decision'] == 'promote-challenger'


def test_model_improvement_accepts_no_gain_and_keeps_champion(tmp_path: Path):
    artifacts, task, result, _ = _model_improvement_result(tmp_path, improved=False)

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.ACCEPTED
    assert validation.verified_metrics['optimization.promotion_eligible'] is False
    assert validation.verified_metrics['optimization.decision'] == 'keep-champion'
    assert validation.verified_metrics['optimization.macro_f1_delta_vs_baseline'] == 0.0


def test_model_improvement_rejects_declared_metric_not_derived_from_confusion(
    tmp_path: Path,
):
    artifacts, task, result, _ = _model_improvement_result(
        tmp_path, corrupt_evaluation_metric=True
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'does not match the confusion matrix' in validation.reasons[0]


def test_model_improvement_rejects_changed_evaluation_boundary(tmp_path: Path):
    artifacts, task, result, _ = _model_improvement_result(
        tmp_path, changed_dataset=True
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'evaluation boundary changed' in validation.reasons[0]


def test_model_improvement_rejects_unbound_history(tmp_path: Path):
    artifacts, task, result, _ = _model_improvement_result(
        tmp_path, wrong_history_hash=True
    )

    validation = build_rightarm_validator_registry(artifacts).validate(task, result)

    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'history does not cover every staged trial report' in validation.reasons[0]


def test_model_improvement_rejects_non_allowlisted_or_over_budget_trial(
    tmp_path: Path,
):
    artifacts, task, result, _ = _model_improvement_result(
        tmp_path, strategy='edit-arbitrary-code'
    )
    validation = build_rightarm_validator_registry(artifacts).validate(task, result)
    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'strategy is not allowlisted' in validation.reasons[0]

    artifacts, task, result, _ = _model_improvement_result(
        tmp_path / 'budget', epochs=51, policy_max_epochs=50
    )
    validation = build_rightarm_validator_registry(artifacts).validate(task, result)
    assert validation.disposition is ValidationDisposition.TERMINAL
    assert 'exceeds max_epochs policy' in validation.reasons[0]


def _model_improvement_result(
    tmp_path: Path,
    *,
    improved: bool = True,
    corrupt_evaluation_metric: bool = False,
    changed_dataset: bool = False,
    wrong_history_hash: bool = False,
    strategy: str = 'fine-tune',
    epochs: int = 50,
    policy_max_epochs: int = 50,
) -> tuple[LocalArtifactStore, TaskSpec, TaskResult, float]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifacts = LocalArtifactStore(tmp_path / 'cas')
    dataset_path = tmp_path / 'dataset.pkl'
    dataset_path.write_bytes(b'immutable-dataset')
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    baseline_checkpoint_path = tmp_path / 'incumbent.pth'
    baseline_checkpoint_path.write_bytes(b'incumbent-checkpoint')
    baseline_checkpoint_sha = hashlib.sha256(
        baseline_checkpoint_path.read_bytes()
    ).hexdigest()
    baseline_config_path = tmp_path / 'incumbent-config.json'
    baseline_config_path.write_text('{}')
    baseline_config_sha = hashlib.sha256(baseline_config_path.read_bytes()).hexdigest()

    baseline_confusion = _improvement_confusion(perfect=False)
    baseline = _checkpoint_evaluation(
        dataset_sha=dataset_sha,
        dataset_size=dataset_path.stat().st_size,
        checkpoint_sha=baseline_checkpoint_sha,
        checkpoint_size=baseline_checkpoint_path.stat().st_size,
        config_sha=baseline_config_sha,
        config_size=baseline_config_path.stat().st_size,
        confusion=baseline_confusion,
    )
    baseline_path = tmp_path / 'baseline-evaluation.json'
    baseline_path.write_text(json.dumps(baseline, sort_keys=True))
    baseline_candidate = artifacts.stage(
        baseline_path, kind='baseline-evaluation-report'
    )

    trial_checkpoint_path = tmp_path / 'trial-checkpoint.pth'
    trial_checkpoint_path.write_bytes(b'trial-checkpoint')
    trial_checkpoint_sha = hashlib.sha256(
        trial_checkpoint_path.read_bytes()
    ).hexdigest()
    training_config_path = tmp_path / 'training-config.json'
    training_config_path.write_text('{"trial":1}')
    training_log_path = tmp_path / 'training.log'
    training_log_path.write_text('trained')
    training_metrics_path = tmp_path / 'training-metrics.json'
    training_metrics_path.write_text('{"loss":0.1}')
    trial_confusion = _improvement_confusion(perfect=improved)
    trial_dataset_sha = 'd' * 64 if changed_dataset else dataset_sha
    trial_evaluation = _checkpoint_evaluation(
        dataset_sha=trial_dataset_sha,
        dataset_size=dataset_path.stat().st_size,
        checkpoint_sha=trial_checkpoint_sha,
        checkpoint_size=trial_checkpoint_path.stat().st_size,
        config_sha=hashlib.sha256(training_config_path.read_bytes()).hexdigest(),
        config_size=training_config_path.stat().st_size,
        confusion=trial_confusion,
    )
    if corrupt_evaluation_metric:
        trial_evaluation['metrics']['offline.macro_f1'] = 0.0
    trial_evaluation_path = tmp_path / 'trial-evaluation.json'
    trial_evaluation_path.write_text(json.dumps(trial_evaluation, sort_keys=True))

    output_paths = {
        'checkpoint': trial_checkpoint_path,
        'training_config': training_config_path,
        'training_log': training_log_path,
        'training_metrics': training_metrics_path,
        'evaluation_report': trial_evaluation_path,
    }
    output_identities = {
        f'{name}_{field}': (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if field == 'sha256'
            else path.stat().st_size
        )
        for name, path in output_paths.items()
        for field in ('sha256', 'size_bytes')
    }
    trial_parameters = {
        'lr': 0.01,
        'weight_decay': 0.0005,
        'dropout': 0.35,
        'gate_threshold': 0.45,
        'gate_loss_weight': 0.5,
        'contrast_loss_weight': 0.01,
        'augmentation_preset': 'standard',
        'motion_preset': 'standard',
        'batch_size': 32,
        'epochs': epochs,
        'seed': 42,
    }
    trial = {
        'schema_version': 'rightarm.gesture.optimization-trial/v1',
        'trial_number': 1,
        'strategy': strategy,
        'parameters': trial_parameters,
        'input': {
            'dataset_sha256': trial_dataset_sha,
            'dataset_size_bytes': dataset_path.stat().st_size,
            'base_model_sha256': baseline_checkpoint_sha,
            'base_model_size_bytes': baseline_checkpoint_path.stat().st_size,
            'initialization_checkpoint_sha256': baseline_checkpoint_sha,
            'initialization_checkpoint_size_bytes': (
                baseline_checkpoint_path.stat().st_size
            ),
            'evaluation_split': 'val',
            'preprocessing_version': 'rightarm.gesture.preprocessing/v1',
            'preprocessing_fingerprint': 'f' * 64,
        },
        'outputs': output_identities,
        'evaluation': trial_evaluation,
        'evaluation_candidates': [
            {
                'role': 'best',
                'checkpoint_sha256': trial_checkpoint_sha,
                'evaluation_report_sha256': output_identities[
                    'evaluation_report_sha256'
                ],
                'metrics': trial_evaluation['metrics'],
            }
        ],
        'selection': {
            'policy': 'rightarm.gesture.product-selection/v1',
            'winner_role': 'best',
            'reason_codes': [
                'none-far-feasibility-first',
                'macro-f1-primary',
                'min-recall-secondary',
                'accuracy-tertiary',
            ],
        },
    }
    trial_path = tmp_path / 'trial-report.json'
    trial_path.write_text(json.dumps(trial, sort_keys=True))
    trial_candidate = artifacts.stage(trial_path, kind='optimization-trial-report')

    selected_training_candidates = {
        'training-config': artifacts.stage(
            training_config_path, kind='training-config'
        ),
        'training-log': artifacts.stage(training_log_path, kind='training-log'),
        'training-metrics': artifacts.stage(
            training_metrics_path, kind='training-metrics'
        ),
    }
    if improved:
        selected_checkpoint = artifacts.stage(trial_checkpoint_path, kind='checkpoint')
        selected_evaluation = artifacts.stage(
            trial_evaluation_path, kind='evaluation-report'
        )
    else:
        selected_checkpoint = artifacts.stage(
            baseline_checkpoint_path, kind='checkpoint'
        )
        selected_evaluation = artifacts.stage(baseline_path, kind='evaluation-report')

    baseline_metrics = baseline['metrics']
    trial_metrics = trial_evaluation['metrics']
    deltas = {
        'offline.accuracy_delta_vs_baseline': (
            trial_metrics['offline.accuracy'] - baseline_metrics['offline.accuracy']
        ),
        'offline.macro_f1_delta_vs_baseline': (
            trial_metrics['offline.macro_f1'] - baseline_metrics['offline.macro_f1']
        ),
        'offline.min_per_class_recall_delta_vs_baseline': (
            trial_metrics['offline.min_per_class_recall']
            - baseline_metrics['offline.min_per_class_recall']
        ),
        'offline.none_false_accept_rate_delta_vs_baseline': (
            trial_metrics['offline.none_false_accept_rate']
            - baseline_metrics['offline.none_false_accept_rate']
        ),
    }
    trial_summary = {
        'trial_number': 1,
        'trial_report_sha256': trial_candidate.sha256,
        **{
            f'{name}_sha256': output_identities[f'{name}_sha256']
            for name in output_paths
        },
        'metrics': trial_metrics,
    }
    winner = {
        'source': 'trial' if improved else 'baseline',
        'trial_number': 1 if improved else None,
        'trial_report_sha256': trial_candidate.sha256 if improved else None,
        'checkpoint_sha256': (
            trial_checkpoint_sha if improved else baseline_checkpoint_sha
        ),
        'training_config_sha256': (
            output_identities['training_config_sha256'] if improved else None
        ),
        'training_log_sha256': (
            output_identities['training_log_sha256'] if improved else None
        ),
        'training_metrics_sha256': (
            output_identities['training_metrics_sha256'] if improved else None
        ),
        'evaluation_report_sha256': (
            output_identities['evaluation_report_sha256']
            if improved
            else baseline_candidate.sha256
        ),
    }
    final = {
        'schema_version': 'rightarm.gesture.model-improvement/v1',
        'baseline_evaluation_sha256': baseline_candidate.sha256,
        'baseline_evaluation': baseline,
        'trial_report_sha256s': [
            'e' * 64 if wrong_history_hash else trial_candidate.sha256
        ],
        'trials': [trial_summary],
        'best_trial': trial_summary,
        'winner': winner,
        'decision': 'promote-challenger' if improved else 'keep-champion',
        'stop_reason': 'target-reached' if improved else 'trial-budget-exhausted',
        'policy': {
            'max_trials': 1,
            'patience': 1,
            'target_macro_f1_delta': 0.001,
            'min_improvement': 0.0005,
            'max_epochs': policy_max_epochs,
            'batch_size': 32,
            'workers': 12,
            'gpus': 1,
        },
        'deltas_vs_baseline': deltas,
        'reason_codes': (
            [
                'macro-f1-minimum-improvement-reached',
                'minimum-recall-guard-passed',
                'none-far-guard-passed',
            ]
            if improved
            else ['challenger-did-not-pass-product-selection-policy']
        ),
    }
    analysis_document = {
        'summary': 'synthetic validator fixture',
        'recommended_action': final['decision'],
    }
    analysis_bytes = json.dumps(
        analysis_document,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')
    final['analysis'] = {
        'sha256': hashlib.sha256(analysis_bytes).hexdigest(),
        'document': analysis_document,
    }
    final_path = tmp_path / 'model-improvement.json'
    final_path.write_text(json.dumps(final, sort_keys=True))
    final_candidate = artifacts.stage(final_path, kind='model-improvement-report')

    task = TaskSpec(
        name='improve-challenger',
        kind='improve-model',
        objective='bounded model improvement',
        agent_profile='default',
        validator='rightarm-model-improvement-v1',
        max_attempts=1,
        timeout_seconds=60,
        completion_contract=CompletionContract(
            required_artifact_kinds=[
                'baseline-evaluation-report',
                'checkpoint',
                'training-config',
                'training-log',
                'training-metrics',
                'evaluation-report',
                'optimization-trial-report',
                'model-improvement-report',
            ],
            required_evidence=True,
            gates=[
                MetricGate(metric='optimization.trial_count', op='gte', value=1),
                MetricGate(metric='optimization.trial_count', op='lte', value=1),
            ],
        ),
    )
    result = TaskResult(
        task_id='task_improve',
        attempt_id='attempt_improve',
        status='succeeded',
        summary='bounded improvement finished',
        artifact_candidates=[
            baseline_candidate,
            selected_checkpoint,
            *selected_training_candidates.values(),
            selected_evaluation,
            trial_candidate,
            final_candidate,
        ],
        metrics={'optimization.trial_count': 99},
        evidence_refs=['job:benchmark', 'job:trial-1', 'job:record'],
    )
    return (
        artifacts,
        task,
        result,
        deltas['offline.macro_f1_delta_vs_baseline'],
    )


def _improvement_confusion(*, perfect: bool) -> dict[str, dict[str, int]]:
    labels = [
        'WAVE_IN',
        'WAVE_OUT',
        'FIVE',
        'CLICK',
        'ROTATE',
        'SLIDE_DOWN',
        'SLIDE_UP',
        'OK',
        'CURSOR',
        'GUN',
        'NONE',
    ]
    confusion = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for label in labels:
        confusion[label][label] = 10
    if not perfect:
        confusion['WAVE_IN']['WAVE_IN'] = 9
        confusion['WAVE_IN']['WAVE_OUT'] = 1
    return confusion


def _checkpoint_evaluation(
    *,
    dataset_sha: str,
    dataset_size: int,
    checkpoint_sha: str,
    checkpoint_size: int,
    config_sha: str,
    config_size: int,
    confusion: dict[str, dict[str, int]],
) -> dict:
    labels = list(confusion)
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in labels)
    per_class = {}
    f1_values = []
    recalls = []
    for label in labels:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in labels)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }
        f1_values.append(f1)
        recalls.append(recall)
    none_support = sum(confusion['NONE'].values())
    metrics = {
        'offline.sample_count': total,
        'offline.accuracy': correct / total,
        'offline.macro_f1': sum(f1_values) / len(f1_values),
        'offline.min_per_class_recall': min(recalls),
        'offline.none_false_accept_rate': (
            (none_support - confusion['NONE']['NONE']) / none_support
            if none_support
            else 0.0
        ),
    }
    return {
        'schema_version': 'rightarm.gesture.checkpoint-evaluation/v2',
        'identity': {
            'dataset_sha256': dataset_sha,
            'dataset_size_bytes': dataset_size,
            'checkpoint_sha256': checkpoint_sha,
            'checkpoint_size_bytes': checkpoint_size,
            'config_sha256': config_sha,
            'config_size_bytes': config_size,
            'evaluation_split': 'val',
            'preprocessing_version': 'rightarm.gesture.preprocessing/v1',
            'preprocessing_fingerprint': 'f' * 64,
        },
        'labels': labels,
        'metrics': metrics,
        'per_class': per_class,
        'confusion_matrix': confusion,
        'evidence': {
            'test_log_sha256': 'a' * 64,
            'test_log_size_bytes': 1,
            'confusion_matrix_sha256': 'b' * 64,
            'confusion_matrix_size_bytes': 1,
        },
    }
