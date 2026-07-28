# The pinned GPU training image runs Python 3.8; keep typing.Dict/List/Tuple.
# ruff: noqa: UP006
"""Deterministic training/evaluation primitives for model improvement trials.

The OpenHands Conversation may choose a bounded hypothesis, but this module
owns the executable parameter allowlist, evaluation math, and winner selection.
It deliberately does not accept arbitrary MMEngine ``cfg-options``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pprint
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rightarm_mlops.protocol import atomic_json, sha256_file

LABELS: Tuple[str, ...] = (
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
)
NONE_INDEX = LABELS.index('NONE')

EVALUATION_SCHEMA = 'rightarm.gesture.checkpoint-evaluation/v2'
TRIAL_SCHEMA = 'rightarm.gesture.optimization-trial/v1'
IMPROVEMENT_SCHEMA = 'rightarm.gesture.model-improvement/v1'
PREPROCESSING_VERSION = 'rightarm.gesture.preprocessing/v1'
SELECTION_POLICY = 'rightarm.gesture.product-selection/v1'

ALLOWED_STRATEGIES = frozenset(
    {
        'fine-tune',
        'optimizer',
        'regularization',
        'augmentation',
        'motion-ablation',
        'gesture-gate',
    }
)
ALLOWED_BATCH_SIZES = frozenset({4, 8, 16, 32, 64})
ALLOWED_AUGMENTATION_PRESETS = frozenset({'standard', 'light', 'strong', 'none'})
ALLOWED_MOTION_PRESETS = frozenset({'standard', 'no-delta-z'})
MAX_HISTORY_TRIALS = 8

DEFAULT_PARAMETERS: Dict[str, Any] = {
    'lr': 0.01,
    'weight_decay': 0.0005,
    'dropout': 0.35,
    'gate_threshold': 0.45,
    'gate_loss_weight': 0.5,
    'contrast_loss_weight': 0.01,
    'augmentation_preset': 'standard',
    'motion_preset': 'standard',
    'batch_size': 32,
    'epochs': 50,
    'seed': 42,
}
PARAMETER_KEYS = frozenset(DEFAULT_PARAMETERS)

DEFAULT_POLICY: Dict[str, Any] = {
    'max_trials': 4,
    'patience': 2,
    'target_macro_f1_delta': 0.002,
    'min_improvement': 0.0005,
    'max_epochs': 50,
    'batch_size': 32,
    'workers': 8,
    'gpus': 1,
}
POLICY_KEYS = frozenset(DEFAULT_POLICY)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    digest, size = sha256_file(path.resolve(strict=True))
    return {'sha256': digest, 'size_bytes': size}


def executor_image_identity(
    requested_image: Optional[str] = None,
) -> Optional[str]:
    """Prefer the executor-owned image identity and reject conflicting claims."""

    injected = os.environ.get('RIGHTARM_MLOPS_EXECUTOR_IMAGE', '').strip()
    requested = (requested_image or '').strip()
    if injected and requested and injected != requested:
        raise ValueError('--image-digest conflicts with RIGHTARM_MLOPS_EXECUTOR_IMAGE')
    return injected or requested or None


def mmaction_subprocess_environment(mmaction_root: Path) -> Dict[str, str]:
    """Use the mounted workspace code instead of an image-baked editable checkout."""

    root = mmaction_root.resolve(strict=True)
    workspace_root = root.parent.parent
    environment = dict(os.environ)
    entries = [str(root), str(workspace_root)]
    existing = environment.get('PYTHONPATH', '')
    if existing:
        entries.append(existing)
    environment['PYTHONPATH'] = os.pathsep.join(entries)
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return environment


def normalize_experiment_parameters(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError('experiment parameters must be a JSON object')
    unknown = sorted(set(raw) - PARAMETER_KEYS)
    if unknown:
        raise ValueError(f'unsupported experiment parameter keys: {unknown}')
    values = dict(DEFAULT_PARAMETERS)
    values.update(raw)

    _bounded_float(values, 'lr', 1e-5, 0.05)
    _bounded_float(values, 'weight_decay', 0.0, 0.01)
    _bounded_float(values, 'dropout', 0.0, 0.75)
    _bounded_float(values, 'gate_threshold', 0.2, 0.8)
    _bounded_float(values, 'gate_loss_weight', 0.05, 2.0)
    _bounded_float(values, 'contrast_loss_weight', 0.0, 0.1)
    _enum_value(
        values,
        'augmentation_preset',
        ALLOWED_AUGMENTATION_PRESETS,
    )
    _enum_value(values, 'motion_preset', ALLOWED_MOTION_PRESETS)
    _integer_value(values, 'batch_size', minimum=1)
    if values['batch_size'] not in ALLOWED_BATCH_SIZES:
        raise ValueError(f'batch_size must be one of {sorted(ALLOWED_BATCH_SIZES)}')
    _integer_value(values, 'epochs', minimum=1, maximum=100)
    _integer_value(values, 'seed', minimum=0, maximum=2_147_483_647)
    return values


def normalize_policy(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError('optimization policy must be a JSON object')
    unknown = sorted(set(raw) - POLICY_KEYS)
    if unknown:
        raise ValueError(f'unsupported optimization policy keys: {unknown}')
    values = dict(DEFAULT_POLICY)
    values.update(raw)
    _integer_value(values, 'max_trials', minimum=1, maximum=MAX_HISTORY_TRIALS)
    _integer_value(values, 'patience', minimum=1)
    if values['patience'] > values['max_trials']:
        raise ValueError('policy patience cannot exceed max_trials')
    _bounded_float(values, 'target_macro_f1_delta', 0.0, 1.0)
    _bounded_float(values, 'min_improvement', 0.0, 0.05)
    _integer_value(values, 'max_epochs', minimum=1, maximum=100)
    _integer_value(values, 'batch_size', minimum=1)
    if values['batch_size'] not in ALLOWED_BATCH_SIZES:
        raise ValueError(
            f'policy batch_size must be one of {sorted(ALLOWED_BATCH_SIZES)}'
        )
    _integer_value(values, 'workers', minimum=0, maximum=32)
    _integer_value(values, 'gpus', minimum=1, maximum=8)
    return values


def _bounded_float(
    values: Dict[str, Any], key: str, minimum: float, maximum: float
) -> None:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{key} must be numeric')
    value = float(value)
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f'{key} must be in [{minimum}, {maximum}]')
    values[key] = value


def _integer_value(
    values: Dict[str, Any],
    key: str,
    *,
    minimum: int,
    maximum: Optional[int] = None,
) -> None:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{key} must be an integer')
    if value < minimum or (maximum is not None and value > maximum):
        upper = f', {maximum}' if maximum is not None else ''
        raise ValueError(f'{key} must be in [{minimum}{upper}]')


def _enum_value(values: Dict[str, Any], key: str, allowed: Iterable[str]) -> None:
    value = values[key]
    allowed_set = frozenset(allowed)
    if value not in allowed_set:
        raise ValueError(f'{key} must be one of {sorted(allowed_set)}')


def validate_strategy(strategy: str) -> str:
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f'strategy must be one of {sorted(ALLOWED_STRATEGIES)}')
    return strategy


def parse_json_object(raw: str, *, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{label} must be valid JSON') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object')
    return value


def validate_source_config(config: Path, repo_root: Path) -> Path:
    """Restrict improvement jobs to the reviewed RGB-gated 11-class config."""

    resolved = config.resolve(strict=True)
    allowed_root = (
        repo_root.resolve(strict=True)
        / 'training'
        / 'mmaction2'
        / 'handoff'
        / 'stgcn_hand21_3d'
        / 'configs'
    ).resolve(strict=True)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            'model-improvement config must be in the reviewed handoff config directory'
        ) from exc
    allowed_names = {
        'stgcn_hand21-rgb-bs64-gated-v3-ft-rtm_add-11cls.py',
    }
    if resolved.name not in allowed_names:
        raise ValueError(
            f'model-improvement config is not allowlisted: {resolved.name}'
        )
    return resolved


def _literal_config_namespace(config: Path) -> Dict[str, Any]:
    """Resolve the checked-in standalone config without importing torch."""

    namespace = runpy.run_path(str(config))
    result: Dict[str, Any] = {}
    for key, value in namespace.items():
        if key.startswith('_'):
            continue
        if _is_literal_config_value(value):
            result[key] = value
    required = {
        'model',
        'train_dataloader',
        'val_dataloader',
        'test_dataloader',
        'train_cfg',
        'param_scheduler',
        'optim_wrapper',
        'test_evaluator',
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f'reviewed config is missing required sections: {missing}')
    return result


def _is_literal_config_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_literal_config_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, (bool, int, float, str)) and _is_literal_config_value(item)
            for key, item in value.items()
        )
    return False


def _training_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = config['train_dataloader']['dataset']
    if dataset.get('type') == 'RepeatDataset':
        dataset = dataset['dataset']
    if not isinstance(dataset, dict):
        raise ValueError('train dataset config is invalid')
    return dataset


def _set_dataset_and_runtime(
    config: Dict[str, Any],
    *,
    dataset: Path,
    evaluation_split: str,
    batch_size: int,
    workers: int,
    seed: int,
) -> None:
    _training_dataset_config(config)['ann_file'] = str(dataset)
    _training_dataset_config(config)['split'] = 'train'
    for loader_name in ('val_dataloader', 'test_dataloader'):
        loader = config[loader_name]
        loader['batch_size'] = batch_size
        loader['num_workers'] = workers
        loader['dataset']['ann_file'] = str(dataset)
        loader['dataset']['split'] = evaluation_split
    config['train_dataloader']['batch_size'] = batch_size
    config['train_dataloader']['num_workers'] = workers
    config['randomness'] = {'seed': seed, 'deterministic': True}
    if isinstance(config.get('env_cfg'), dict):
        config['env_cfg']['cudnn_benchmark'] = False


def _use_checkpoint_backed_rgb_initialization(config: Dict[str, Any]) -> None:
    """Avoid external backbone downloads when a full checkpoint will be loaded."""

    model = config.get('model')
    rgb_branch = model.get('rgb_branch') if isinstance(model, dict) else None
    if not isinstance(rgb_branch, dict) or rgb_branch.get('type') != 'RGBBranch':
        raise ValueError('reviewed config must define an RGBBranch')
    rgb_branch['pretrained'] = False


def _pipeline(config: Dict[str, Any], loader: str) -> List[Dict[str, Any]]:
    if loader == 'train_dataloader':
        dataset = _training_dataset_config(config)
    else:
        dataset = config[loader]['dataset']
    pipeline = dataset.get('pipeline')
    if not isinstance(pipeline, list) or not all(
        isinstance(item, dict) for item in pipeline
    ):
        raise ValueError(f'{loader} has no mutable pipeline')
    return pipeline


def _find_transform(
    pipeline: Sequence[Dict[str, Any]], transform_type: str
) -> Dict[str, Any]:
    matches = [item for item in pipeline if item.get('type') == transform_type]
    if len(matches) != 1:
        raise ValueError(
            f'expected exactly one {transform_type} transform, found {len(matches)}'
        )
    return matches[0]


def _apply_augmentation_preset(config: Dict[str, Any], preset: str) -> None:
    if preset == 'standard':
        return
    pipeline = _pipeline(config, 'train_dataloader')
    rotate = _find_transform(pipeline, 'JointRandomRotate')
    keypoint = _find_transform(pipeline, 'JointKeypointJitter')
    rgb = _find_transform(pipeline, 'RGBQualityJitter')
    if preset == 'none':
        rotate.update({'discrete_prob': 0.0, 'continuous_prob': 0.0})
        keypoint['prob'] = 0.0
        rgb['prob'] = 0.0
    elif preset == 'light':
        rotate.update(
            {
                'discrete_prob': 0.0,
                'continuous_prob': 0.1,
                'continuous_max_deg': 30.0,
            }
        )
        keypoint['prob'] = 0.25
        rgb['prob'] = 0.25
    elif preset == 'strong':
        rotate.update(
            {
                'discrete_prob': 0.0,
                'continuous_prob': 0.4,
                'continuous_max_deg': 45.0,
            }
        )
        keypoint['prob'] = 0.75
        rgb['prob'] = 0.75
    else:  # normalize_experiment_parameters should make this unreachable.
        raise ValueError(f'unsupported augmentation preset: {preset}')


def _apply_motion_preset(config: Dict[str, Any], preset: str) -> None:
    if preset == 'standard':
        return
    if preset != 'no-delta-z':
        raise ValueError(f'unsupported motion preset: {preset}')
    for loader in (
        'train_dataloader',
        'val_dataloader',
        'test_dataloader',
    ):
        transform = _find_transform(_pipeline(config, loader), 'FormatGlobalTrajectory')
        transform['type'] = 'MLOpsFormatGlobalTrajectory'
        transform['zero_channels'] = (2,)
    imports = config.setdefault(
        'custom_imports',
        {'imports': [], 'allow_failed_imports': False},
    ).setdefault('imports', [])
    if 'rightarm_mlops.model_improvement' not in imports:
        imports.append('rightarm_mlops.model_improvement')


def _render_config(path: Path, config: Mapping[str, Any]) -> None:
    lines = [
        '# Deterministically generated by rightarm_mlops.model_improvement.',
        '# Arbitrary cfg-options are intentionally unsupported.',
        '',
    ]
    for key in sorted(config):
        lines.append(
            f'{key} = {pprint.pformat(config[key], width=88, sort_dicts=True)}'
        )
        lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def preprocessing_descriptor(
    config: Mapping[str, Any], evaluation_split: str
) -> Dict[str, Any]:
    test_dataset = config['test_dataloader']['dataset']
    pipeline = []
    for raw in test_dataset.get('pipeline', []):
        item = dict(raw)
        # Model hypotheses may change the deployed preprocessing implementation,
        # but every score comparison remains bound to the same immutable
        # product evaluation contract. Record those hypotheses in the config and
        # trial report rather than making otherwise comparable evaluations use
        # unrelated boundary fingerprints.
        if item.get('type') == 'MLOpsFormatGlobalTrajectory':
            item['type'] = 'FormatGlobalTrajectory'
            item.pop('zero_channels', None)
        pipeline.append(item)
    return {
        'schema_version': PREPROCESSING_VERSION,
        'evaluation_split': evaluation_split,
        'categories': list(config.get('categories', LABELS)),
        'dataset_type': test_dataset.get('type'),
        'canonical_test_pipeline': pipeline,
    }


def resolve_benchmark_config(
    *,
    source_config: Path,
    dataset: Path,
    evaluation_split: str,
    batch_size: int,
    workers: int,
    seed: int,
    output_path: Path,
) -> Tuple[Dict[str, Any], str]:
    config = _literal_config_namespace(source_config)
    _set_dataset_and_runtime(
        config,
        dataset=dataset,
        evaluation_split=evaluation_split,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
    )
    _use_checkpoint_backed_rgb_initialization(config)
    for evaluator in config['test_evaluator']:
        if evaluator.get('type') != 'ConfusionMatrix':
            continue
        evaluator['save_path'] = str(output_path.parent / 'confusion_matrix_test.npy')
        if evaluator.get('misclassified_save_path') is not None:
            evaluator['misclassified_save_path'] = str(
                output_path.parent / 'misclassified_test.txt'
            )
        if evaluator.get('classified_save_path') is not None:
            evaluator['classified_save_path'] = str(
                output_path.parent / 'classified_test.txt'
            )
        if evaluator.get('summary_save_path') is not None:
            evaluator['summary_save_path'] = str(
                output_path.parent / 'confusion_matrix_test_summary.txt'
            )
    descriptor = preprocessing_descriptor(config, evaluation_split)
    fingerprint = canonical_sha256(descriptor)
    _render_config(output_path, config)
    return descriptor, fingerprint


def resolve_trial_config(
    *,
    source_config: Path,
    dataset: Path,
    initialization_checkpoint: Path,
    evaluation_split: str,
    workers: int,
    parameters: Mapping[str, Any],
    output_path: Path,
) -> Tuple[Dict[str, Any], str]:
    normalized = normalize_experiment_parameters(parameters)
    config = _literal_config_namespace(source_config)
    _set_dataset_and_runtime(
        config,
        dataset=dataset,
        evaluation_split=evaluation_split,
        batch_size=normalized['batch_size'],
        workers=workers,
        seed=normalized['seed'],
    )
    _use_checkpoint_backed_rgb_initialization(config)
    config['load_from'] = str(initialization_checkpoint)
    config['resume'] = False
    config['train_cfg']['max_epochs'] = normalized['epochs']
    config['default_hooks']['checkpoint']['max_keep_ckpts'] = 3
    optimizer = config['optim_wrapper']['optimizer']
    optimizer['lr'] = normalized['lr']
    optimizer['weight_decay'] = normalized['weight_decay']
    head = config['model']['cls_head']
    head['dropout'] = normalized['dropout']
    head['gate_threshold'] = normalized['gate_threshold']
    head['gate_loss_weight'] = normalized['gate_loss_weight']
    head['contrast_loss_weight'] = normalized['contrast_loss_weight']
    _apply_augmentation_preset(config, normalized['augmentation_preset'])
    _apply_motion_preset(config, normalized['motion_preset'])

    epochs = normalized['epochs']
    warmup = min(3, max(epochs - 1, 0))
    schedulers: List[Dict[str, Any]] = []
    if warmup:
        schedulers.append(
            {
                'type': 'LinearLR',
                'start_factor': 0.01,
                'by_epoch': True,
                'begin': 0,
                'end': warmup,
                'convert_to_iter_based': True,
            }
        )
    schedulers.append(
        {
            'type': 'CosineAnnealingLR',
            'eta_min': 0,
            'T_max': max(epochs - warmup, 1),
            'by_epoch': True,
            'begin': warmup,
            'end': epochs,
            'convert_to_iter_based': True,
        }
    )
    config['param_scheduler'] = schedulers
    descriptor = preprocessing_descriptor(config, evaluation_split)
    fingerprint = canonical_sha256(descriptor)
    _render_config(output_path, config)
    return descriptor, fingerprint


def checkpoint_evaluation_report(
    *,
    confusion: np.ndarray,
    dataset: Path,
    checkpoint: Path,
    config: Path,
    evaluation_split: str,
    preprocessing_fingerprint: str,
    test_log: Path,
    confusion_path: Path,
    executor_image: Optional[str],
) -> Dict[str, Any]:
    matrix = np.asarray(confusion)
    if matrix.shape != (len(LABELS), len(LABELS)):
        raise ValueError(f'confusion matrix must be 11x11, got {matrix.shape!r}')
    if not np.issubdtype(matrix.dtype, np.integer):
        if not np.all(np.equal(matrix, np.floor(matrix))):
            raise ValueError('confusion matrix contains non-integer values')
        matrix = matrix.astype(np.int64)
    else:
        matrix = matrix.astype(np.int64, copy=False)
    if np.any(matrix < 0):
        raise ValueError('confusion matrix contains negative values')
    sample_count = int(matrix.sum())
    if sample_count < 1:
        raise ValueError('evaluation has no samples')

    diagonal = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        diagonal,
        predicted,
        out=np.zeros_like(diagonal),
        where=predicted != 0,
    )
    recall = np.divide(
        diagonal,
        support,
        out=np.zeros_like(diagonal),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) != 0,
    )
    none_total = int(support[NONE_INDEX])
    none_false_accept = int(matrix[NONE_INDEX].sum() - matrix[NONE_INDEX, NONE_INDEX])
    metrics = {
        'offline.sample_count': sample_count,
        'offline.accuracy': float(diagonal.sum() / sample_count),
        'offline.macro_f1': float(f1.mean()),
        'offline.min_per_class_recall': float(recall.min()),
        'offline.none_false_accept_rate': (
            float(none_false_accept / none_total) if none_total else 0.0
        ),
    }
    dataset_id = file_identity(dataset)
    checkpoint_id = file_identity(checkpoint)
    config_id = file_identity(config)
    log_id = file_identity(test_log)
    confusion_id = file_identity(confusion_path)
    identity = {
        'dataset_sha256': dataset_id['sha256'],
        'dataset_size_bytes': dataset_id['size_bytes'],
        'checkpoint_sha256': checkpoint_id['sha256'],
        'checkpoint_size_bytes': checkpoint_id['size_bytes'],
        'config_sha256': config_id['sha256'],
        'config_size_bytes': config_id['size_bytes'],
        'evaluation_split': evaluation_split,
        'preprocessing_version': PREPROCESSING_VERSION,
        'preprocessing_fingerprint': preprocessing_fingerprint,
    }
    return {
        'schema_version': EVALUATION_SCHEMA,
        'evaluation_id': canonical_sha256(identity),
        'identity': identity,
        'labels': list(LABELS),
        'metrics': metrics,
        'per_class': {
            label: {
                'precision': float(precision[index]),
                'recall': float(recall[index]),
                'f1': float(f1[index]),
                'support': int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        'confusion_matrix': {
            truth: {
                prediction: int(matrix[truth_index, prediction_index])
                for prediction_index, prediction in enumerate(LABELS)
            }
            for truth_index, truth in enumerate(LABELS)
        },
        'evidence': {
            'test_log_sha256': log_id['sha256'],
            'test_log_size_bytes': log_id['size_bytes'],
            'confusion_matrix_sha256': confusion_id['sha256'],
            'confusion_matrix_size_bytes': confusion_id['size_bytes'],
            'executor_image': executor_image,
        },
    }


def run_checkpoint_evaluation(
    *,
    source_config: Path,
    dataset: Path,
    identity_dataset: Optional[Path] = None,
    identity_config: Optional[Path] = None,
    checkpoint: Path,
    mmaction_root: Path,
    evaluation_split: str,
    batch_size: int,
    workers: int,
    seed: int,
    output_dir: Path,
    executor_image: Optional[str],
) -> Tuple[Dict[str, Any], Path, Path, List[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / 'resolved-evaluation-config.py'
    _, preprocessing_fingerprint = resolve_benchmark_config(
        source_config=source_config,
        dataset=dataset,
        evaluation_split=evaluation_split,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        output_path=resolved_config,
    )
    command = [
        sys.executable,
        str(mmaction_root / 'tools' / 'test.py'),
        str(resolved_config),
        str(checkpoint),
        '--work-dir',
        str(output_dir),
    ]
    log_path = output_dir / 'evaluation.log'
    with log_path.open('wb') as log:
        try:
            subprocess.run(
                command,
                cwd=output_dir,
                env=mmaction_subprocess_environment(mmaction_root),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            setattr(exc, 'mlops_log_path', log_path)
            raise
    confusion_path = output_dir / 'confusion_matrix_test.npy'
    if not confusion_path.is_file():
        candidates = sorted(output_dir.rglob('confusion_matrix*.npy'))
        if len(candidates) != 1:
            raise ValueError('MMAction test did not emit one confusion matrix')
        confusion_path = candidates[0]
    report = checkpoint_evaluation_report(
        confusion=np.load(confusion_path, allow_pickle=False),
        dataset=identity_dataset or dataset,
        checkpoint=checkpoint,
        config=identity_config or source_config,
        evaluation_split=evaluation_split,
        preprocessing_fingerprint=preprocessing_fingerprint,
        test_log=log_path,
        confusion_path=confusion_path,
        executor_image=executor_image,
    )
    report_path = output_dir / 'checkpoint-evaluation.json'
    atomic_json(report_path, report)
    return report, report_path, resolved_config, command


def evaluation_rank(report: Mapping[str, Any]) -> Tuple[Any, ...]:
    metrics = report['metrics']
    none_far = float(metrics['offline.none_false_accept_rate'])
    return (
        float(metrics['offline.macro_f1']),
        float(metrics['offline.min_per_class_recall']),
        -none_far,
    )


def choose_trial_evaluation(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError('trial emitted no checkpoint evaluations')
    role_priority = {'retained': 0, 'last': 1, 'best': 2}
    return max(
        candidates,
        key=lambda item: (
            float(item['evaluation']['metrics']['offline.none_false_accept_rate'])
            <= 0.01,
            float(item['evaluation']['metrics']['offline.macro_f1']),
            float(item['evaluation']['metrics']['offline.min_per_class_recall']),
            float(item['evaluation']['metrics']['offline.accuracy']),
            -float(item['evaluation']['metrics']['offline.none_false_accept_rate']),
            role_priority[item['role']],
        ),
    )


def build_trial_report(
    *,
    trial_number: int,
    strategy: str,
    parameters: Mapping[str, Any],
    dataset: Path,
    base_checkpoint: Path,
    initialization_checkpoint: Path,
    source_config: Path,
    resolved_config: Path,
    training_log: Path,
    training_metrics: Path,
    best_checkpoint: Path,
    last_checkpoint: Path,
    evaluation_candidates: Sequence[Mapping[str, Any]],
    selected_report_path: Path,
    preprocessing_fingerprint: str,
    evaluation_split: str,
    executor_image: Optional[str],
) -> Dict[str, Any]:
    if trial_number < 1 or trial_number > MAX_HISTORY_TRIALS:
        raise ValueError(f'trial_number must be in [1, {MAX_HISTORY_TRIALS}]')
    strategy = validate_strategy(strategy)
    normalized = normalize_experiment_parameters(parameters)
    selected = choose_trial_evaluation(evaluation_candidates)
    selected_checkpoint = Path(str(selected['checkpoint'])).resolve(strict=True)
    selected_evaluation = selected['evaluation']

    dataset_id = file_identity(dataset)
    base_id = file_identity(base_checkpoint)
    init_id = file_identity(initialization_checkpoint)
    source_config_id = file_identity(source_config)
    resolved_config_id = file_identity(resolved_config)
    training_log_id = file_identity(training_log)
    training_metrics_id = file_identity(training_metrics)
    selected_checkpoint_id = file_identity(selected_checkpoint)
    selected_evaluation_id = file_identity(selected_report_path)
    best_id = file_identity(best_checkpoint)
    last_id = file_identity(last_checkpoint)
    input_record = {
        'dataset_sha256': dataset_id['sha256'],
        'dataset_size_bytes': dataset_id['size_bytes'],
        'base_model_sha256': base_id['sha256'],
        'base_model_size_bytes': base_id['size_bytes'],
        'initialization_checkpoint_sha256': init_id['sha256'],
        'initialization_checkpoint_size_bytes': init_id['size_bytes'],
        'source_config_sha256': source_config_id['sha256'],
        'source_config_size_bytes': source_config_id['size_bytes'],
        'evaluation_split': evaluation_split,
        'preprocessing_version': PREPROCESSING_VERSION,
        'preprocessing_fingerprint': preprocessing_fingerprint,
    }
    output_record = {
        'checkpoint_sha256': selected_checkpoint_id['sha256'],
        'checkpoint_size_bytes': selected_checkpoint_id['size_bytes'],
        'training_config_sha256': resolved_config_id['sha256'],
        'training_config_size_bytes': resolved_config_id['size_bytes'],
        'training_log_sha256': training_log_id['sha256'],
        'training_log_size_bytes': training_log_id['size_bytes'],
        'training_metrics_sha256': training_metrics_id['sha256'],
        'training_metrics_size_bytes': training_metrics_id['size_bytes'],
        'evaluation_report_sha256': selected_evaluation_id['sha256'],
        'evaluation_report_size_bytes': selected_evaluation_id['size_bytes'],
    }
    return {
        'schema_version': TRIAL_SCHEMA,
        'trial_number': trial_number,
        'trial_id': canonical_sha256(
            {
                'trial_number': trial_number,
                'strategy': strategy,
                'parameters': normalized,
                'input': input_record,
            }
        ),
        'strategy': strategy,
        'parameters': normalized,
        'input': input_record,
        'outputs': output_record,
        'training': {
            'best_checkpoint_sha256': best_id['sha256'],
            'best_checkpoint_size_bytes': best_id['size_bytes'],
            'last_checkpoint_sha256': last_id['sha256'],
            'last_checkpoint_size_bytes': last_id['size_bytes'],
            'executor_image': executor_image,
        },
        'evaluation': selected_evaluation,
        'evaluation_candidates': [
            {
                'role': item['role'],
                'checkpoint_sha256': item['evaluation']['identity'][
                    'checkpoint_sha256'
                ],
                'evaluation_report_sha256': item['report_sha256'],
                'metrics': item['evaluation']['metrics'],
            }
            for item in evaluation_candidates
        ],
        'selection': {
            'policy': SELECTION_POLICY,
            'winner_role': selected['role'],
            'reason_codes': [
                'none-far-feasibility-first',
                'macro-f1-primary',
                'min-recall-secondary',
                'accuracy-tertiary',
            ],
        },
    }


def _validate_evaluation_document(value: Mapping[str, Any]) -> None:
    if value.get('schema_version') != EVALUATION_SCHEMA:
        raise ValueError('history baseline evaluation schema is invalid')
    if value.get('labels') != list(LABELS):
        raise ValueError('history evaluation labels are invalid')
    confusion = value.get('confusion_matrix')
    if not isinstance(confusion, dict) or set(confusion) != set(LABELS):
        raise ValueError('history evaluation confusion matrix is incomplete')
    if any(
        not isinstance(confusion[label], dict) or set(confusion[label]) != set(LABELS)
        for label in LABELS
    ):
        raise ValueError('history evaluation confusion rows are incomplete')


def _trial_summary(report: Mapping[str, Any], report_sha256: str) -> Dict[str, Any]:
    outputs = report['outputs']
    return {
        'trial_number': report['trial_number'],
        'trial_report_sha256': report_sha256,
        'checkpoint_sha256': outputs['checkpoint_sha256'],
        'training_config_sha256': outputs['training_config_sha256'],
        'training_log_sha256': outputs['training_log_sha256'],
        'training_metrics_sha256': outputs['training_metrics_sha256'],
        'evaluation_report_sha256': outputs['evaluation_report_sha256'],
        'metrics': report['evaluation']['metrics'],
    }


def _history_path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f'{label} must be a non-empty path string')
    return Path(raw).resolve(strict=True)


def build_model_improvement_report(
    *,
    history: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Dict[str, Any]:
    allowed_history = {'baseline_evaluation', 'trial_reports', 'policy'}
    unknown_history = sorted(set(history) - allowed_history)
    if unknown_history:
        raise ValueError(f'history-json contains unsupported keys: {unknown_history}')
    baseline_path = _history_path(
        history.get('baseline_evaluation'),
        label='baseline_evaluation',
    )
    raw_trials = history.get('trial_reports')
    if (
        not isinstance(raw_trials, list)
        or not raw_trials
        or len(raw_trials) > MAX_HISTORY_TRIALS
    ):
        raise ValueError(f'trial_reports must contain 1..{MAX_HISTORY_TRIALS} paths')
    if not isinstance(analysis, Mapping):
        raise ValueError('analysis-json must be an object')
    analysis_text = _canonical_json(analysis)
    if len(analysis_text.encode('utf-8')) > 32 * 1024:
        raise ValueError('analysis-json exceeds 32 KiB')
    policy = normalize_policy(history.get('policy', {}))

    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if not isinstance(baseline, dict):
        raise ValueError('baseline evaluation must be a JSON object')
    _validate_evaluation_document(baseline)
    baseline_sha256, _ = sha256_file(baseline_path)
    baseline_identity = baseline['identity']

    trial_records = []
    seen_numbers = set()
    for index, raw_path in enumerate(raw_trials):
        trial_path = _history_path(raw_path, label=f'trial_reports[{index}]')
        trial = json.loads(trial_path.read_text(encoding='utf-8'))
        if not isinstance(trial, dict) or trial.get('schema_version') != TRIAL_SCHEMA:
            raise ValueError(f'trial report schema is invalid: {trial_path}')
        trial_number = trial.get('trial_number')
        if (
            isinstance(trial_number, bool)
            or not isinstance(trial_number, int)
            or trial_number < 1
            or trial_number > MAX_HISTORY_TRIALS
            or trial_number in seen_numbers
        ):
            raise ValueError('trial numbers must be unique integers in range')
        seen_numbers.add(trial_number)
        if trial['input']['dataset_sha256'] != baseline_identity['dataset_sha256']:
            raise ValueError('trial and baseline evaluation datasets differ')
        if trial['input']['evaluation_split'] != baseline_identity['evaluation_split']:
            raise ValueError('trial and baseline evaluation splits differ')
        _validate_evaluation_document(trial['evaluation'])
        trial_sha256, _ = sha256_file(trial_path)
        trial_records.append((trial, trial_sha256))
    trial_records.sort(key=lambda item: item[0]['trial_number'])
    if len(trial_records) > policy['max_trials']:
        raise ValueError('trial history exceeds policy max_trials')

    baseline_metrics = baseline['metrics']
    eligible_records = [
        item
        for item in trial_records
        if (
            float(item[0]['evaluation']['metrics']['offline.min_per_class_recall'])
            - float(baseline_metrics['offline.min_per_class_recall'])
            >= -0.02
            and float(
                item[0]['evaluation']['metrics']['offline.none_false_accept_rate']
            )
            <= 0.01
        )
    ]
    best_trial, best_trial_sha256 = max(
        eligible_records or trial_records,
        key=lambda item: (
            *evaluation_rank(item[0]['evaluation']),
            -int(item[0]['trial_number']),
        ),
    )
    best_summary = _trial_summary(best_trial, best_trial_sha256)
    best_metrics = best_trial['evaluation']['metrics']
    deltas = {
        'offline.accuracy_delta_vs_baseline': (
            float(best_metrics['offline.accuracy'])
            - float(baseline_metrics['offline.accuracy'])
        ),
        'offline.macro_f1_delta_vs_baseline': (
            float(best_metrics['offline.macro_f1'])
            - float(baseline_metrics['offline.macro_f1'])
        ),
        'offline.min_per_class_recall_delta_vs_baseline': (
            float(best_metrics['offline.min_per_class_recall'])
            - float(baseline_metrics['offline.min_per_class_recall'])
        ),
        'offline.none_false_accept_rate_delta_vs_baseline': (
            float(best_metrics['offline.none_false_accept_rate'])
            - float(baseline_metrics['offline.none_false_accept_rate'])
        ),
    }
    eligible = (
        any(record[0] is best_trial for record in eligible_records)
        and deltas['offline.macro_f1_delta_vs_baseline'] >= policy['min_improvement']
    )
    if eligible:
        decision = 'promote-challenger'
        winner = {
            'source': 'trial',
            **{key: value for key, value in best_summary.items() if key != 'metrics'},
        }
        reason_codes = [
            'macro-f1-minimum-improvement-reached',
            'minimum-recall-guard-passed',
            'none-far-guard-passed',
        ]
    else:
        decision = 'keep-champion'
        winner = {
            'source': 'baseline',
            'trial_number': None,
            'trial_report_sha256': None,
            'checkpoint_sha256': baseline_identity['checkpoint_sha256'],
            'training_config_sha256': None,
            'training_log_sha256': None,
            'training_metrics_sha256': None,
            'evaluation_report_sha256': baseline_sha256,
        }
        reason_codes = ['challenger-did-not-pass-product-selection-policy']
    if (
        eligible_records
        and deltas['offline.macro_f1_delta_vs_baseline']
        >= policy['target_macro_f1_delta']
    ):
        stop_reason = 'target-reached'
    elif len(trial_records) == policy['max_trials']:
        stop_reason = 'trial-budget-exhausted'
    elif (
        _trailing_no_improvement_count(
            trial_records,
            baseline_metrics=baseline_metrics,
            minimum=policy['min_improvement'],
        )
        >= policy['patience']
    ):
        stop_reason = 'patience-exhausted'
    else:
        raise ValueError(
            'history has not reached a deterministic optimization stop condition'
        )
    return {
        'schema_version': IMPROVEMENT_SCHEMA,
        'baseline_evaluation_sha256': baseline_sha256,
        'trial_report_sha256s': [report_sha256 for _, report_sha256 in trial_records],
        'decision': decision,
        'stop_reason': stop_reason,
        'policy': policy,
        'baseline_evaluation': baseline,
        'trials': [
            _trial_summary(trial, report_sha256)
            for trial, report_sha256 in trial_records
        ],
        'best_trial': best_summary,
        'winner': winner,
        'deltas_vs_baseline': deltas,
        'reason_codes': reason_codes,
        'analysis': {
            'sha256': canonical_sha256(analysis),
            'document': dict(analysis),
        },
    }


def _trailing_no_improvement_count(
    trial_records: Sequence[Tuple[Mapping[str, Any], str]],
    *,
    baseline_metrics: Mapping[str, Any],
    minimum: float,
) -> int:
    best = float(baseline_metrics['offline.macro_f1'])
    trailing = 0
    for trial, _ in trial_records:
        metrics = trial['evaluation']['metrics']
        guardrail_eligible = (
            float(metrics['offline.min_per_class_recall'])
            - float(baseline_metrics['offline.min_per_class_recall'])
            >= -0.02
            and float(metrics['offline.none_false_accept_rate']) <= 0.01
        )
        value = float(metrics['offline.macro_f1'])
        if guardrail_eligible and value >= best + minimum:
            best = value
            trailing = 0
        else:
            trailing += 1
    return trailing


# The no-delta-z preset keeps the five-channel tensor contract and only zeros
# delta-z. Registration is optional so importing this module for a dry run or
# unit test never requires torch/MMCV.
try:  # pragma: no cover - exercised only in the MMAction training image.
    from mmaction.registry import TRANSFORMS
    from mmcv.transforms import BaseTransform

    @TRANSFORMS.register_module()
    class MLOpsFormatGlobalTrajectory(BaseTransform):
        def __init__(
            self,
            num_traj_channels: int = 2,
            zero_channels: Sequence[int] = (),
        ) -> None:
            self.num_traj_channels = int(num_traj_channels)
            self.zero_channels = tuple(int(value) for value in zero_channels)
            if any(
                value < 0 or value >= self.num_traj_channels
                for value in self.zero_channels
            ):
                raise ValueError('zero_channels are outside trajectory channels')

        def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
            trajectory = np.asarray(results['global_traj'], dtype=np.float32)
            if trajectory.shape[-1] < self.num_traj_channels:
                raise ValueError('global trajectory has too few channels')
            trajectory = trajectory[..., : self.num_traj_channels].copy()
            for channel in self.zero_channels:
                trajectory[..., channel] = 0.0
            total_frames = trajectory.shape[0]
            num_clips = int(results.get('num_clips', 1))
            if total_frames % num_clips:
                raise ValueError('trajectory frames are not divisible by num_clips')
            results['global_traj'] = np.ascontiguousarray(
                trajectory.reshape(
                    num_clips,
                    total_frames // num_clips,
                    self.num_traj_channels,
                )
            )
            return results

except Exception:  # noqa: BLE001 - optional runtime registration.
    pass
