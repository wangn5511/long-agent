"""Operator-owned manual entry points for the rightArm MLOps loop."""

from __future__ import annotations

import math
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openhands.mlops_loop.models import TaskSpec
from openhands.mlops_loop.spec import load_yaml, task_specs_from_documents


class ManualRunMode(StrEnum):
    DATA = 'data'
    TRAIN = 'train'
    FULL = 'full'
    EXPORT = 'export'
    IMPROVE = 'improve'


MODE_TASK_NAMES: dict[ManualRunMode, tuple[str, ...]] = {
    ManualRunMode.DATA: (
        'inspect-data',
        'build-dataset',
        'review-dataset-distribution',
    ),
    ManualRunMode.TRAIN: (
        'train-challenger',
        'evaluate-challenger',
        'package-rknn',
    ),
    ManualRunMode.FULL: (
        'inspect-data',
        'build-dataset',
        'review-dataset-distribution',
        'train-challenger',
        'evaluate-challenger',
        'package-rknn',
    ),
    ManualRunMode.EXPORT: ('evaluate-challenger', 'package-rknn'),
    ManualRunMode.IMPROVE: ('improve-challenger',),
}


MODE_REQUIRED_KINDS: dict[ManualRunMode, frozenset[str]] = {
    ManualRunMode.DATA: frozenset(
        {
            'rtm-detector',
            'rtm-person-pose',
            'rtm-hand-pose',
            'baseline-mmaction-dataset',
        }
    ),
    ManualRunMode.TRAIN: frozenset({'mmaction-dataset', 'base-model'}),
    ManualRunMode.FULL: frozenset(
        {
            'rtm-detector',
            'rtm-person-pose',
            'rtm-hand-pose',
            'base-model',
            'baseline-mmaction-dataset',
        }
    ),
    ManualRunMode.EXPORT: frozenset({'checkpoint', 'mmaction-dataset'}),
    ManualRunMode.IMPROVE: frozenset({'mmaction-dataset', 'base-model'}),
}


MODE_OBJECTIVES: dict[ManualRunMode, str] = {
    ManualRunMode.DATA: (
        'Materialize one sealed Tencent COS batch, validate and safely land it, '
        'build an incremental right-arm 11-class dataset, independently review '
        'its distribution against the trusted baseline, then merge it into the '
        'traceable training dataset artifact.'
    ),
    ManualRunMode.TRAIN: (
        'Train from the supplied immutable PKL and base model, evaluate the '
        'challenger, then export and verify ONNX/RKNN artifacts.'
    ),
    ManualRunMode.FULL: (
        'Materialize one sealed Tencent COS batch, build an incremental dataset, '
        'independently review and merge it with the trusted baseline, then execute '
        'training, evaluation and ONNX/RKNN packaging as one audited Run.'
    ),
    ManualRunMode.EXPORT: (
        'Evaluate the supplied checkpoint on the immutable dataset, export '
        'ONNX/RKNN artifacts and verify conversion parity.'
    ),
    ManualRunMode.IMPROVE: (
        'Evaluate the supplied incumbent model, run a bounded sequence of '
        'independently measured training trials, diagnose validation errors, '
        'and retain the best policy-compliant right-arm gesture checkpoint.'
    ),
}

IMPROVEMENT_PARAMETER_DEFAULTS: dict[str, int | float] = {
    'max_trials': 4,
    'patience': 2,
    'target_macro_f1_delta': 0.002,
    'min_improvement': 0.0005,
    'max_epochs': 50,
    'batch_size': 32,
    'workers': 12,
    'gpus': 1,
}

_IMPROVEMENT_PASSTHROUGH_PARAMETERS = frozenset(
    {'triggered_by', 'training_image', 'evaluation_image'}
)


def validate_cos_source_uri(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != 'cos'
        or not parsed.netloc
        or not parsed.path.strip('/')
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError('source URI must be cos://<bucket>/<non-empty-prefix>/')
    return f'cos://{parsed.netloc}/{parsed.path.strip("/")}/'


def validate_manual_inputs(
    mode: ManualRunMode,
    *,
    source_uri: str | None,
    artifacts: list[dict[str, Any]],
) -> str | None:
    if mode in {ManualRunMode.DATA, ManualRunMode.FULL}:
        if source_uri is None:
            raise ValueError(f'{mode} mode requires a Tencent COS source URI')
        source_uri = validate_cos_source_uri(source_uri)
    elif source_uri is not None:
        raise ValueError(f'{mode} mode does not accept a COS source URI')

    kinds = {str(artifact['kind']) for artifact in artifacts}
    missing = sorted(MODE_REQUIRED_KINDS[mode] - kinds)
    if missing:
        raise ValueError(
            f'{mode} mode is missing immutable input artifact kinds: {missing}'
        )
    if mode is ManualRunMode.IMPROVE:
        counts: dict[str, int] = {}
        for artifact in artifacts:
            kind = str(artifact['kind'])
            counts[kind] = counts.get(kind, 0) + 1
        ambiguous = sorted(
            kind
            for kind in (*MODE_REQUIRED_KINDS[mode], 'training-init-model')
            if counts.get(kind, 0) > 1
        )
        if ambiguous:
            raise ValueError(
                'improve mode requires at most one immutable artifact for each '
                f'model role: {ambiguous}'
            )
    return source_uri


def validate_improvement_parameters(
    parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize the bounded search policy at the controller API boundary."""

    supplied = dict(parameters or {})
    unknown = sorted(
        set(supplied)
        - set(IMPROVEMENT_PARAMETER_DEFAULTS)
        - _IMPROVEMENT_PASSTHROUGH_PARAMETERS
    )
    if unknown:
        raise ValueError(f'improve mode has unsupported parameters: {unknown}')

    normalized = {
        **supplied,
        **{
            key: supplied.get(key, default)
            for key, default in IMPROVEMENT_PARAMETER_DEFAULTS.items()
        },
    }
    integer_ranges = {
        'max_trials': (1, 8),
        'patience': (1, 8),
        'max_epochs': (1, 100),
        'workers': (0, 32),
        'gpus': (1, 8),
    }
    for name, (minimum, maximum) in integer_ranges.items():
        value = normalized[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f'improve parameter {name} must be an integer in [{minimum}, {maximum}]'
            )
    if normalized['batch_size'] not in {4, 8, 16, 32, 64}:
        raise ValueError(
            'improve parameter batch_size must be one of [4, 8, 16, 32, 64]'
        )
    if normalized['patience'] > normalized['max_trials']:
        raise ValueError('improve parameter patience must not exceed max_trials')

    for name, upper_bound in (
        ('target_macro_f1_delta', 1.0),
        ('min_improvement', 0.05),
    ):
        numeric_value = normalized[name]
        if (
            not isinstance(numeric_value, (int, float))
            or isinstance(numeric_value, bool)
            or not math.isfinite(float(numeric_value))
            or not 0.0 <= float(numeric_value) <= upper_bound
        ):
            raise ValueError(
                f'improve parameter {name} must be a finite number in [0, {upper_bound}]'
            )
        normalized[name] = float(numeric_value)
    return normalized


def _optimization_overrides(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        'maxTrials': parameters['max_trials'],
        'patience': parameters['patience'],
        'targetMacroF1Delta': parameters['target_macro_f1_delta'],
        'minImprovement': parameters['min_improvement'],
        'maxEpochs': parameters['max_epochs'],
        'batchSize': parameters['batch_size'],
        'workers': parameters['workers'],
        'gpus': parameters['gpus'],
    }


def build_manual_run(
    mode: ManualRunMode,
    *,
    request_id: str,
    source_uri: str | None,
    artifacts: list[dict[str, Any]],
    parameters: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[TaskSpec]]:
    parameters = parameters or {}
    if mode is ManualRunMode.IMPROVE:
        parameters = validate_improvement_parameters(parameters)
    source_uri = validate_manual_inputs(
        mode, source_uri=source_uri, artifacts=artifacts
    )
    spec_root = Path(__file__).with_name('specs')
    if mode is ManualRunMode.IMPROVE:
        loop_spec = load_yaml(spec_root / 'right-arm-gesture-improve-v1-loop.yaml')
        canonical_run = load_yaml(spec_root / 'right-arm-gesture-improve-v1-run.yaml')
        canonical_run['spec']['optimization'].update(
            _optimization_overrides(parameters)
        )
    else:
        loop_spec = load_yaml(spec_root / 'right-arm-gesture-loop-v1-loop.yaml')
        canonical_run = load_yaml(spec_root / 'right-arm-gesture-loop-v1-run.yaml')
    selected_names = set(MODE_TASK_NAMES[mode])
    selected_loop = deepcopy(loop_spec)
    selected_loop['spec']['taskSequence'] = [
        item
        for item in loop_spec['spec']['taskSequence']
        if item['name'] in selected_names
    ]
    tasks = task_specs_from_documents(selected_loop, canonical_run)
    if [task.name for task in tasks] != list(MODE_TASK_NAMES[mode]):
        raise RuntimeError(f'manual task template is incomplete for mode {mode}')

    artifact_ids = [str(artifact['id']) for artifact in artifacts]
    artifact_ids_by_kind: dict[str, list[str]] = {}
    for artifact in artifacts:
        artifact_ids_by_kind.setdefault(str(artifact['kind']), []).append(
            str(artifact['id'])
        )
    run_spec: dict[str, Any] = {
        'apiVersion': 'mlops.openhands.dev/v1',
        'kind': 'MLOpsRun',
        'metadata': {
            'name': f'rightarm-manual-{mode}',
            'labels': {
                'loop': 'rightarm-manual-v1',
                'mode': str(mode),
                'target': 'rk3588',
            },
        },
        'spec': {
            'objective': MODE_OBJECTIVES[mode],
            'inputArtifactIds': artifact_ids,
            'inputArtifactsByKind': artifact_ids_by_kind,
            'manualTrigger': {
                'requestId': request_id,
                'mode': str(mode),
                'parameters': parameters,
            },
            'evaluation': deepcopy(canonical_run['spec']['evaluation']),
            'policy': deepcopy(canonical_run['spec']['policy']),
        },
    }
    if source_uri is not None:
        run_spec['spec']['source'] = {
            'type': 'cos',
            'uri': source_uri,
            'credentialsRef': 'rightarm-cos-reader',
            'completionMarker': '_SUCCESS',
            'manifestName': 'batch.manifest.json',
        }
    run_spec['spec']['evaluation']['image'] = parameters.get(
        'evaluation_image', 'model-agent-runtime:configured-evaluation'
    )
    if mode in {
        ManualRunMode.TRAIN,
        ManualRunMode.FULL,
        ManualRunMode.IMPROVE,
    }:
        run_spec['spec']['training'] = deepcopy(canonical_run['spec']['training'])
        run_spec['spec']['training']['repository'] = '/workspace/rightArmPerception'
        run_spec['spec']['training']['image'] = parameters.get(
            'training_image', 'model-agent-runtime:configured-training'
        )
        run_spec['spec']['training']['environment']['MLOPS_BASE_MODEL_ARTIFACT'] = (
            artifact_ids_by_kind['base-model'][0]
        )
    if mode is ManualRunMode.IMPROVE:
        optimization = deepcopy(canonical_run['spec']['optimization'])
        run_spec['spec']['optimization'] = optimization
        run_spec['spec']['training']['resources']['gpu'] = parameters['gpus']
        run_spec['spec']['training']['parameters'] = {
            'maxEpochs': parameters['max_epochs'],
            'batchSize': parameters['batch_size'],
            'workers': parameters['workers'],
            'gpus': parameters['gpus'],
        }
        init_models = artifact_ids_by_kind.get('training-init-model', [])
        if init_models:
            run_spec['spec']['training']['environment'][
                'MLOPS_TRAINING_INIT_MODEL_ARTIFACT'
            ] = init_models[0]
    if mode in {ManualRunMode.DATA, ManualRunMode.FULL}:
        run_spec['spec']['data'] = deepcopy(canonical_run['spec']['data'])
    return run_spec, tasks
