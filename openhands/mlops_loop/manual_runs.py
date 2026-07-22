"""Operator-owned manual entry points for the rightArm MLOps loop."""

from __future__ import annotations

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


MODE_TASK_NAMES: dict[ManualRunMode, tuple[str, ...]] = {
    ManualRunMode.DATA: ('inspect-data', 'build-dataset'),
    ManualRunMode.TRAIN: (
        'train-challenger',
        'evaluate-challenger',
        'package-rknn',
    ),
    ManualRunMode.FULL: (
        'inspect-data',
        'build-dataset',
        'train-challenger',
        'evaluate-challenger',
        'package-rknn',
    ),
    ManualRunMode.EXPORT: ('evaluate-challenger', 'package-rknn'),
}


MODE_REQUIRED_KINDS: dict[ManualRunMode, frozenset[str]] = {
    ManualRunMode.DATA: frozenset({'rtm-detector', 'rtm-person-pose', 'rtm-hand-pose'}),
    ManualRunMode.TRAIN: frozenset({'mmaction-dataset', 'base-model'}),
    ManualRunMode.FULL: frozenset(
        {
            'rtm-detector',
            'rtm-person-pose',
            'rtm-hand-pose',
            'base-model',
        }
    ),
    ManualRunMode.EXPORT: frozenset({'checkpoint', 'mmaction-dataset'}),
}


MODE_OBJECTIVES: dict[ManualRunMode, str] = {
    ManualRunMode.DATA: (
        'Materialize one sealed Tencent COS batch, validate and safely land it, '
        'then build the traceable right-arm 11-class dataset artifact.'
    ),
    ManualRunMode.TRAIN: (
        'Train from the supplied immutable PKL and base model, evaluate the '
        'challenger, then export and verify ONNX/RKNN artifacts.'
    ),
    ManualRunMode.FULL: (
        'Materialize one sealed Tencent COS batch and execute data preparation, '
        'training, evaluation and ONNX/RKNN packaging as one audited Run.'
    ),
    ManualRunMode.EXPORT: (
        'Evaluate the supplied checkpoint on the immutable dataset, export '
        'ONNX/RKNN artifacts and verify conversion parity.'
    ),
}


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
    return source_uri


def build_manual_run(
    mode: ManualRunMode,
    *,
    request_id: str,
    source_uri: str | None,
    artifacts: list[dict[str, Any]],
    parameters: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[TaskSpec]]:
    parameters = parameters or {}
    source_uri = validate_manual_inputs(
        mode, source_uri=source_uri, artifacts=artifacts
    )
    spec_root = Path(__file__).with_name('specs')
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
    if mode in {ManualRunMode.TRAIN, ManualRunMode.FULL}:
        run_spec['spec']['training'] = deepcopy(canonical_run['spec']['training'])
        run_spec['spec']['training']['repository'] = '/workspace/rightArmPerception'
        run_spec['spec']['training']['image'] = parameters.get(
            'training_image', 'model-agent-runtime:configured-training'
        )
        run_spec['spec']['training']['environment']['MLOPS_BASE_MODEL_ARTIFACT'] = (
            artifact_ids_by_kind['base-model'][0]
        )
    if mode in {ManualRunMode.DATA, ManualRunMode.FULL}:
        run_spec['spec']['data'] = deepcopy(canonical_run['spec']['data'])
    return run_spec, tasks
