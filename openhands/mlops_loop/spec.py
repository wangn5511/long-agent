"""Load and resolve LoopSpec/RunSpec documents into immutable task specs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from openhands.mlops_loop.models import CompletionContract, MetricGate, TaskSpec

RIGHTARM_REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    'inspect-data': ['source-manifest', 'data-validation-report', 'raw-dataset'],
    'build-dataset': ['incremental-mmaction-dataset', 'dataset-report'],
    'review-dataset-distribution': [
        'dataset-distribution-report',
        'dataset-merge-plan',
        'dataset-merge-report',
        'mmaction-dataset',
    ],
    'train': [
        'checkpoint',
        'last-checkpoint',
        'training-config',
        'training-log',
        'training-metrics',
    ],
    'improve-model': [
        'baseline-evaluation-report',
        'checkpoint',
        'training-config',
        'training-log',
        'training-metrics',
        'evaluation-report',
        'optimization-trial-report',
        'model-improvement-report',
    ],
    'evaluate': ['evaluation-report'],
    'package': ['onnx-model', 'rknn-model', 'parity-report'],
    'verify-board': ['board-report', 'release-manifest'],
    'deploy': ['deployment-observation'],
    'monitor': ['deployment-observation'],
    'rollback': ['deployment-observation'],
}

RIGHTARM_RELEASE_REQUIRED_KINDS = frozenset(
    {
        'raw-dataset',
        'mmaction-dataset',
        'base-model',
        'code',
        'container-image',
        'training-config',
        'checkpoint',
        'evaluation-report',
        'onnx-model',
        'rknn-model',
        'label-mapping',
        'tensor-contract',
        'parity-report',
        'board-report',
    }
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'expected a YAML object in {path}')
    return value


def resolve_template(value: Any, replacements: dict[str, str]) -> Any:
    """Recursively resolve exact ``<name>`` placeholders and inline uses."""

    if isinstance(value, str):
        resolved = value
        for name, replacement in replacements.items():
            resolved = resolved.replace(f'<{name}>', replacement)
        return resolved
    if isinstance(value, list):
        return [resolve_template(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: resolve_template(item, replacements) for key, item in value.items()
        }
    return deepcopy(value)


def task_specs_from_documents(
    loop_spec: dict[str, Any], run_spec: dict[str, Any]
) -> list[TaskSpec]:
    run_gates = {
        gate['metric']: MetricGate.model_validate(gate)
        for gate in run_spec.get('spec', {}).get('evaluation', {}).get('gates', [])
    }
    tasks: list[TaskSpec] = []
    for raw in loop_spec['spec']['taskSequence']:
        kind = raw['kind']
        prefixes = {
            'evaluate': ('offline.',),
            'package': ('conversion.',),
            'verify-board': ('edge.',),
        }.get(kind, ())
        gates = [
            gate
            for metric, gate in run_gates.items()
            if any(metric.startswith(prefix) for prefix in prefixes)
        ]
        if kind == 'improve-model':
            max_trials = (
                run_spec.get('spec', {}).get('optimization', {}).get('maxTrials')
            )
            if (
                not isinstance(max_trials, int)
                or isinstance(max_trials, bool)
                or not 1 <= max_trials <= 8
            ):
                raise ValueError(
                    'improve-model requires optimization.maxTrials in [1, 8]'
                )
            gates.extend(
                [
                    MetricGate(
                        metric='optimization.trial_count',
                        op='gte',
                        value=1,
                    ),
                    MetricGate(
                        metric='optimization.trial_count',
                        op='lte',
                        value=max_trials,
                    ),
                ]
            )
        contract = CompletionContract(
            required_artifact_kinds=RIGHTARM_REQUIRED_ARTIFACTS.get(kind, []),
            required_evidence=True,
            gates=gates,
            allow_skip_training=kind == 'inspect-data',
        )
        objective = (
            'Compare the trusted baseline and incremental dataset using their '
            'deterministic distribution reports. Only a green review may execute '
            'the merge-all-unique policy; otherwise submit status=needs_decision. '
            'Produce the dataset distribution report, merge plan, merge report, '
            'and final MMAction dataset required by the completion contract.'
            if kind == 'review-dataset-distribution'
            else (
                'Evaluate the incumbent on the immutable validation split, then '
                'run a bounded sequence of independently measured training and '
                'evaluation trials. Diagnose each machine report before choosing '
                'the next allowlisted change, retain the best policy-compliant '
                'checkpoint, and stop at the target, patience, or trial budget. '
                'Produce the complete optimization history and every artifact '
                'required by the completion contract.'
                if kind == 'improve-model'
                else (
                    f'Complete the {kind} stage for immutable RunSpec '
                    f'{run_spec["metadata"]["name"]}; produce every artifact and '
                    'piece of evidence in the completion contract.'
                )
            )
        )
        tasks.append(
            TaskSpec(
                name=raw['name'],
                kind=kind,
                objective=objective,
                agent_profile=raw['agentProfile'],
                validator=raw['validator'],
                max_attempts=raw['maxAttempts'],
                timeout_seconds=raw['timeoutSeconds'],
                completion_contract=contract,
                requires_human_confirmation=raw.get('requiresHumanConfirmation', False),
            )
        )
    return tasks
