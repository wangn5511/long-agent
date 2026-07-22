"""Load and resolve LoopSpec/RunSpec documents into immutable task specs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from openhands.mlops_loop.models import CompletionContract, MetricGate, TaskSpec

RIGHTARM_REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    'inspect-data': ['source-manifest', 'data-validation-report', 'raw-dataset'],
    'build-dataset': ['mmaction-dataset', 'dataset-report'],
    'train': [
        'checkpoint',
        'last-checkpoint',
        'training-config',
        'training-log',
        'training-metrics',
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
        contract = CompletionContract(
            required_artifact_kinds=RIGHTARM_REQUIRED_ARTIFACTS.get(kind, []),
            required_evidence=True,
            gates=gates,
            allow_skip_training=kind == 'inspect-data',
        )
        tasks.append(
            TaskSpec(
                name=raw['name'],
                kind=kind,
                objective=(
                    f'Complete the {kind} stage for immutable RunSpec '
                    f'{run_spec["metadata"]["name"]}; produce every artifact and '
                    'piece of evidence in the completion contract.'
                ),
                agent_profile=raw['agentProfile'],
                validator=raw['validator'],
                max_attempts=raw['maxAttempts'],
                timeout_seconds=raw['timeoutSeconds'],
                completion_contract=contract,
                requires_human_confirmation=raw.get('requiresHumanConfirmation', False),
            )
        )
    return tasks
