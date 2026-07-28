import json
from pathlib import Path

import pytest
from rightarm_mlops.model_agent_cli import (
    _improvement_parameters,
    _input_plan,
    _parser,
    _trigger,
)


@pytest.mark.parametrize('mode', ('data', 'full'))
def test_data_modes_plan_a_distinct_baseline_dataset_input(
    tmp_path: Path, mode: str
) -> None:
    baseline = tmp_path / 'baseline.pkl'
    detector = tmp_path / 'detector.onnx'
    person = tmp_path / 'person.onnx'
    hand = tmp_path / 'hand.onnx'
    argv = [
        'trigger',
        mode,
        '--cos-uri',
        'cos://gesture-bucket/batch-001/',
        '--baseline-pkl',
        str(baseline),
        '--det-onnx',
        str(detector),
        '--person-pose-onnx',
        str(person),
        '--hand-pose-onnx',
        str(hand),
    ]
    if mode == 'full':
        argv.extend(['--base-model', str(tmp_path / 'base.pth')])

    plan = _input_plan(_parser().parse_args(argv))

    by_kind = {item['kind']: item for item in plan}
    assert by_kind['baseline-mmaction-dataset']['path'] == str(baseline)
    assert 'mmaction-dataset' not in by_kind


def test_data_mode_uses_configured_baseline_when_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / 'configured-baseline.pkl'
    monkeypatch.setenv('MLOPS_DEFAULT_DATASET_PATH', str(baseline))
    monkeypatch.delenv('MLOPS_BASELINE_MMACTION_DATASET_ARTIFACT', raising=False)
    for name, value in (
        ('MLOPS_RTM_DETECTOR_PATH', tmp_path / 'detector.onnx'),
        ('MLOPS_RTM_PERSON_POSE_PATH', tmp_path / 'person.onnx'),
        ('MLOPS_RTM_HAND_POSE_PATH', tmp_path / 'hand.onnx'),
    ):
        monkeypatch.setenv(name, str(value))
        monkeypatch.delenv(name.replace('_PATH', '_ARTIFACT'), raising=False)

    args = _parser().parse_args(
        [
            'trigger',
            'data',
            '--cos-uri',
            'cos://gesture-bucket/batch-001/',
        ]
    )

    by_kind = {item['kind']: item for item in _input_plan(args)}
    assert by_kind['baseline-mmaction-dataset']['path'] == str(baseline)


@pytest.mark.parametrize('incumbent_option', ('--checkpoint', '--base-model'))
def test_improve_mode_plans_incumbent_and_optional_initialization(
    tmp_path: Path, incumbent_option: str
) -> None:
    dataset = tmp_path / 'dataset.pkl'
    incumbent = tmp_path / 'incumbent.pth'
    initialization = tmp_path / 'pretrained.pth'
    args = _parser().parse_args(
        [
            'trigger',
            'improve',
            '--pkl',
            str(dataset),
            incumbent_option,
            str(incumbent),
            '--init-checkpoint',
            str(initialization),
            '--max-trials',
            '6',
            '--patience',
            '3',
            '--target-macro-f1-delta',
            '0.01',
            '--min-improvement',
            '0.001',
            '--max-epochs',
            '80',
            '--batch-size',
            '16',
            '--workers',
            '4',
            '--gpus',
            '2',
        ]
    )

    by_kind = {item['kind']: item for item in _input_plan(args)}
    assert by_kind == {
        'mmaction-dataset': {
            'kind': 'mmaction-dataset',
            'path': str(dataset),
        },
        'training-init-model': {
            'kind': 'training-init-model',
            'path': str(initialization),
        },
        'base-model': {'kind': 'base-model', 'path': str(incumbent)},
    }
    assert _improvement_parameters(args) == {
        'max_trials': 6,
        'patience': 3,
        'target_macro_f1_delta': 0.01,
        'min_improvement': 0.001,
        'max_epochs': 80,
        'batch_size': 16,
        'workers': 4,
        'gpus': 2,
    }


def test_improve_mode_uses_configured_inputs_and_gpu_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / 'configured.pkl'
    incumbent = tmp_path / 'configured.pth'
    monkeypatch.setenv('MLOPS_DEFAULT_DATASET_PATH', str(dataset))
    monkeypatch.delenv('MLOPS_DEFAULT_DATASET_ARTIFACT', raising=False)
    monkeypatch.setenv('MLOPS_BASE_MODEL_PATH', str(incumbent))
    monkeypatch.delenv('MLOPS_CHAMPION_BASE_MODEL_ARTIFACT', raising=False)
    monkeypatch.setenv('MLOPS_TRAIN_GPU_COUNT', '3')

    args = _parser().parse_args(['trigger', 'improve'])

    by_kind = {item['kind']: item for item in _input_plan(args)}
    assert by_kind['mmaction-dataset']['path'] == str(dataset)
    assert by_kind['base-model']['path'] == str(incumbent)
    assert _improvement_parameters(args)['gpus'] == 3


@pytest.mark.parametrize(
    'arguments',
    [
        ['--max-trials', '9'],
        ['--target-macro-f1-delta', 'nan'],
        ['--target-macro-f1-delta', '1.0001'],
        ['--min-improvement', '-0.1'],
        ['--max-epochs', '0'],
        ['--gpus', '0'],
    ],
)
def test_improve_mode_rejects_out_of_bounds_cli_values(
    tmp_path: Path, arguments: list[str]
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                'trigger',
                'improve',
                '--pkl',
                str(tmp_path / 'dataset.pkl'),
                '--checkpoint',
                str(tmp_path / 'incumbent.pth'),
                *arguments,
            ]
        )


def test_improve_mode_rejects_patience_above_trial_budget(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            'trigger',
            'improve',
            '--pkl',
            str(tmp_path / 'dataset.pkl'),
            '--checkpoint',
            str(tmp_path / 'incumbent.pth'),
            '--max-trials',
            '2',
            '--patience',
            '3',
        ]
    )

    with pytest.raises(ValueError, match='patience'):
        _input_plan(args)


def test_improve_dry_run_includes_the_normalized_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / 'dataset.pkl'
    incumbent = tmp_path / 'incumbent.pth'
    dataset.write_bytes(b'dataset')
    incumbent.write_bytes(b'checkpoint')
    args = _parser().parse_args(
        [
            'trigger',
            'improve',
            '--pkl',
            str(dataset),
            '--checkpoint',
            str(incumbent),
            '--request-id',
            'improve-dry-run',
            '--gpus',
            '1',
            '--dry-run',
        ]
    )

    assert _trigger(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['mode'] == 'improve'
    assert output['parameters'] == {
        'triggered_by': output['parameters']['triggered_by'],
        'max_trials': 4,
        'patience': 2,
        'target_macro_f1_delta': 0.002,
        'min_improvement': 0.0005,
        'max_epochs': 50,
        'batch_size': 32,
        'workers': 12,
        'gpus': 1,
    }
    assert output['will_dispatch'] is False
