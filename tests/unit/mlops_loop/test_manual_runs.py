import pytest

from openhands.mlops_loop.manual_runs import (
    IMPROVEMENT_PARAMETER_DEFAULTS,
    MODE_REQUIRED_KINDS,
    MODE_TASK_NAMES,
    ManualRunMode,
    build_manual_run,
    validate_cos_source_uri,
    validate_improvement_parameters,
)


@pytest.mark.parametrize('mode', list(ManualRunMode))
def test_manual_modes_select_complete_task_subsequences(mode: ManualRunMode):
    artifacts = [
        {'id': f'{kind}:sha256:' + 'a' * 64, 'kind': kind}
        for kind in MODE_REQUIRED_KINDS[mode]
    ]
    source_uri = (
        'cos://gesture-bucket/batch-001/'
        if mode in {ManualRunMode.DATA, ManualRunMode.FULL}
        else None
    )

    spec, tasks = build_manual_run(
        mode,
        request_id='manual-test',
        source_uri=source_uri,
        artifacts=artifacts,
    )

    assert [task.name for task in tasks] == list(MODE_TASK_NAMES[mode])
    assert spec['metadata']['labels']['mode'] == mode
    assert spec['spec']['manualTrigger']['requestId'] == 'manual-test'
    assert '{{' not in str(spec)


def test_improve_mode_uses_an_independent_bounded_task_contract():
    artifacts = [
        {
            'id': 'mmaction-dataset:sha256:' + 'a' * 64,
            'kind': 'mmaction-dataset',
        },
        {'id': 'base-model:sha256:' + 'b' * 64, 'kind': 'base-model'},
        {
            'id': 'training-init-model:sha256:' + 'c' * 64,
            'kind': 'training-init-model',
        },
    ]
    parameters = {
        'max_trials': 6,
        'patience': 3,
        'target_macro_f1_delta': 0.01,
        'min_improvement': 0.001,
        'max_epochs': 80,
        'batch_size': 16,
        'workers': 4,
        'gpus': 2,
    }

    spec, tasks = build_manual_run(
        ManualRunMode.IMPROVE,
        request_id='improve-test',
        source_uri=None,
        artifacts=artifacts,
        parameters=parameters,
    )

    assert [task.name for task in tasks] == ['improve-challenger']
    task = tasks[0]
    assert task.kind == 'improve-model'
    assert task.validator == 'rightarm-model-improvement-v1'
    assert [gate.model_dump() for gate in task.completion_contract.gates] == [
        {
            'metric': 'optimization.trial_count',
            'op': 'gte',
            'value': 1,
        },
        {
            'metric': 'optimization.trial_count',
            'op': 'lte',
            'value': 6,
        },
    ]
    assert task.completion_contract.required_artifact_kinds == [
        'baseline-evaluation-report',
        'checkpoint',
        'training-config',
        'training-log',
        'training-metrics',
        'evaluation-report',
        'optimization-trial-report',
        'model-improvement-report',
    ]
    assert spec['metadata']['name'] == 'rightarm-manual-improve'
    assert spec['spec']['manualTrigger']['parameters'] == parameters
    assert spec['spec']['optimization'] == {
        'policyVersion': 'rightarm.gesture.model-improvement-policy/v1',
        'trialJobKind': 'train-eval-trial',
        'maxTrials': 6,
        'patience': 3,
        'targetMacroF1Delta': 0.01,
        'minImprovement': 0.001,
        'maxEpochs': 80,
        'batchSize': 16,
        'workers': 4,
        'gpus': 2,
        'allowedChangeFamilies': [
            'fine-tune',
            'optimizer',
            'regularization',
            'augmentation',
            'motion-ablation',
            'gesture-gate',
        ],
        'stopReasons': [
            'target-reached',
            'patience-exhausted',
            'trial-budget-exhausted',
        ],
    }
    assert spec['spec']['training']['resources']['gpu'] == 2
    assert (
        spec['spec']['training']['environment']['MLOPS_BASE_MODEL_ARTIFACT']
        == artifacts[1]['id']
    )
    assert (
        spec['spec']['training']['environment']['MLOPS_TRAINING_INIT_MODEL_ARTIFACT']
        == artifacts[2]['id']
    )


def test_improve_mode_populates_default_search_policy():
    artifacts = [
        {'id': f'{kind}:sha256:' + 'a' * 64, 'kind': kind}
        for kind in MODE_REQUIRED_KINDS[ManualRunMode.IMPROVE]
    ]

    spec, _ = build_manual_run(
        ManualRunMode.IMPROVE,
        request_id='improve-defaults',
        source_uri=None,
        artifacts=artifacts,
    )

    assert spec['spec']['manualTrigger']['parameters'] == IMPROVEMENT_PARAMETER_DEFAULTS


@pytest.mark.parametrize(
    ('parameters', 'message'),
    [
        ({'max_trials': 9}, 'max_trials'),
        ({'max_trials': 2, 'patience': 3}, 'patience'),
        ({'target_macro_f1_delta': float('nan')}, 'target_macro_f1_delta'),
        ({'target_macro_f1_delta': 1.0001}, 'target_macro_f1_delta'),
        ({'min_improvement': -0.1}, 'min_improvement'),
        ({'max_epochs': 0}, 'max_epochs'),
        ({'max_epochs': 101}, 'max_epochs'),
        ({'batch_size': 12}, 'batch_size'),
        ({'workers': 33}, 'workers'),
        ({'gpus': 9}, 'gpus'),
        ({'arbitrary_cfg_options': 'unsafe'}, 'unsupported parameters'),
    ],
)
def test_improve_mode_revalidates_api_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        validate_improvement_parameters(parameters)


def test_existing_manual_mode_task_lists_are_unchanged():
    assert MODE_TASK_NAMES[ManualRunMode.DATA] == (
        'inspect-data',
        'build-dataset',
        'review-dataset-distribution',
    )
    assert MODE_TASK_NAMES[ManualRunMode.TRAIN] == (
        'train-challenger',
        'evaluate-challenger',
        'package-rknn',
    )
    assert MODE_TASK_NAMES[ManualRunMode.FULL] == (
        'inspect-data',
        'build-dataset',
        'review-dataset-distribution',
        'train-challenger',
        'evaluate-challenger',
        'package-rknn',
    )
    assert MODE_TASK_NAMES[ManualRunMode.EXPORT] == (
        'evaluate-challenger',
        'package-rknn',
    )


@pytest.mark.parametrize(
    'mode',
    [ManualRunMode.DATA, ManualRunMode.FULL],
)
def test_data_modes_review_distribution_before_training(mode: ManualRunMode):
    assert 'baseline-mmaction-dataset' in MODE_REQUIRED_KINDS[mode]
    artifacts = [
        {'id': f'{kind}:sha256:' + 'a' * 64, 'kind': kind}
        for kind in MODE_REQUIRED_KINDS[mode]
    ]

    spec, tasks = build_manual_run(
        mode,
        request_id='distribution-review-test',
        source_uri='cos://gesture-bucket/batch-001/',
        artifacts=artifacts,
    )

    assert 'trusted baseline' in spec['spec']['objective']
    names = [task.name for task in tasks]
    assert names.index('build-dataset') < names.index('review-dataset-distribution')
    if mode is ManualRunMode.FULL:
        assert names.index('review-dataset-distribution') < names.index(
            'train-challenger'
        )
    review = next(task for task in tasks if task.name == 'review-dataset-distribution')
    assert review.kind == 'review-dataset-distribution'
    assert review.validator == 'rightarm-dataset-distribution-review-v1'
    assert 'trusted baseline and incremental dataset' in review.objective
    assert 'status=needs_decision' in review.objective
    assert review.completion_contract.required_artifact_kinds == [
        'dataset-distribution-report',
        'dataset-merge-plan',
        'dataset-merge-report',
        'mmaction-dataset',
    ]


def test_manual_mode_rejects_missing_immutable_input():
    with pytest.raises(ValueError, match='base-model'):
        build_manual_run(
            ManualRunMode.TRAIN,
            request_id='missing-base',
            source_uri=None,
            artifacts=[{'id': 'dataset-id', 'kind': 'mmaction-dataset'}],
        )


@pytest.mark.parametrize(
    'uri',
    [
        'https://bucket.example/batch/',
        'cos://bucket/',
        'cos:///batch/',
        'cos://bucket/batch/?version=1',
    ],
)
def test_cos_source_uri_is_strict(uri: str):
    with pytest.raises(ValueError, match='cos://'):
        validate_cos_source_uri(uri)
