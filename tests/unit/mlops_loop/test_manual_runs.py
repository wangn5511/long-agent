import pytest

from openhands.mlops_loop.manual_runs import (
    MODE_REQUIRED_KINDS,
    MODE_TASK_NAMES,
    ManualRunMode,
    build_manual_run,
    validate_cos_source_uri,
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
