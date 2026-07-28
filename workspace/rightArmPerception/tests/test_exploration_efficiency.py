import pytest
from rightarm_mlops.exploration_efficiency import (
    build_exploration_efficiency_report,
)


def _trial_job(
    number: int,
    *,
    started_at: float,
    finished_at: float,
    duration: float,
    status: str = 'succeeded',
) -> dict:
    report_sha = f'{number:064x}'
    result = {
        'job_kind': 'train-eval-trial',
        'status': status,
        'outputs': [
            {
                'kind': 'optimization-trial-report',
                'path': f'/output/trial-{number}.json',
                'sha256': report_sha,
            }
        ],
        'runtime': {
            'job_started_at': f'1970-01-01T00:{int(started_at / 60):02d}:00Z',
            'job_finished_at': f'1970-01-01T00:{int(finished_at / 60):02d}:00Z',
            'job_duration_seconds': duration,
            'gpus': 8,
        },
    }
    return {
        'id': f'job-{number}',
        'task_attempt_id': 'attempt-accepted',
        'status': status,
        'result': result,
        'failure': None,
        'argv': ['train-eval-trial'],
        'created_at': started_at,
        'updated_at': finished_at,
    }


def _audit(trials: list[dict]) -> dict:
    trial_candidates = [
        {
            'candidate_id': f'candidate-{number}',
            'kind': 'optimization-trial-report',
            'sha256': trial['result']['outputs'][0]['sha256'],
        }
        for number, trial in enumerate(trials, start=1)
    ]
    return {
        'protocol': 'mlops.openhands.dev/run-audit/v1',
        'run': {
            'id': 'run-adaptive',
            'state': 'COMPLETED',
            'created_at': 100.0,
            'updated_at': 1900.0,
            'spec': {
                'spec': {
                    'manualTrigger': {'mode': 'improve'},
                    'optimization': {
                        'maxTrials': 4,
                        'patience': 2,
                        'targetDeltaMacroF1': 0.002,
                    },
                }
            },
        },
        'tasks': [
            {
                'id': 'task-improve',
                'spec': {'kind': 'improve-model'},
            }
        ],
        'attempts': [
            {
                'id': 'attempt-accepted',
                'task_id': 'task-improve',
                'status': 'ACCEPTED',
                'finished_at': 1900.0,
                'result': {
                    'status': 'succeeded',
                    'artifact_candidates': trial_candidates,
                },
                'validator_result': {
                    'disposition': 'accepted',
                    'accepted_candidate_ids': [
                        item['candidate_id'] for item in trial_candidates
                    ],
                },
            }
        ],
        'job_attempts': [
            {
                'id': 'baseline',
                'task_attempt_id': 'attempt-accepted',
                'status': 'succeeded',
                'result': {
                    'job_kind': 'benchmark-checkpoint',
                    'status': 'succeeded',
                },
                'created_at': 100.0,
                'updated_at': 160.0,
            },
            *trials,
            {
                'id': 'selection',
                'task_attempt_id': 'attempt-accepted',
                'status': 'succeeded',
                'result': {
                    'job_kind': 'record-model-improvement',
                    'status': 'succeeded',
                },
                'created_at': 1840.0,
                'updated_at': 1900.0,
            },
        ],
    }


def test_agent_selected_four_trials_compare_with_paired_human_baseline() -> None:
    trials = [
        _trial_job(
            number,
            started_at=number * 300.0,
            finished_at=number * 300.0 + 240.0,
            duration=200.0,
        )
        for number in range(1, 5)
    ]

    agent_only = build_exploration_efficiency_report(_audit(trials), expected_trials=4)
    human_manifest = {
        'schema_version': 'rightarm.gesture.human-exploration-baseline/v1',
        'baseline_id': 'human-paired-001',
        'measurement_scope': 'run-created-to-accepted-attempt-finished',
        'agent_comparison_fingerprint_sha256': agent_only[
            'comparison_fingerprint_sha256'
        ],
        'valid_trial_count': 4,
        'wall_seconds': 3600.0,
        'started_at': '1970-01-01T00:00:00Z',
        'finished_at': '1970-01-01T01:00:00Z',
        'evidence_uri': 'artifact://human-baselines/human-paired-001',
    }

    report = build_exploration_efficiency_report(
        _audit(trials),
        expected_trials=4,
        human_baseline_manifest=human_manifest,
    )

    assert report['max_trial_budget'] == 4
    assert report['agent_selected_trial_count'] == 4
    assert report['exploration_completed'] is True
    assert report['agent_wall_seconds'] == pytest.approx(1800.0)
    assert report['allocated_gpu_hours'] == pytest.approx(6400 / 3600)
    assert report['comparison']['agent_to_human_time_ratio'] == pytest.approx(0.5)
    assert report['comparison']['time_saving_fraction'] == pytest.approx(0.5)
    assert report['comparison']['speedup'] == pytest.approx(2.0)
    assert report['comparison']['uses_at_most_half_human_time'] is True
    assert report['comparison']['evidence_status'] == 'structured-self-attested'
    assert report['claim_eligible'] is False


def test_agent_can_stop_after_three_trials_and_compare_with_same_count() -> None:
    trials = [
        _trial_job(
            number,
            started_at=number * 300.0,
            finished_at=number * 300.0 + 240.0,
            duration=200.0,
        )
        for number in range(1, 4)
    ]

    report = build_exploration_efficiency_report(
        _audit(trials),
        human_wall_seconds=3600.0,
        human_baseline_id='provisional-human-baseline',
        human_valid_trials=3,
    )

    assert report['max_trial_budget'] == 4
    assert report['agent_selected_trial_count'] == 3
    assert report['exploration_completed'] is True
    assert report['agent_wall_seconds'] == pytest.approx(1800.0)
    assert report['comparison']['human_baseline_id'] == ('provisional-human-baseline')
    assert report['claim_eligible'] is False


def test_ad_hoc_human_time_is_explicitly_not_claim_eligible() -> None:
    trials = [
        _trial_job(
            number,
            started_at=number * 300.0,
            finished_at=number * 300.0 + 240.0,
            duration=200.0,
        )
        for number in range(1, 5)
    ]

    report = build_exploration_efficiency_report(
        _audit(trials),
        human_wall_seconds=3600.0,
        human_baseline_id='operator-note-only',
        human_valid_trials=4,
    )

    assert report['comparison']['evidence_status'] == 'unverified-operator-input'
    assert report['claim_eligible'] is False


def test_duplicate_job_for_one_accepted_trial_does_not_inflate_iteration_count() -> (
    None
):
    trials = [
        _trial_job(
            number,
            started_at=number * 300.0,
            finished_at=number * 300.0 + 240.0,
            duration=200.0,
        )
        for number in range(1, 5)
    ]
    duplicate = dict(trials[-1])
    duplicate['id'] = 'job-duplicate'
    audit = _audit(trials)
    audit['job_attempts'].insert(-1, duplicate)

    report = build_exploration_efficiency_report(audit)

    assert report['valid_trial_count'] == 3
    assert report['exploration_completed'] is False
    assert report['duplicate_accepted_trial_shas'] == [f'{4:064x}']


def test_trials_above_the_configured_safety_cap_are_not_complete() -> None:
    trials = [
        _trial_job(
            number,
            started_at=number * 300.0,
            finished_at=number * 300.0 + 240.0,
            duration=200.0,
        )
        for number in range(1, 6)
    ]

    report = build_exploration_efficiency_report(_audit(trials))

    assert report['max_trial_budget'] == 4
    assert report['agent_selected_trial_count'] == 5
    assert report['exploration_completed'] is False
    assert report['agent_wall_seconds'] is None


def test_expected_trials_only_asserts_the_agent_selected_count() -> None:
    trials = [
        _trial_job(
            number,
            started_at=number * 300.0,
            finished_at=number * 300.0 + 240.0,
            duration=200.0,
        )
        for number in range(1, 4)
    ]

    with pytest.raises(
        ValueError,
        match='expected_trials does not match the Agent-selected valid trial count',
    ):
        build_exploration_efficiency_report(_audit(trials), expected_trials=4)


def test_human_trial_count_must_match_the_agent_selected_count() -> None:
    trials = [
        _trial_job(
            number,
            started_at=number * 300.0,
            finished_at=number * 300.0 + 240.0,
            duration=200.0,
        )
        for number in range(1, 4)
    ]

    with pytest.raises(
        ValueError,
        match='human valid trial count must match the Agent-selected trial count',
    ):
        build_exploration_efficiency_report(
            _audit(trials),
            human_wall_seconds=3600.0,
            human_baseline_id='wrong-count-baseline',
            human_valid_trials=4,
        )
