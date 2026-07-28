"""Build an auditable Agent-versus-human exploration efficiency summary."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

SCHEMA_VERSION = 'rightarm.gesture.exploration-efficiency/v1'
HUMAN_BASELINE_SCHEMA_VERSION = 'rightarm.gesture.human-exploration-baseline/v1'
AGENT_WALL_SCOPE = 'run-created-to-accepted-attempt-finished'
_COMPLETED_RUN_STATES = frozenset({'COMPLETED', 'COMPLETED_NO_TRAINING'})


def _finite_positive(value: Optional[float], label: str) -> Optional[float]:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f'{label} must be a finite positive number')
    return parsed


def _job_kind(job: Mapping[str, Any]) -> Optional[str]:
    for key in ('result', 'failure'):
        payload = job.get(key)
        if isinstance(payload, Mapping) and isinstance(payload.get('job_kind'), str):
            return str(payload['job_kind'])
    argv = job.get('argv')
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        known = {
            'benchmark-checkpoint',
            'train-eval-trial',
            'record-model-improvement',
        }
        for value in argv:
            if value in known:
                return str(value)
    return None


def _runtime(job: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ('result', 'failure'):
        payload = job.get(key)
        if isinstance(payload, Mapping) and isinstance(payload.get('runtime'), Mapping):
            return payload['runtime']
    return {}


def _parse_utc(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + '+00:00' if value.endswith('Z') else value
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _json_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _accepted_trial_context(
    attempts: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> tuple[Optional[str], frozenset[str], bool]:
    accepted = [item for item in attempts if item.get('status') == 'ACCEPTED']
    if len(accepted) != 1:
        return None, frozenset(), False
    attempt = accepted[0]
    task_id = attempt.get('task_id')
    matching_tasks = [item for item in tasks if item.get('id') == task_id]
    if len(matching_tasks) != 1:
        return None, frozenset(), False
    task_spec = _json_mapping(
        matching_tasks[0].get('spec_json') or matching_tasks[0].get('spec')
    )
    if task_spec.get('kind') != 'improve-model':
        return None, frozenset(), False
    result = _json_mapping(attempt.get('result_json') or attempt.get('result'))
    validation = _json_mapping(
        attempt.get('validator_result_json') or attempt.get('validator_result')
    )
    if (
        result.get('status') != 'succeeded'
        or validation.get('disposition') != 'accepted'
    ):
        return None, frozenset(), False
    accepted_candidate_ids = validation.get('accepted_candidate_ids')
    candidates = result.get('artifact_candidates')
    if not isinstance(accepted_candidate_ids, list) or not isinstance(candidates, list):
        return None, frozenset(), False
    accepted_ids = {value for value in accepted_candidate_ids if isinstance(value, str)}
    trial_shas = frozenset(
        str(item['sha256'])
        for item in candidates
        if isinstance(item, Mapping)
        and item.get('candidate_id') in accepted_ids
        and item.get('kind') == 'optimization-trial-report'
        and isinstance(item.get('sha256'), str)
    )
    attempt_id = attempt.get('id')
    if not isinstance(attempt_id, str) or not trial_shas:
        return None, frozenset(), False
    return attempt_id, trial_shas, True


def _comparison_fingerprint(
    domain_spec: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    *,
    max_trial_budget: int,
) -> tuple[dict[str, Any], str]:
    executor_images = set()
    gpu_hardware: list[Any] = []
    for job in jobs:
        if _job_kind(job) not in {'benchmark-checkpoint', 'train-eval-trial'}:
            continue
        runtime = _runtime(job)
        image = runtime.get('executor_image') or runtime.get('image_digest')
        if isinstance(image, str) and image:
            executor_images.add(image)
        hardware = runtime.get('gpu_hardware')
        if isinstance(hardware, list):
            gpu_hardware.extend(hardware)
    payload = {
        'measurement_scope': AGENT_WALL_SCOPE,
        'max_trial_budget': max_trial_budget,
        'input_artifacts_by_kind': domain_spec.get('inputArtifactsByKind'),
        'optimization': domain_spec.get('optimization'),
        'training': domain_spec.get('training'),
        'evaluation': domain_spec.get('evaluation'),
        'executor_images': sorted(executor_images),
        'gpu_hardware': sorted(
            gpu_hardware,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        ),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')
    return payload, hashlib.sha256(canonical).hexdigest()


def _validated_human_baseline(
    manifest: Mapping[str, Any],
    *,
    comparison_fingerprint_sha256: str,
    valid_trial_count: int,
) -> dict[str, Any]:
    required = {
        'schema_version',
        'baseline_id',
        'measurement_scope',
        'agent_comparison_fingerprint_sha256',
        'valid_trial_count',
        'wall_seconds',
        'started_at',
        'finished_at',
        'evidence_uri',
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f'human baseline manifest is missing {missing}')
    if manifest.get('schema_version') != HUMAN_BASELINE_SCHEMA_VERSION:
        raise ValueError('unsupported human baseline manifest schema')
    baseline_id = manifest.get('baseline_id')
    evidence_uri = manifest.get('evidence_uri')
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        raise ValueError('human baseline manifest requires baseline_id')
    if not isinstance(evidence_uri, str) or not evidence_uri.strip():
        raise ValueError('human baseline manifest requires evidence_uri')
    if manifest.get('measurement_scope') != AGENT_WALL_SCOPE:
        raise ValueError('human baseline measurement_scope does not match Agent')
    if (
        manifest.get('agent_comparison_fingerprint_sha256')
        != comparison_fingerprint_sha256
    ):
        raise ValueError('human baseline comparison fingerprint does not match Run')
    if manifest.get('valid_trial_count') != valid_trial_count:
        raise ValueError(
            'human baseline valid_trial_count does not match Agent-selected trials'
        )
    wall_seconds = _finite_positive(manifest.get('wall_seconds'), 'wall_seconds')
    started = _parse_utc(manifest.get('started_at'))
    finished = _parse_utc(manifest.get('finished_at'))
    if started is None or finished is None or finished <= started:
        raise ValueError('human baseline timestamps must be ordered ISO-8601 values')
    assert wall_seconds is not None
    if abs((finished - started) - wall_seconds) > max(1.0, wall_seconds * 0.001):
        raise ValueError('human baseline wall_seconds disagrees with timestamps')
    manifest_canonical = json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')
    return {
        'human_baseline_id': baseline_id,
        'human_wall_seconds': wall_seconds,
        'human_baseline_evidence_uri': evidence_uri,
        'human_baseline_manifest_sha256': hashlib.sha256(
            manifest_canonical
        ).hexdigest(),
    }


def _job_bounds(job: Mapping[str, Any]) -> tuple[float, float, float, str]:
    runtime = _runtime(job)
    started = _parse_utc(runtime.get('job_started_at'))
    finished = _parse_utc(runtime.get('job_finished_at'))
    duration = runtime.get('job_duration_seconds')
    if (
        started is not None
        and finished is not None
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(float(duration))
        and float(duration) >= 0
    ):
        return started, finished, float(duration), 'job-monotonic-runtime'
    created = float(job['created_at'])
    updated = float(job['updated_at'])
    finished = max(created, updated)
    return created, finished, finished - created, 'controller-observed'


def _gpu_count(job: Mapping[str, Any]) -> int:
    runtime = _runtime(job)
    value = runtime.get('gpus')
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    argv = job.get('argv')
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        try:
            index = list(argv).index('--gpus')
            parsed = int(argv[index + 1])
        except (IndexError, TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0
    return 0


def _accepted_trial_sha(
    job: Mapping[str, Any],
    *,
    accepted_attempt_id: Optional[str],
    accepted_trial_shas: frozenset[str],
) -> Optional[str]:
    if job.get('task_attempt_id') != accepted_attempt_id:
        return None
    result = job.get('result')
    if job.get('status') != 'succeeded' or not isinstance(result, Mapping):
        return None
    if result.get('status') != 'succeeded':
        return None
    outputs = result.get('outputs')
    if not isinstance(outputs, list):
        return None
    matched = {
        str(item['sha256'])
        for item in outputs
        if isinstance(item, Mapping)
        and item.get('kind') == 'optimization-trial-report'
        and item.get('sha256') in accepted_trial_shas
    }
    return next(iter(matched)) if len(matched) == 1 else None


def _job_summary(
    job: Mapping[str, Any],
    *,
    accepted_attempt_id: Optional[str],
    accepted_trial_shas: frozenset[str],
) -> dict[str, Any]:
    started, finished, duration, timing_source = _job_bounds(job)
    return {
        'job_id': job.get('id'),
        'status': job.get('status'),
        'started_at_epoch': started,
        'finished_at_epoch': finished,
        'duration_seconds': duration,
        'timing_source': timing_source,
        'gpus': _gpu_count(job),
        'accepted_trial_sha256': _accepted_trial_sha(
            job,
            accepted_attempt_id=accepted_attempt_id,
            accepted_trial_shas=accepted_trial_shas,
        ),
        'valid_iteration': False,
    }


def _peak_concurrent_gpus(job_summaries: Sequence[Mapping[str, Any]]) -> int:
    events = []
    for item in job_summaries:
        gpus = item.get('gpus')
        if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus <= 0:
            continue
        events.append((float(item['started_at_epoch']), gpus))
        events.append((float(item['finished_at_epoch']), -gpus))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def build_exploration_efficiency_report(
    audit: Mapping[str, Any],
    *,
    expected_trials: Optional[int] = None,
    human_wall_seconds: Optional[float] = None,
    human_baseline_id: Optional[str] = None,
    human_valid_trials: Optional[int] = None,
    human_baseline_manifest: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Summarize one improve Run and optionally compare a paired human baseline."""

    if audit.get('protocol') != 'mlops.openhands.dev/run-audit/v1':
        raise ValueError('unsupported Run audit protocol')
    if expected_trials is not None and not 1 <= expected_trials <= 8:
        raise ValueError('expected_trials must be in [1, 8]')
    if human_baseline_manifest is not None and (
        human_wall_seconds is not None
        or human_baseline_id is not None
        or human_valid_trials is not None
    ):
        raise ValueError(
            'human baseline manifest cannot be combined with ad-hoc baseline values'
        )
    ad_hoc_human_values = (
        human_wall_seconds,
        human_baseline_id,
        human_valid_trials,
    )
    if any(value is not None for value in ad_hoc_human_values) and not all(
        value is not None for value in ad_hoc_human_values
    ):
        raise ValueError(
            'human wall time, baseline id and valid trial count must be supplied together'
        )
    if human_valid_trials is not None and (
        isinstance(human_valid_trials, bool)
        or not isinstance(human_valid_trials, int)
        or not 1 <= human_valid_trials <= 8
    ):
        raise ValueError('human_valid_trials must be in [1, 8]')
    human_wall_seconds = _finite_positive(human_wall_seconds, 'human_wall_seconds')
    run = audit.get('run')
    jobs = audit.get('job_attempts')
    attempts = audit.get('attempts')
    tasks = audit.get('tasks')
    if (
        not isinstance(run, Mapping)
        or not isinstance(jobs, list)
        or not isinstance(attempts, list)
        or not isinstance(tasks, list)
    ):
        raise ValueError('Run audit is missing run/tasks/attempts/job_attempts')

    run_document = _json_mapping(run.get('spec'))
    domain_spec = _json_mapping(run_document.get('spec'))
    manual_trigger = _json_mapping(domain_spec.get('manualTrigger'))
    run_is_improve = manual_trigger.get('mode') == 'improve'
    optimization = _json_mapping(domain_spec.get('optimization'))
    configured_trial_budget = optimization.get('maxTrials')
    if (
        not isinstance(configured_trial_budget, int)
        or isinstance(configured_trial_budget, bool)
        or not 1 <= configured_trial_budget <= 8
    ):
        raise ValueError('improve RunSpec is missing a valid optimization.maxTrials')
    fingerprint_payload, fingerprint_sha256 = _comparison_fingerprint(
        domain_spec,
        [item for item in jobs if isinstance(item, Mapping)],
        max_trial_budget=configured_trial_budget,
    )
    verified_human_baseline = None

    accepted_attempt_id, accepted_trial_shas, accepted_attempt_verified = (
        _accepted_trial_context(
            [item for item in attempts if isinstance(item, Mapping)],
            [item for item in tasks if isinstance(item, Mapping)],
        )
    )
    accepted_attempt_finished_at = next(
        (
            float(item['finished_at'])
            for item in attempts
            if isinstance(item, Mapping)
            and item.get('id') == accepted_attempt_id
            and isinstance(item.get('finished_at'), (int, float))
            and not isinstance(item.get('finished_at'), bool)
        ),
        None,
    )

    trial_jobs = [job for job in jobs if _job_kind(job) == 'train-eval-trial']
    baseline_jobs = [job for job in jobs if _job_kind(job) == 'benchmark-checkpoint']
    selection_jobs = [
        job for job in jobs if _job_kind(job) == 'record-model-improvement'
    ]
    trial_summaries = [
        _job_summary(
            job,
            accepted_attempt_id=accepted_attempt_id,
            accepted_trial_shas=accepted_trial_shas,
        )
        for job in trial_jobs
    ]
    matching_trials_by_sha: dict[str, list[dict[str, Any]]] = {}
    for item in trial_summaries:
        matched_sha = item['accepted_trial_sha256']
        if matched_sha is not None:
            matching_trials_by_sha.setdefault(matched_sha, []).append(item)
    valid_trials = []
    for matched_jobs in matching_trials_by_sha.values():
        if len(matched_jobs) == 1:
            matched_jobs[0]['valid_iteration'] = True
            valid_trials.append(matched_jobs[0])
    duplicate_accepted_trial_shas = sorted(
        sha
        for sha, matched_jobs in matching_trials_by_sha.items()
        if len(matched_jobs) > 1
    )
    failed_trials = [
        item
        for item in trial_summaries
        if item['status'] in {'failed', 'lost', 'cancelled'}
    ]
    ordered_valid = sorted(valid_trials, key=lambda item: item['started_at_epoch'])
    trial_makespan = None
    decision_latency = 0.0
    trial_overlap = 0.0
    if ordered_valid:
        trial_makespan = max(
            0.0,
            max(item['finished_at_epoch'] for item in ordered_valid)
            - ordered_valid[0]['started_at_epoch'],
        )
        for current, following in zip(ordered_valid, ordered_valid[1:], strict=False):
            decision_latency += max(
                0.0,
                following['started_at_epoch'] - current['finished_at_epoch'],
            )
            trial_overlap += max(
                0.0,
                current['finished_at_epoch'] - following['started_at_epoch'],
            )

    run_state = str(run.get('state'))
    run_completed = run_state in _COMPLETED_RUN_STATES
    accepted_selection_jobs = [
        job
        for job in selection_jobs
        if job.get('task_attempt_id') == accepted_attempt_id
    ]
    selection_succeeded = any(
        isinstance(job.get('result'), Mapping)
        and job.get('status') == 'succeeded'
        and job['result'].get('status') == 'succeeded'
        for job in accepted_selection_jobs
    )
    exploration_completed = (
        bool(valid_trials)
        and len(valid_trials) <= configured_trial_budget
        and len(valid_trials) == len(accepted_trial_shas)
        and not duplicate_accepted_trial_shas
        and selection_succeeded
        and run_completed
        and run_is_improve
        and accepted_attempt_verified
    )
    if expected_trials is not None and len(valid_trials) != expected_trials:
        raise ValueError(
            'expected_trials does not match the Agent-selected valid trial count'
        )
    if human_valid_trials is not None and human_valid_trials != len(valid_trials):
        raise ValueError(
            'human valid trial count must match the Agent-selected trial count'
        )
    if human_baseline_manifest is not None:
        verified_human_baseline = _validated_human_baseline(
            human_baseline_manifest,
            comparison_fingerprint_sha256=fingerprint_sha256,
            valid_trial_count=len(valid_trials),
        )
        human_wall_seconds = verified_human_baseline['human_wall_seconds']
        human_baseline_id = verified_human_baseline['human_baseline_id']
    run_created = float(run['created_at'])
    agent_wall_seconds = (
        max(0.0, accepted_attempt_finished_at - run_created)
        if exploration_completed and accepted_attempt_finished_at is not None
        else None
    )
    elapsed_seconds = max(0.0, time.time() - run_created)

    comparison = None
    if human_wall_seconds is not None and agent_wall_seconds is not None:
        ratio = agent_wall_seconds / human_wall_seconds
        comparison = {
            'human_baseline_id': human_baseline_id,
            'human_wall_seconds': human_wall_seconds,
            'evidence_status': (
                'structured-self-attested'
                if verified_human_baseline is not None
                else 'unverified-operator-input'
            ),
            'agent_to_human_time_ratio': ratio,
            'time_saving_fraction': 1.0 - ratio,
            'speedup': human_wall_seconds / agent_wall_seconds
            if agent_wall_seconds > 0
            else None,
            'uses_at_most_half_human_time': ratio <= 0.5,
        }
        if verified_human_baseline is not None:
            comparison.update(verified_human_baseline)

    gpu_job_summaries = [
        _job_summary(
            job,
            accepted_attempt_id=accepted_attempt_id,
            accepted_trial_shas=accepted_trial_shas,
        )
        for job in jobs
        if isinstance(job, Mapping) and _gpu_count(job) > 0
    ]
    gpu_seconds = sum(
        item['duration_seconds'] * item['gpus'] for item in gpu_job_summaries
    )
    return {
        'schema_version': SCHEMA_VERSION,
        'run_id': run.get('id'),
        'run_state': run_state,
        'max_trial_budget': configured_trial_budget,
        'agent_selected_trial_count': len(valid_trials),
        'valid_trial_count': len(valid_trials),
        'failed_trial_count': len(failed_trials),
        'attempt_count': len(attempts) if isinstance(attempts, list) else None,
        'accepted_attempt_id': accepted_attempt_id,
        'accepted_trial_candidate_count': len(accepted_trial_shas),
        'duplicate_accepted_trial_shas': duplicate_accepted_trial_shas,
        'baseline_job_count': len(baseline_jobs),
        'selection_job_count': len(selection_jobs),
        'accepted_selection_job_count': len(accepted_selection_jobs),
        'exploration_completed': exploration_completed,
        'agent_wall_seconds': agent_wall_seconds,
        'agent_wall_scope': AGENT_WALL_SCOPE,
        'comparison_fingerprint': fingerprint_payload,
        'comparison_fingerprint_sha256': fingerprint_sha256,
        'elapsed_seconds': elapsed_seconds,
        'valid_trial_makespan_seconds': trial_makespan,
        'decision_latency_seconds': decision_latency,
        'trial_overlap_seconds': trial_overlap,
        'serial_trial_execution': trial_overlap == 0.0,
        'peak_concurrent_trial_gpus': _peak_concurrent_gpus(valid_trials),
        'allocated_gpu_hours': gpu_seconds / 3600.0,
        'gpu_job_count': len(gpu_job_summaries),
        'valid_trials_per_hour': (
            len(valid_trials) * 3600.0 / trial_makespan
            if trial_makespan and trial_makespan > 0
            else None
        ),
        'trial_jobs': trial_summaries,
        'comparison': comparison,
        'claim_eligible': False,
        'formal_claim_blockers': [
            'The human timing evidence is not yet sealed and independently verified.',
            'The executed dirty workspace is not yet sealed as an immutable code artifact.',
        ],
        'notes': [
            'A valid iteration must be accepted by the final Validator and match an optimization-trial-report job output by sha256.',
            'Failed/retried GPU jobs consume time and GPU-hours but do not count as valid iterations.',
            'A paired human baseline must freeze dataset, checkpoint, code, image, hardware and search policy, and complete the same Agent-selected number of valid trials.',
        ],
    }
