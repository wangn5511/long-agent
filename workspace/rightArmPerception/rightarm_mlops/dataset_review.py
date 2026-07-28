"""Deterministic distribution review and guarded merging for PoseDataset PKLs.

This module deliberately keeps review separate from mutation:

* :func:`build_distribution_report` loads two datasets, canonicalizes their
  label identities, and writes a deterministic
  ``dataset-distribution-report.json``.
* :func:`merge_reviewed_datasets` recomputes that plan, verifies the caller's
  digest, and only merges when the deterministic recommendation is green.

Pickle is not a safe interchange format.  Dataset loading therefore uses a
restricted unpickler that only permits the small set of NumPy constructors
needed by ndarray pickles.  It never imports a class named by the pickle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import shutil
import struct
import tarfile
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from rightarm_mlops.protocol import (
    atomic_json,
    deterministic_tar,
    safe_extract_tar,
    sha256_file,
)

CANONICAL_LABELS = (
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
_LABEL_TO_ID = {label: index for index, label in enumerate(CANONICAL_LABELS)}

REPORT_SCHEMA = 'rightarm.gesture.dataset-distribution-report/v1'
PLAN_SCHEMA = 'rightarm.gesture.dataset-merge-plan/v1'
MERGE_SCHEMA = 'rightarm.gesture.dataset-merge-report/v1'
DATASET_SCHEMA = 'rightarm.gesture.mmaction-pose/v1'

_GREEN_THRESHOLDS = {
    'jensen_shannon_divergence': 0.05,
    'max_class_proportion_delta': 0.03,
    'none_proportion_delta': 0.02,
    'incremental_to_baseline_ratio': 0.25,
}
_MEDIA_FIELDS = ('video_path', 'clip_video_path')
_VOLATILE_SAMPLE_FIELDS = frozenset(
    {
        'frame_dir',
        'label',
        'label_name',
        'video_path',
        'clip_video_path',
        'source_video',
        'annotation_json_path',
    }
)


try:  # NumPy 2.x
    from numpy._core import multiarray as _np_multiarray
    from numpy._core import numeric as _np_numeric
except ImportError:  # pragma: no cover - exercised by NumPy 1.x environments
    from numpy.core import multiarray as _np_multiarray
    from numpy.core import numeric as _np_numeric


_ALLOWED_PICKLE_GLOBALS = {
    ('numpy', 'dtype'): np.dtype,
    ('numpy', 'ndarray'): np.ndarray,
    ('numpy.core.numeric', '_frombuffer'): _np_numeric._frombuffer,
    ('numpy._core.numeric', '_frombuffer'): _np_numeric._frombuffer,
    ('numpy.core.multiarray', '_reconstruct'): _np_multiarray._reconstruct,
    ('numpy._core.multiarray', '_reconstruct'): _np_multiarray._reconstruct,
    ('numpy.core.multiarray', 'scalar'): _np_multiarray.scalar,
    ('numpy._core.multiarray', 'scalar'): _np_multiarray.scalar,
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Unpickle primitive containers and explicitly allowlisted NumPy values."""

    def find_class(self, module: str, name: str) -> Any:
        value = _ALLOWED_PICKLE_GLOBALS.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(
                f'dataset pickle contains forbidden global {module}.{name}'
            )
        return value


@dataclass
class _LoadedDataset:
    role: str
    artifact_id: str
    input_path: Path
    input_format: str
    input_sha256: str
    input_size_bytes: int
    dataset_path: Path
    media_root: Path
    value: dict[str, Any]

    def input_record(self) -> dict[str, Any]:
        return {
            'role': self.role,
            'artifact_id': self.artifact_id,
            'format': self.input_format,
            'sha256': self.input_sha256,
            'size_bytes': self.input_size_bytes,
        }


@dataclass
class _SampleRecord:
    role: str
    position: int
    sample_id: str
    split: str
    label: str
    group_id: str | None
    fingerprint: str
    sample: dict[str, Any]
    media_root: Path
    disposition: str = 'keep'

    def index_record(self) -> dict[str, Any]:
        return {
            'dataset': self.role,
            'fingerprint': self.fingerprint,
            'group_id': self.group_id,
            'label': self.label,
            'merge_disposition': self.disposition,
            'sample_id': self.sample_id,
            'split': self.split,
        }


def build_distribution_report(
    baseline_path: str | Path,
    incremental_path: str | Path,
    *,
    baseline_artifact_id: str,
    incremental_artifact_id: str,
    work_dir: str | Path,
) -> dict[str, Any]:
    """Build and persist a deterministic comparison of two dataset artifacts.

    ``baseline_path`` and ``incremental_path`` may point to a raw PKL or to a
    tar bundle containing exactly one PKL.  The returned value is also written
    to ``<work_dir>/dataset-distribution-report.json`` with sorted JSON keys.
    """

    _validate_artifact_id(baseline_artifact_id, 'baseline_artifact_id')
    _validate_artifact_id(incremental_artifact_id, 'incremental_artifact_id')
    destination = Path(work_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        baseline = stack.enter_context(
            _open_dataset(
                baseline_path,
                artifact_id=baseline_artifact_id,
                role='baseline',
                work_dir=destination,
            )
        )
        incremental = stack.enter_context(
            _open_dataset(
                incremental_path,
                artifact_id=incremental_artifact_id,
                role='new-batch',
                work_dir=destination,
            )
        )
        report, _, _ = _build_report(baseline, incremental)

    atomic_json(destination / 'dataset-distribution-report.json', report)
    return report


def merge_reviewed_datasets(
    baseline_path: str | Path,
    incremental_path: str | Path,
    *,
    baseline_artifact_id: str,
    incremental_artifact_id: str,
    distribution_report_sha256: str,
    reason_codes: Sequence[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge two reviewed datasets when the recomputed plan is green.

    Exact duplicates with the same sample id and semantic fingerprint retain
    the baseline copy.  Reusing a sample id for different semantic content is
    a hard conflict and can never be overridden with ``reason_codes``.
    """

    _validate_artifact_id(baseline_artifact_id, 'baseline_artifact_id')
    _validate_artifact_id(incremental_artifact_id, 'incremental_artifact_id')
    if not re.fullmatch(r'[0-9a-f]{64}', distribution_report_sha256):
        raise ValueError('distribution_report_sha256 must be 64 lowercase hex digits')
    normalized_reasons = _normalize_reason_codes(reason_codes)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    distribution_path = destination / 'dataset-distribution-report.json'
    plan_path = destination / 'dataset-merge-plan.json'

    with ExitStack() as stack:
        baseline = stack.enter_context(
            _open_dataset(
                baseline_path,
                artifact_id=baseline_artifact_id,
                role='baseline',
                work_dir=destination,
            )
        )
        incremental = stack.enter_context(
            _open_dataset(
                incremental_path,
                artifact_id=incremental_artifact_id,
                role='new-batch',
                work_dir=destination,
            )
        )
        report, baseline_records, incremental_records = _build_report(
            baseline, incremental
        )
        atomic_json(distribution_path, report)
        actual_report_sha256, _ = sha256_file(distribution_path)
        if actual_report_sha256 != distribution_report_sha256:
            raise ValueError(
                'distribution report digest does not match the recomputed plan'
            )
        recommendation = report['recommendation']
        if recommendation['action'] != 'merge-all-unique':
            raise ValueError(
                "dataset merge is forbidden unless recommendation is 'merge-all-unique'"
            )
        merge_plan: dict[str, Any] = {
            'schema_version': PLAN_SCHEMA,
            'distribution_report_sha256': distribution_report_sha256,
            'inputs': report['inputs'],
            'decision': 'merge-all-unique',
            'requires_human_decision': False,
            'reason_codes': normalized_reasons,
        }
        atomic_json(plan_path, merge_plan)
        dataset_merge_plan_sha256, _ = sha256_file(plan_path)

        retained_baseline = [
            record
            for record in baseline_records
            if record.disposition == 'keep-baseline'
        ]
        retained_incremental = [
            record
            for record in incremental_records
            if record.disposition == 'add-incremental'
        ]
        retained = retained_baseline + retained_incremental

        raw_samples, bundle_samples, media_paths = _materialize_merged_media(
            retained, destination
        )
        raw_dataset = _merged_dataset_value(
            baseline.value,
            raw_samples,
            retained,
            baseline_artifact_id=baseline_artifact_id,
            incremental_artifact_id=incremental_artifact_id,
            distribution_report_sha256=distribution_report_sha256,
            dataset_merge_plan_sha256=dataset_merge_plan_sha256,
        )
        bundle_dataset = _merged_dataset_value(
            baseline.value,
            bundle_samples,
            retained,
            baseline_artifact_id=baseline_artifact_id,
            incremental_artifact_id=incremental_artifact_id,
            distribution_report_sha256=distribution_report_sha256,
            dataset_merge_plan_sha256=dataset_merge_plan_sha256,
        )

        merged_pkl = destination / 'merged-rightarm-gesture-11.pkl'
        merged_tar = destination / 'merged-rightarm-gesture-11.tar'
        _atomic_pickle(merged_pkl, raw_dataset)
        _write_bundle(merged_tar, bundle_dataset, media_paths, destination)

    duplicate_count = sum(
        record.disposition == 'drop-exact-duplicate' for record in incremental_records
    )
    dataset_sha256, dataset_size_bytes = sha256_file(merged_tar)
    merge_report: dict[str, Any] = {
        'schema_version': MERGE_SCHEMA,
        'distribution_report_sha256': distribution_report_sha256,
        'dataset_merge_plan_sha256': dataset_merge_plan_sha256,
        'inputs': report['inputs'],
        'reason_codes': normalized_reasons,
        'recommendation': recommendation,
        'sample_counts': {
            'baseline_retained': len(retained_baseline),
            'exact_incremental_duplicates_dropped': duplicate_count,
            'incremental_added': len(retained_incremental),
            'merged': len(retained),
        },
        'split_counts': _split_counts(retained),
        'class_counts': _class_counts(retained),
        'output': {
            'sha256': dataset_sha256,
            'size_bytes': dataset_size_bytes,
            'sample_count': len(retained),
            'counts': _split_label_counts(retained),
        },
    }
    atomic_json(destination / 'dataset-merge-report.json', merge_report)
    return merge_report


def _build_report(
    baseline: _LoadedDataset, incremental: _LoadedDataset
) -> tuple[dict[str, Any], list[_SampleRecord], list[_SampleRecord]]:
    hard_failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    baseline_records = _canonical_records(baseline, hard_failures, warnings)
    incremental_records = _canonical_records(incremental, hard_failures, warnings)

    baseline_unique = _deduplicate_within(baseline_records, hard_failures, warnings)
    incremental_unique = _deduplicate_within(
        incremental_records, hard_failures, warnings
    )
    _mark_cross_dataset_dispositions(
        baseline_unique, incremental_unique, hard_failures, warnings
    )

    effective_incremental = [
        item for item in incremental_unique if item.disposition == 'add-incremental'
    ]
    if not baseline_unique:
        hard_failures.append(
            {'code': 'baseline-has-no-unique-samples', 'dataset': 'baseline'}
        )
    if not incremental_unique:
        hard_failures.append(
            {
                'code': 'incremental-has-no-unique-samples',
                'dataset': 'new-batch',
            }
        )
    if incremental_unique and not effective_incremental:
        warnings.append(
            {
                'code': 'incremental-has-no-new-unique-samples',
                'dataset': 'new-batch',
            }
        )

    group_overlap = _group_overlap(
        baseline_unique,
        incremental_unique,
        hard_failures=hard_failures,
        warnings=warnings,
    )
    baseline_summary = _dataset_summary(baseline_unique)
    incremental_summary = _dataset_summary(incremental_unique)
    new_summary = _dataset_summary(effective_incremental)
    combined_summary = _dataset_summary(baseline_unique + effective_incremental)
    metrics = _distribution_metrics(baseline_unique, effective_incremental)

    threshold_reasons = _threshold_reason_codes(metrics)
    sorted_failures = _sorted_issues(hard_failures)
    sorted_warnings = _sorted_issues(warnings)
    if sorted_failures:
        recommendation_action = 'reject'
        recommendation_reasons = sorted({item['code'] for item in sorted_failures})
    elif threshold_reasons:
        recommendation_action = 'manual-review'
        recommendation_reasons = threshold_reasons
    else:
        recommendation_action = 'merge-all-unique'
        recommendation_reasons = ['all-green-thresholds-passed']

    all_records = baseline_unique + incremental_unique
    sample_index = [
        item.index_record()
        for item in sorted(
            all_records,
            key=lambda value: (
                value.role,
                value.sample_id,
                value.fingerprint,
                value.position,
            ),
        )
    ]
    samples = [
        {
            'id': f'{item.role}:{item.sample_id}:{item.position:08d}',
            'origin': item.role,
            'split': item.split,
            'label': item.label,
            'fingerprint': item.fingerprint,
            'group_id': item.group_id,
            'merge_disposition': item.disposition,
        }
        for item in sorted(
            (
                item
                for item in all_records
                if item.split in {'train', 'val', 'test'} and item.label in _LABEL_TO_ID
            ),
            key=lambda value: (
                value.role,
                value.sample_id,
                value.fingerprint,
                value.position,
            ),
        )
    ]
    report: dict[str, Any] = {
        'schema_version': REPORT_SCHEMA,
        'canonical_categories': list(CANONICAL_LABELS),
        'inputs': [baseline.input_record(), incremental.input_record()],
        'samples': samples,
        'sample_index': sample_index,
        'summary': {
            'counts': _origin_split_label_counts(samples),
            'baseline': baseline_summary,
            'new_batch': incremental_summary,
            'new_batch_unique_after_baseline': new_summary,
            'combined_unique': combined_summary,
        },
        'distribution_metrics': metrics,
        'thresholds': dict(_GREEN_THRESHOLDS),
        'group_overlap': group_overlap,
        'hard_failures': sorted_failures,
        'warnings': sorted_warnings,
        'recommendation': {
            'action': recommendation_action,
            'reason_codes': recommendation_reasons,
        },
    }
    return report, baseline_unique, incremental_unique


def _canonical_records(
    loaded: _LoadedDataset,
    hard_failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> list[_SampleRecord]:
    value = loaded.value
    annotations = value.get('annotations')
    splits = value.get('split')
    if not isinstance(annotations, list):
        raise ValueError(f'{loaded.role} dataset annotations must be a list')
    if not isinstance(splits, dict):
        raise ValueError(f'{loaded.role} dataset split must be an object')

    categories = value.get('categories')
    if (
        not isinstance(categories, (list, tuple))
        or len(categories) != len(CANONICAL_LABELS)
        or set(categories) != set(CANONICAL_LABELS)
    ):
        hard_failures.append(
            {
                'code': 'noncanonical-label-ontology',
                'dataset': loaded.role,
            }
        )

    memberships: dict[str, list[str]] = defaultdict(list)
    for raw_split, raw_ids in sorted(splits.items(), key=lambda item: str(item[0])):
        split = str(raw_split)
        if not isinstance(raw_ids, (list, tuple)):
            hard_failures.append(
                {
                    'code': 'invalid-split-members',
                    'dataset': loaded.role,
                    'split': split,
                }
            )
            continue
        seen_in_split: set[str] = set()
        for raw_id in raw_ids:
            sample_id = str(raw_id)
            if sample_id in seen_in_split:
                hard_failures.append(
                    {
                        'code': 'duplicate-id-in-split',
                        'dataset': loaded.role,
                        'sample_id': sample_id,
                        'split': split,
                    }
                )
            seen_in_split.add(sample_id)
            memberships[sample_id].append(split)

    records: list[_SampleRecord] = []
    annotation_ids: set[str] = set()
    for position, raw_sample in enumerate(annotations):
        if not isinstance(raw_sample, dict):
            hard_failures.append(
                {
                    'code': 'annotation-is-not-an-object',
                    'dataset': loaded.role,
                    'position': position,
                }
            )
            continue
        sample = dict(raw_sample)
        sample_id_value = sample.get('frame_dir', sample.get('clip_id'))
        if not isinstance(sample_id_value, str) or not sample_id_value:
            sample_id = f'__missing_sample_id_{position:08d}'
            hard_failures.append(
                {
                    'code': 'missing-sample-id',
                    'dataset': loaded.role,
                    'position': position,
                }
            )
        else:
            sample_id = sample_id_value
        annotation_ids.add(sample_id)

        sample_memberships = sorted(set(memberships.get(sample_id, [])))
        if not sample_memberships:
            split = 'unassigned'
            hard_failures.append(
                {
                    'code': 'sample-is-not-in-a-split',
                    'dataset': loaded.role,
                    'sample_id': sample_id,
                }
            )
        elif len(sample_memberships) > 1:
            split = 'ambiguous'
            hard_failures.append(
                {
                    'code': 'sample-is-in-multiple-splits',
                    'dataset': loaded.role,
                    'sample_id': sample_id,
                    'splits': sample_memberships,
                }
            )
        else:
            split = sample_memberships[0]

        label = _resolve_label(value, sample)
        if label is None:
            label = '__INVALID__'
            hard_failures.append(
                {
                    'code': 'unresolvable-sample-label',
                    'dataset': loaded.role,
                    'sample_id': sample_id,
                }
            )
        else:
            sample['label'] = _LABEL_TO_ID[label]
            sample['label_name'] = label

        quality_issues = _sample_quality_issues(sample)
        for issue in quality_issues:
            hard_failures.append(
                {
                    'code': issue,
                    'dataset': loaded.role,
                    'sample_id': sample_id,
                }
            )
        group_id = _resolve_group_id(sample)
        if group_id is None:
            warnings.append(
                {
                    'code': 'unresolved-group-id',
                    'dataset': loaded.role,
                    'sample_id': sample_id,
                }
            )
        _check_media_references(
            loaded,
            sample,
            sample_id=sample_id,
            hard_failures=hard_failures,
            warnings=warnings,
        )
        fingerprint = _sample_fingerprint(sample, label)
        records.append(
            _SampleRecord(
                role=loaded.role,
                position=position,
                sample_id=sample_id,
                split=split,
                label=label,
                group_id=group_id,
                fingerprint=fingerprint,
                sample=sample,
                media_root=loaded.media_root,
            )
        )

    dangling = sorted(set(memberships) - annotation_ids)
    for sample_id in dangling:
        hard_failures.append(
            {
                'code': 'split-id-has-no-annotation',
                'dataset': loaded.role,
                'sample_id': sample_id,
            }
        )
    return records


def _deduplicate_within(
    records: list[_SampleRecord],
    hard_failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> list[_SampleRecord]:
    by_id: dict[str, _SampleRecord] = {}
    unique: list[_SampleRecord] = []
    for record in records:
        existing = by_id.get(record.sample_id)
        if existing is None:
            record.disposition = (
                'keep-baseline' if record.role == 'baseline' else 'add-incremental'
            )
            by_id[record.sample_id] = record
            unique.append(record)
            continue
        if existing.fingerprint != record.fingerprint:
            record.disposition = 'conflict'
            hard_failures.append(
                {
                    'code': 'conflicting-duplicate-sample-id',
                    'dataset': record.role,
                    'sample_id': record.sample_id,
                }
            )
        else:
            record.disposition = 'drop-exact-duplicate'
            warnings.append(
                {
                    'code': 'exact-duplicate-within-dataset',
                    'dataset': record.role,
                    'sample_id': record.sample_id,
                }
            )
    return unique


def _mark_cross_dataset_dispositions(
    baseline: list[_SampleRecord],
    incremental: list[_SampleRecord],
    hard_failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    baseline_by_id = {item.sample_id: item for item in baseline}
    for record in incremental:
        existing = baseline_by_id.get(record.sample_id)
        if existing is None:
            record.disposition = 'add-incremental'
        elif existing.fingerprint == record.fingerprint:
            record.disposition = 'drop-exact-duplicate'
            warnings.append(
                {
                    'code': 'exact-incremental-duplicate-baseline-retained',
                    'dataset': 'new-batch',
                    'sample_id': record.sample_id,
                }
            )
        else:
            record.disposition = 'conflict'
            hard_failures.append(
                {
                    'code': 'baseline-incremental-sample-conflict',
                    'dataset': 'new-batch',
                    'sample_id': record.sample_id,
                }
            )


def _dataset_summary(records: Sequence[_SampleRecord]) -> dict[str, Any]:
    frame_lengths = [
        int(item.sample.get('total_frames', 0))
        for item in records
        if isinstance(item.sample.get('total_frames'), (int, np.integer))
        and int(item.sample['total_frames']) >= 0
    ]
    class_counts = _class_counts(records)
    total = len(records)
    proportions = {
        label: _stable_float(class_counts[label] / total if total else 0.0)
        for label in CANONICAL_LABELS
    }
    return {
        'sample_count': total,
        'split_counts': _split_counts(records),
        'class_counts': class_counts,
        'class_proportions': proportions,
        'frame_count': sum(frame_lengths),
        'frame_length': _integer_distribution(frame_lengths),
        'group_count': len(
            {item.group_id for item in records if item.group_id is not None}
        ),
        'unresolved_group_count': sum(item.group_id is None for item in records),
    }


def _distribution_metrics(
    baseline: Sequence[_SampleRecord],
    incremental: Sequence[_SampleRecord],
) -> dict[str, Any]:
    baseline_train = [item for item in baseline if item.split == 'train']
    incremental_train = [item for item in incremental if item.split == 'train']
    projected_train = baseline_train + incremental_train
    baseline_counts = _class_counts(baseline_train)
    projected_counts = _class_counts(projected_train)
    baseline_total = len(baseline)
    incremental_total = len(incremental)
    baseline_train_total = len(baseline_train)
    projected_train_total = len(projected_train)
    baseline_p = np.asarray(
        [
            baseline_counts[label] / baseline_train_total
            if baseline_train_total
            else 0.0
            for label in CANONICAL_LABELS
        ],
        dtype=np.float64,
    )
    projected_p = np.asarray(
        [
            projected_counts[label] / projected_train_total
            if projected_train_total
            else 0.0
            for label in CANONICAL_LABELS
        ],
        dtype=np.float64,
    )
    deltas = projected_p - baseline_p
    absolute = np.abs(deltas)
    maximum_index = int(np.argmax(absolute)) if absolute.size else 0
    none_index = _LABEL_TO_ID['NONE']
    ratio = incremental_total / baseline_total if baseline_total else math.inf
    return {
        'jensen_shannon_divergence': _stable_float(
            _jensen_shannon_divergence(baseline_p, projected_p)
        ),
        'max_class_proportion_delta': {
            'absolute': _stable_float(float(absolute[maximum_index])),
            'label': CANONICAL_LABELS[maximum_index],
            'signed': _stable_float(float(deltas[maximum_index])),
        },
        'none_proportion_delta': {
            'absolute': _stable_float(float(absolute[none_index])),
            'signed': _stable_float(float(deltas[none_index])),
        },
        'incremental_to_baseline_ratio': (
            _stable_float(ratio) if math.isfinite(ratio) else None
        ),
        'comparison': 'baseline-train-vs-projected-combined-train',
        'baseline_train_sample_count': baseline_train_total,
        'projected_combined_train_sample_count': projected_train_total,
    }


def _threshold_reason_codes(metrics: Mapping[str, Any]) -> list[str]:
    reasons = []
    jsd = metrics['jensen_shannon_divergence']
    max_delta = metrics['max_class_proportion_delta']['absolute']
    none_delta = metrics['none_proportion_delta']['absolute']
    ratio = metrics['incremental_to_baseline_ratio']
    if jsd > _GREEN_THRESHOLDS['jensen_shannon_divergence']:
        reasons.append('jensen-shannon-divergence-above-green-threshold')
    if max_delta > _GREEN_THRESHOLDS['max_class_proportion_delta']:
        reasons.append('max-class-delta-above-green-threshold')
    if none_delta > _GREEN_THRESHOLDS['none_proportion_delta']:
        reasons.append('none-delta-above-green-threshold')
    if ratio is None or ratio > _GREEN_THRESHOLDS['incremental_to_baseline_ratio']:
        reasons.append('incremental-ratio-above-green-threshold')
    return sorted(reasons)


def _group_overlap(
    baseline: Sequence[_SampleRecord],
    incremental: Sequence[_SampleRecord],
    *,
    hard_failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_groups = _groups_by_split(baseline)
    incremental_groups = _groups_by_split(incremental)
    baseline_overlap = _multi_split_groups(baseline_groups)
    incremental_overlap = _multi_split_groups(incremental_groups)
    shared = sorted(set(baseline_groups) & set(incremental_groups))
    cross_split_conflicts = sorted(
        group
        for group in shared
        if len(baseline_groups[group] | incremental_groups[group]) > 1
    )

    if baseline_overlap:
        warnings.append(
            {
                'code': 'baseline-existing-group-split-overlap',
                'group_count': len(baseline_overlap),
            }
        )
    if incremental_overlap:
        hard_failures.append(
            {
                'code': 'incremental-group-split-overlap',
                'dataset': 'new-batch',
                'group_count': len(incremental_overlap),
            }
        )
    if shared:
        warnings.append(
            {
                'code': 'baseline-incremental-group-overlap',
                'group_count': len(shared),
            }
        )
    if cross_split_conflicts:
        hard_failures.append(
            {
                'code': 'cross-dataset-group-split-conflict',
                'group_count': len(cross_split_conflicts),
            }
        )
    return {
        'baseline_existing': {
            'count': len(baseline_overlap),
            'groups': baseline_overlap,
        },
        'new_batch_existing': {
            'count': len(incremental_overlap),
            'groups': incremental_overlap,
        },
        'baseline_incremental': {
            'count': len(shared),
            'groups': shared,
        },
        'cross_dataset_split_conflicts': {
            'count': len(cross_split_conflicts),
            'groups': cross_split_conflicts,
        },
    }


def _groups_by_split(
    records: Sequence[_SampleRecord],
) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for item in records:
        if item.group_id is not None:
            groups[item.group_id].add(item.split)
    return groups


def _multi_split_groups(groups: Mapping[str, set[str]]) -> list[str]:
    return sorted(group for group, splits in groups.items() if len(splits) > 1)


def _resolve_label(dataset: Mapping[str, Any], sample: Mapping[str, Any]) -> str | None:
    candidates: list[str] = []
    explicit = sample.get('label_name')
    if isinstance(explicit, str):
        candidates.append(explicit)

    raw_label = sample.get('label')
    categories = dataset.get('categories')
    if (
        isinstance(raw_label, (int, np.integer))
        and isinstance(categories, (list, tuple))
        and 0 <= int(raw_label) < len(categories)
        and isinstance(categories[int(raw_label)], str)
    ):
        candidates.append(categories[int(raw_label)])

    mapping = dataset.get('label_mapping')
    if isinstance(mapping, dict):
        mapped = mapping.get(raw_label, mapping.get(str(raw_label)))
        if isinstance(mapped, str):
            candidates.append(mapped)

    label_map = dataset.get('label_map')
    if isinstance(label_map, dict) and isinstance(raw_label, (int, np.integer)):
        inverse = {
            int(value): str(name)
            for name, value in label_map.items()
            if isinstance(name, str) and isinstance(value, (int, np.integer))
        }
        if int(raw_label) in inverse:
            candidates.append(inverse[int(raw_label)])

    canonical = {candidate for candidate in candidates if candidate in _LABEL_TO_ID}
    if len(canonical) != 1:
        return None
    return next(iter(canonical))


def _sample_quality_issues(sample: Mapping[str, Any]) -> list[str]:
    issues = []
    total_frames = sample.get('total_frames')
    if not isinstance(total_frames, (int, np.integer)) or int(total_frames) <= 0:
        issues.append('invalid-total-frames')
        expected_frames = None
    else:
        expected_frames = int(total_frames)

    keypoint = sample.get('keypoint')
    score = sample.get('keypoint_score')
    if not isinstance(keypoint, np.ndarray) or keypoint.ndim != 4:
        issues.append('invalid-keypoint-array')
    else:
        if keypoint.shape[-2:] != (21, 3):
            issues.append('invalid-keypoint-shape')
        if expected_frames is not None and keypoint.shape[1] != expected_frames:
            issues.append('keypoint-frame-count-mismatch')
        if not np.isfinite(keypoint).all():
            issues.append('nonfinite-keypoint-values')
    if not isinstance(score, np.ndarray) or score.ndim != 3:
        issues.append('invalid-keypoint-score-array')
    else:
        if score.shape[-1] != 21:
            issues.append('invalid-keypoint-score-shape')
        if expected_frames is not None and score.shape[1] != expected_frames:
            issues.append('keypoint-score-frame-count-mismatch')
        if not np.isfinite(score).all():
            issues.append('nonfinite-keypoint-scores')
    if (
        isinstance(keypoint, np.ndarray)
        and isinstance(score, np.ndarray)
        and keypoint.ndim == 4
        and score.ndim == 3
        and keypoint.shape[:-1] != score.shape
    ):
        issues.append('keypoint-score-shape-mismatch')
    return sorted(set(issues))


def _resolve_group_id(sample: Mapping[str, Any]) -> str | None:
    for field in (
        'subject_group_id',
        'source_group_id',
        'group_id',
        'session_id',
        'source_session_id',
    ):
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return f'{field}:{value.strip()}'

    strings = [
        str(sample.get(field))
        for field in (
            'source_video',
            'video_path',
            'clip_video_path',
            'annotation_json_path',
            'frame_dir',
            'clip_id',
        )
        if isinstance(sample.get(field), (str, Path))
    ]
    patterns = (
        re.compile(r'(debug_\d{8}_\d{6})'),
        re.compile(r'(\d{8}_(?:base|top)_\d{6})', re.IGNORECASE),
        re.compile(r'(?:^|[/_])(session[-_][A-Za-z0-9.-]+)(?:[/_]|$)'),
    )
    for value in strings:
        for pattern in patterns:
            match = pattern.search(value)
            if match:
                return match.group(1)
    return None


def _sample_fingerprint(sample: Mapping[str, Any], label: str) -> str:
    semantic = {
        key: value
        for key, value in sample.items()
        if key not in _VOLATILE_SAMPLE_FIELDS
    }
    semantic['canonical_label'] = label
    digest = hashlib.sha256()
    _update_value_digest(digest, semantic)
    return digest.hexdigest()


def _update_value_digest(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b'N')
    elif isinstance(value, bool):
        digest.update(b'B1' if value else b'B0')
    elif isinstance(value, (int, np.integer)):
        raw = str(int(value)).encode('ascii')
        digest.update(b'I' + len(raw).to_bytes(8, 'big') + raw)
    elif isinstance(value, (float, np.floating)):
        digest.update(b'F' + struct.pack('>d', float(value)))
    elif isinstance(value, str):
        raw = value.encode('utf-8')
        digest.update(b'S' + len(raw).to_bytes(8, 'big') + raw)
    elif isinstance(value, bytes):
        digest.update(b'Y' + len(value).to_bytes(8, 'big') + value)
    elif isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        dtype = contiguous.dtype.str.encode('ascii')
        digest.update(b'A' + len(dtype).to_bytes(4, 'big') + dtype)
        _update_value_digest(digest, tuple(int(item) for item in contiguous.shape))
        raw = contiguous.tobytes(order='C')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    elif isinstance(value, (list, tuple)):
        digest.update(b'L' + len(value).to_bytes(8, 'big'))
        for item in value:
            _update_value_digest(digest, item)
    elif isinstance(value, dict):
        keys = sorted(value, key=_stable_mapping_key)
        digest.update(b'D' + len(keys).to_bytes(8, 'big'))
        for key in keys:
            _update_value_digest(digest, key)
            _update_value_digest(digest, value[key])
    else:
        raise ValueError(
            f'dataset sample contains unsupported value type {type(value).__name__}'
        )


def _stable_mapping_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    if not left.size or float(left.sum()) == 0.0 or float(right.sum()) == 0.0:
        return 1.0
    midpoint = (left + right) / 2.0

    def divergence(value: np.ndarray) -> float:
        mask = value > 0
        return float(np.sum(value[mask] * np.log2(value[mask] / midpoint[mask])))

    return (divergence(left) + divergence(right)) / 2.0


def _integer_distribution(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {
            'min': None,
            'p50': None,
            'p95': None,
            'max': None,
        }
    ordered = sorted(int(value) for value in values)
    return {
        'min': ordered[0],
        'p50': _nearest_rank(ordered, 0.50),
        'p95': _nearest_rank(ordered, 0.95),
        'max': ordered[-1],
    }


def _nearest_rank(ordered: Sequence[int], quantile: float) -> int:
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return int(ordered[index])


def _stable_float(value: float) -> float:
    return round(float(value), 12)


def _class_counts(records: Sequence[_SampleRecord]) -> dict[str, int]:
    counts = Counter(item.label for item in records)
    return {label: int(counts.get(label, 0)) for label in CANONICAL_LABELS}


def _split_counts(records: Sequence[_SampleRecord]) -> dict[str, int]:
    counts = Counter(item.split for item in records)
    return {split: int(counts[split]) for split in sorted(counts)}


def _split_label_counts(
    records: Sequence[_SampleRecord],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        counts[item.split][item.label] += 1
    return {
        split: {
            label: int(counts[split][label])
            for label in CANONICAL_LABELS
            if counts[split][label]
        }
        for split in sorted(counts)
    }


def _origin_split_label_counts(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for sample in samples:
        counts[str(sample['origin'])][str(sample['split'])][str(sample['label'])] += 1
    return {
        origin: {
            split: {
                label: int(counts[origin][split][label])
                for label in CANONICAL_LABELS
                if counts[origin][split][label]
            }
            for split in sorted(counts[origin])
        }
        for origin in sorted(counts)
    }


def _sorted_issues(issues: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for issue in issues:
        normalized = dict(issue)
        key = json.dumps(
            normalized, ensure_ascii=False, sort_keys=True, separators=(',', ':')
        )
        unique[key] = normalized
    return [unique[key] for key in sorted(unique)]


def _check_media_references(
    loaded: _LoadedDataset,
    sample: Mapping[str, Any],
    *,
    sample_id: str,
    hard_failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    for field in _MEDIA_FIELDS:
        raw = sample.get(field)
        if raw is None:
            continue
        if not isinstance(raw, (str, Path)) or not str(raw):
            hard_failures.append(
                {
                    'code': 'invalid-media-reference',
                    'dataset': loaded.role,
                    'field': field,
                    'sample_id': sample_id,
                }
            )
            continue
        path = Path(raw)
        if path.is_absolute():
            if not path.is_file():
                warnings.append(
                    {
                        'code': 'missing-absolute-media',
                        'dataset': loaded.role,
                        'field': field,
                        'sample_id': sample_id,
                    }
                )
            continue
        try:
            resolved = _resolve_relative_media(loaded.media_root, path)
        except ValueError:
            hard_failures.append(
                {
                    'code': 'unsafe-relative-media',
                    'dataset': loaded.role,
                    'field': field,
                    'sample_id': sample_id,
                }
            )
            continue
        if not resolved.is_file():
            hard_failures.append(
                {
                    'code': 'missing-relative-media',
                    'dataset': loaded.role,
                    'field': field,
                    'sample_id': sample_id,
                }
            )


def _resolve_relative_media(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'unsafe relative media path: {relative}')
    canonical_root = root.resolve(strict=True)
    resolved = (canonical_root / relative).resolve()
    if not resolved.is_relative_to(canonical_root):
        raise ValueError(f'relative media path escapes bundle: {relative}')
    return resolved


def _materialize_merged_media(
    records: Sequence[_SampleRecord], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    raw_samples: list[dict[str, Any]] = []
    bundle_samples: list[dict[str, Any]] = []
    media_paths: dict[str, Path] = {}
    media_dir = output_dir / 'media'

    for record in records:
        raw_sample = dict(record.sample)
        bundle_sample = dict(record.sample)
        for field in _MEDIA_FIELDS:
            raw_value = record.sample.get(field)
            if raw_value is None:
                continue
            source = Path(raw_value)
            if source.is_absolute():
                # Existing absolute baseline references are intentionally
                # retained; this avoids copying the large SFT media corpus.
                raw_sample[field] = str(source)
                bundle_sample[field] = str(source)
                continue
            source = _resolve_relative_media(record.media_root, source)
            if not source.is_file():
                raise ValueError(
                    f'relative media is missing for sample {record.sample_id!r}'
                )
            digest, _ = sha256_file(source)
            suffix = source.suffix.lower() or '.bin'
            relative = Path('media') / f'{digest}{suffix}'
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                existing_digest, _ = sha256_file(target)
                if existing_digest != digest:
                    raise ValueError(
                        f'media destination collision for {relative.as_posix()}'
                    )
            else:
                _copy_file_atomic(source, target)
            media_paths[relative.as_posix()] = target
            raw_sample[field] = str(target.resolve())
            bundle_sample[field] = relative.as_posix()
        raw_samples.append(raw_sample)
        bundle_samples.append(bundle_sample)
    if media_paths:
        media_dir.mkdir(parents=True, exist_ok=True)
    return raw_samples, bundle_samples, media_paths


def _merged_dataset_value(
    baseline_value: Mapping[str, Any],
    samples: Sequence[dict[str, Any]],
    records: Sequence[_SampleRecord],
    *,
    baseline_artifact_id: str,
    incremental_artifact_id: str,
    distribution_report_sha256: str,
    dataset_merge_plan_sha256: str,
) -> dict[str, Any]:
    value = {
        key: item
        for key, item in baseline_value.items()
        if key
        not in {'split', 'annotations', 'categories', 'label_map', 'label_mapping'}
    }
    splits: dict[str, list[str]] = defaultdict(list)
    for record in records:
        splits[record.split].append(record.sample_id)
    value.update(
        {
            'split': {key: splits[key] for key in sorted(splits)},
            'annotations': list(samples),
            'categories': list(CANONICAL_LABELS),
            'label_mapping': {
                index: label for index, label in enumerate(CANONICAL_LABELS)
            },
            'label_map': dict(_LABEL_TO_ID),
            'schema_version': DATASET_SCHEMA,
            'merge_review': {
                'baseline_artifact_id': baseline_artifact_id,
                'incremental_artifact_id': incremental_artifact_id,
                'distribution_report_sha256': distribution_report_sha256,
                'dataset_merge_plan_sha256': dataset_merge_plan_sha256,
            },
        }
    )
    return value


def _write_bundle(
    output_tar: Path,
    dataset: Mapping[str, Any],
    media_paths: Mapping[str, Path],
    output_dir: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix='.merged-bundle-'
    ) as temporary_name:
        root = Path(temporary_name)
        _atomic_pickle(root / 'rightarm-gesture-11.pkl', dict(dataset))
        for relative_name, source in sorted(media_paths.items()):
            target = root / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_file_atomic(source, target)
        deterministic_tar(root, output_tar)


def _copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp'
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_pickle(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _open_dataset(
    path: str | Path,
    *,
    artifact_id: str,
    role: str,
    work_dir: Path,
) -> Iterator[_LoadedDataset]:
    input_path = Path(path).resolve(strict=True)
    if not input_path.is_file():
        raise ValueError(f'{role} dataset input is not a file')
    digest, size = sha256_file(input_path)
    is_bundle = _is_tar_bundle(input_path)
    if is_bundle:
        with tempfile.TemporaryDirectory(
            dir=work_dir, prefix=f'.{role}-dataset-'
        ) as temporary_name:
            root = Path(temporary_name)
            safe_extract_tar(input_path, root)
            dataset_path = _find_bundle_pickle(root)
            value = _restricted_pickle_load(dataset_path)
            yield _LoadedDataset(
                role=role,
                artifact_id=artifact_id,
                input_path=input_path,
                input_format='tar',
                input_sha256=digest,
                input_size_bytes=size,
                dataset_path=dataset_path,
                media_root=dataset_path.parent,
                value=value,
            )
    else:
        value = _restricted_pickle_load(input_path)
        yield _LoadedDataset(
            role=role,
            artifact_id=artifact_id,
            input_path=input_path,
            input_format='pkl',
            input_sha256=digest,
            input_size_bytes=size,
            dataset_path=input_path,
            media_root=input_path.parent,
            value=value,
        )


def _is_tar_bundle(path: Path) -> bool:
    if path.suffix.lower() == '.pkl':
        return False
    try:
        return tarfile.is_tarfile(path)
    except OSError:
        return False


def _find_bundle_pickle(root: Path) -> Path:
    preferred = sorted(root.rglob('rightarm-gesture-11.pkl'))
    if len(preferred) == 1:
        return preferred[0]
    candidates = sorted(root.rglob('*.pkl'))
    if len(candidates) != 1:
        raise ValueError('dataset bundle must contain exactly one PKL')
    return candidates[0]


def _restricted_pickle_load(path: Path) -> dict[str, Any]:
    with path.open('rb') as stream:
        value = _RestrictedNumpyUnpickler(stream).load()
        if stream.read(1):
            raise ValueError('dataset pickle contains trailing data')
    if not isinstance(value, dict):
        raise ValueError('dataset pickle root must be an object')
    return value


def _validate_artifact_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')


def _normalize_reason_codes(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError('reason_codes must be a string array')
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError('reason_codes must contain non-empty strings')
        normalized.append(value.strip())
    return sorted(set(normalized))
