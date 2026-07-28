"""Independent validators for rightArmPerception task artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from openhands.mlops_loop.artifact_store import ArtifactStore
from openhands.mlops_loop.models import (
    TaskResult,
    TaskSpec,
    ValidationDisposition,
    ValidationResult,
)
from openhands.mlops_loop.validators import ContractValidator, ValidatorRegistry

RIGHTARM_LABELS = frozenset(
    {
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
    }
)

RIGHTARM_LABEL_ORDER = (
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


class JSONReportValidator(ContractValidator):
    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        report_kind: str,
        schema_version: str | None,
    ):
        super().__init__(artifact_store)
        self.report_kind = report_kind
        self.schema_version = schema_version

    def __call__(self, task: TaskSpec, result: TaskResult) -> ValidationResult:
        if result.status != 'succeeded':
            return super().__call__(task, result)
        reports = [
            item for item in result.artifact_candidates if item.kind == self.report_kind
        ]
        if len(reports) != 1:
            return self._invalid(f'exactly one {self.report_kind} artifact is required')
        try:
            report = self._load_json(reports[0].uri)
            if (
                self.schema_version is not None
                and report.get('schema_version') != self.schema_version
            ):
                raise ValueError(
                    f'expected schema_version {self.schema_version!r}, got '
                    f'{report.get("schema_version")!r}'
                )
            metrics = self.extract_metrics(report)
            self.validate_report(report)
        except (
            ArithmeticError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return self._invalid(f'{self.report_kind} is invalid: {exc}')
        # Replace Agent-claimed metrics with values recomputed/read from the
        # immutable machine report before evaluating gates.
        verified_result = result.model_copy(update={'metrics': metrics})
        return super().__call__(task, verified_result)

    def extract_metrics(self, report: dict[str, Any]) -> dict[str, Any]:
        metrics = report.get('metrics', {})
        if not isinstance(metrics, dict):
            raise ValueError('report metrics must be an object')
        return metrics

    def validate_report(self, report: dict[str, Any]) -> None:
        return None

    @staticmethod
    def _load_json(uri: str) -> dict[str, Any]:
        parsed = urlparse(uri)
        if parsed.scheme != 'file' or parsed.netloc not in {'', 'localhost'}:
            raise ValueError('staging validator requires a local file URI')
        value = json.loads(Path(unquote(parsed.path)).read_text())
        if not isinstance(value, dict):
            raise ValueError('report must be a JSON object')
        return value

    @staticmethod
    def _invalid(reason: str) -> ValidationResult:
        return ValidationResult(
            validation_id=f'validation:{uuid4().hex}',
            disposition=ValidationDisposition.TERMINAL,
            reasons=[reason],
        )


class DataInspectionValidator(JSONReportValidator):
    def __call__(self, task: TaskSpec, result: TaskResult) -> ValidationResult:
        if result.status != 'succeeded':
            return super().__call__(task, result)
        by_kind = {
            kind: [item for item in result.artifact_candidates if item.kind == kind]
            for kind in ('source-manifest', 'data-validation-report', 'raw-dataset')
        }
        if any(len(values) != 1 for values in by_kind.values()):
            return self._invalid(
                'data inspection requires exactly one source manifest, report, '
                'and raw dataset'
            )
        try:
            report = self._load_json(by_kind['data-validation-report'][0].uri)
            source = self._load_json(by_kind['source-manifest'][0].uri)
            if source != report.get('source_manifest'):
                raise ValueError(
                    'source manifest does not match the independently inspected report'
                )
            self._validate_source_manifest(source)
            self._validate_raw_dataset(
                by_kind['raw-dataset'][0].uri, source=source, report=report
            )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            tarfile.TarError,
        ) as exc:
            return self._invalid(f'landed dataset is invalid: {exc}')
        return super().__call__(task, result)

    def validate_report(self, report: dict[str, Any]) -> None:
        if int(report.get('object_count', 0)) < 1:
            raise ValueError('batch contains no source objects')
        annotations = report.get('annotations', {})
        if int(annotations.get('annotation_count', 0)) < 1:
            raise ValueError('batch contains no annotations')

    @staticmethod
    def _validate_source_manifest(source: dict[str, Any]) -> None:
        if source.get('schema_version') != 'rightarm.gesture.batch/v1':
            raise ValueError('source manifest schema_version is invalid')
        if source.get('label_ontology') != 'rightarm-gesture-11/v1':
            raise ValueError('source manifest label ontology is invalid')
        if not isinstance(source.get('batch_id'), str) or not source['batch_id']:
            raise ValueError('source manifest has no batch id')
        if not isinstance(source.get('objects'), list) or not source['objects']:
            raise ValueError('source manifest has no objects')
        for item in [*source['objects'], source.get('annotations')]:
            if not isinstance(item, dict):
                raise ValueError('source manifest object entry is invalid')
            digest = item.get('sha256')
            size = item.get('bytes', item.get('size_bytes'))
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(value not in '0123456789abcdef' for value in digest)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ValueError('source manifest object identity is invalid')

    @classmethod
    def _validate_raw_dataset(
        cls, uri: str, *, source: dict[str, Any], report: dict[str, Any]
    ) -> None:
        path = cls._local_path(uri, 'raw dataset')
        with tarfile.open(path, mode='r:*') as archive:
            members = {member.name: member for member in archive.getmembers()}
            for member in members.values():
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or '..' in relative.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f'unsafe raw dataset member: {member.name!r}')
            raw_manifest = cls._tar_json(archive, members, 'raw-dataset.manifest.json')
            if raw_manifest.get('schema_version') != 'rightarm.gesture.raw-dataset/v1':
                raise ValueError('raw dataset manifest schema_version is invalid')
            if raw_manifest.get('source_manifest') != source:
                raise ValueError('raw dataset does not retain its source manifest')
            objects = raw_manifest.get('objects')
            if not isinstance(objects, list) or len(objects) != report['object_count']:
                raise ValueError('raw dataset object count does not match inspection')
            for item in objects:
                cls._verify_tar_record(archive, members, item)
            annotations = raw_manifest.get('annotations')
            if not isinstance(annotations, dict):
                raise ValueError('raw dataset annotations record is invalid')
            cls._verify_tar_record(archive, members, annotations)

    @staticmethod
    def _local_path(uri: str, label: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != 'file' or parsed.netloc not in {'', 'localhost'}:
            raise ValueError(f'{label} validator requires a local staging URI')
        return Path(unquote(parsed.path)).resolve(strict=True)

    @staticmethod
    def _tar_json(
        archive: tarfile.TarFile,
        members: dict[str, tarfile.TarInfo],
        name: str,
    ) -> dict[str, Any]:
        member = members.get(name)
        if member is None or not member.isfile():
            raise ValueError(f'raw dataset is missing {name}')
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f'raw dataset cannot read {name}')
        value = json.loads(stream.read())
        if not isinstance(value, dict):
            raise ValueError(f'{name} must be a JSON object')
        return value

    @staticmethod
    def _verify_tar_record(
        archive: tarfile.TarFile,
        members: dict[str, tarfile.TarInfo],
        record: dict[str, Any],
    ) -> None:
        name = record.get('path')
        if not isinstance(name, str) or name not in members:
            raise ValueError(f'raw dataset is missing declared object {name!r}')
        member = members[name]
        if not member.isfile():
            raise ValueError(f'raw dataset object is not a file: {name!r}')
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f'raw dataset cannot read object: {name!r}')
        digest = hashlib.sha256()
        size = 0
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
        declared_size = record.get('size_bytes', record.get('bytes'))
        if digest.hexdigest() != record.get('sha256') or size != declared_size:
            raise ValueError(f'raw dataset object hash mismatch: {name!r}')

    def extract_metrics(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            'object_count': int(report['object_count']),
            'annotation_count': int(report['annotations']['annotation_count']),
            'duplicate_object_count': int(report.get('duplicate_object_count', 0)),
            'skip_training': bool(report.get('skip_training', False)),
        }


class DatasetValidator(JSONReportValidator):
    def validate_report(self, report: dict[str, Any]) -> None:
        if int(report.get('train_samples', 0)) < 1:
            raise ValueError('compiled dataset has no training samples')
        if int(report.get('val_samples', 0)) < 1:
            raise ValueError('compiled dataset has no validation samples')
        frozen = report.get('frozen_test_set')
        if frozen is not None and not frozen.get('excluded_from_training_splits'):
            raise ValueError('frozen test set was not excluded from training splits')

    def extract_metrics(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            'train_samples': int(report['train_samples']),
            'val_samples': int(report['val_samples']),
            'low_quality_samples': int(report.get('low_quality_samples', 0)),
        }


class DatasetDistributionReviewValidator(ContractValidator):
    """Validate the machine profile, Agent plan, and resulting merged dataset."""

    _KINDS = (
        'dataset-distribution-report',
        'dataset-merge-plan',
        'dataset-merge-report',
        'mmaction-dataset',
    )
    _SCHEMAS = {
        'dataset-distribution-report': (
            'rightarm.gesture.dataset-distribution-report/v1'
        ),
        'dataset-merge-plan': 'rightarm.gesture.dataset-merge-plan/v1',
        'dataset-merge-report': 'rightarm.gesture.dataset-merge-report/v1',
    }
    _ORIGINS = frozenset({'baseline', 'new-batch'})
    _SPLITS = frozenset({'train', 'val', 'test'})

    def __call__(self, task: TaskSpec, result: TaskResult) -> ValidationResult:
        if result.status != 'succeeded':
            return super().__call__(task, result)
        by_kind = {
            kind: [
                candidate
                for candidate in result.artifact_candidates
                if candidate.kind == kind
            ]
            for kind in self._KINDS
        }
        invalid_cardinality = [
            kind for kind, candidates in by_kind.items() if len(candidates) != 1
        ]
        if invalid_cardinality:
            return JSONReportValidator._invalid(
                'dataset distribution review requires exactly one artifact for '
                f'each completion kind; invalid kinds: {invalid_cardinality}'
            )

        try:
            distribution_candidate = by_kind['dataset-distribution-report'][0]
            plan_candidate = by_kind['dataset-merge-plan'][0]
            merge_candidate = by_kind['dataset-merge-report'][0]
            dataset_candidate = by_kind['mmaction-dataset'][0]
            distribution = JSONReportValidator._load_json(distribution_candidate.uri)
            plan = JSONReportValidator._load_json(plan_candidate.uri)
            merge = JSONReportValidator._load_json(merge_candidate.uri)
            for kind, document in (
                ('dataset-distribution-report', distribution),
                ('dataset-merge-plan', plan),
                ('dataset-merge-report', merge),
            ):
                expected = self._SCHEMAS[kind]
                if document.get('schema_version') != expected:
                    raise ValueError(f'{kind} schema_version must be {expected!r}')

            if plan.get('distribution_report_sha256') != distribution_candidate.sha256:
                raise ValueError(
                    'dataset merge plan does not bind the distribution report hash'
                )
            if merge.get('distribution_report_sha256') != distribution_candidate.sha256:
                raise ValueError(
                    'dataset merge report does not bind the distribution report hash'
                )
            if merge.get('dataset_merge_plan_sha256') != plan_candidate.sha256:
                raise ValueError(
                    'dataset merge report does not bind the dataset merge plan hash'
                )

            distribution_inputs = self._input_identities(distribution)
            if self._input_identities(plan) != distribution_inputs:
                raise ValueError(
                    'dataset merge plan input identities differ from the '
                    'distribution report'
                )
            if self._input_identities(merge) != distribution_inputs:
                raise ValueError(
                    'dataset merge report input identities differ from the '
                    'distribution report'
                )

            sample_counts, origin_counts = self._validate_distribution(distribution)
            if distribution.get('hard_failures') != []:
                raise ValueError('dataset distribution report has hard failures')
            recommendation = distribution.get('recommendation')
            if (
                not isinstance(recommendation, dict)
                or recommendation.get('action') != 'merge-all-unique'
            ):
                raise ValueError(
                    'dataset distribution recommendation must be merge-all-unique'
                )
            if plan.get('decision') != 'merge-all-unique':
                raise ValueError('dataset merge plan decision must be merge-all-unique')
            if plan.get('requires_human_decision') is not False:
                raise ValueError('dataset merge plan must not require a human decision')

            output = merge.get('output')
            if not isinstance(output, dict):
                raise ValueError('dataset merge report output must be an object')
            if (
                output.get('sha256') != dataset_candidate.sha256
                or output.get('size_bytes') != dataset_candidate.size_bytes
            ):
                raise ValueError(
                    'dataset merge report output does not match the final '
                    'mmaction dataset artifact'
                )
            output_count = self._validate_output_counts(output)
            missing_train_classes = sum(
                1
                for label in RIGHTARM_LABEL_ORDER
                if not any(
                    origin_counts.get(origin, {}).get('train', {}).get(label, 0)
                    for origin in self._ORIGINS
                )
            )
            metrics = {
                'distribution.sample_count': sample_counts,
                'distribution.baseline_sample_count': sum(
                    sum(labels.values())
                    for labels in origin_counts.get('baseline', {}).values()
                ),
                'distribution.new_batch_sample_count': sum(
                    sum(labels.values())
                    for labels in origin_counts.get('new-batch', {}).values()
                ),
                'distribution.hard_failure_count': 0,
                'distribution.missing_train_class_count': missing_train_classes,
                'merge.plan_actionable': True,
                'merge.requires_human_decision': False,
                'merge.output_sample_count': output_count,
            }
        except (
            ArithmeticError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return JSONReportValidator._invalid(
                f'dataset distribution review is invalid: {exc}'
            )

        verified_result = result.model_copy(update={'metrics': metrics})
        return super().__call__(task, verified_result)

    @classmethod
    def _input_identities(cls, document: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
        raw_inputs = document.get('inputs')
        if not isinstance(raw_inputs, list) or len(raw_inputs) != 2:
            raise ValueError(
                'dataset review inputs must contain baseline and new-batch'
            )
        identities: dict[str, tuple[Any, ...]] = {}
        for item in raw_inputs:
            if not isinstance(item, dict):
                raise ValueError('dataset review input identity must be an object')
            origin = cls._origin(item.get('role'))
            artifact_id = item.get('artifact_id')
            digest = item.get('sha256')
            size = item.get('size_bytes')
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError('dataset review input artifact_id is invalid')
            if not cls._is_sha256(digest):
                raise ValueError('dataset review input sha256 is invalid')
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError('dataset review input size_bytes is invalid')
            if origin in identities:
                raise ValueError(f'duplicate dataset review input role: {origin}')
            identities[origin] = (origin, artifact_id, digest, size)
        if set(identities) != cls._ORIGINS:
            raise ValueError(
                'dataset review inputs must contain baseline and new-batch'
            )
        return tuple(identities[origin] for origin in sorted(identities))

    @classmethod
    def _validate_distribution(
        cls, report: dict[str, Any]
    ) -> tuple[int, dict[str, dict[str, dict[str, int]]]]:
        if report.get('canonical_categories') != list(RIGHTARM_LABEL_ORDER):
            raise ValueError('dataset distribution categories are not canonical')
        samples = report.get('samples')
        if not isinstance(samples, list) or not samples:
            raise ValueError('dataset distribution report contains no samples')
        counts: dict[str, dict[str, dict[str, int]]] = {}
        seen_ids: set[str] = set()
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError('dataset distribution sample must be an object')
            sample_id = sample.get('id')
            origin = cls._origin(sample.get('origin'))
            split = sample.get('split')
            label = sample.get('label')
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError('dataset distribution sample id is invalid')
            if sample_id in seen_ids:
                raise ValueError(
                    f'dataset distribution sample id is not unique: {sample_id!r}'
                )
            seen_ids.add(sample_id)
            if split not in cls._SPLITS:
                raise ValueError(
                    f'dataset distribution sample split is invalid: {split!r}'
                )
            if label not in RIGHTARM_LABELS:
                raise ValueError(
                    f'dataset distribution sample label is invalid: {label!r}'
                )
            split_counts = counts.setdefault(origin, {}).setdefault(split, {})
            split_counts[label] = split_counts.get(label, 0) + 1

        summary = report.get('summary')
        if not isinstance(summary, dict):
            raise ValueError('dataset distribution summary must be an object')
        declared = cls._normalize_origin_counts(summary.get('counts'))
        if declared != counts:
            raise ValueError(
                'dataset distribution summary counts do not match its samples'
            )
        return len(samples), counts

    @classmethod
    def _normalize_origin_counts(
        cls, raw_counts: Any
    ) -> dict[str, dict[str, dict[str, int]]]:
        if not isinstance(raw_counts, dict):
            raise ValueError('dataset distribution summary counts must be an object')
        normalized: dict[str, dict[str, dict[str, int]]] = {}
        for raw_origin, raw_splits in raw_counts.items():
            origin = cls._origin(raw_origin)
            if origin in normalized:
                raise ValueError(f'duplicate distribution count origin: {origin}')
            normalized[origin] = cls._normalize_split_counts(raw_splits)
            if not normalized[origin]:
                normalized.pop(origin)
        return normalized

    @classmethod
    def _normalize_split_counts(cls, raw_splits: Any) -> dict[str, dict[str, int]]:
        if not isinstance(raw_splits, dict):
            raise ValueError('dataset split counts must be an object')
        normalized: dict[str, dict[str, int]] = {}
        for split, raw_labels in raw_splits.items():
            if split not in cls._SPLITS:
                raise ValueError(f'dataset count split is invalid: {split!r}')
            if not isinstance(raw_labels, dict):
                raise ValueError('dataset label counts must be an object')
            labels: dict[str, int] = {}
            for label, count in raw_labels.items():
                if label not in RIGHTARM_LABELS:
                    raise ValueError(f'dataset count label is invalid: {label!r}')
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError('dataset label count must be non-negative')
                if count:
                    labels[label] = count
            if labels:
                normalized[split] = labels
        return normalized

    @classmethod
    def _validate_output_counts(cls, output: dict[str, Any]) -> int:
        sample_count = output.get('sample_count')
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 1
        ):
            raise ValueError('dataset merge output sample_count must be positive')
        counts = cls._normalize_split_counts(output.get('counts'))
        counted = sum(sum(labels.values()) for labels in counts.values())
        if counted != sample_count:
            raise ValueError(
                'dataset merge output counts do not match its sample_count'
            )
        return sample_count

    @classmethod
    def _origin(cls, value: Any) -> str:
        if value == 'new':
            return 'new-batch'
        if value not in cls._ORIGINS:
            raise ValueError(f'dataset sample origin is invalid: {value!r}')
        return value

    @staticmethod
    def _is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in '0123456789abcdef' for character in value)
        )


class ModelImprovementValidator(ContractValidator):
    """Independently validate a bounded train/evaluate/improve Conversation."""

    _SINGLE_KINDS = (
        'baseline-evaluation-report',
        'checkpoint',
        'training-config',
        'training-log',
        'training-metrics',
        'evaluation-report',
        'model-improvement-report',
    )
    _TRIAL_KIND = 'optimization-trial-report'
    _EVALUATION_SCHEMA = 'rightarm.gesture.checkpoint-evaluation/v2'
    _TRIAL_SCHEMA = 'rightarm.gesture.optimization-trial/v1'
    _FINAL_SCHEMA = 'rightarm.gesture.model-improvement/v1'
    _PREPROCESSING_VERSION = 'rightarm.gesture.preprocessing/v1'
    _ALLOWED_STRATEGIES = frozenset(
        {
            'fine-tune',
            'optimizer',
            'regularization',
            'augmentation',
            'motion-ablation',
            'gesture-gate',
        }
    )
    _ALLOWED_STOP_REASONS = frozenset(
        {
            'target-reached',
            'patience-exhausted',
            'trial-budget-exhausted',
        }
    )
    _ALLOWED_DECISIONS = frozenset({'promote-challenger', 'keep-champion'})

    def __call__(self, task: TaskSpec, result: TaskResult) -> ValidationResult:
        if result.status != 'succeeded':
            return super().__call__(task, result)
        by_kind = {
            kind: [
                candidate
                for candidate in result.artifact_candidates
                if candidate.kind == kind
            ]
            for kind in (*self._SINGLE_KINDS, self._TRIAL_KIND)
        }
        invalid_single = [
            kind for kind in self._SINGLE_KINDS if len(by_kind[kind]) != 1
        ]
        trial_candidates = by_kind[self._TRIAL_KIND]
        if invalid_single or not 1 <= len(trial_candidates) <= 8:
            return JSONReportValidator._invalid(
                'model improvement requires exactly one baseline evaluation, '
                'selected checkpoint/config/log/metrics/evaluation/final report '
                'and between one and eight optimization trial reports; '
                f'invalid singleton kinds: {invalid_single}, '
                f'trial count: {len(trial_candidates)}'
            )

        try:
            baseline_candidate = by_kind['baseline-evaluation-report'][0]
            selected_evaluation_candidate = by_kind['evaluation-report'][0]
            final_candidate = by_kind['model-improvement-report'][0]
            baseline = JSONReportValidator._load_json(baseline_candidate.uri)
            selected_evaluation = JSONReportValidator._load_json(
                selected_evaluation_candidate.uri
            )
            final = JSONReportValidator._load_json(final_candidate.uri)
            baseline_facts = self._validate_evaluation(baseline, role='baseline')
            selected_evaluation_facts = self._validate_evaluation(
                selected_evaluation, role='selected'
            )
            self._require_same_evaluation_boundary(
                baseline_facts, selected_evaluation_facts
            )
            trials = self._validate_trials(
                trial_candidates, baseline_facts=baseline_facts
            )
            metrics = self._validate_final(
                final,
                final_candidate=final_candidate,
                baseline=baseline,
                baseline_candidate=baseline_candidate,
                baseline_facts=baseline_facts,
                trials=trials,
                selected_evaluation=selected_evaluation,
                selected_evaluation_candidate=selected_evaluation_candidate,
                selected_evaluation_facts=selected_evaluation_facts,
                selected_candidates={
                    kind: by_kind[kind][0]
                    for kind in (
                        'checkpoint',
                        'training-config',
                        'training-log',
                        'training-metrics',
                    )
                },
            )
        except (
            ArithmeticError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return JSONReportValidator._invalid(
                f'model improvement result is invalid: {exc}'
            )

        # Never use Agent-claimed scores or promotion decisions for metric gates.
        verified = result.model_copy(update={'metrics': metrics})
        return super().__call__(task, verified)

    @classmethod
    def _validate_evaluation(
        cls, report: dict[str, Any], *, role: str
    ) -> dict[str, Any]:
        if report.get('schema_version') != cls._EVALUATION_SCHEMA:
            raise ValueError(
                f'{role} evaluation schema_version must be {cls._EVALUATION_SCHEMA!r}'
            )
        labels = report.get('labels')
        if labels != list(RIGHTARM_LABEL_ORDER):
            raise ValueError(
                f'{role} evaluation labels must use the canonical 11-class order'
            )
        identity = cls._require_dict(report.get('identity'), f'{role} identity')
        for prefix in ('dataset', 'checkpoint', 'config'):
            cls._require_sha(
                identity.get(f'{prefix}_sha256'), f'{role} {prefix} sha256'
            )
            cls._require_size(
                identity.get(f'{prefix}_size_bytes'), f'{role} {prefix} size'
            )
        split = identity.get('evaluation_split')
        if split != 'val':
            raise ValueError(f'{role} evaluation must use immutable split "val"')
        if identity.get('preprocessing_version') != cls._PREPROCESSING_VERSION:
            raise ValueError(f'{role} evaluation preprocessing version is invalid')
        cls._require_sha(
            identity.get('preprocessing_fingerprint'),
            f'{role} preprocessing fingerprint',
        )

        confusion = cls._validate_confusion(report.get('confusion_matrix'), role=role)
        recomputed = cls._metrics_from_confusion(confusion)
        declared_metrics = cls._require_dict(
            report.get('metrics'), f'{role} evaluation metrics'
        )
        for name, expected in recomputed.items():
            if name == 'per_class':
                continue
            cls._require_metric_equal(
                declared_metrics.get(name),
                expected,
                f'{role} evaluation metric {name}',
            )

        per_class = cls._require_dict(
            report.get('per_class'), f'{role} per-class metrics'
        )
        if set(per_class) != RIGHTARM_LABELS:
            raise ValueError(f'{role} evaluation per-class metrics are incomplete')
        for label, expected in recomputed['per_class'].items():
            actual = cls._require_dict(
                per_class.get(label), f'{role} per-class metrics for {label}'
            )
            for name, value in expected.items():
                cls._require_metric_equal(
                    actual.get(name),
                    value,
                    f'{role} {label} {name}',
                )

        evidence = cls._require_dict(
            report.get('evidence'), f'{role} evaluation evidence'
        )
        for prefix in ('test_log', 'confusion_matrix'):
            cls._require_sha(
                evidence.get(f'{prefix}_sha256'), f'{role} {prefix} evidence sha256'
            )
            cls._require_size(
                evidence.get(f'{prefix}_size_bytes'),
                f'{role} {prefix} evidence size',
            )
        return {
            'identity': identity,
            'metrics': {
                key: value for key, value in recomputed.items() if key != 'per_class'
            },
            'per_class': recomputed['per_class'],
        }

    @classmethod
    def _validate_trials(
        cls,
        candidates: list[Any],
        *,
        baseline_facts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        trials: list[dict[str, Any]] = []
        for candidate in candidates:
            report = JSONReportValidator._load_json(candidate.uri)
            if report.get('schema_version') != cls._TRIAL_SCHEMA:
                raise ValueError(
                    f'optimization trial schema_version must be {cls._TRIAL_SCHEMA!r}'
                )
            trial_number = report.get('trial_number')
            if (
                not isinstance(trial_number, int)
                or isinstance(trial_number, bool)
                or not 1 <= trial_number <= 8
            ):
                raise ValueError('optimization trial_number must be in [1, 8]')
            strategy = report.get('strategy')
            if strategy not in cls._ALLOWED_STRATEGIES:
                raise ValueError(
                    f'optimization trial strategy is not allowlisted: {strategy!r}'
                )
            parameters = cls._validate_trial_parameters(report.get('parameters'))
            inputs = cls._require_dict(
                report.get('input'), f'trial {trial_number} input'
            )
            cls._validate_trial_input(
                inputs, baseline_facts=baseline_facts, trial_number=trial_number
            )
            outputs = cls._require_dict(
                report.get('outputs'), f'trial {trial_number} outputs'
            )
            for prefix in (
                'checkpoint',
                'training_config',
                'training_log',
                'training_metrics',
                'evaluation_report',
            ):
                cls._require_sha(
                    outputs.get(f'{prefix}_sha256'),
                    f'trial {trial_number} {prefix} sha256',
                )
                cls._require_size(
                    outputs.get(f'{prefix}_size_bytes'),
                    f'trial {trial_number} {prefix} size',
                )

            evaluation = cls._require_dict(
                report.get('evaluation'), f'trial {trial_number} evaluation'
            )
            evaluation_facts = cls._validate_evaluation(
                evaluation, role=f'trial {trial_number}'
            )
            cls._require_same_evaluation_boundary(baseline_facts, evaluation_facts)
            if evaluation_facts['identity']['dataset_sha256'] != inputs.get(
                'dataset_sha256'
            ) or evaluation_facts['identity']['dataset_size_bytes'] != inputs.get(
                'dataset_size_bytes'
            ):
                raise ValueError(
                    f'trial {trial_number} evaluation does not bind its dataset input'
                )
            for identity_name, output_name in (
                ('checkpoint', 'checkpoint'),
                ('config', 'training_config'),
            ):
                if (
                    evaluation_facts['identity'][f'{identity_name}_sha256']
                    != outputs[f'{output_name}_sha256']
                    or evaluation_facts['identity'][f'{identity_name}_size_bytes']
                    != outputs[f'{output_name}_size_bytes']
                ):
                    raise ValueError(
                        f'trial {trial_number} evaluation does not bind its '
                        f'{output_name} output'
                    )
            cls._validate_evaluation_candidates(
                report,
                trial_number=trial_number,
                selected=evaluation_facts,
                selected_report_sha256=outputs['evaluation_report_sha256'],
            )
            trials.append(
                {
                    'candidate': candidate,
                    'document': report,
                    'trial_number': trial_number,
                    'strategy': strategy,
                    'parameters': parameters,
                    'inputs': inputs,
                    'outputs': outputs,
                    'evaluation': evaluation,
                    'evaluation_facts': evaluation_facts,
                }
            )

        trials.sort(key=lambda item: item['trial_number'])
        numbers = [item['trial_number'] for item in trials]
        if numbers != list(range(1, len(trials) + 1)):
            raise ValueError(
                'optimization trial numbers must be unique and contiguous from 1'
            )
        return trials

    @classmethod
    def _validate_trial_input(
        cls,
        inputs: dict[str, Any],
        *,
        baseline_facts: dict[str, Any],
        trial_number: int,
    ) -> None:
        for prefix in ('dataset', 'base_model', 'initialization_checkpoint'):
            cls._require_sha(
                inputs.get(f'{prefix}_sha256'),
                f'trial {trial_number} {prefix} sha256',
            )
            cls._require_size(
                inputs.get(f'{prefix}_size_bytes'),
                f'trial {trial_number} {prefix} size',
            )
        baseline_identity = baseline_facts['identity']
        if (
            inputs['dataset_sha256'] != baseline_identity['dataset_sha256']
            or inputs['dataset_size_bytes'] != baseline_identity['dataset_size_bytes']
        ):
            raise ValueError(
                f'trial {trial_number} dataset differs from baseline evaluation'
            )
        if (
            inputs['base_model_sha256'] != baseline_identity['checkpoint_sha256']
            or inputs['base_model_size_bytes']
            != baseline_identity['checkpoint_size_bytes']
        ):
            raise ValueError(
                f'trial {trial_number} base model differs from the incumbent'
            )
        if inputs.get('evaluation_split') != baseline_identity['evaluation_split']:
            raise ValueError(f'trial {trial_number} evaluation split changed')
        if (
            inputs.get('preprocessing_version')
            != baseline_identity['preprocessing_version']
            or inputs.get('preprocessing_fingerprint')
            != baseline_identity['preprocessing_fingerprint']
        ):
            raise ValueError(f'trial {trial_number} preprocessing changed')

    @classmethod
    def _validate_evaluation_candidates(
        cls,
        report: dict[str, Any],
        *,
        trial_number: int,
        selected: dict[str, Any],
        selected_report_sha256: str,
    ) -> None:
        candidates = report.get('evaluation_candidates')
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                f'trial {trial_number} evaluation_candidates must be non-empty'
            )
        roles: set[str] = set()
        identities: set[tuple[str, str]] = set()
        normalized: dict[str, dict[str, Any]] = {}
        for item in candidates:
            item = cls._require_dict(item, f'trial {trial_number} evaluation candidate')
            role = item.get('role')
            if role not in {'best', 'last', 'retained'} or role in roles:
                raise ValueError(
                    f'trial {trial_number} evaluation candidate role is invalid'
                )
            roles.add(role)
            cls._require_sha(
                item.get('checkpoint_sha256'),
                f'trial {trial_number} evaluation candidate checkpoint',
            )
            cls._require_sha(
                item.get('evaluation_report_sha256'),
                f'trial {trial_number} evaluation candidate report',
            )
            metrics = cls._require_dict(
                item.get('metrics'),
                f'trial {trial_number} evaluation candidate metrics',
            )
            for name in (
                'offline.sample_count',
                'offline.accuracy',
                'offline.macro_f1',
                'offline.min_per_class_recall',
                'offline.none_false_accept_rate',
            ):
                value = metrics.get(name)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f'trial {trial_number} candidate metric {name} is invalid'
                    )
            identity = (
                item['checkpoint_sha256'],
                item['evaluation_report_sha256'],
            )
            if identity in identities:
                raise ValueError(
                    f'trial {trial_number} evaluation candidates are not deduplicated'
                )
            identities.add(identity)
            normalized[role] = {**item, 'metrics': metrics}
        selection = cls._require_dict(
            report.get('selection'), f'trial {trial_number} selection'
        )
        if selection.get('policy') != 'rightarm.gesture.product-selection/v1':
            raise ValueError(f'trial {trial_number} selection policy is invalid')
        winner_role = selection.get('winner_role')
        if winner_role not in normalized:
            raise ValueError(f'trial {trial_number} winner role is not a candidate')
        expected_winner_role = max(
            normalized,
            key=lambda role: cls._evaluation_candidate_rank(
                normalized[role], role=role
            ),
        )
        if winner_role != expected_winner_role:
            raise ValueError(
                f'trial {trial_number} winner is not selected deterministically'
            )
        winner = normalized[winner_role]
        if (
            winner['checkpoint_sha256'] != selected['identity']['checkpoint_sha256']
            or winner['evaluation_report_sha256'] != selected_report_sha256
        ):
            raise ValueError(
                f'trial {trial_number} selected evaluation identity differs '
                'from its winner candidate'
            )
        for name, value in selected['metrics'].items():
            cls._require_metric_equal(
                winner['metrics'].get(name),
                value,
                f'trial {trial_number} selected candidate metric {name}',
            )
        reason_codes = selection.get('reason_codes')
        if reason_codes != [
            'none-far-feasibility-first',
            'macro-f1-primary',
            'min-recall-secondary',
            'accuracy-tertiary',
        ]:
            raise ValueError(f'trial {trial_number} selection reason_codes invalid')

    @staticmethod
    def _evaluation_candidate_rank(
        candidate: dict[str, Any], *, role: str
    ) -> tuple[bool, float, float, float, float, int]:
        metrics = candidate['metrics']
        role_priority = {'retained': 0, 'last': 1, 'best': 2}
        return (
            metrics['offline.none_false_accept_rate'] <= 0.01,
            float(metrics['offline.macro_f1']),
            float(metrics['offline.min_per_class_recall']),
            float(metrics['offline.accuracy']),
            -float(metrics['offline.none_false_accept_rate']),
            role_priority[role],
        )

    @classmethod
    def _validate_trial_parameters(cls, raw: Any) -> dict[str, Any]:
        parameters = cls._require_dict(raw, 'optimization trial parameters')
        allowed = {
            'lr',
            'weight_decay',
            'dropout',
            'gate_threshold',
            'gate_loss_weight',
            'contrast_loss_weight',
            'augmentation_preset',
            'motion_preset',
            'batch_size',
            'epochs',
            'seed',
        }
        if set(parameters) != allowed:
            raise ValueError(
                'optimization trial parameters must be the complete normalized '
                f'allowlist; missing={sorted(allowed - set(parameters))}, '
                f'unknown={sorted(set(parameters) - allowed)}'
            )
        for name, (minimum, maximum) in {
            'epochs': (1, 100),
            'seed': (0, 2_147_483_647),
        }.items():
            value = parameters[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f'optimization parameter {name} is outside its bounded range'
                )
        batch_size = parameters['batch_size']
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size not in {4, 8, 16, 32, 64}
        ):
            raise ValueError(
                'optimization parameter batch_size is outside its allowlist'
            )
        for name, (lower_bound, upper_bound, inclusive_minimum) in {
            'lr': (0.00001, 0.05, True),
            'weight_decay': (0.0, 0.01, True),
            'dropout': (0.0, 0.75, True),
            'gate_threshold': (0.2, 0.8, True),
            'gate_loss_weight': (0.05, 2.0, True),
            'contrast_loss_weight': (0.0, 0.1, True),
        }.items():
            numeric_value = parameters[name]
            if (
                not isinstance(numeric_value, (int, float))
                or isinstance(numeric_value, bool)
                or not math.isfinite(float(numeric_value))
                or (
                    float(numeric_value) < lower_bound
                    if inclusive_minimum
                    else float(numeric_value) <= lower_bound
                )
                or float(numeric_value) > upper_bound
            ):
                raise ValueError(
                    f'optimization parameter {name} is outside its bounded range'
                )
        if parameters['augmentation_preset'] not in {
            'standard',
            'light',
            'strong',
            'none',
        }:
            raise ValueError(
                'optimization augmentation_preset is outside its allowlist'
            )
        if parameters['motion_preset'] not in {'standard', 'no-delta-z'}:
            raise ValueError('optimization motion_preset is outside its allowlist')
        return parameters

    @classmethod
    def _validate_final(
        cls,
        report: dict[str, Any],
        *,
        final_candidate: Any,
        baseline: dict[str, Any],
        baseline_candidate: Any,
        baseline_facts: dict[str, Any],
        trials: list[dict[str, Any]],
        selected_evaluation: dict[str, Any],
        selected_evaluation_candidate: Any,
        selected_evaluation_facts: dict[str, Any],
        selected_candidates: dict[str, Any],
    ) -> dict[str, Any]:
        if report.get('schema_version') != cls._FINAL_SCHEMA:
            raise ValueError(
                f'model improvement report schema_version must be {cls._FINAL_SCHEMA!r}'
            )
        if report.get('baseline_evaluation') != baseline:
            raise ValueError(
                'model improvement report embedded baseline evaluation differs '
                'from the staged baseline artifact'
            )
        if report.get('baseline_evaluation_sha256') != baseline_candidate.sha256:
            raise ValueError(
                'model improvement report does not bind the baseline evaluation hash'
            )

        policy = cls._validate_policy(report.get('policy'), trial_count=len(trials))
        for trial in trials:
            if trial['parameters']['epochs'] > policy['max_epochs']:
                raise ValueError(
                    f'trial {trial["trial_number"]} exceeds max_epochs policy'
                )
            if trial['parameters']['batch_size'] > policy['batch_size']:
                raise ValueError(
                    f'trial {trial["trial_number"]} exceeds batch_size policy'
                )

        trial_hashes = [trial['candidate'].sha256 for trial in trials]
        if report.get('trial_report_sha256s') != trial_hashes:
            raise ValueError(
                'model improvement report history does not cover every staged '
                'trial report in trial order'
            )
        cls._validate_final_trial_history(report.get('trials'), trials=trials)

        baseline_metrics = baseline_facts['metrics']
        eligible = [
            trial
            for trial in trials
            if cls._is_trial_guardrail_eligible(
                trial['evaluation_facts']['metrics'], baseline_metrics
            )
        ]
        best_trial = max(
            eligible or trials,
            key=cls._trial_rank,
        )
        best_metrics = best_trial['evaluation_facts']['metrics']
        macro_delta = (
            best_metrics['offline.macro_f1'] - baseline_metrics['offline.macro_f1']
        )
        min_recall_delta = (
            best_metrics['offline.min_per_class_recall']
            - baseline_metrics['offline.min_per_class_recall']
        )
        none_delta = (
            best_metrics['offline.none_false_accept_rate']
            - baseline_metrics['offline.none_false_accept_rate']
        )
        accuracy_delta = (
            best_metrics['offline.accuracy'] - baseline_metrics['offline.accuracy']
        )
        promotion_eligible = (
            best_trial in eligible and macro_delta >= policy['min_improvement']
        )
        expected_decision = (
            'promote-challenger' if promotion_eligible else 'keep-champion'
        )
        decision = report.get('decision')
        if decision not in cls._ALLOWED_DECISIONS or decision != expected_decision:
            raise ValueError(
                'model improvement decision does not match independently '
                'recomputed promotion eligibility'
            )

        cls._validate_best_trial(report.get('best_trial'), best_trial=best_trial)
        cls._validate_deltas(
            report.get('deltas_vs_baseline'),
            accuracy_delta=accuracy_delta,
            macro_delta=macro_delta,
            min_recall_delta=min_recall_delta,
            none_delta=none_delta,
        )
        stop_reason = report.get('stop_reason')
        cls._validate_stop_reason(
            stop_reason,
            trials=trials,
            eligible=eligible,
            baseline_metrics=baseline_metrics,
            policy=policy,
            best_macro_delta=macro_delta,
        )
        reason_codes = report.get('reason_codes')
        expected_reason_codes = (
            [
                'macro-f1-minimum-improvement-reached',
                'minimum-recall-guard-passed',
                'none-far-guard-passed',
            ]
            if decision == 'promote-challenger'
            else ['challenger-did-not-pass-product-selection-policy']
        )
        if reason_codes != expected_reason_codes:
            raise ValueError('model improvement reason_codes are invalid')
        cls._validate_analysis(report.get('analysis'))

        winner = cls._require_dict(report.get('winner'), 'model improvement winner')
        cls._validate_winner(
            winner,
            decision=decision,
            baseline_candidate=baseline_candidate,
            baseline_facts=baseline_facts,
            best_trial=best_trial,
        )

        best_outputs = best_trial['outputs']
        for kind, prefix in (
            ('training-config', 'training_config'),
            ('training-log', 'training_log'),
            ('training-metrics', 'training_metrics'),
        ):
            cls._assert_artifact_identity(
                selected_candidates[kind],
                sha256=best_outputs[f'{prefix}_sha256'],
                size_bytes=best_outputs[f'{prefix}_size_bytes'],
                label=f'selected {kind}',
            )

        if decision == 'promote-challenger':
            cls._assert_artifact_identity(
                selected_candidates['checkpoint'],
                sha256=best_outputs['checkpoint_sha256'],
                size_bytes=best_outputs['checkpoint_size_bytes'],
                label='selected checkpoint',
            )
            cls._assert_artifact_identity(
                selected_evaluation_candidate,
                sha256=best_outputs['evaluation_report_sha256'],
                size_bytes=best_outputs['evaluation_report_size_bytes'],
                label='selected evaluation',
            )
            if selected_evaluation != best_trial['evaluation']:
                raise ValueError(
                    'selected evaluation artifact differs from deterministic '
                    'best trial evaluation'
                )
        else:
            baseline_identity = baseline_facts['identity']
            cls._assert_artifact_identity(
                selected_candidates['checkpoint'],
                sha256=baseline_identity['checkpoint_sha256'],
                size_bytes=baseline_identity['checkpoint_size_bytes'],
                label='retained incumbent checkpoint',
            )
            if selected_evaluation != baseline:
                raise ValueError(
                    'keep-champion decision must retain the baseline evaluation'
                )
            if selected_evaluation_candidate.sha256 != baseline_candidate.sha256:
                raise ValueError(
                    'retained incumbent evaluation hash differs from baseline'
                )
            if (
                selected_evaluation_facts['identity']['checkpoint_sha256']
                != baseline_identity['checkpoint_sha256']
            ):
                raise ValueError(
                    'retained incumbent evaluation belongs to another checkpoint'
                )

        # The final report is deliberately not allowed to identify itself as an
        # input or winner. Its staging hash is still independently verified by
        # ContractValidator below.
        if final_candidate.sha256 in {
            baseline_candidate.sha256,
            *trial_hashes,
        }:
            raise ValueError('model improvement report aliases a history artifact')

        return {
            'optimization.trial_count': len(trials),
            'optimization.baseline_macro_f1': baseline_metrics['offline.macro_f1'],
            'optimization.best_macro_f1': best_metrics['offline.macro_f1'],
            'optimization.macro_f1_delta_vs_baseline': macro_delta,
            'optimization.min_per_class_recall_delta_vs_baseline': (min_recall_delta),
            'optimization.none_false_accept_rate': best_metrics[
                'offline.none_false_accept_rate'
            ],
            'optimization.promotion_eligible': promotion_eligible,
            'optimization.decision': decision,
            'optimization.stop_reason': stop_reason,
            'optimization.best_trial_number': best_trial['trial_number'],
        }

    @classmethod
    def _validate_policy(cls, raw: Any, *, trial_count: int) -> dict[str, Any]:
        policy = cls._require_dict(raw, 'model improvement policy')
        expected_keys = {
            'max_trials',
            'patience',
            'target_macro_f1_delta',
            'min_improvement',
            'max_epochs',
            'batch_size',
            'workers',
            'gpus',
        }
        if set(policy) != expected_keys:
            raise ValueError(
                'model improvement policy must contain only the normalized '
                f'bounded keys; missing={sorted(expected_keys - set(policy))}, '
                f'unknown={sorted(set(policy) - expected_keys)}'
            )
        integer_ranges = {
            'max_trials': (1, 8),
            'patience': (1, 8),
            'max_epochs': (1, 100),
            'workers': (0, 32),
            'gpus': (1, 8),
        }
        for name, (minimum, maximum) in integer_ranges.items():
            value = policy[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f'model improvement policy {name} is outside its bounded range'
                )
        if policy['patience'] > policy['max_trials']:
            raise ValueError('model improvement patience exceeds max_trials')
        if trial_count > policy['max_trials']:
            raise ValueError('staged trial history exceeds max_trials policy')
        if policy['batch_size'] not in {4, 8, 16, 32, 64}:
            raise ValueError('model improvement batch_size is not allowlisted')
        for name, upper_bound in (
            ('target_macro_f1_delta', 1.0),
            ('min_improvement', 0.05),
        ):
            numeric_value = policy[name]
            if (
                not isinstance(numeric_value, (int, float))
                or isinstance(numeric_value, bool)
                or not math.isfinite(float(numeric_value))
                or not 0.0 <= float(numeric_value) <= upper_bound
            ):
                raise ValueError(
                    f'model improvement policy {name} is outside its bounded range'
                )
            policy[name] = float(numeric_value)
        return policy

    @classmethod
    def _validate_final_trial_history(
        cls, raw: Any, *, trials: list[dict[str, Any]]
    ) -> None:
        if not isinstance(raw, list) or len(raw) != len(trials):
            raise ValueError(
                'model improvement trials summary must cover every trial report'
            )
        for summary, trial in zip(raw, trials, strict=True):
            summary = cls._require_dict(
                summary, f'trial {trial["trial_number"]} final summary'
            )
            report_hash = summary.get(
                'trial_report_sha256', summary.get('report_sha256')
            )
            if (
                summary.get('trial_number') != trial['trial_number']
                or report_hash != trial['candidate'].sha256
            ):
                raise ValueError(
                    'model improvement trial summary does not bind its staged '
                    'trial report'
                )
            for prefix in (
                'checkpoint',
                'training_config',
                'training_log',
                'training_metrics',
                'evaluation_report',
            ):
                if (
                    summary.get(f'{prefix}_sha256')
                    != trial['outputs'][f'{prefix}_sha256']
                ):
                    raise ValueError(
                        f'trial {trial["trial_number"]} final summary does not '
                        f'bind {prefix}'
                    )
            metrics = cls._require_dict(
                summary.get('metrics'),
                f'trial {trial["trial_number"]} final summary metrics',
            )
            for name, expected in trial['evaluation_facts']['metrics'].items():
                cls._require_metric_equal(
                    metrics.get(name),
                    expected,
                    f'trial {trial["trial_number"]} final summary metric {name}',
                )

    @classmethod
    def _validate_best_trial(cls, raw: Any, *, best_trial: dict[str, Any]) -> None:
        summary = cls._require_dict(raw, 'model improvement best_trial')
        report_hash = summary.get('trial_report_sha256', summary.get('report_sha256'))
        if (
            summary.get('trial_number') != best_trial['trial_number']
            or report_hash != best_trial['candidate'].sha256
        ):
            raise ValueError(
                'model improvement best_trial is not the deterministic winner'
            )
        outputs = best_trial['outputs']
        for prefix in (
            'checkpoint',
            'training_config',
            'training_log',
            'training_metrics',
            'evaluation_report',
        ):
            if summary.get(f'{prefix}_sha256') != outputs[f'{prefix}_sha256']:
                raise ValueError(f'model improvement best_trial does not bind {prefix}')

    @classmethod
    def _validate_winner(
        cls,
        winner: dict[str, Any],
        *,
        decision: str,
        baseline_candidate: Any,
        baseline_facts: dict[str, Any],
        best_trial: dict[str, Any],
    ) -> None:
        if decision == 'promote-challenger':
            if winner.get('source') != 'trial':
                raise ValueError('promoted winner source must be trial')
            if (
                winner.get('trial_number') != best_trial['trial_number']
                or winner.get('trial_report_sha256') != best_trial['candidate'].sha256
            ):
                raise ValueError('promoted winner does not bind the best trial')
            outputs = best_trial['outputs']
            for prefix in (
                'checkpoint',
                'training_config',
                'training_log',
                'training_metrics',
                'evaluation_report',
            ):
                if winner.get(f'{prefix}_sha256') != outputs[f'{prefix}_sha256']:
                    raise ValueError(f'promoted winner does not bind {prefix}')
            return

        if winner.get('source') != 'baseline':
            raise ValueError('keep-champion winner source must be baseline')
        if (
            winner.get('trial_number') is not None
            or winner.get('trial_report_sha256') is not None
        ):
            raise ValueError('baseline winner must not claim a trial identity')
        baseline_identity = baseline_facts['identity']
        if (
            winner.get('checkpoint_sha256') != baseline_identity['checkpoint_sha256']
            or winner.get('evaluation_report_sha256') != baseline_candidate.sha256
        ):
            raise ValueError('baseline winner hashes do not bind the incumbent')
        for prefix in ('training_config', 'training_log', 'training_metrics'):
            if winner.get(f'{prefix}_sha256') is not None:
                raise ValueError(f'baseline winner {prefix}_sha256 must be null')

    @classmethod
    def _validate_deltas(
        cls,
        raw: Any,
        *,
        accuracy_delta: float,
        macro_delta: float,
        min_recall_delta: float,
        none_delta: float,
    ) -> None:
        deltas = cls._require_dict(raw, 'model improvement deltas_vs_baseline')
        expected = {
            'offline.accuracy_delta_vs_baseline': accuracy_delta,
            'offline.macro_f1_delta_vs_baseline': macro_delta,
            'offline.min_per_class_recall_delta_vs_baseline': min_recall_delta,
            'offline.none_false_accept_rate_delta_vs_baseline': none_delta,
        }
        if set(deltas) != set(expected):
            raise ValueError('model improvement deltas_vs_baseline keys are invalid')
        for name, value in expected.items():
            cls._require_metric_equal(
                deltas.get(name), value, f'model improvement delta {name}'
            )

    @classmethod
    def _validate_stop_reason(
        cls,
        stop_reason: Any,
        *,
        trials: list[dict[str, Any]],
        eligible: list[dict[str, Any]],
        baseline_metrics: dict[str, Any],
        policy: dict[str, Any],
        best_macro_delta: float,
    ) -> None:
        if stop_reason not in cls._ALLOWED_STOP_REASONS:
            raise ValueError('model improvement stop_reason is invalid')
        if eligible and best_macro_delta >= policy['target_macro_f1_delta']:
            expected = 'target-reached'
        elif len(trials) == policy['max_trials']:
            expected = 'trial-budget-exhausted'
        elif (
            cls._trailing_no_improvement_count(
                trials,
                baseline_metrics=baseline_metrics,
                minimum=policy['min_improvement'],
            )
            >= policy['patience']
        ):
            expected = 'patience-exhausted'
        else:
            raise ValueError(
                'model improvement history has not reached a bounded stop condition'
            )
        if stop_reason != expected:
            raise ValueError(
                'model improvement stop_reason is not the deterministic result '
                'of its trial history'
            )

    @classmethod
    def _trailing_no_improvement_count(
        cls,
        trials: list[dict[str, Any]],
        *,
        baseline_metrics: dict[str, Any],
        minimum: float,
    ) -> int:
        best = baseline_metrics['offline.macro_f1']
        trailing = 0
        for trial in trials:
            metrics = trial['evaluation_facts']['metrics']
            if (
                cls._is_trial_guardrail_eligible(metrics, baseline_metrics)
                and metrics['offline.macro_f1'] >= best + minimum
            ):
                best = metrics['offline.macro_f1']
                trailing = 0
            else:
                trailing += 1
        return trailing

    @staticmethod
    def _is_trial_guardrail_eligible(
        metrics: dict[str, Any], baseline_metrics: dict[str, Any]
    ) -> bool:
        return (
            metrics['offline.min_per_class_recall']
            - baseline_metrics['offline.min_per_class_recall']
            >= -0.02
            and metrics['offline.none_false_accept_rate'] <= 0.01
        )

    @staticmethod
    def _trial_rank(trial: dict[str, Any]) -> tuple[float, float, float, int]:
        metrics = trial['evaluation_facts']['metrics']
        return (
            metrics['offline.macro_f1'],
            metrics['offline.min_per_class_recall'],
            -metrics['offline.none_false_accept_rate'],
            -trial['trial_number'],
        )

    @staticmethod
    def _assert_artifact_identity(
        candidate: Any,
        *,
        sha256: str,
        size_bytes: int,
        label: str,
    ) -> None:
        if candidate.sha256 != sha256 or candidate.size_bytes != size_bytes:
            raise ValueError(f'{label} does not match the bound output identity')

    @classmethod
    def _validate_analysis(cls, raw: Any) -> None:
        analysis = cls._require_dict(raw, 'model improvement analysis')
        document = cls._require_dict(
            analysis.get('document'), 'model improvement analysis document'
        )
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
        if len(encoded) > 32 * 1024:
            raise ValueError('model improvement analysis exceeds 32 KiB')
        if analysis.get('sha256') != hashlib.sha256(encoded).hexdigest():
            raise ValueError('model improvement analysis hash is invalid')

    @classmethod
    def _validate_confusion(cls, raw: Any, *, role: str) -> dict[str, dict[str, int]]:
        confusion = cls._require_dict(raw, f'{role} confusion matrix')
        if set(confusion) != RIGHTARM_LABELS:
            raise ValueError(f'{role} confusion matrix rows are incomplete')
        normalized: dict[str, dict[str, int]] = {}
        for truth in RIGHTARM_LABEL_ORDER:
            row = cls._require_dict(
                confusion.get(truth), f'{role} confusion row {truth}'
            )
            if set(row) != RIGHTARM_LABELS:
                raise ValueError(f'{role} confusion matrix row {truth} is incomplete')
            normalized[truth] = {}
            for prediction in RIGHTARM_LABEL_ORDER:
                count = row[prediction]
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(
                        f'{role} confusion count {truth}/{prediction} is invalid'
                    )
                normalized[truth][prediction] = count
        if sum(sum(row.values()) for row in normalized.values()) < 1:
            raise ValueError(f'{role} confusion matrix has no samples')
        return normalized

    @classmethod
    def _metrics_from_confusion(
        cls, confusion: dict[str, dict[str, int]]
    ) -> dict[str, Any]:
        total = sum(sum(row.values()) for row in confusion.values())
        correct = sum(confusion[label][label] for label in RIGHTARM_LABEL_ORDER)
        per_class: dict[str, dict[str, float | int]] = {}
        f1_values: list[float] = []
        recalls: list[float] = []
        for label in RIGHTARM_LABEL_ORDER:
            true_positive = confusion[label][label]
            support = sum(confusion[label].values())
            predicted = sum(confusion[truth][label] for truth in RIGHTARM_LABEL_ORDER)
            precision = cls._divide(true_positive, predicted)
            recall = cls._divide(true_positive, support)
            f1 = cls._divide(2.0 * precision * recall, precision + recall)
            per_class[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
            }
            f1_values.append(f1)
            recalls.append(recall)
        none_support = sum(confusion['NONE'].values())
        none_false_accepts = none_support - confusion['NONE']['NONE']
        return {
            'offline.sample_count': total,
            'offline.accuracy': cls._divide(correct, total),
            'offline.macro_f1': sum(f1_values) / len(f1_values),
            'offline.min_per_class_recall': min(recalls),
            'offline.none_false_accept_rate': cls._divide(
                none_false_accepts, none_support
            ),
            'per_class': per_class,
        }

    @staticmethod
    def _divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    @classmethod
    def _require_same_evaluation_boundary(
        cls, baseline: dict[str, Any], candidate: dict[str, Any]
    ) -> None:
        left = baseline['identity']
        right = candidate['identity']
        for key in (
            'dataset_sha256',
            'dataset_size_bytes',
            'evaluation_split',
            'preprocessing_version',
            'preprocessing_fingerprint',
        ):
            if left[key] != right[key]:
                raise ValueError(
                    f'evaluation boundary changed between baseline and candidate: {key}'
                )

    @staticmethod
    def _require_dict(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f'{label} must be an object')
        return value

    @staticmethod
    def _require_sha(value: Any, label: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)
        ):
            raise ValueError(f'{label} must be a lowercase SHA-256 digest')
        return value

    @staticmethod
    def _require_size(value: Any, label: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{label} must be a non-negative byte count')
        return value

    @staticmethod
    def _require_metric_equal(actual: Any, expected: Any, label: str) -> None:
        if isinstance(expected, int):
            if (
                not isinstance(actual, int)
                or isinstance(actual, bool)
                or actual != expected
            ):
                raise ValueError(f'{label} does not match the confusion matrix')
            return
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError(f'{label} does not match the confusion matrix')


class EvaluationValidator(JSONReportValidator):
    def validate_report(self, report: dict[str, Any]) -> None:
        labels = report.get('labels')
        if not isinstance(labels, list) or set(labels) != RIGHTARM_LABELS:
            raise ValueError('evaluation does not cover the fixed 11-class ontology')
        per_class = report.get('per_class')
        confusion = report.get('confusion_matrix')
        if not isinstance(per_class, dict) or set(per_class) != RIGHTARM_LABELS:
            raise ValueError('evaluation per-class metrics are incomplete')
        if not isinstance(confusion, dict) or set(confusion) != RIGHTARM_LABELS:
            raise ValueError('evaluation confusion matrix is incomplete')
        if any(
            not isinstance(row, dict) or set(row) != RIGHTARM_LABELS
            for row in confusion.values()
        ):
            raise ValueError('evaluation confusion matrix rows are incomplete')
        metrics = report.get('metrics', {})
        sample_count = metrics.get('offline.sample_count')
        if not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError('evaluation sample count must be positive')
        for name, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'evaluation metric is non-finite: {name}')


class TrainingValidator(JSONReportValidator):
    def validate_report(self, report: dict[str, Any]) -> None:
        if int(report.get('record_count', 0)) < 1:
            raise ValueError('training metrics contain no scalar records')
        latest = report.get('latest')
        if not isinstance(latest, dict) or not latest:
            raise ValueError('training metrics have no latest values')
        sources = report.get('sources')
        if not isinstance(sources, list) or not sources:
            raise ValueError('training metrics have no immutable scalar sources')
        for item in sources:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get('sha256'), str)
                or len(item['sha256']) != 64
                or not isinstance(item.get('size_bytes'), int)
            ):
                raise ValueError('training metric source record is invalid')

    def extract_metrics(self, report: dict[str, Any]) -> dict[str, Any]:
        latest = report.get('latest', {})
        return {
            f'training.{key}': value
            for key, value in latest.items()
            if isinstance(value, (bool, int, float, str)) or value is None
        }


class BoardValidator(JSONReportValidator):
    def validate_report(self, report: dict[str, Any]) -> None:
        for path in (
            ('hardware', 'model'),
            ('runtime', 'version'),
            ('deployment_binary_sha',),
            ('preprocessing_version',),
        ):
            value: Any = report
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(f'missing board report field: {".".join(path)}')
                value = value[key]
        layout = report.get('input_layout', {}).get('keypoint_physical')
        if layout not in {'1,V,C,T', '(1,V,C,T)', 'NHWC:(1,V,C,T)'}:
            raise ValueError(
                'board report does not prove keypoint (1,T,V,C)->(1,V,C,T) layout'
            )


class ParityValidator(JSONReportValidator):
    def extract_metrics(self, report: dict[str, Any]) -> dict[str, Any]:
        metrics = report.get('metrics')
        if isinstance(metrics, dict):
            return metrics
        summary = report.get('summary', {})
        total = int(summary.get('n_total', 0))
        if total <= 0:
            raise ValueError('parity report has no samples')
        samples = report.get('per_sample', [])
        gate_pairs = [
            (row.get('gate_onnx'), row.get('gate_rknn'))
            for row in samples
            if row.get('gate_onnx') is not None and row.get('gate_rknn') is not None
        ]
        gate_matches = sum(
            (left >= 0.45) == (right >= 0.45) for left, right in gate_pairs
        )
        return {
            'conversion.onnx_rknn_top1_match_rate': int(summary.get('n_top1_match', 0))
            / total,
            'conversion.gate_decision_match_rate': gate_matches / len(gate_pairs)
            if gate_pairs
            else 0.0,
        }


class DeploymentObservationValidator(JSONReportValidator):
    def __call__(self, task: TaskSpec, result: TaskResult) -> ValidationResult:
        validation = super().__call__(task, result)
        if (
            validation.disposition is ValidationDisposition.ACCEPTED
            and task.kind != 'monitor'
        ):
            reports = [
                item
                for item in result.artifact_candidates
                if item.kind == self.report_kind
            ]
            if self._load_json(reports[0].uri).get('observed_state') != 'HEALTHY':
                return self._invalid(
                    f'{task.kind} requires a HEALTHY deployment observation'
                )
        return validation

    def validate_report(self, report: dict[str, Any]) -> None:
        if (
            not isinstance(report.get('deployment_id'), str)
            or not report['deployment_id']
        ):
            raise ValueError('deployment observation has no deployment_id')
        if not isinstance(report.get('revision'), str) or not report['revision']:
            raise ValueError('deployment observation has no revision')
        observed_state = report.get('observed_state')
        if observed_state not in {'HEALTHY', 'UNHEALTHY'}:
            raise ValueError('deployment observation state is invalid')
        smoke = report.get('smoke_test', {})
        if not isinstance(smoke.get('passed'), bool):
            raise ValueError('deployment smoke test result is missing')
        if observed_state == 'HEALTHY' and smoke['passed'] is not True:
            raise ValueError('deployment smoke test did not pass')


def build_rightarm_validator_registry(
    artifact_store: ArtifactStore,
) -> ValidatorRegistry:
    default = ContractValidator(artifact_store)
    registry = ValidatorRegistry(default)
    registry.register(
        'rightarm-data-inspection-v1',
        DataInspectionValidator(
            artifact_store,
            report_kind='data-validation-report',
            schema_version='rightarm.gesture.data-validation/v1',
        ),
    )
    registry.register(
        'rightarm-dataset-v1',
        DatasetValidator(
            artifact_store,
            report_kind='dataset-report',
            schema_version='rightarm.gesture.dataset-report/v1',
        ),
    )
    registry.register(
        'rightarm-dataset-distribution-review-v1',
        DatasetDistributionReviewValidator(artifact_store),
    )
    registry.register(
        'rightarm-model-improvement-v1',
        ModelImprovementValidator(artifact_store),
    )
    registry.register(
        'rightarm-training-v1',
        TrainingValidator(
            artifact_store,
            report_kind='training-metrics',
            schema_version='rightarm.gesture.training-metrics/v1',
        ),
    )
    registry.register(
        'rightarm-evaluation-v1',
        EvaluationValidator(
            artifact_store,
            report_kind='evaluation-report',
            schema_version='rightarm.gesture.evaluation/v1',
        ),
    )
    registry.register(
        'rightarm-rknn-package-v1',
        ParityValidator(
            artifact_store,
            report_kind='parity-report',
            schema_version=None,
        ),
    )
    registry.register(
        'rightarm-rk3588-v1',
        BoardValidator(
            artifact_store,
            report_kind='board-report',
            schema_version='rightarm.gesture.board-report/v1',
        ),
    )
    registry.register(
        'rightarm-deployment-v1',
        DeploymentObservationValidator(
            artifact_store,
            report_kind='deployment-observation',
            schema_version='rightarm.gesture.deployment-observation/v1',
        ),
    )
    registry.register(
        'rightarm-canary-v1',
        DeploymentObservationValidator(
            artifact_store,
            report_kind='deployment-observation',
            schema_version='rightarm.gesture.deployment-observation/v1',
        ),
    )
    return registry
