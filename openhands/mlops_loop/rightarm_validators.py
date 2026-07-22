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
