"""Host-side client for the operator-facing model-agent entry points."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from rightarm_mlops.exploration_efficiency import (
    build_exploration_efficiency_report,
)

DEPENDENCIES: dict[str, tuple[tuple[str, str, str], ...]] = {
    'data': (
        (
            'baseline-mmaction-dataset',
            'MLOPS_BASELINE_MMACTION_DATASET_ARTIFACT',
            'MLOPS_DEFAULT_DATASET_PATH',
        ),
        ('rtm-detector', 'MLOPS_RTM_DETECTOR_ARTIFACT', 'MLOPS_RTM_DETECTOR_PATH'),
        (
            'rtm-person-pose',
            'MLOPS_RTM_PERSON_POSE_ARTIFACT',
            'MLOPS_RTM_PERSON_POSE_PATH',
        ),
        (
            'rtm-hand-pose',
            'MLOPS_RTM_HAND_POSE_ARTIFACT',
            'MLOPS_RTM_HAND_POSE_PATH',
        ),
    ),
    'train': (
        ('base-model', 'MLOPS_CHAMPION_BASE_MODEL_ARTIFACT', 'MLOPS_BASE_MODEL_PATH'),
    ),
    'full': (
        (
            'baseline-mmaction-dataset',
            'MLOPS_BASELINE_MMACTION_DATASET_ARTIFACT',
            'MLOPS_DEFAULT_DATASET_PATH',
        ),
        ('rtm-detector', 'MLOPS_RTM_DETECTOR_ARTIFACT', 'MLOPS_RTM_DETECTOR_PATH'),
        (
            'rtm-person-pose',
            'MLOPS_RTM_PERSON_POSE_ARTIFACT',
            'MLOPS_RTM_PERSON_POSE_PATH',
        ),
        (
            'rtm-hand-pose',
            'MLOPS_RTM_HAND_POSE_ARTIFACT',
            'MLOPS_RTM_HAND_POSE_PATH',
        ),
        ('base-model', 'MLOPS_CHAMPION_BASE_MODEL_ARTIFACT', 'MLOPS_BASE_MODEL_PATH'),
    ),
    'export': (
        (
            'mmaction-dataset',
            'MLOPS_DEFAULT_DATASET_ARTIFACT',
            'MLOPS_DEFAULT_DATASET_PATH',
        ),
    ),
    'improve': (
        (
            'mmaction-dataset',
            'MLOPS_DEFAULT_DATASET_ARTIFACT',
            'MLOPS_DEFAULT_DATASET_PATH',
        ),
        ('base-model', 'MLOPS_CHAMPION_BASE_MODEL_ARTIFACT', 'MLOPS_BASE_MODEL_PATH'),
    ),
}


def _bounded_integer(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError('expected an integer') from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f'value must be in [{minimum}, {maximum}]')
        return parsed

    return parse


def _bounded_delta(maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError('expected a number') from exc
        if not 0.0 <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f'value must be in [0, {maximum}]')
        return parsed

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='model-agent',
        description='Trigger and inspect the rightArm OpenHands MLOps loop.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    trigger = subparsers.add_parser('trigger', help='start one audited MLOps Run')
    modes = trigger.add_subparsers(dest='mode', required=True)

    data = modes.add_parser('data', help='COS download through PKL generation')
    data.add_argument('--cos-uri', required=True)
    data.add_argument(
        '--baseline-pkl',
        type=Path,
        help='override the configured trusted SFT baseline PKL',
    )
    _rtm_arguments(data)
    _common_trigger_arguments(data)

    train = modes.add_parser('train', help='train, evaluate and export from a PKL')
    train.add_argument('--pkl', required=True, type=Path)
    train.add_argument('--base-model', type=Path)
    _common_trigger_arguments(train)

    full = modes.add_parser('full', help='run data, train, evaluate and export')
    full.add_argument('--cos-uri', required=True)
    full.add_argument('--base-model', type=Path)
    full.add_argument(
        '--baseline-pkl',
        type=Path,
        help='override the configured trusted SFT baseline PKL',
    )
    _rtm_arguments(full)
    _common_trigger_arguments(full)

    export = modes.add_parser('export', help='evaluate and export a checkpoint')
    export.add_argument('--checkpoint', required=True, type=Path)
    export.add_argument(
        '--pkl', type=Path, help='override the configured evaluation PKL'
    )
    _common_trigger_arguments(export)

    improve = modes.add_parser(
        'improve',
        help='run bounded train/evaluate/diagnose trials from an incumbent model',
    )
    improve.add_argument(
        '--pkl',
        type=Path,
        help='override the configured immutable training/evaluation PKL',
    )
    incumbent = improve.add_mutually_exclusive_group()
    incumbent.add_argument(
        '--checkpoint',
        dest='base_model',
        type=Path,
        help='incumbent checkpoint to evaluate and improve',
    )
    incumbent.add_argument(
        '--base-model',
        dest='base_model',
        type=Path,
        help='alias for --checkpoint',
    )
    improve.add_argument(
        '--init-checkpoint',
        type=Path,
        help='optional distinct initialization checkpoint for training trials',
    )
    improve.add_argument('--max-trials', type=_bounded_integer(1, 8), default=4)
    improve.add_argument('--patience', type=_bounded_integer(1, 8), default=2)
    improve.add_argument(
        '--target-macro-f1-delta', type=_bounded_delta(1.0), default=0.002
    )
    improve.add_argument('--min-improvement', type=_bounded_delta(0.05), default=0.0005)
    improve.add_argument('--max-epochs', type=_bounded_integer(1, 100), default=50)
    improve.add_argument(
        '--batch-size', type=int, choices=(4, 8, 16, 32, 64), default=32
    )
    improve.add_argument('--workers', type=_bounded_integer(0, 32), default=2)
    improve.add_argument(
        '--gpus',
        type=_bounded_integer(1, 8),
        default=os.environ.get('MLOPS_TRAIN_GPU_COUNT', '1'),
    )
    _common_trigger_arguments(improve)

    status = subparsers.add_parser('status', help='show recent controller Runs')
    status.add_argument('--limit', type=int, default=20)
    status.add_argument(
        '--timeout',
        type=_bounded_integer(1, 300),
        default=60,
        help='controller request timeout in seconds',
    )
    status.add_argument('--json', action='store_true', dest='as_json')

    efficiency = subparsers.add_parser(
        'efficiency', help='compare one improve Run with a paired human baseline'
    )
    efficiency.add_argument('--run-id', required=True)
    efficiency.add_argument(
        '--expected-trials',
        type=_bounded_integer(1, 8),
        help='optional assertion; must equal the Agent-selected valid trial count',
    )
    efficiency.add_argument('--human-wall-minutes', type=float)
    efficiency.add_argument('--human-baseline-id')
    efficiency.add_argument(
        '--human-valid-trials',
        type=_bounded_integer(1, 8),
        help='valid iterations completed by the paired human baseline',
    )
    efficiency.add_argument(
        '--human-baseline-json',
        type=Path,
        help='auditable paired-human baseline manifest for a claim-eligible result',
    )
    efficiency.add_argument('--json', action='store_true', dest='as_json')
    return parser


def _rtm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--det-onnx', type=Path)
    parser.add_argument('--person-pose-onnx', type=Path)
    parser.add_argument('--hand-pose-onnx', type=Path)


def _common_trigger_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--request-id')
    parser.add_argument('--dry-run', action='store_true')


class ControllerClient:
    def __init__(self) -> None:
        port = os.environ.get('MLOPS_PORT')
        state = os.environ.get('MODEL_AGENT_STATE_DIR')
        if not port or not state:
            raise ValueError('run this command through ./model-agent after ./setup.sh')
        self.base_url = os.environ.get(
            'MODEL_AGENT_CONTROLLER_URL', f'http://127.0.0.1:{port}'
        ).rstrip('/')
        self.import_root = Path(state).resolve() / 'mlops' / 'imports'
        token_path = Path(state).resolve() / 'secrets' / 'mlops-operator-token'
        try:
            self.token = token_path.read_text(encoding='utf-8').strip()
        except FileNotFoundError as exc:
            raise ValueError('controller token is missing; run ./setup.sh') from exc
        if not self.token:
            raise ValueError('controller token is empty; run ./setup.sh')

    def get(self, path: str, *, timeout: float = 60) -> Any:
        return self._request('GET', path, None, timeout=timeout)

    def post(self, path: str, payload: dict[str, Any], *, timeout: float = 60) -> Any:
        return self._request('POST', path, payload, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout: float = 60,
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            f'{self.base_url}{path}',
            data=data,
            method=method,
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'X-MLOps-Operator-Token': self.token,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            try:
                detail = json.loads(body).get('detail', body)
            except json.JSONDecodeError:
                detail = body
            raise RuntimeError(
                f'controller rejected the request ({exc.code}): {detail}'
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f'cannot reach the controller at {self.base_url}; run ./start.sh first'
            ) from exc

    def bootstrap(self, source: Path, kind: str) -> dict[str, Any]:
        source = source.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f'input is not a regular file: {source}')
        digest, size_bytes = _sha256_file(source)
        self.import_root.mkdir(parents=True, exist_ok=True)
        suffix = ''.join(source.suffixes)[-32:]
        destination = self.import_root / f'{kind}-{digest}{suffix}'
        if destination.exists():
            existing_digest, existing_size = _sha256_file(destination)
            if (existing_digest, existing_size) != (digest, size_bytes):
                raise RuntimeError(f'existing import is corrupt: {destination}')
        else:
            temporary = destination.with_name(
                f'.{destination.name}.{uuid.uuid4().hex}.tmp'
            )
            try:
                shutil.copyfile(source, temporary)
                copied_digest, copied_size = _sha256_file(temporary)
                if (copied_digest, copied_size) != (digest, size_bytes):
                    raise RuntimeError(f'input changed while it was imported: {source}')
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        container_path = f'/var/lib/rightarm-mlops/imports/{destination.name}'
        return self.post(
            '/api/v1/mlops/artifacts/bootstrap',
            {
                'path': container_path,
                'kind': kind,
                'sha256': digest,
                'size_bytes': size_bytes,
                'metadata': {'source_name': source.name},
            },
            timeout=3600,
        )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _path_override(args: argparse.Namespace, kind: str) -> Path | None:
    names = {
        'base-model': 'base_model',
        'mmaction-dataset': 'pkl',
        'baseline-mmaction-dataset': 'baseline_pkl',
        'rtm-detector': 'det_onnx',
        'rtm-person-pose': 'person_pose_onnx',
        'rtm-hand-pose': 'hand_pose_onnx',
    }
    value = getattr(args, names[kind], None)
    return value if isinstance(value, Path) else None


def _input_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    if args.mode == 'improve':
        _improvement_parameters(args)
    if args.mode == 'train':
        inputs.append({'kind': 'mmaction-dataset', 'path': str(args.pkl)})
    elif args.mode == 'export':
        inputs.append({'kind': 'checkpoint', 'path': str(args.checkpoint)})
    if args.mode == 'improve' and args.init_checkpoint is not None:
        inputs.append(
            {'kind': 'training-init-model', 'path': str(args.init_checkpoint)}
        )

    for kind, artifact_env, path_env in DEPENDENCIES[args.mode]:
        override = _path_override(args, kind)
        artifact_id = os.environ.get(artifact_env, '').strip()
        configured_path = os.environ.get(path_env, '').strip()
        if override is not None:
            inputs.append({'kind': kind, 'path': str(override)})
        elif artifact_id:
            inputs.append({'kind': kind, 'artifact_id': artifact_id})
        elif configured_path:
            inputs.append({'kind': kind, 'path': configured_path})
        else:
            raise ValueError(
                f'missing {kind}; pass its file option or set '
                f'{artifact_env}/{path_env} in .env'
            )
    return inputs


def _improvement_parameters(args: argparse.Namespace) -> dict[str, int | float]:
    if args.patience > args.max_trials:
        raise ValueError('improve --patience must not exceed --max-trials')
    return {
        'max_trials': args.max_trials,
        'patience': args.patience,
        'target_macro_f1_delta': args.target_macro_f1_delta,
        'min_improvement': args.min_improvement,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'gpus': args.gpus,
    }


def _trigger(args: argparse.Namespace) -> int:
    inputs = _input_plan(args)
    for item in inputs:
        if 'path' in item:
            path = Path(item['path']).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f'input is not a regular file: {path}')
            item['path'] = str(path)
    request_id = args.request_id or uuid.uuid4().hex
    source_uri = getattr(args, 'cos_uri', None)
    parameters: dict[str, Any] = {'triggered_by': os.environ.get('USER', 'operator')}
    if args.mode == 'improve':
        parameters.update(_improvement_parameters(args))
    if args.dry_run:
        print(
            json.dumps(
                {
                    'mode': args.mode,
                    'request_id': request_id,
                    'source_uri': source_uri,
                    'inputs': inputs,
                    'parameters': parameters,
                    'will_dispatch': False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    client = ControllerClient()
    artifact_ids: list[str] = []
    for item in inputs:
        if 'artifact_id' in item:
            artifact_id = item['artifact_id']
        else:
            print(
                f'登记不可变输入：{item["kind"]} ({Path(item["path"]).name})',
                file=sys.stderr,
            )
            artifact_id = client.bootstrap(Path(item['path']), item['kind'])['id']
        if artifact_id not in artifact_ids:
            artifact_ids.append(artifact_id)
    result = client.post(
        '/api/v1/mlops/manual-runs',
        {
            'mode': args.mode,
            'request_id': request_id,
            'source_uri': source_uri,
            'initial_artifact_ids': artifact_ids,
            'parameters': parameters,
        },
    )
    print(f'Run 已创建：{result["id"]}')
    print(f'模式：{args.mode}  状态：{result["state"]}')
    print('步骤：' + ' → '.join(result['tasks']))
    if result.get('conversation_id'):
        print(f'OpenHands 会话：{result["conversation_id"]}')
    return 0


def _status(args: argparse.Namespace) -> int:
    result = ControllerClient().get(
        f'/api/v1/mlops/runs?limit={args.limit}',
        timeout=args.timeout,
    )
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    runs = result['runs']
    if not runs:
        print('当前还没有 MLOps Run。')
        return 0
    print(f'{"RUN ID":38} {"MODE":8} {"STATE":22} CURRENT TASK')
    for run in runs:
        print(
            f'{run["id"][:38]:38} {(run["mode"] or "-"):8} '
            f'{run["state"]:22} {run["current_task"] or "-"}'
        )
    return 0


def _efficiency(args: argparse.Namespace) -> int:
    human_seconds = None
    if args.human_wall_minutes is not None:
        human_seconds = args.human_wall_minutes * 60.0
    human_manifest = None
    if args.human_baseline_json is not None:
        raw_manifest = json.loads(
            args.human_baseline_json.resolve(strict=True).read_text(encoding='utf-8')
        )
        if not isinstance(raw_manifest, dict):
            raise ValueError('human baseline JSON must contain one object')
        human_manifest = raw_manifest
    audit = ControllerClient().get(f'/api/v1/mlops/runs/{args.run_id}/audit')
    report = build_exploration_efficiency_report(
        audit,
        expected_trials=args.expected_trials,
        human_wall_seconds=human_seconds,
        human_baseline_id=args.human_baseline_id,
        human_valid_trials=args.human_valid_trials,
        human_baseline_manifest=human_manifest,
    )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    print(
        f'Run {report["run_id"]}: '
        f'Agent 自主完成 {report["valid_trial_count"]} 次有效迭代'
    )
    if report['agent_wall_seconds'] is None:
        print(f'当前状态：{report["run_state"]}，尚不能计算最终墙钟时间。')
    else:
        print(f'Agent 端到端墙钟：{report["agent_wall_seconds"] / 60:.2f} 分钟')
    comparison = report['comparison']
    if comparison is not None:
        print(
            f'相对人工时间比：{comparison["agent_to_human_time_ratio"]:.3f}；'
            f'节省：{comparison["time_saving_fraction"]:.1%}；'
            f'加速：{comparison["speedup"]:.2f}x'
        )
        if report['claim_eligible']:
            print('对比证据：已通过同口径人工基线清单校验，可用于正式结论。')
        else:
            print(
                '对比证据：当前为结构化人工自证，尚未独立封存，不能作为正式效率结论。'
            )
    else:
        print('未提供可追踪的人工墙钟基线；当前只报告 Agent 侧效率。')
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == 'trigger':
            return _trigger(args)
        if args.command == 'status':
            return _status(args)
        return _efficiency(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'model-agent: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
