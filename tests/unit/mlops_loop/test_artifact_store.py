from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from openhands.mlops_loop.artifact_store import S3ArtifactStore
from openhands.mlops_loop.errors import InvariantViolation
from openhands.mlops_loop.models import ArtifactRef


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.put_count = 0

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            item = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise ClientError(
                {'Error': {'Code': '404', 'Message': 'not found'}}, 'HeadObject'
            ) from exc
        return {
            'ContentLength': len(item['body']),
            'Metadata': dict(item['metadata']),
        }

    def put_object(self, **request: Any) -> None:
        identity = (request['Bucket'], request['Key'])
        if identity in self.objects and request.get('IfNoneMatch') == '*':
            raise ClientError(
                {
                    'Error': {
                        'Code': 'PreconditionFailed',
                        'Message': 'already exists',
                    }
                },
                'PutObject',
            )
        body = request['Body'].read()
        self.objects[identity] = {
            'body': body,
            'metadata': dict(request['Metadata']),
        }
        self.put_count += 1

    def download_file(self, bucket: str, key: str, destination: str) -> None:
        Path(destination).write_bytes(self.objects[(bucket, key)]['body'])


def test_s3_store_seals_once_and_materializes_verified_content(tmp_path: Path):
    client = FakeS3Client()
    store = S3ArtifactStore(
        tmp_path / 'cache', bucket='artifacts', prefix='rightarm', client=client
    )
    source = tmp_path / 'model.onnx'
    source.write_bytes(b'portable-model')
    candidate = store.stage(source, kind='onnx-model')

    first = store.seal(candidate)
    second = store.seal(candidate)

    assert first.uri == second.uri
    assert first.uri.startswith('s3://artifacts/rightarm/objects/sha256/')
    assert first.validation_uri == candidate.uri
    assert 'validation_uri' not in first.model_dump(mode='json')
    assert client.put_count == 1

    artifact = ArtifactRef(
        id=f'onnx-model:sha256:{first.sha256}',
        kind=first.kind,
        uri=first.uri,
        sha256=first.sha256,
        size_bytes=first.size_bytes,
    )
    destination = store.materialize(artifact, tmp_path / 'inputs' / 'model.onnx')
    assert destination.read_bytes() == b'portable-model'
    assert store.materialize(artifact, destination) == destination


def test_candidate_key_is_idempotent_and_cannot_change_content(tmp_path: Path):
    store = S3ArtifactStore(
        tmp_path / 'cache', bucket='artifacts', client=FakeS3Client()
    )
    source = tmp_path / 'output.bin'
    source.write_bytes(b'first')

    first = store.stage(source, kind='model', candidate_key='attempt:job:model')
    repeated = store.stage(source, kind='model', candidate_key='attempt:job:model')

    assert repeated.candidate_id == first.candidate_id
    assert repeated.uri == first.uri
    source.write_bytes(b'different')
    with pytest.raises(InvariantViolation, match='reused for different content'):
        store.stage(source, kind='model', candidate_key='attempt:job:model')


def test_s3_store_rejects_corrupt_existing_cas_metadata(tmp_path: Path):
    client = FakeS3Client()
    store = S3ArtifactStore(tmp_path / 'cache', bucket='artifacts', client=client)
    source = tmp_path / 'checkpoint.pt'
    source.write_bytes(b'checkpoint')
    candidate = store.stage(source, kind='best-checkpoint')
    key = store._object_key(candidate.sha256)
    client.objects[('artifacts', key)] = {
        'body': source.read_bytes(),
        'metadata': {'sha256': '0' * 64, 'size-bytes': str(candidate.size_bytes)},
    }

    with pytest.raises(InvariantViolation, match='metadata is corrupt'):
        store.seal(candidate)


def test_s3_store_rejects_artifact_outside_configured_prefix(tmp_path: Path):
    store = S3ArtifactStore(
        tmp_path / 'cache', bucket='artifacts', prefix='rightarm', client=FakeS3Client()
    )
    artifact = ArtifactRef(
        id=f'model:sha256:{"0" * 64}',
        kind='model',
        uri=f's3://artifacts/another-prefix/{"0" * 64}',
        sha256='0' * 64,
        size_bytes=1,
    )

    with pytest.raises(InvariantViolation, match='immutable S3 object prefix'):
        store.materialize(artifact, tmp_path / 'input')
