"""Immutable content-addressed storage for outer-loop artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from openhands.mlops_loop.errors import InvariantViolation
from openhands.mlops_loop.models import ArtifactCandidate, ArtifactRef


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class ArtifactStore(Protocol):
    """Storage boundary used by the deterministic controller and MCP gateway."""

    def stage(
        self,
        source: str | Path,
        *,
        kind: str,
        media_type: str | None = None,
        metadata: dict[str, object] | None = None,
        candidate_key: str | None = None,
    ) -> ArtifactCandidate: ...

    def verify(self, candidate: ArtifactCandidate) -> None: ...

    def seal(self, candidate: ArtifactCandidate) -> ArtifactCandidate: ...

    def materialize(self, artifact: ArtifactRef, destination: str | Path) -> Path: ...


class LocalArtifactStore:
    """A local CAS with the same immutable semantics expected from S3.

    Files are first staged as untrusted candidates.  Only the outer-loop
    validator can seal them into ``objects/sha256``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.staging = self.root / 'staging'
        self.objects = self.root / 'objects' / 'sha256'
        self.staging.mkdir(parents=True, exist_ok=True)
        self.objects.mkdir(parents=True, exist_ok=True)

    def stage(
        self,
        source: str | Path,
        *,
        kind: str,
        media_type: str | None = None,
        metadata: dict[str, object] | None = None,
        candidate_key: str | None = None,
    ) -> ArtifactCandidate:
        source_path = Path(source).resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f'artifact source is not a file: {source_path}')
        digest, size = sha256_file(source_path)
        candidate_id = (
            f'artifact-candidate:{hashlib.sha256(candidate_key.encode()).hexdigest()}'
            if candidate_key is not None
            else f'artifact-candidate:{uuid4().hex}'
        )
        staged_path = self.staging / candidate_id.replace(':', '_')
        if staged_path.exists():
            existing_digest, existing_size = sha256_file(staged_path)
            if (existing_digest, existing_size) != (digest, size):
                raise InvariantViolation(
                    'artifact candidate key was reused for different content'
                )
        else:
            try:
                self._copy_exclusive(source_path, staged_path)
            except FileExistsError:
                existing_digest, existing_size = sha256_file(staged_path)
                if (existing_digest, existing_size) != (digest, size):
                    raise InvariantViolation(
                        'artifact candidate key raced with different content'
                    ) from None
        return ArtifactCandidate(
            candidate_id=candidate_id,
            kind=kind,
            uri=staged_path.as_uri(),
            sha256=digest,
            size_bytes=size,
            media_type=media_type,
            metadata=metadata or {},
        )

    def seal(self, candidate: ArtifactCandidate) -> ArtifactCandidate:
        source_path = self._staged_path(candidate.uri)
        actual_digest, actual_size = sha256_file(source_path)
        if (actual_digest, actual_size) != (candidate.sha256, candidate.size_bytes):
            raise InvariantViolation(
                f'artifact candidate {candidate.candidate_id} changed after staging'
            )
        target = self.objects / actual_digest[:2] / actual_digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing_digest, existing_size = sha256_file(target)
            if (existing_digest, existing_size) != (actual_digest, actual_size):
                raise InvariantViolation(f'CAS object is corrupt: {target}')
        else:
            self._copy_exclusive(source_path, target)
        return candidate.model_copy(update={'uri': target.as_uri()})

    def verify(self, candidate: ArtifactCandidate) -> None:
        path = self._staged_path(candidate.uri)
        digest, size = sha256_file(path)
        if digest != candidate.sha256 or size != candidate.size_bytes:
            raise InvariantViolation(
                f'artifact {candidate.candidate_id} does not match its manifest'
            )

    def materialize(self, artifact: ArtifactRef, destination: str | Path) -> Path:
        """Copy one sealed object to an executor-owned input path."""

        source = self._sealed_path(artifact.uri)
        return self._materialize_file(source, artifact, Path(destination))

    def _staged_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != 'file' or parsed.netloc not in {'', 'localhost'}:
            raise ValueError(f'LocalArtifactStore only accepts file URIs, got {uri!r}')
        path = Path(unquote(parsed.path)).resolve(strict=True)
        if not path.is_relative_to(self.staging):
            raise InvariantViolation(
                'artifact candidate is outside the untrusted staging area'
            )
        return path

    def _sealed_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != 'file' or parsed.netloc not in {'', 'localhost'}:
            raise ValueError(f'LocalArtifactStore only accepts file URIs, got {uri!r}')
        path = Path(unquote(parsed.path)).resolve(strict=True)
        if not path.is_relative_to(self.objects):
            raise InvariantViolation('artifact is outside the sealed object area')
        return path

    @classmethod
    def _materialize_file(
        cls, source: Path, artifact: ArtifactRef, destination: Path
    ) -> Path:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            digest, size = sha256_file(destination)
            if (digest, size) != (artifact.sha256, artifact.size_bytes):
                raise InvariantViolation(
                    f'materialized artifact destination is corrupt: {destination}'
                )
            return destination
        temporary = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
        try:
            cls._copy_exclusive(source, temporary)
            digest, size = sha256_file(temporary)
            if (digest, size) != (artifact.sha256, artifact.size_bytes):
                raise InvariantViolation(
                    f'artifact {artifact.id} does not match its sealed manifest'
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @staticmethod
    def _copy_exclusive(source: Path, target: Path) -> None:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            with os.fdopen(descriptor, 'wb') as output, source.open('rb') as input_:
                shutil.copyfileobj(input_, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            target.unlink(missing_ok=True)
            raise


class S3ArtifactStore(LocalArtifactStore):
    """S3-backed CAS with local untrusted staging and controlled downloads."""

    def __init__(
        self,
        root: str | Path,
        *,
        bucket: str,
        prefix: str = 'agentic-mlops',
        endpoint_url: str | None = None,
        region_name: str | None = None,
        client: Any | None = None,
    ):
        if not bucket.strip():
            raise ValueError('S3 artifact bucket must not be empty')
        super().__init__(root)
        self.bucket = bucket
        self.prefix = prefix.strip('/')
        self.client = client or boto3.client(
            's3', endpoint_url=endpoint_url, region_name=region_name
        )

    def seal(self, candidate: ArtifactCandidate) -> ArtifactCandidate:
        self.verify(candidate)
        source = self._staged_path(candidate.uri)
        key = self._object_key(candidate.sha256)
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if self._error_code(exc) not in {'404', 'NoSuchKey', 'NotFound'}:
                raise
            try:
                with source.open('rb') as stream:
                    self.client.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=stream,
                        ContentLength=candidate.size_bytes,
                        ContentType=candidate.media_type or 'application/octet-stream',
                        Metadata={
                            'sha256': candidate.sha256,
                            'size-bytes': str(candidate.size_bytes),
                        },
                        IfNoneMatch='*',
                    )
            except ClientError as put_exc:
                if self._error_code(put_exc) not in {
                    '409',
                    '412',
                    'ConditionalRequestConflict',
                    'PreconditionFailed',
                }:
                    raise
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        self._verify_remote_head(candidate, head)
        sealed_uri = f's3://{self.bucket}/{key}'
        return candidate.model_copy(
            update={'uri': sealed_uri, 'validation_uri': candidate.uri}
        )

    def materialize(self, artifact: ArtifactRef, destination: str | Path) -> Path:
        parsed = urlparse(artifact.uri)
        if parsed.scheme == 'file':
            return super().materialize(artifact, destination)
        if parsed.scheme != 's3' or parsed.netloc != self.bucket:
            raise InvariantViolation(
                f'artifact {artifact.id} is outside the configured S3 bucket'
            )
        key = unquote(parsed.path.lstrip('/'))
        if key != self._object_key(artifact.sha256):
            raise InvariantViolation(
                f'artifact {artifact.id} is outside the immutable S3 object prefix'
            )
        destination_path = Path(destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            digest, size = sha256_file(destination_path)
            if (digest, size) != (artifact.sha256, artifact.size_bytes):
                raise InvariantViolation(
                    f'materialized artifact destination is corrupt: {destination_path}'
                )
            return destination_path
        temporary = destination_path.with_name(
            f'.{destination_path.name}.{uuid4().hex}.tmp'
        )
        try:
            self.client.download_file(self.bucket, key, str(temporary))
            digest, size = sha256_file(temporary)
            if (digest, size) != (artifact.sha256, artifact.size_bytes):
                raise InvariantViolation(
                    f'downloaded artifact {artifact.id} failed hash verification'
                )
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)
        return destination_path

    def _object_key(self, digest: str) -> str:
        suffix = f'objects/sha256/{digest[:2]}/{digest}'
        return f'{self.prefix}/{suffix}' if self.prefix else suffix

    @staticmethod
    def _error_code(exc: ClientError) -> str:
        return str(exc.response.get('Error', {}).get('Code', ''))

    @staticmethod
    def _verify_remote_head(
        candidate: ArtifactCandidate, head: Mapping[str, Any]
    ) -> None:
        metadata = head.get('Metadata')
        if not isinstance(metadata, Mapping):
            metadata = {}
        if (
            head.get('ContentLength') != candidate.size_bytes
            or metadata.get('sha256') != candidate.sha256
            or metadata.get('size-bytes') != str(candidate.size_bytes)
        ):
            raise InvariantViolation(
                f'S3 CAS object metadata is corrupt for sha256:{candidate.sha256}'
            )
