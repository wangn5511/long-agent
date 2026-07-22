import hashlib
import json
import shutil
from pathlib import Path

from rightarm_mlops.cos_materialize import materialize_cos_batch


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_cos_uses_private_runtime_and_verifies_batch(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / 'remote-batch'
    source.mkdir()
    video = source / 'wave.mp4'
    video.write_bytes(b'video')
    annotations = source / 'annotations.csv'
    annotations.write_text(
        '视频名,起始帧号,终止帧号,类别名\nwave.mp4,0,4,WAVE_IN\n',
        encoding='utf-8',
    )
    manifest = {
        'schema_version': 'rightarm.gesture.batch/v1',
        'batch_id': 'batch-001',
        'created_at': '2026-07-22T00:00:00Z',
        'producer': 'test',
        'label_ontology': 'rightarm-gesture-11/v1',
        'objects': [
            {
                'path': video.name,
                'sha256': _digest(video),
                'size_bytes': video.stat().st_size,
            }
        ],
        'annotations': {
            'path': annotations.name,
            'sha256': _digest(annotations),
            'size_bytes': annotations.stat().st_size,
        },
    }
    (source / 'batch.manifest.json').write_text(json.dumps(manifest))
    (source / '_SUCCESS').write_bytes(b'')
    coscli = tmp_path / 'coscli'
    coscli.write_bytes(b'fake')
    coscli.chmod(0o700)
    config = tmp_path / 'cos.yaml'
    config.write_text('{}')
    config.chmod(0o600)
    monkeypatch.setenv('RIGHTARM_COSCLI_PATH', str(coscli))
    monkeypatch.setenv('RIGHTARM_COS_CONFIG_PATH', str(config))
    monkeypatch.setenv('RIGHTARM_COS_DISK_RESERVE_BYTES', '0')

    def fake_run(argv, check):
        assert check is True
        assert '--secret-id' not in argv
        remote = argv[2]
        destination = Path(argv[3])
        if remote.endswith('batch.manifest.json'):
            shutil.copyfile(source / 'batch.manifest.json', destination)
        elif remote.endswith('_SUCCESS'):
            shutil.copyfile(source / '_SUCCESS', destination)
        else:
            shutil.copytree(source, destination / 'batch-001')

    monkeypatch.setattr('rightarm_mlops.cos_materialize.subprocess.run', fake_run)
    output = tmp_path / 'output'
    output.mkdir()

    report = materialize_cos_batch('cos://bucket/prefix/batch-001/', output)

    assert report['batch_id'] == 'batch-001'
    assert report['verified_object_count'] == 1
    assert report['materialized_source_uri'].startswith('file://')
    assert (output / 'cos-materialization-report.json').is_file()
