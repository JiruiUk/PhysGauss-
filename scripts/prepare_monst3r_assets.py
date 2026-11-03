"""Fetch pinned official research assets without installing or executing them."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import ssl
import time
import urllib.request
import zipfile

import certifi

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '574cc77ad278bad582f470e5382624e01f8769a7'
CROCO_COMMIT = 'bd6f4e07d5c4f13ae5388efc052dadf142aff754'
HF_COMMIT = '1ea2a9eade23a10898ec73be3fd44b5aa4ea0255'
HF_ROOT = 'https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt'
WEIGHTS_SHA = '53089f3bf16ef8d546d9c00232474f2b42490ce359986a1795795c6dc07a3e57'
WEIGHTS_BYTES = 2284790056
CONFIG_SHA = 'fccd9c31cacd66a9a7a511e1debf8ddc43525d252b2378ffdff584ac8e237071'
SOURCE_VERIFICATION_FILES = {
    'dust3r/model.py': '2e253e3e77a18005e882e5dbb872fa4419bdad16b46af6772b11cfc608ce3ee6',
    'dust3r/cloud_opt/optimizer.py': 'a761b1b0012fb0ad3ea851fbcfc873a01d0fd99f64f0123214597bf635edc21e',
    'dust3r/cloud_opt/base_opt.py': 'aed21d7a60a37e8c7498d169c82e1c14c3441445f1abf6bade17bbb684c132d0'}
REPORT = ROOT / 'experiments/staged_reconstruction/monst3r_geometry'
SOURCE = ROOT / 'third_party/monst3r_574cc77'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def download(url, path):
    assert path.resolve().is_relative_to(ROOT)
    assert not path.exists(), f'Preserve existing asset: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    assert not partial.exists(), f'Inspect unfinished transfer before retry: {partial}'
    last = time.monotonic()
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=60, context=context) as response, partial.open('xb') as output:
        while part := response.read(8 * 1024 * 1024):
            output.write(part)
            if time.monotonic() - last > 20:
                print('MONST3R_DOWNLOAD_BYTES', path.name, output.tell(), flush=True)
                last = time.monotonic()
    partial.rename(path)


def extract_archive(archive, destination):
    assert destination.resolve().is_relative_to(ROOT / 'third_party')
    with zipfile.ZipFile(archive) as stream:
        roots = {PurePosixPath(n.filename).parts[0] for n in stream.infolist()}
        assert len(roots) == 1
        for info in stream.infolist():
            relative = PurePosixPath(info.filename)
            assert '..' not in relative.parts and not relative.is_absolute()
            tail = Path(*relative.parts[1:])
            target = destination / tail
            assert target.resolve().is_relative_to(destination.resolve())
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                assert not target.exists(), f'No overwrite: {target}'
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(info) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-source', action='store_true')
    parser.add_argument('--download-weights', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.download_source:
        receipt = REPORT / 'asset_source.json'
        assert not receipt.exists() and not SOURCE.exists()
        archive = ROOT / 'tmp/monst3r_source_audit/official_source.zip'
        croco = ROOT / 'tmp/monst3r_source_audit/official_croco.zip'
        download(f'https://codeload.github.com/Junyi42/monst3r/zip/{SOURCE_COMMIT}', archive)
        download(f'https://codeload.github.com/junyi42/croco/zip/{CROCO_COMMIT}', croco)
        extract_archive(archive, SOURCE)
        extract_archive(croco, SOURCE / 'croco')
        # Confirm whole-archive source matches the independently fetched files.
        for relative, expected in SOURCE_VERIFICATION_FILES.items():
            assert sha(SOURCE / relative) == expected
        result = {'source_commit': SOURCE_COMMIT, 'croco_commit': CROCO_COMMIT,
                  'source_root': SOURCE.relative_to(ROOT).as_posix(),
                  'source_archive_sha256': sha(archive), 'croco_archive_sha256': sha(croco),
                  'source_files_sha256': {p.relative_to(SOURCE).as_posix(): sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()},
                  'installed_or_executed_downloaded_code': False}
        receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print('MONST3R_SOURCE_FETCHED', SOURCE, flush=True)
    if args.download_weights:
        receipt = REPORT / 'asset_weights.json'
        assert not receipt.exists()
        directory = ROOT / 'models/monst3r'
        config, weights = directory / 'config.json', directory / 'model.safetensors'
        download(f'{HF_ROOT}/resolve/{HF_COMMIT}/config.json', config)
        download(f'{HF_ROOT}/resolve/{HF_COMMIT}/model.safetensors', weights)
        assert weights.stat().st_size == WEIGHTS_BYTES and sha(weights) == WEIGHTS_SHA
        assert sha(config) == CONFIG_SHA
        result = {'official_hf_revision': HF_COMMIT, 'repository': HF_ROOT,
                  'weights_path': weights.relative_to(ROOT).as_posix(),
                  'weights_sha256': WEIGHTS_SHA, 'weights_bytes': WEIGHTS_BYTES,
                  'config_sha256': sha(config), 'format': 'safetensors',
                  'model_loaded_or_executed': False, 'geometry_qualified_for_training': False}
        receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print('MONST3R_WEIGHTS_FETCHED_AND_HASH_VERIFIED', WEIGHTS_SHA, flush=True)


if __name__ == '__main__':
    main()
