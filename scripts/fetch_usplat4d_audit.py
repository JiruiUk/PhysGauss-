"""Read pinned official text sources for auditing; never import or execute them."""
import argparse
import hashlib
from pathlib import Path
import certifi
import requests
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--download-audit-text', action='store_true'); args = p.parse_args()
    if not args.download_audit_text:
        raise SystemExit('Official-source text download is opt-in')
    exp = ROOT/'experiments/staged_reconstruction/uncertainty_motion'
    tree = read(exp/'official_tree.json'); commit = tree['commit']
    assert commit == 'c7a55211395c4020fc28fd61a09b172abf773710'
    destination = ROOT/'third_party/usplat4d_audit_c7a5521'
    sources = ['README.md', 'USplat4d_vMoSca/mosca_reconstruct.py']
    sources += ['USplat4d_vMoSca/lib_ugraph/'+name for name in
        ['ugraph_gs.py', 'ugraph_helper.py', 'ugraph_model4.py', 'ugraph_utils.py', 'loss_dy.py', 'loss_rigidity.py']]
    sources += ['.gitmodules', 'install_usplat4d_env.sh',
        'USplat4d_vMoSca/lib_usplat4d_prep/usplat4d_prep.py',
        'USplat4d_vMoSca/lib_render/gauspl_renderer_native_add3.py',
        'USplat4d_vMoSca/lib_render/diff-gaussian-rasterization-alphadep-add3/cuda_rasterizer/forward.cu',
        'USplat4d_vMoSca/mosca_cfg.py', 'USplat4d_vMoSca/profile/demo/demo_fit.yaml']
    lookup = {r['path']: r for r in tree['tree']['tree']}
    session = requests.Session(); session.verify = certifi.where()
    result = []
    for name in sources:
        expected = lookup[name]
        assert expected['type'] == 'blob'
        path = destination/name
        if path.exists():
            value = path.read_bytes()
        else:
            response = session.get(f'https://raw.githubusercontent.com/TAMU-Visual-AI/usplat4d/{commit}/{name}', timeout=30)
            response.raise_for_status(); value = response.content
        assert hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest() == expected['sha']
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
        result.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'official_blob_sha': expected['sha']})
    write(exp/'official_source.json', {'official_repository': 'https://github.com/TAMU-Visual-AI/usplat4d',
        'commit': commit, 'downloaded_audit_text': result, 'script_sha256': sha(__file__),
        'remote_code_executed': False, 'dependency_installations': 0})
    print('USPLAT_OFFICIAL_TEXT_SOURCES_VERIFIED', len(result), commit, flush=True)


if __name__ == '__main__':
    main()
