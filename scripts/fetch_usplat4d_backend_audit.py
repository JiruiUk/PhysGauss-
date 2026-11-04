"""Pin and read the official linked gsplat fork; no remote code execution."""
import argparse
import hashlib
import certifi
import requests
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-audit-text', action='store_true')
    args = parser.parse_args()
    if not args.download_audit_text:
        raise SystemExit('Source-text download is opt-in')
    exp = ROOT/'experiments/staged_reconstruction/uncertainty_motion'
    official = read(exp/'official_tree.json')
    link = next(r for r in official['tree']['tree'] if r['path'] == 'external/gsplat')
    assert link['type'] == 'commit'
    commit = link['sha']
    assert commit == '2486a4ad5343a4e1a6f57fcdd2845c17577816d2'
    session = requests.Session(); session.verify = certifi.where()
    def get(url):
        response = session.get(url, timeout=40); response.raise_for_status()
        return response
    tree_path = exp/'backend_tree.json'
    if tree_path.exists():
        tree = read(tree_path)
        assert tree['commit'] == commit
    else:
        value = get(f'https://api.github.com/repos/percool/gsplat/git/trees/{commit}?recursive=1').json()
        assert not value['truncated']
        tree = {'repository': 'https://github.com/percool/gsplat', 'commit': commit, 'tree': value}
        write(tree_path, tree)
    sources = [r for r in tree['tree']['tree'] if r['type'] == 'blob' and (
        r['path'] == 'gsplat/rendering.py' or
        ('RasterizeToPixels3DGS' in r['path'] and r['path'].endswith(('.cu', '.h'))))]
    assert sources
    result = []
    for row in sources:
        path = ROOT/'third_party/usplat4d_gsplat_audit_2486a4a'/row['path']
        value = path.read_bytes() if path.exists() else get(f'https://raw.githubusercontent.com/percool/gsplat/{commit}/{row["path"]}').content
        assert hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest() == row['sha']
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
        result.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'official_blob_sha': row['sha']})
    write(exp/'backend_source.json', {'repository': tree['repository'], 'commit': commit,
        'linked_by_official_commit': official['commit'], 'downloaded_audit_text': result,
        'script_sha256': sha(__file__), 'remote_code_executed': False, 'dependency_installations': 0})
    print('USPLAT_BACKEND_SOURCES_VERIFIED', commit, [r['path'] for r in result], flush=True)


if __name__ == '__main__':
    main()
