"""Pin official public CoTracker3 assets; no repository code is executed."""
import hashlib
import io
from pathlib import Path
import zipfile
import requests
from ..prepare_data import ROOT, REPORT, read, write, sha

EXP=REPORT/'learned_tracks'
SOURCE=ROOT/'third_party/cotracker_research'
WEIGHTS=ROOT/'models/cotracker3/scaled_offline.pth'


def download(url,path,expected):
    if path.exists():
        assert sha(path)==expected
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.download')
    with requests.get(url,stream=True,timeout=45) as response:
        response.raise_for_status()
        with temp.open('wb') as stream:
            for chunk in response.iter_content(1024*1024): stream.write(chunk)
    assert sha(temp)==expected
    temp.replace(path)


def main():
    receipt=EXP/'source_lock.json'
    if receipt.exists():
        lock=read(receipt)
        assert sha(WEIGHTS)==lock['model_sha256']
        assert all(sha(SOURCE/name)==value for name,value in lock['code_sha256'].items())
        print('LEARNED_TRACK_ASSETS_VERIFIED',flush=True); return
    response=requests.get('https://api.github.com/repos/facebookresearch/co-tracker/commits/main',timeout=45)
    response.raise_for_status(); commit=response.json()['sha']
    url=f'https://codeload.github.com/facebookresearch/co-tracker/zip/{commit}'
    archive=requests.get(url,timeout=45); archive.raise_for_status()
    assert not SOURCE.exists(),'Preserve existing source without a provenance receipt'
    SOURCE.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        for info in bundle.infolist():
            relative=Path(*Path(info.filename).parts[1:])
            if not relative.parts or info.is_dir(): continue
            target=(SOURCE/relative).resolve()
            assert target.is_relative_to(SOURCE.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(bundle.read(info))
    response=requests.get('https://huggingface.co/api/models/facebook/cotracker3?blobs=true',timeout=45)
    response.raise_for_status(); model=response.json()
    entry=next(v for v in model['siblings'] if v['rfilename']=='scaled_offline.pth')
    expected=entry['lfs']['sha256']
    model_url=f'https://huggingface.co/facebook/cotracker3/resolve/{model["sha"]}/scaled_offline.pth'
    print('DOWNLOADING_OFFICIAL_COTRACKER3',entry['size'],flush=True)
    download(model_url,WEIGHTS,expected)
    lock={'source_url':'https://github.com/facebookresearch/co-tracker','commit':commit,
          'archive_sha256':hashlib.sha256(archive.content).hexdigest(),
          'model_revision':model['sha'],'model_url':model_url,'model_sha256':expected,'model_bytes':WEIGHTS.stat().st_size,
          'code_sha256':{p.relative_to(SOURCE).as_posix():sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()},
          'user_data_uploaded':False,'repository_code_executed':False,'optimizer_updates':0}
    write(receipt,lock); print('LEARNED_TRACK_ASSETS_PINNED',commit,expected,flush=True)


if __name__=='__main__': main()
