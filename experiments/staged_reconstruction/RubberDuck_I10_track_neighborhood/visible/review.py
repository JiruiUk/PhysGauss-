"""Independent CPU accounting and read-only source-RGB neighborhood diagrams."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variants.staged_reconstruction.i10_tracking.io import ROOT, DATA, read, write, sha
from variants.staged_reconstruction.i10_tracking.core import project_world

BASE = Path(__file__).parent; REPORT = BASE/'v1'
OUT = ROOT/'output/staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1'


def main():
    p = read(BASE/'protocol.json'); result = read(REPORT/'audit.json')
    assert read(REPORT/'independent_replay.json')['passed']
    for receipt in [BASE/'preregistration.json', REPORT/'source_freeze.json']:
        d = read(receipt)
        for name, h in d.get('files', d).items(): assert sha(ROOT/name) == h, name
    asset = np.load(ROOT/p['trajectories']); xyz, views, k = [asset[n] for n in ['xyz', 'views', 'k']]
    projected, depth = project_world(xyz, views, k)
    source = {r['id']:r for r in read(ROOT/p['evidence'])['rows'] if r['stationarity']['label'] == 'motion_candidate'}
    assert len(source) == len(result['rows']) == 222
    bands = []
    for band in range(3):
        rows = [r for r in result['rows'] if r['band'] == band]
        bands.append(dict(band=band,motion_candidates=len(rows),fit_candidate=sum('row' in r['binding'] for r in rows),
                          reciprocal=sum(r['reciprocal']['passed'] for r in rows),
                          visibility=sum(r['visibility']['passed'] for r in rows),
                          accepted=sum(r['accepted'] for r in rows),
                          graph=sum(r['accepted'] and r['neighborhood']['qualified'] for r in rows)))
    for row in result['rows']:
        assert row['uv'] == source[row['id']]['uv'] and row['start'] == source[row['id']]['start']
        sl = slice(row['start'], row['start']+12)
        errors = np.linalg.norm(projected[sl]-np.array(row['uv'])[:,None], axis=-1)
        for key, label, fit, held in [('binding','fit',p['fit_observations'],p['held_observations']),
                                      ('swapped_binding','held',p['held_observations'],p['fit_observations'])]:
            saved = row[key]; score = np.sqrt(np.mean(errors[fit]**2, axis=0))
            expected = []
            for i in row['candidate_rows'][label]:
                v = row['visibility_by_row'][str(i)]
                if sum(v[t]['passed'] for t in fit) >= 4 and np.isfinite(errors[:,i]).all() and (depth[sl,i]>.01).all() and score[i]<=2:
                    expected.append(i)
            expected.sort(key=lambda i:(score[i],i))
            assert expected == saved['eligible_rows']
            if not expected:
                assert 'row' not in saved; continue
            i = expected[0]; assert i == saved['row']
            assert abs(float(score[i])-saved['fit_rms_px']) < 1e-12
            assert abs(float(np.sqrt(np.mean(errors[held,i]**2)))-saved['held_rms_px']) < 1e-12
            assert abs(float(errors[:,i].max())-saved['max_px']) < 1e-12
            rivals=[float(score[j]) for j in expected[1:] if np.linalg.norm(xyz[row['start']+6,i]-xyz[row['start']+6,j])>=.05]
            assert (min(rivals) if rivals else None) == saved['alternative_fit_rms_px']
        graph = row['neighborhood']
        if 'anchor' in graph:
            delta = xyz[sl].astype(np.float64)-xyz[sl,graph['anchor']:graph['anchor']+1].astype(np.float64)
            dist = np.linalg.norm(delta,axis=-1)
            candidates = [i for i in range(xyz.shape[1]) if i!=graph['anchor'] and dist[6,i]<=.05]
            candidates.sort(key=lambda i:(dist[6,i],i));candidates=candidates[:8]
            expected=[i for i in candidates if dist[:,i].max()<=.05 and np.sqrt(np.mean(np.sum((delta[:,i]-delta[6,i])**2,axis=-1)))<=.02]
            assert candidates==graph['candidate_rows'] and expected==graph['rows']
    accepted=[r for r in result['rows'] if r['accepted']]
    graphs=[r for r in accepted if r['neighborhood']['qualified']]
    assert [r['accepted'] for r in bands] == result['summary']['bindings']['by_band']
    assert [r['graph'] for r in bands] == result['summary']['graph']['by_band']
    graph_ids={i for r in graphs for i in r['neighborhood']['rows']}
    assert len(graph_ids)==677 and len(graphs)==116 and len(accepted)==122
    changes=dict(previously_admitted=sum(bool((r.get('old_binding') or {}).get('accepted')) for r in accepted),
                 newly_verified=sum(not (r.get('old_binding') or {}).get('accepted',False) for r in accepted),
                 same_old_ID=sum((r.get('old_binding') or {}).get('row')==r['binding']['row'] for r in accepted))
    write(REPORT/'independent_accounting.json',dict(passed=True,bands=bands,changes=changes,
          all222_source_UV_and_split_rankings_and_costs=True,all_persistent_graph_distances_and_rows=True,
          no_new_GT_or_training=True,source_sha256=sha(Path(__file__))))
    meta=read(DATA/'transforms_train.json');OUT.mkdir(parents=True,exist_ok=True)
    sheet=Image.new('RGB',(1200,750),'#151923');draw=ImageDraw.Draw(sheet)
    for y,start in enumerate([0,52,108]):
        rows=[r for r in graphs if r['start']==start]
        for col,local in enumerate([0,6,11]):
            index=start+local;frame=meta['frames'][index]
            rgba=np.array(Image.open(DATA/(frame['file_path']+'.png')).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
            tile=Image.fromarray(np.uint8(rgba[...,:3].astype(float)*rgba[...,3:4]/255));d=ImageDraw.Draw(tile)
            for row in rows:
                anchor=projected[index,row['binding']['row']];target=np.array(row['uv'][local])
                for neighbor in row['neighborhood']['rows']:
                    point=projected[index,neighbor]
                    d.line([tuple(anchor),tuple(point)],fill='#00c8da',width=1)
                    d.ellipse((point[0]-1,point[1]-1,point[0]+1,point[1]+1),fill='#00c8da')
                d.ellipse((target[0]-3,target[1]-3,target[0]+3,target[1]+3),outline='#33ff77',width=1)
            x0,y0=col*400,y*250
            draw.text((x0+3,y0+3),f'train{index}: green=RGB track; cyan=inferred neighbors',fill='white')
            sheet.paste(tile,(x0,y0+23))
    path=OUT/'local_neighborhoods_RGB.png';sheet.save(path)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,values,title in [(axes[0],[r['motion_candidates'] for r in bands],'Original RGB motion candidates'),
                            (axes[1],[r['graph'] for r in bands],'Qualified local anchor tracks')]:
        ax.bar(['Early','Middle','Late'],values,color=['#2879ad','#d98f25','#669c39'])
        for i,value in enumerate(values):ax.text(i,value+2,str(value),ha='center')
        ax.set_ylim(0,max(values)*1.2);ax.set_title(title);ax.set_ylabel('Track count')
    axes[1].axhline(10,color='#b62930',ls='--',label='Frozen minimum per third:10');axes[1].legend(fontsize=8)
    fig.suptitle('Model-conditioned binding audit; no new Gaussian training or independent 3D GT')
    fig.savefig(OUT/'temporal_coverage.png',dpi=160);plt.close(fig)
    write(REPORT/'diagrams.json',{f.relative_to(ROOT).as_posix():sha(f) for f in [path,OUT/'temporal_coverage.png']})
    print('T4_INDEPENDENT_ACCOUNTING_AND_DIAGRAMS_PASS',bands,changes,flush=True)


if __name__=='__main__':main()
