"""Review-only analytic plot and lossless aggregate archives, no GT arrays."""
from pathlib import Path
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha

BASE=ROOT/'experiments/staged_reconstruction/S1_photometric_oracle'
REPORT=BASE/'v1'; OUT=ROOT/'output/staged_reconstruction/S1_photometric_oracle/v1'


def main():
    assert read(REPORT/'independent_audit.json')['passed']
    values=read(REPORT/'analytic_controls.json')['controls']
    fig,axes=plt.subplots(3,2,figsize=(10,9),constrained_layout=True)
    for row,(name,c) in enumerate(values.items()):
        grid=np.asarray(c['grid_depth']); curves=np.asarray(c['fit_curve'])
        for j in range(2):
            ax=axes[row,j]; gt=c['true_depth'][j]; minimum=c['minimum_depth'][j]
            ax.plot(grid,curves[j],label='original fit cost')
            ax.axvline(gt,color='#238b45',label=f'true {gt:.5f}')
            ax.axvline(minimum,color='#cb181d',linestyle='--',label=f'grid min {minimum:.5f}')
            ax.scatter([gt],[c['true_fit'][j]],color='#238b45',s=24,zorder=5)
            ax.set_xlim(3.7,4.4); ax.set_ylim(-.00005,.008)
            ax.set_title(f'{name}, pixel {c["uv"][j]}'); ax.set_xlabel('reference camera depth'); ax.set_ylabel('best3 of4 RGB fit cost')
            ax.legend(fontsize=8)
    fig.suptitle('D5 fixed analytic controls: evaluator only; no optimized reconstruction')
    plot=OUT/'analytic_control_curves.png'; assert not plot.exists(); fig.savefig(plot,dpi=140); plt.close(fig)
    archives={}
    for case in ['diffuse','glossy']:
        source=REPORT/f'{case}_per_view.json'; target=source.with_suffix('.json.gz')
        assert not target.exists(); target.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
        assert gzip.decompress(target.read_bytes())==source.read_bytes()
        archives[case]={'original':source.relative_to(ROOT).as_posix(),'original_sha256':sha(source),
            'lossless_archive':target.relative_to(ROOT).as_posix(),'archive_sha256':sha(target),
            'original_bytes':source.stat().st_size,'archive_bytes':target.stat().st_size}
    write(REPORT/'exports.json',{'archives':archives,'plot':{plot.relative_to(ROOT).as_posix():sha(plot)},
        'source_sha256':sha(Path(__file__)),'no_privileged_pixel_array_or_GT_training_asset':True})
    print('D5_REVIEW_EXPORT_COMPLETE',archives,flush=True)


if __name__=='__main__': main()
