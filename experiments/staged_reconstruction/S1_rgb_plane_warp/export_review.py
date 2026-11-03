"""Export a declared control comparison; independent original GT hash audit."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha

BASE=ROOT/'experiments/staged_reconstruction/S1_rgb_plane_warp'
REPORT=BASE/'v1'; OUT=ROOT/'output/staged_reconstruction/S1_rgb_plane_warp/v1'
D5=ROOT/'experiments/staged_reconstruction/S1_photometric_oracle/v1'
SOURCE=ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'


def main():
    assert read(REPORT/'input_audit.json')['passed'] and (REPORT/'evaluation.json').exists()
    # Reuse the already frozen original v2 GT receipt, not a new result-dependent GT.
    freeze=read(D5/'source_freeze.json'); relative=(SOURCE/'dataset.json').relative_to(ROOT).as_posix()
    assert sha(SOURCE/'dataset.json')==freeze[relative]
    dataset=read(SOURCE/'dataset.json')
    for name,h in dataset['files'].items(): assert sha(ROOT/name)==h,name
    old=read(D5/'analytic_controls.json')['controls']; new=read(REPORT/'analytic_controls.json')['controls']
    fig,axes=plt.subplots(3,2,figsize=(10,9),constrained_layout=True)
    for row,(name,c) in enumerate(new.items()):
        for col in range(2):
            ax=axes[row,col]; grid=np.asarray(c['grid_depth']); curve=np.asarray(c['fit_curve'][col],dtype=float)
            ax.plot(grid,old[name]['fit_curve'][col],color='#999999',label='original unwarped')
            ax.plot(grid,curve,color='#2171b5',label='D6 raw RGB plane warp')
            ax.axvline(c['true_depth'][col],color='#238b45',label=f'true {c["true_depth"][col]:.5f}')
            ax.axvline(c['minimum_depth'][col],color='#cb181d',linestyle='--',label=f'D6 min {c["minimum_depth"][col]:.5f}')
            ax.set_xlim(3.7,4.4); ax.set_ylim(-.00005,.008)
            ax.set_title(name+f', pixel{c["uv"][col]}'); ax.set_xlabel('reference camera depth'); ax.set_ylabel('best3 of4 RGB fit cost'); ax.legend(fontsize=8)
    fig.suptitle('Fixed analytic controls: exact analytic input slopes, NOT dataset slope accuracy')
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/'analytic_control_comparison.png'; assert not path.exists()
    fig.savefig(path,dpi=140); plt.close(fig)
    write(REPORT/'exports.json',{'images':{path.relative_to(ROOT).as_posix():sha(path)},
        'original_GT_receipt_sha256':sha(SOURCE/'dataset.json'),'original_GT_receipt_matches_preexisting_D5_freeze':True,
        'all_original_dataset_hashes_verified':True,'source_sha256':sha(Path(__file__)),
        'control_input_analytic_slopes_never_used_for_dataset':True})
    print('D6_EXPORT_AND_ORIGINAL_GT_INTEGRITY_PASSED',flush=True)


if __name__=='__main__': main()
