"""Clarify local validity at every observation of the display-only track."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variants.staged_reconstruction.i10_joint_scale import io


def main():
    a=io.read(io.REPORT/'audit_v2.json')
    passing=sorted([q for q in a['candidate']['motion'] if q['passed']],key=lambda q:q['id'])
    assert passing
    q=passing[0];fig,axes=plt.subplots(3,4,figsize=(13,6.5))
    for t,ax in enumerate(axes.flat):
        good=q['local_eligible'][t];ax.imshow(io.rgb(q['start']+t))
        ax.scatter(*q['uv'][t],marker='+',color='#00ff55' if good else '#ff9f20',s=100);ax.axis('off')
        ax.set_title(f'train {q["start"]+t}: {"fit" if t%2==0 else "check"}; {"local valid" if good else "unknown"}',fontsize=9)
    fig.suptitle(f'POSTHOC observed-RGB illustration: {q["id"]}, row {q["row"]}\nGreen=locally usable; orange=unknown. Global gate failed; no training. Parent motion is not GT.',fontsize=11)
    fig.tight_layout();path=io.OUT/'local_track_validity.png';fig.savefig(path,dpi=125);plt.close(fig)
    io.write(io.REPORT/'track_figure.json',dict(path=path.relative_to(io.ROOT).as_posix(),sha256=io.sha(path),
        source_sha256=io.sha(Path(__file__)),original_gallery_preserved=True,
        reason='Original gallery marked observed UVs green even at uncalibrated first frame; new display distinguishes validity without changing data or gates.',
        local_valid_observations=q['valid'],fit_observations=q['fit_count'],check_observations=q['held_count'],new_training_result=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--enable-joint-scale-audit',action='store_true');args=p.parse_args()
    if not args.enable_joint_scale_audit:raise SystemExit('Joint scale audit default off')
    with io.lock():main()
