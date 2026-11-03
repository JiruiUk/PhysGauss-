"""Freeze original step-zero initialization and unchanged training budget."""
from ..prepare_data import ROOT, REPORT, RUNS, read, write, sha

EXP = REPORT/'early_depth'
WORK = RUNS/'early_depth'
INITIAL = RUNS/'S3_union/warmup/formal/initial.pt'
INITIAL_SHA = '6a4dde83c07f4418dde4549bf2f1cc810f1c001a66614d63505767846dce9b98'


def main():
    assert not (EXP/'protocol.json').exists()
    assert sha(INITIAL) == INITIAL_SHA
    base = read(REPORT/'S3_union/protocol.json')
    plan = read(REPORT/'ordinal_depth/protocol.json')
    plan.update(arm='I4_early_depth', initial=INITIAL.relative_to(ROOT).as_posix(), initial_sha256=INITIAL_SHA,
                original_protocol_sha256=sha(REPORT/'S3_union/protocol.json'),
                warmup_updates=base['warmup_updates'], physics_updates=base['physics_updates'],
                actual_updates=base['warmup_updates']+base['physics_updates'],
                remove_inherited_I1_graph_prior=True,
                matched_control='Original S3-U: same original initial full state, 16k warmup +6k physics and camera/width schedule; only ordinal supervision is added',
                comparison_to_I3_is_not_single_switch=True,
                warmup_width_schedule=base['warmup_width_schedule'], smoke_updates=200,
                physics_phase_admission='Original S3-U numerical warmup guardrails, no final completeness claim')
    write(EXP/'protocol.json',plan)
    WORK.mkdir(parents=True,exist_ok=True)
    print('EARLY_DEPTH_PREPARED',plan['actual_updates'],flush=True)


if __name__=='__main__': main()
