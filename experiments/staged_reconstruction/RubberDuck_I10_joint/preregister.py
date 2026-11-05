"""Fix the new human-authorized source combination and single matched trial."""
from variants.staged_reconstruction.prepare_data import ROOT,read,sha,write

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_joint'


def main():
    assert not (BASE/'protocol.json').exists()
    p=read(ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot/protocol.json')
    p.update(stage='J6 I10 dynamic plus accepted J4 static RGB joint pilot',
        human_authorization='当前效果可以接受，同时把I10模型也放入其中进行联合训练',
        parent='runs/pidg_staged_reconstruction/background_residual/formal/I10_static_residual/checkpoint.pt',
        parent_sha256='1a87d0daa1d5a39c94c84611568b38d94d13c5ad6225eb556ba82ecbbf926d1a',
        static_parent='runs/pidg_staged_reconstruction/RubberDuck_joint_pilot/v1/formal/joint_depth_trust/checkpoint.pt',
        static_parent_sha256='4123471807797cc2f1319e0dbed2908fcdba83eaeba846a5887ab6aca7354b62',
        parent_state_policy='All non-static I10 model+full dynamic Adam/RNG/sampler/physics graphs; J4 static model+static Adam only',
        source_static_Adam='J4 mixed counters: geometry12000, opacity/SH14000; exact inherited state',
        static_rates='exact original optimizer_for and accepted J4 complete static Adam; no reset',
        dynamic_rates='exact original union.optimizer and I10 complete dynamic/physics Adam; no reset',
        self_depth='unchanged J3 loss against accepted J4 frozen static teacher; opacity-only; parent alpha>.5; self-reference not GT',
        I10_background_target_neutralization_enabled=False,
        new_full_RGB_objective_not_a_repeat_of_I10=True,
        initial_label='I10_J4_initial',human_accepts_static_start=True,
        old_nonpromotion_and_failure_records_unchanged=True,
        continuation_end='Complete this fixed trial and report; no automatic extension',
        static_visual_nonregression_required=True)
    for key in ['parent','static_parent']:assert sha(ROOT/p[key])==p[key+'_sha256']
    write(BASE/'protocol.json',p)
    write(BASE/'preregistration.json',{'files':{x.relative_to(ROOT).as_posix():sha(x) for x in [BASE/'plan.md',BASE/'protocol.json']}})
    write(BASE/'progress.json',{'status':'preregistered_not_trained','formal_updates':0,'old_paused_tasks_not_resumed':True})
    print('J6_PREREGISTERED',sha(BASE/'protocol.json'))


if __name__=='__main__':main()
