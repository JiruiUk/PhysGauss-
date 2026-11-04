from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_visibility_audit'

def main():
    assert not (BASE/'protocol.json').exists()
    old=ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot'
    q=read(old/'protocol.json');a=read(old/'v1/evaluation.json')['checkpoint_audits']
    models={'parent6k':{'path':q['parent'],'sha256':q['parent_sha256']}}
    for arm in q['arms']:
        models[arm]={'path':f'runs/pidg_staged_reconstruction/RubberDuck_joint_pilot/v1/formal/{arm}/checkpoint.pt',
            'sha256':a[arm]['checkpoint_sha256']}
    p={'stage':'J5 read-only RubberDuck visibility/coverage audit','default_enabled':False,
        'models':models,'width':1600,'expected_train_views':120,'fixed_cameras':[86,188],
        'fixed_times':[2,100,200],'observed_source_indices':[0,24,72],
        'dark_RGB_mean_lt':.05,'high_alpha_gt':.5,'native_ID_capacity':20,
        'source_support':'same model canonical IDs deformed at source time, native depth-sorted override-color visibility',
        'stored_weights_not_assumed_compositing_weights':True,'packed_groups_per_native_RGB_pass':3,
        'native_single_control':{'width':64,'height':36,'z':4.,'scale':.08,'opacity':.8},
        'state_before_after':'exact model+twoAdam/bindings/RNG/sampler/physics graphs; no updates or backward',
        'native_parity':'RGB/alpha/D exact; packed/unpacked visible weight atol1e-7 rtol1e-6',
        'independent_processes':2,'missing_denominators':'None/unknown; no invented zero error',
        'same_time_camera_evidence':'metadata time keys and exact transform hashes; no assumption of simultaneous views',
        'new_semantic_static_dynamic_or_instance_labels':False,'training_or_GT_exports':False,
        'val_test_RGB_read':False,'threshold_weight_sweep':False,'promotion_authorized':False}
    for row in models.values():assert sha(ROOT/row['path'])==row['sha256']
    write(BASE/'protocol.json',p)
    files={x.relative_to(ROOT).as_posix():sha(x) for x in [BASE/'plan.md',BASE/'protocol.json']}
    write(BASE/'preregistration.json',{'files':files})
    print('J5_PREREGISTERED',sha(BASE/'protocol.json'))

if __name__=='__main__':main()
