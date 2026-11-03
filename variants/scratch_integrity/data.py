"""Rebuild all row bindings from the new common state, using image-only evidence."""
from pathlib import Path
import shutil
import json
import numpy as np
from variants.transfer_integrity import data as parent
from variants.metric_tuning.runtime import read_json,write_json,sha256
from .prefix import ROOT,REPORT,RUNS,PREFIX,plan,step


def prepare():
    n=step();assert n in plan()['common_prefix_candidates']
    report=REPORT/f'binding_{n}';runs=RUNS/f'binding_{n}'
    report.mkdir(parents=True,exist_ok=True);runs.mkdir(parents=True,exist_ok=True)
    source=RUNS/f'common_candidates/{n}/rubber_duck'
    digest=sha256(source/'checkpoints/resume.pt')
    assert digest==sha256(PREFIX/'checkpoints/resume.pt')
    if (report/'cold_binding.json').exists():
        result=read_json(report/'cold_binding.json')
        assert result['source_checkpoint_sha256']==digest and sha256(Path(result['path']))==result['sha256']
        return result
    write_json(report/'plan.json',{'source_sha256':digest,'source_iteration':n,'protocol_sha256':sha256(REPORT/'plan.json'),
               'historical_trained_states_used':False,'reliability_gates':'unchanged v1 gates'})
    # This independent static hypothesis audit contains only image tracks and
    # camera triangulation, no model or Gaussian row IDs.
    image_audit=ROOT/'experiments/transfer_integrity/static_image_audit.json'
    assert sha256(image_audit)==read_json(REPORT/'image_evidence_lock.json')['experiments/transfer_integrity/static_image_audit.json']
    shutil.copyfile(image_audit,report/'static_image_audit.json')
    parent.SOURCE,parent.REPORT,parent.RUNS=source,report,runs
    result=parent.prepare()
    with np.load(result['path']) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['source_iteration']=np.asarray(n)
    # Freeze the own early bank for later support tests; never borrow C1's IDs.
    arrays['early_labels']=np.full(int(arrays['point_count']),-1,dtype=np.int64)
    for j in range(int(arrays['batch_count'])):
        if int(arrays[f'b{j}_clip'])==0:
            arrays['early_labels'][arrays[f'b{j}_ids']]=arrays[f'b{j}_local_labels'];break
    assert (arrays['early_labels']>=0).sum()>=40
    np.savez_compressed(result['path'],**arrays)
    result.update(sha256=sha256(Path(result['path'])),source_iteration=n,source_directory=str(source),
                  random_initial_checkpoint_sha256=read_json(REPORT/'initialization.json')['random_initial_checkpoint_sha256'],
                  historical_Gaussian_row_ids_used=False,pretrained_model_used=False,
                  own_early_bank_count=int((arrays['early_labels']>=0).sum()),wrapper_sha256=sha256(Path(__file__)))
    write_json(report/'data_binding.json',result);write_json(report/'cold_binding.json',result)
    write_json(REPORT/'data_binding.json',result)
    return result


if __name__=='__main__':
    import torch
    torch.set_num_threads(4)
    try:print(json.dumps(prepare(),indent=2),flush=True)
    except AssertionError as error:
        write_json(REPORT/f'binding_{step()}/gate_rejected.json',{'status':'rejected','iteration':step(),'reason':str(error)})
        raise SystemExit(42)
