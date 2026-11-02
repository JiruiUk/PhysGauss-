"""Read-only V2 observation check at the candidate's actual training time step."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
from variants.coherent_momentum.evaluate import motion_module
from variants.coherent_momentum.run import RUNS,NAME,REPORTS,assert_frozen
assert_frozen(ROOT/'variants/coherent_momentum/code');m=motion_module()
original=m.temporal.velocity
def operational(*args,**kwargs):
    if len(args)<8 and 'h' not in kwargs:kwargs['h']=1/199
    return original(*args,**kwargs)
m.temporal.velocity=operational
try:
    result=m.evaluate(RUNS/NAME/'rubber_duck','best',m.read(m.CACHE/'manifest.json'))
finally:m.temporal.velocity=original
(REPORTS/'operational_motion.json').write_text(json.dumps({'status':'complete','h':1/199,
    'purpose':'secondary diagnostic at actual N1 training velocity step; prospective comparison remains frozen V2 h=.001 for both models',
    'result':result},indent=2)+'\n',encoding='utf8')
print('Operational h motion:',result['metrics'])
