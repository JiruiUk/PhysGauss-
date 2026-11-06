"""Presentation-only subsets: all three declared anchors, offsets -24/0/+24."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().with_name('report.py')
spec=importlib.util.spec_from_file_location('v0_report',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
p=r.read(path.with_name('protocol.json'))
for a in p['anchors']:
    ids=[a-24,a,a+24];q=r.read(r.REPORT/'J7.json')
    rows={(v['camera_index'],v['time_index']):v for v in q['rows']}
    paths=[r.OUT/'J7'/'native1600'/f'c{c:03d}_t{a:03d}_{kind}.png' for kind in ['joint','gray'] for c in ids]
    labels=[f'{kind} | camera {c} | angle {rows[c,a]["camera_time_distance"]["view_direction_deg"]:.1f} deg' for kind in ['joint','gray'] for c in ids]
    r.panel(paths,labels,f'J7, fixed time index {a}. Top: RGB. Bottom: gray dynamic support. Green: observed camera.',
        r.OUT/'figures'/f'J7_a{a}_explanation.png',3,[c==a for kind in range(2) for c in ids],size=(480,270))
print('V0_PRESENTATION_ONLY no new measurements')
