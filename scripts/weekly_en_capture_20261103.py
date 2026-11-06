"""Capture the frozen Chinese report's authoring calls without rendering or inference."""
import ast
import json
import re
from pathlib import Path
from datetime import datetime
from weekly_image_atlas_20261103 import render_atlas
from weekly_s6_update_20261103 import render_s6
from weekly_test_comparison_20261103 import render_comparison

ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'tmp/pdfs/weekly_20261103/evidence'
DEST=ROOT/'tmp/pdfs/weekly_en_20261103'
DEST.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def capture():
    pages=[]
    env={'PAGE':0}
    def start(*args):
        pages.append({'start':list(args),'blocks':[]})
        env['PAGE']=len(pages)
    def block(kind):
        def call(*args,**kwargs):pages[-1]['blocks'].append({'kind':kind,'args':args,'kwargs':kwargs})
        return call
    src=(ROOT/'scripts/build_weekly_experiment_summary_20261103.py').read_text(encoding='utf-8')
    tree=ast.parse(src)
    refs=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='REFS' for x in n.targets))
    manifest=read(ROOT/'tmp/pdfs/weekly_20261103/snapshot.json')
    s6=read(SNAP/'s6_revision/manifest.json')
    cmp=read(ROOT/'output/pdf/PIDG_Weekly_Experiment_Summary_CN_20261103.test_comparison.json')
    base=SNAP/'s6_revision/experiments/staged_reconstruction/S6'
    env.update(start=start,**{k:block(k) for k in ['p','fig','table','box','heading','compact_comparison_table']},
        REFS=ast.literal_eval(refs.value),CUTOFF=datetime.fromisoformat(manifest['cutoff_utc']).strftime('%Y-%m-%d %H:%M UTC'),
        S6_CUTOFF=datetime.fromisoformat(s6['captured_utc']).strftime('%Y-%m-%d %H:%M UTC'),MANIFEST=manifest,S6_MANIFEST=s6,
        ATLAS=read(SNAP/'atlas/manifest.json')['figure_paths'],render_atlas=render_atlas,render_s6=render_s6,render_comparison=render_comparison,
        S6_ASSETS=s6['figure_paths'],S6_DATA={'val':read(base/'S6_val_04000.json')['metrics'],'test':read(base/'S6_test_04000.json')['metrics'],
        'quality':read(base/'native_full_quality.json'),'final':read(base/'final_verification.json')},COMPARE_DATA=cmp,
        COMPARE_ASSETS={'curves':'test_revision/all_frames.png',**{f'example{i}':f'test_revision/{Path(r["path"]).name}' for i,r in enumerate(cmp['selected_examples'])}})
    first=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='start')
    last=next(i for i in range(first,len(tree.body)) if isinstance(tree.body[i],ast.Expr) and isinstance(tree.body[i].value,ast.Call) and isinstance(tree.body[i].value.func,ast.Attribute) and isinstance(tree.body[i].value.func.value,ast.Name) and tree.body[i].value.func.value.id=='QA')
    exec(compile(ast.Module(body=tree.body[first:last],type_ignores=[]),'<captured-report>','exec'),env)
    assert len(pages)==49
    strings=[]
    def walk(x):
        if isinstance(x,str) and re.search('[\u3400-\u9fff]',x) and x not in strings:strings.append(x)
        elif isinstance(x,(list,tuple)):
            for v in x:walk(v)
        elif isinstance(x,dict):
            for v in x.values():walk(v)
    for page in pages:walk(page)
    (DEST/'pages_cn.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    (DEST/'strings_cn.json').write_text(json.dumps(dict(enumerate(strings)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pages':len(pages),'strings':len(strings),'characters':sum(map(len,strings))}))
    return pages,strings

if __name__=='__main__':capture()
