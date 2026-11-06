"""English edition of the frozen 49-page report; no model execution."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import re
import shutil
import subprocess
import importlib.util
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
import pdfplumber
from weekly_en_capture_20261103 import capture
from weekly_en_figures_20261103 import prepare as prepare_figures

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
TMP=ROOT/'tmp/pdfs/weekly_en_20261103'
SNAP=ROOT/'tmp/pdfs/weekly_20261103/evidence'
CN='PIDG_Weekly_Experiment_Summary_CN_20261103'
BASE='PIDG_Weekly_Experiment_Summary_EN_20261103'
PDF=OUT/(BASE+'.pdf')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
W,H=A4;M=46;CW=W-2*M
NAVY=colors.HexColor('#16364D');INK=colors.HexColor('#1B2B3C')
TEAL=colors.HexColor('#087C83');GRAY=colors.HexColor('#596A78')
LIGHT=colors.HexColor('#EEF4F6');RULE=colors.HexColor('#D9E3E8')
pdfmetrics.registerFont(TTFont('EN','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ENB','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('EN',normal='EN',bold='ENB',italic='EN',boldItalic='ENB')

def norm(s):
    return str(s).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')
def fmt(s):
    return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',escape(norm(s))).replace('\n','<br/>')

def style(kind,scale):
    options={'body':(9.6,13.2,INK,'EN'),'small':(8.1,10.9,GRAY,'EN'),
        'source':(7.9,10.2,GRAY,'EN'),'cell':(8.3,10.6,INK,'EN'),
        'head':(8.0,10.2,colors.white,'ENB'),'heading':(11.5,14.3,TEAL,'ENB'),
        'callout':(10.4,14.1,NAVY,'ENB'),'compact':(7.9,9.6,INK,'EN'),
        'compacthead':(7.8,9.6,colors.white,'ENB')}
    size,leading,color,font=options[kind]
    return ParagraphStyle(kind,fontName=font,fontSize=size*scale,leading=leading*scale,
        textColor=color,splitLongWords=True,alignment=1 if kind.startswith('compact') else 0)

def make_table(args,kwargs,scale,compact=False):
    headers,rows=args[:2]
    weights=[.55]+[1]*8 if compact else (args[2] if len(args)>2 else kwargs.get('widths'))
    widths=[CW*v/sum(weights) for v in weights] if weights else [CW/len(headers)]*len(headers)
    hs=style('compacthead' if compact else 'head',scale)
    cs=style('compact' if compact else 'cell',scale)
    data=[[Paragraph(fmt(s),hs) for s in headers]]+[[Paragraph(fmt(s),cs) for s in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    pad=(2.7 if compact else 5.0)*scale
    side=(3 if compact else 6)*scale
    cmds=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE' if compact else 'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),side),('RIGHTPADDING',(0,0),(-1,-1),side),
        ('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad),
        ('LINEBELOW',(0,0),(-1,0),.5,NAVY),('LINEBELOW',(0,1),(-1,-1),.3,RULE)]
    for i in range(2,len(data),2):cmds.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#F1F6F7')))
    highlights=args[3] if len(args)>3 else kwargs.get('highlights',[])
    for i in highlights:cmds.append(('BACKGROUND',(0,i+1),(-1,i+1),colors.HexColor('#E0F0ED')))
    t.setStyle(TableStyle(cmds));_,height=t.wrap(CW,H)
    return t,height

def plan(page,scale):
    items=[]
    def paragraph(text,kind='body',gap=8,width=CW):
        ob=Paragraph(fmt(text),style(kind,scale));_,height=ob.wrap(width,H)
        items.append(('paragraph',height,gap*scale,ob,0))
    if len(page['start'])>2 and page['start'][2]:paragraph(page['start'][2],'small',13)
    for block in page['blocks']:
        kind,args,kwargs=block['kind'],block['args'],block['kwargs']
        if kind=='p':paragraph(args[0],args[1] if len(args)>1 else 'body',min(8,args[2]) if len(args)>2 else 8)
        elif kind=='heading':paragraph(args[0],'heading',7)
        elif kind=='box':
            ob=Paragraph(fmt(args[0]),style('callout',scale));_,height=ob.wrap(CW-28,H)
            items.append(('box',height+22*scale,12*scale,ob,height))
        elif kind in ('table','compact_comparison_table'):
            ob,height=make_table(args,kwargs,scale,kind=='compact_comparison_table')
            items.append(('table',height,4*scale if kind=='compact_comparison_table' else 11*scale,ob,0))
        elif kind=='fig':
            rel,max_h,caption=args
            path=SNAP/rel
            with Image.open(path) as im:iw,ih=im.size
            ratio=min(CW/iw,max_h*scale/ih)
            width,height=iw*ratio,ih*ratio
            items.append(('figure',height,6*scale,path,width))
            paragraph(caption,'small',11)
        else:raise ValueError(kind)
    return items,sum(x[1]+x[2] for x in items)

def main():
    original_hashes={str(p):sha(p) for p in OUT.glob(CN+'.*')}
    pages,strings=capture()
    translations={}
    for line in (ROOT/'scripts/weekly_report_en_20261103.tsv').read_text(encoding='utf-8').splitlines():
        key,value=line.split('\t',1)
        assert int(key) not in translations
        translations[int(key)]=value.replace('\\n','\n')
    assert set(translations)==set(range(len(strings))),set(range(len(strings)))-set(translations)
    mapping={s:translations[i] for i,s in enumerate(strings)}
    def translate(x):
        if isinstance(x,str):return mapping.get(x,x)
        if isinstance(x,(list,tuple)):return [translate(v) for v in x]
        if isinstance(x,dict):return {k:translate(v) for k,v in x.items()}
        return x
    english=translate(pages)
    if importlib.util.find_spec('matplotlib'):
        overrides,figure_record=prepare_figures()
    else:
        subprocess.run([str(ROOT/'envs/pidg-cu128/python.exe'),str(ROOT/'scripts/weekly_en_figures_20261103.py')],check=True,cwd=ROOT)
        figure_record=read(TMP/'english_figure_sources.json')
        overrides={'output/fixed_camera_temporal/key_comparison.png':str(TMP/'figures/key_comparison_en.png'),
            'atlas/velocity.png':str(TMP/'figures/velocity_fit_en.png')}
    for page in english:
        for block in page['blocks']:
            if block['kind']=='fig' and block['args'][0] in overrides:
                block['args'][0]=overrides[block['args'][0]]
    serialized=json.dumps(english,ensure_ascii=False)
    assert not re.search('[\u3400-\u9fff]',serialized)
    numeric_cells=0
    for original,translated in zip(pages,english):
        for a,b in zip(original['blocks'],translated['blocks']):
            if a['kind'] in ('table','compact_comparison_table'):
                for ra,rb in zip(a['args'][1],b['args'][1]):
                    for ca,cb in zip(ra,rb):
                        if isinstance(ca,str) and re.fullmatch(r'[+\-0-9.,%()/\s]+',ca):
                            assert ca==cb,(ca,cb)
                            numeric_cells+=1
    (TMP/'pages_en.json').write_text(json.dumps(english,ensure_ascii=False,indent=2),encoding='utf-8')
    c=canvas.Canvas(str(PDF),pagesize=A4,pageCompression=1)
    c.setTitle('PIDG: Two Weeks of Experiments | 2026.10.26 - 2026.11.03 | English')
    c.setAuthor('PhysGauss / PIDG experiment records')
    c.setSubject('English edition: experiment evidence, formal S6 results and per-frame test comparison')
    layout=[];images={};text=[]
    for number,page in enumerate(english,1):
        selected=None
        for scale in [1.,.98,.96,.94,.92,.90,.88,.86,.84,.82]:
            items,total=plan(page,scale)
            if total<=H-148:
                selected=(items,total,scale);break
        if selected is None:raise ValueError(('Page overflow',number,total))
        items,total,scale=selected
        kicker,title=page['start'][:2]
        c.setFillColor(TEAL);c.rect(M,H-38,26,3,fill=1,stroke=0)
        c.setFillColor(GRAY);c.setFont('EN',7.8)
        c.drawString(M+34,H-38,kicker);c.drawRightString(W-M,H-38,'PIDG / Two-week experiment record')
        ts=min(20.5,503/max(1,pdfmetrics.stringWidth(title,'ENB',1)))
        c.setFont('ENB',ts);c.setFillColor(NAVY);c.drawString(M,H-79,title)
        c.bookmarkPage(f'p{number}');c.addOutlineEntry(title,f'p{number}',0,False)
        y=H-99
        text.extend(['\f',f'{number:02d} {title}'])
        for kind,height,gap,ob,extra in items:
            if kind in ('paragraph','table'):ob.drawOn(c,M,y-height)
            elif kind=='box':
                c.setFillColor(LIGHT);c.roundRect(M,y-height,CW,height,5,fill=1,stroke=0)
                c.setFillColor(TEAL);c.rect(M,y-height,3,height,fill=1,stroke=0)
                ob.drawOn(c,M+14,y-11*scale-extra)
            elif kind=='figure':
                with Image.open(ob) as im:
                    c.drawImage(ImageReader(im.convert('RGB')),M+(CW-extra)/2,y-height,extra,height)
                images[str(ob.relative_to(ROOT))]=sha(ob)
            if kind in ('paragraph','box'):text.append(ob.getPlainText())
            y-=height+gap
        assert y>=49
        c.setStrokeColor(RULE);c.setLineWidth(.5);c.line(M,37,W-M,37)
        c.setFillColor(GRAY);c.setFont('EN',7.4)
        c.drawString(M,24,'2026.10.26 - 2026.11.03 | Rubber Duck')
        c.drawRightString(W-M,24,f'{number:02d} / 49')
        layout.append({'page':number,'scale':scale,'bottom_content_y':round(y,2)})
        c.showPage()
    c.save()
    for suffix in ['.sources.json','.sources.images.json','.sources.s6.json','.test_comparison.csv']:
        shutil.copyfile(OUT/(CN+suffix),OUT/(BASE+suffix))
    comparison=read(OUT/(CN+'.test_comparison.json'))
    comparison['english_edition']={'source_report':CN+'.pdf','numerical_data_unchanged':True}
    for row,reason in zip(comparison['selected_examples'],['Largest full-frame PSNR regression vs H3','Largest full-frame PSNR delta vs H3','Full-frame PSNR delta near the median']):row['reason']=reason
    comparison['CSV']=str((OUT/(BASE+'.test_comparison.csv')).relative_to(ROOT))
    (OUT/(BASE+'.test_comparison.json')).write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/(BASE+'.source.txt')).write_text('\n'.join(text),encoding='utf-8')
    reader=PdfReader(PDF);assert len(reader.pages)==49
    texts=[p.extract_text() for p in reader.pages]
    assert not re.search('[\u3400-\u9fff]','\n'.join(texts))
    assert all(f'{i:02d} / 49' in t for i,t in enumerate(texts,1))
    with pdfplumber.open(PDF) as doc:
        outliers=[(i+1,v['text']) for i,p in enumerate(doc.pages) for v in p.chars if v['x0']<M-1 or v['x1']>W-M+1 or v['top']<20 or v['bottom']>H-17]
        assert not outliers,outliers[:20]
        comparison_values=0
        for page_no in range(37,47):
            expected=english[page_no-1]['blocks'][0]['args'][1]
            extracted=doc.pages[page_no-1].extract_text(x_tolerance=1,y_tolerance=2)
            for row in expected:
                for cell in row[1:]:
                    if cell.startswith('-\n'):continue
                    for value in cell.split('\n'):
                        assert value in extracted,(page_no,value)
                        comparison_values+=1
    for path,value in original_hashes.items():assert sha(path)==value,path
    checks={
        'pages':49,'language':'English','translated_unique_strings':len(mapping),'numeric_table_cells_unchanged':numeric_cells,
        'per_frame_printed_values_verified':comparison_values,'image_count':len(images),'image_sources_sha256':images,
        'source_CN_pdf_sha256':sha(OUT/(CN+'.pdf')),'pdf_sha256':sha(PDF),
        'translation_sha256':sha(ROOT/'scripts/weekly_report_en_20261103.tsv'),
        'generator_sha256':sha(__file__),'source_CN_artifacts_unchanged':True,
        'same_per_frame_CSV_sha256':sha(OUT/(BASE+'.test_comparison.csv'))==sha(OUT/(CN+'.test_comparison.csv')),
        'english_figure_annotations':figure_record,
        'no_chinese_text':True,'text_bounds_passed':True,'layout':layout,'visual_review':'pending',
        'created_utc':datetime.now(timezone.utc).isoformat(),'training_or_model_inference':False}
    (OUT/(BASE+'.qa.json')).write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k not in ['image_sources_sha256','layout']},indent=2))
    print(json.dumps(layout))

if __name__=='__main__':main()
