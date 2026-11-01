"""Reuse verified exports and explain validation-driven before/after changes."""
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw,ImageFont
from .run import RUNS,REPORTS,BASELINE,engine,parent


def export_and_review(name,proposal):
    from variants.integrity_tuning import export as common
    from scripts import export_color_repair_images as collect
    engine.write_json(REPORTS/'diagnosis.json',engine.read_json(parent.REPORTS/'diagnosis.json'))
    common.RUNS,common.REPORTS=RUNS,REPORTS
    common.export_and_assess([name])
    result=engine.read_json(REPORTS/'assessment.json')
    source=parent.RUNS/proposal['source_model']/'rubber_duck'
    comparable=(source/'test/ours_50000/render_complete.json').exists()
    baseline=engine.read_json(parent.REPORTS/'diagnosis.json')
    final=result['models'][0]['validation']
    comparisons=[{'model':'C1_reused','iteration':50000,'metrics':baseline['image_metrics'],'integrity':baseline['mean']}]
    if comparable:
        history=parent.read_history(source);record=next(r for r in reversed(history) if r['iteration']==50000)
        comparisons.append({'model':proposal['source_model'],'iteration':50000,'metrics':record['metrics'],'integrity':record['integrity']})
    comparisons.append({'model':name,'iteration':50000,'metrics':final['metrics'],'integrity':final['integrity']})
    if comparable:
        output=RUNS/'render_exports/final50000';frames=collect.read(collect.SPLIT)['frames'];label=proposal['source_model']
        with ThreadPoolExecutor(max_workers=3) as pool:
            source_records=list(pool.map(lambda item:collect.export_frame(source/'test/ours_50000',output,label,*item),enumerate(frames)))
        collect.write(output/label/'manifest.json',{'source':str(source),'frames':source_records})
        folder=output/'gt_c1_before_after';folder.mkdir(exist_ok=True)
        def combine(index):
            file=f'{index:05d}.png';panels=[('GT',output/'gt'/file),('C1 reused',output/'C1_reused/renders'/file),
                (label,output/label/'renders'/file),(name,output/name/'renders'/file)]
            image=Image.new('RGB',(6400,950),'#edf2f7');draw=ImageDraw.Draw(image)
            font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25);pixels=[]
            for j,(title,path) in enumerate(panels):
                with Image.open(path) as im:
                    im=im.convert('RGB');image.paste(im,(j*1600,50));pixels.append(im.tobytes())
                draw.text((j*1600+12,10),title+' / final50000',font=font,fill='#17304a')
            path=folder/file;image.save(path,compress_level=3)
            with Image.open(path) as im:
                assert all(im.crop((j*1600,50,(j+1)*1600,950)).tobytes()==p for j,p in enumerate(pixels))
            return {'index':index,'filename':file,'sha256':collect.digest(path)}
        with ThreadPoolExecutor(max_workers=3) as pool:images=list(pool.map(combine,range(50)))
        collect.write(folder/'manifest.json',{'images':50,'source_pixels_preserved':True,'panel_order':['GT','C1 reused',label,name],'frames':images})
        collect.atomic_text(folder/'index.html','<!doctype html><meta charset="utf-8"><h1>GT | C1 | before | after</h1>'+''.join(f'<p>{r["filename"]}</p><img style="width:100%" src="{r["filename"]}">' for r in images))
        result['before_after_gallery_absolute']=str(folder/'index.html')
    result.update(proposal=proposal,validation_before_after=comparisons,all_comparisons_same_total_iterations=comparable,
        decisions_used_test_data=False)
    engine.write_json(REPORTS/'revision_assessment.json',result)
    lines=['# Result-driven model revision','',
        'All decisions used validation. The original C1 was reused. Every trained follow-up used the same full3k initialization, total50k.',
        '','Changes and evidence:',*['- '+s for s in proposal['reasons']],
        '','| Model | PSNR val | SSIM val | LPIPS val | MS-SSIM val | Integrity score val | BG spill val |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append('| '+r['model']+' | '+' | '.join(f"{r['metrics'][k]:.7f}" for k in engine.METRICS)+f" | {r['integrity']['integrity_score']:.7f} | {r['integrity']['background_false_coverage']:.7f} |")
    lines+=['','No automatic replacement of the baseline: retain prior models if quality or completeness regresses.',
        'The proxies describe2D coverage, projected fragmentation and appearance, not proof of complete hidden3D surfaces.',
        '','Final test and individual renders: '+str(REPORTS/'assessment.md'),
        'GT / C1 / before / after: '+result.get('before_after_gallery_absolute',result['gallery_absolute'])]
    engine.write_text_atomic(REPORTS/'revision_report.md','\n'.join(lines)+'\n')
