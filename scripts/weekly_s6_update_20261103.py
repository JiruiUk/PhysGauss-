"""S6 formal results for the weekly PDF; existing records and images only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont, ImageChops

S6_PAGE_COUNT = 4

def prepare_s6(root, snap):
    root, snap = Path(root), Path(snap)
    base=snap/'s6_revision'; out=base/'figures'
    out.mkdir(parents=True,exist_ok=True)
    receipt=base/'manifest.json'
    previous=json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
    sources={}; figures=[]
    def track(rel):
        dst=base/rel
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes((root/rel).read_bytes())
        data=dst.read_bytes()
        sources[rel]={'snapshot':str(dst.relative_to(snap)),
            'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        return dst
    records=['assessment.md','protocol.json','S6_val_04000.json','S6_test_04000.json',
        'native_full_quality.json','final_verification.json','manual_visual_review.json',
        'admission.json','S6_exports.json','source_freeze.json','state_objective_audit.json']
    for name in records: track('experiments/staged_reconstruction/S6/'+name)
    for rel in ['experiments/staged_reconstruction/final_summary.md',
        'experiments/staged_reconstruction_plan.md','experiments/staged_reconstruction/completion_integrity.json']:
        track(rel)
    def read(name):
        return json.loads((base/'experiments/staged_reconstruction/S6'/name).read_text(encoding='utf-8'))
    metrics=read('S6_val_04000.json'); test=read('S6_test_04000.json')
    final=read('final_verification.json'); quality=read('native_full_quality.json')
    assert metrics['iteration']==test['iteration']==final['iteration']==4000
    assert final['native_images_reviewed'] and not final['geometry_success_established']
    assert final['checkpoint_sha256']==quality['checkpoint_sha256']
    def load(rel,side=None):
        with Image.open(track(rel)) as im: im=im.convert('RGB')
        if side is not None:
            assert im.size==(3200,900),im.size
            im=im.crop((side*1600,0,(side+1)*1600,900))
        return im
    font=ImageFont.truetype('C:/Windows/Fonts/Dengb.ttf',32)
    def grid(name,panels,ref=None):
        ims=[load(rel,side) for label,rel,side in panels]
        assert len({im.size for im in ims})==1
        crop=(0,0,*ims[0].size)
        if ref is not None:
            im=load(ref,0)
            r,g,b=im.split()
            bounds=ImageChops.lighter(ImageChops.lighter(r,g),b).point(lambda v:255 if v>18 else 0).getbbox()
            x0,y0,x1,y1=bounds
            crop=(max(0,x0-45),max(0,y0-45),min(im.width,x1+45),min(im.height,y1+45))
        w=520; h=round(w*(crop[3]-crop[1])/(crop[2]-crop[0])); gap=12; title=46
        sheet=Image.new('RGB',(len(ims)*(w+gap)-gap,h+title),'#eef3f5'); d=ImageDraw.Draw(sheet)
        for i,(im,(label,rel,side)) in enumerate(zip(ims,panels)):
            x=i*(w+gap)
            d.rectangle((x,0,x+w,title),fill='#183b50')
            d.text((x+12,8),label,font=font,fill='white')
            sheet.paste(im.crop(crop).resize((w,h),Image.Resampling.LANCZOS),(x,title))
        dst=out/(name+'.png'); sheet.save(dst)
        figures.append({'output':str(dst.relative_to(snap)),
            'panels':[{'label':label,'source':rel,'pair_side':side} for label,rel,side in panels],
            'common_crop_xyxy':crop,'operation':'common crop, proportional Lanczos; no image repair'})
        return str(dst.relative_to(snap))
    gt='output/association_training/T1_association/native_GT_Render0086.png'
    assets={'observed':grid('observed86',[('GT t86',gt,0)]+[(f'Stage {s}',f'output/staged_reconstruction/{s}/original_t0086_native.png',None) for s in ['S3','S4','S5','S6']],gt)}
    for frame in [2,86,188]:
        assets[f'fixed{frame}']=grid(f'fixed_t{frame}',[
            (f'{s} cam{cam}',f'output/staged_reconstruction/{s}/fixed{cam}_t{frame:04d}_native.png',None)
            for cam in [86,188] for s in ['S5','S6']])
    rel='output/staged_reconstruction/S6/S6_review_step04000.png'
    im=load(rel); assert im.size==(1200,2205)
    # Each existing row has three panels (observed RGB / render / group contributions).
    rows=[0,2,7]; combined=Image.new('RGB',(1200,245*len(rows)))
    for i,row in enumerate(rows): combined.paste(im.crop((0,row*245,1200,(row+1)*245)),(0,i*245))
    dst=out/'observed_groups_2_42_188.png'; combined.save(dst)
    assets['groups']=str(dst.relative_to(snap))
    figures.append({'output':assets['groups'],'source':rel,
        'operation':'unchanged existing rows; no per-model crop or repair',
        'source_crop_boxes':[[0,row*245,1200,(row+1)*245] for row in rows],
        'frame_order':[2,42,188],'columns':['observed RGB','S6 RGB','model bank contributions, not physical IDs']})
    rel='output/staged_reconstruction/S6/native_crop_review.png'
    dst=out/'native_crop_review.png'; load(rel).save(dst)
    assets['native']=str(dst.relative_to(snap))
    figures.append({'output':assets['native'],'source':rel,'operation':'existing native crop figure, unchanged'})
    for rel,expected in read('manual_visual_review.json')['files_reviewed'].items():
        if rel in sources: assert sources[rel]['sha256']==expected,rel
    manifest={'captured_utc':previous.get('captured_utc',datetime.now(timezone.utc).isoformat()),
        'scope':'S6 formal 4000-step results and final chain status; preserves earlier report evidence snapshot',
        'checkpoint_sha256':final['checkpoint_sha256'],'sources':sources,'derived_figures':figures,
        'figure_paths':assets,'no_model_inference_or_training':True}
    receipt.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return assets,manifest,{'val':metrics['metrics'],'test':test['metrics'],'quality':quality,'final':final}

def render_s6(assets,data,start,p,fig,box,table,heading):
    val,test=data['val'],data['test']; quality=data['quality']; final=data['final']
    start('S6 补充 01 / 04','S6正式终态：指标与训练条件','正式联合微调4000步；预先指定终态，非事后选择best。原生1600×900，30验证/50测试。[37,38]')
    table(['指标 / 口径','验证集','测试集'],[
        ('条件动态区域 PSNR / dB',f"{val['dynamic_PSNR']:.4f}",f"{test['dynamic_PSNR']:.4f}"),
        ('动态分支区域alpha',f"{val['dynamic_alpha']:.6f}",f"{test['dynamic_alpha']:.6f}"),
        ('条件静态区域 PSNR / dB',f"{val['static_PSNR']:.4f}",f"{test['static_PSNR']:.4f}"),
        ('全画幅PSNR / dB（浮点）',f"{val['PSNR']:.4f}",f"{test['PSNR']:.4f}"),
        ('全画幅PSNR / dB（8bit）',*[f"{quality['splits'][s]['metrics']['PSNR']:.4f}" for s in ['val','test']]),
        ('全画幅SSIM（8bit）',*[f"{quality['splits'][s]['metrics']['SSIM']:.6f}" for s in ['val','test']]),
        ('标准VGG LPIPS（8bit）',*[f"{quality['splits'][s]['metrics']['LPIPS']:.6f}" for s in ['val','test']]),
    ],[2.3,1,1])
    p('浮点评估与裁剪/取整8bit PNG评估单列，避免混用；区域掩码为条件图像支持，非独立几何GT。黑色背景会稀释全画幅误差。','small')
    heading('相对阶段与对应误差')
    p('验证动态PSNR：S5 25.6037→S6 28.3270 dB（+2.7233）；相对S2为+4.0736 dB。固定455次合格训练观测的轨迹误差：S3 0.53684→S4 0.78502→S5 0.66467→S6 0.42711 px@400；相对S3下降20.44%。')
    heading('协议和验收')
    p('继承S5，0.25倍学习率联合已有RGB/运动学/CMR/轨迹目标；解冻静态外观，静态几何仍精确保持S1。138组Adam状态、完整采样重放和数值门槛通过；三段视频各200帧解码完成。完整条件训练链46k次更新，另保留两次失败S2的32k。')
    box('S6改善了观测重建和既有轨迹拟合。它是顺序继承训练的结果，不能将收益独立归因于物理项、轨迹项或静态外观。')

    start('S6 补充 02 / 04','S6图像：观测视角的改善','所有图像来自已导出的正式终态。展示训练观测，用于解释局部表现，不替代验证/测试统计。[37,38]')
    fig(assets['observed'],155,'图39  实际相机、原帧86：GT / S3 / S4 / S5 / S6。由同一GT决定共同裁剪，S6鸭体轮廓较清晰；这是顺序终态比较，非同预算单因素消融。')
    fig(assets['groups'],355,'图40  从上至下原训练帧2/42/188，左实际GT、中S6渲染、右模型组贡献。沿用原图像行，仅重排三行；组颜色不是物理身份。帧42仍有跨组支撑，晚期绿色长条泄漏明显。')
    p('原始复核还包含27/43/52/86/166/200帧。本页是预先复核面板中的解释性选帧，不声称代表所有时刻；没有用物体颜色或模型组颜色补遮挡后真实身份。','small')

    start('S6 补充 03 / 04','S5与S6：固定新视角仍不完整','每行列顺序：S5相机86 / S6相机86 / S5相机188 / S6相机188。保留完整视野，不裁异常支撑。[37]')
    fig(assets['fixed2'],165,'图41  时刻2，两固定相机均为新组合，无独立GT。S6仍有颗粒、拉伸和碎片，观测视角的改善未自动转为完整三维形态。')
    fig(assets['fixed86'],165,'图42  时刻86，相机86与当时真实观测相同，相机188为新组合。前两列较完整，后两列明显退化；不得将新组合称为同一帧GT重建。')
    fig(assets['fixed188'],165,'图43  时刻188，相机188与真实观测相同，相机86为新组合。前两列仍有分散支撑与拉丝；缺少新组合GT，不计算其PSNR。')
    p('相机86/188在对应时刻的视角重合仅提供观测检查；其他时刻没有同步GT。固定图像保持1600×900全视野后等比缩放，未补洞或调整模型输出。','small')

    start('S6 补充 04 / 04','S6原生细节：几何与身份边界','既有原生像素裁剪图，显示时等比缩放；未生成新渲染或美化图像。[37]')
    fig(assets['native'],365,'图44  行：时刻2/86/188；列：固定86 / 固定188 / 原观测轨迹。原图均裁x=450:1200、y=160:650，图内保留标签。观测轨迹相对完整，新组合仍有空洞、条纹和拉丝。')
    heading('哪些目标仍未完成')
    p('固定新视角完整几何未恢复，晚期物理ID仍未知。模型组泄漏说明分组贡献不等于持久材料点身份；稀疏训练轨迹改善也不能证明跨遮挡重识别。')
    p('S6的条件静态体先验调用800次、合格观测实例16000次，活动违约与非零先验均为0；S5亦未激活。未证明实际碰撞修正。物体间非穿透、接触力、摩擦、恢复系数、本构与未知力均未验证/关闭。')
    box('六阶段条件重建实验的训练与评估已执行；完整几何、物理身份和原始接触目标仍未完成。后续先补可校准的RGB几何/身份验证证据。')
    return 6
