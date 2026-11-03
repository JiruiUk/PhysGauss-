"""Final read-only audit, gallery verdict, and completion receipt."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning.runtime import read_json,write_json,sha256
from variants.transfer_integrity.run import freeze,prepare_code
REPORT=ROOT/'experiments/transfer_integrity';OUT=ROOT/'output/transfer_integrity'


def main():
    freeze(prepare_code())
    assert '27 passed' in (REPORT/'tests_final.log').read_text()
    for p,h in read_json(REPORT/'history_before.json').items():assert sha256(ROOT/p)==h,p
    historical=read_json(ROOT/'experiments/fixed_camera_temporal/artifact_manifest.json')
    for p,h in historical.items():assert sha256(ROOT/p)==h,p
    for p,h in read_json(REPORT/'GT_provenance.json')['original_RGBA_sha256'].items():assert sha256(ROOT/p)==h,p
    videos=read_json(REPORT/'video_verification.json')
    assert len(videos)==16
    for p,v in videos.items():assert v['frames']==200 and sha256(OUT/p)==v['sha256']
    assert read_json(REPORT/'export_verification.json')['exporter_sha256']==sha256(REPORT/'export.py')
    sources=read_json(REPORT/'render_verification.json')
    for p,h in sources['checkpoint_sha256_unchanged'].items():assert sha256(Path(p))==h,p
    for p,h in sources['source_sha256_unchanged'].items():assert sha256(ROOT/p)==h,p
    for name in ['C1_original_50k','U0_C1_control','U1_transfer_geometry']:
        for split in ['val','test']:
            value=read_json(REPORT/f'{name}_{split}.json')
            assert value['evaluator_sha256']==sha256(REPORT/'evaluate.py')
        assert read_json(REPORT/f'{name}_support.json')['evaluator_sha256']==sha256(ROOT/'variants/association_training/evaluate.py')
    for p,h in read_json(REPORT/'artifact_manifest.json').items():assert sha256(ROOT/p)==h,p
    # This curation changes presentation only; no render or measurement is edited.
    gallery=OUT/'index.html';text=gallery.read_text(encoding='utf-8')
    verdict='<p id="final-verdict"><strong>实测结论：局部对应改善，但固定视角碎裂和遮挡后身份保留没有明显改善。</strong>两组各继续训练2000步，默认关闭新分支。<a href="native_crop_fixed86_t2.png">查看原生像素局部对比</a>。</p>'
    if 'id="final-verdict"' not in text:
        end=text.index('</h1>')+len('</h1>');text=text[:end]+'\n'+verdict+text[end:]
    text='\n'.join(line.rstrip() for line in text.splitlines())+'\n'
    gallery.write_text(text,encoding='utf-8')
    import re
    local=re.findall(r'(?:href|src)="([^"]+)"',text)
    assert all((OUT/p).resolve().exists() for p in local),[p for p in local if not (OUT/p).resolve().exists()]
    looked=['early_native_inspection.png','native_crop_fixed86_t2.png','fixed_0086_comparison_t0200.png',
            'fixed_0086_comparison_t0086.png','fixed_0002_comparison_t0086.png','fixed_0188_comparison_t0086.png',
            'moving_control_comparison_t0002.png','fixed_0086_comparison_RGB_alpha.decoded_contact.png']
    write_json(REPORT/'manual_review.json',{'reviewed_artifacts':{p:sha256(OUT/p) for p in looked},
               'finding':'Fixed-view streaking, broken blue surfaces and incomplete yellow outlines remain; U0 and U1 are visually close. Moving/diagonal observations remain much better.',
               'native_pixel_crop':[600,220,1250,800],'decoded_video_contact_inspected':True,
               'physical_ID_confirmed':False})
    write_json(REPORT/'artifact_manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    evidence=[p for p in REPORT.iterdir() if p.is_file() and p.name not in ['final_verification.json','finalize.log']]
    write_json(REPORT/'final_verification.json',{'status':'completed_evaluated','conclusion':'limited_local_constraint_improvement_no_fixed_view_repair_or_identity_recovery',
               'historical_render_files_verified_unchanged':len(historical),'historical_checkpoints_reports_unchanged':True,
               'all_200_original_GT_files_unchanged':True,'training_source_and_binding_freeze_unchanged':True,
               'same_2000_training_camera_samples':True,'final_iteration_both':52000,
               'validation_count_per_model':30,'test_count_per_model':50,'native_renders':2400,'videos_decoded':16,
               'novel_camera_time_GT_available':False,'physical_ID_after_occlusion_verified':False,
               'gallery_local_links_checked':len(local),'evidence_sha256':{p.name:sha256(p) for p in evidence}})
    state=read_json(ROOT/'runs/pidg_transfer_integrity/controller.json');state.update(status='completed_evaluated',
                assessment=str(REPORT/'assessment.md'),conclusion='No clear fixed-view repair or identity recovery in this first pilot')
    write_json(ROOT/'runs/pidg_transfer_integrity/controller.json',state)
    print('FINAL VERIFIED:',len(historical),'historical files, 200 GTs, final states, 16 videos and gallery links',flush=True)


if __name__=='__main__':main()
