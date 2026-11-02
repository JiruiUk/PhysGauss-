from pathlib import Path
import sys,json,hashlib,zipfile,statistics
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(r'F:/A_Edinburgh/PhysGauss');sys.path.insert(0,str(ROOT))
from variants.monotonic_training.run import assert_frozen,HERE
assert_frozen(HERE/'code')
REPORT=ROOT/'experiments/monotonic_training';RUN=ROOT/'runs/pidg_monotonic_training/J1_monotonic_kinematic/rubber_duck'
OUT=ROOT/'output/monotonic_training/J1_monotonic_kinematic';QA=OUT.parent/'verification_20261102'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
original=read(REPORT/'original_inputs_sha256.json')
assert all(sha(ROOT/p)==d for p,d in original.items())
checkpoint=read(RUN/'checkpoints/resume.json');assert checkpoint['iteration']==50000 and sha(RUN/'checkpoints/resume.pt')==checkpoint['sha256']
manifest=read(OUT/'manifest.json');per=read(RUN/'per_view.json');metrics=read(RUN/'results.json')
expected={f'{i:05d}.png' for i in range(50)};rows=[];video_checks={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
for tag in ['best','50000']:
 folder=OUT/'pairs'/tag
 assert {p.name for p in folder.glob('*.png')}==expected
 signatures=[]
 for i in range(50):
  name=f'{i:05d}.png';raw=RUN/'test'/('ours_'+tag)
  with Image.open(raw/'gt'/name) as im:a=np.array(im.convert('RGB'))
  with Image.open(raw/'renders'/name) as im:b=np.array(im.convert('RGB'))
  with Image.open(folder/name) as im:pair=np.array(im.convert('RGB'))
  assert pair.shape==(950,3200,3) and a.shape==b.shape==(900,1600,3)
  assert np.array_equal(pair[50:,:1600],a) and np.array_equal(pair[50:,1600:],b)
  assert int(manifest['frame_mapping'][i+(50 if tag=='50000' else 0)]['frame_id'])==1+4*i
  mse=np.mean(((a.astype(np.float32)-b)/255)**2,dtype=np.float64);score=-10*np.log10(max(mse,1e-12));delta=score-per['ours_'+tag]['PSNR'][name]
  assert abs(delta)<2e-4
  signatures.append(pair[50::16,::16].astype(np.float32));rows.append({'tag':tag,'index':i,'PSNR':float(score),'stored_difference':float(delta)})
 signatures=np.stack(signatures)
 reader=imageio_ffmpeg.read_frames(str(OUT/f'GT_Render_{tag}.mp4'));meta=next(reader)
 assert tuple(meta['size'])==(3200,950) and meta['fps']==10
 nearest=[];preview=[];mae=[]
 for i,frame in enumerate(reader):
  arr=np.frombuffer(frame,np.uint8).reshape(950,3200,3)
  sig=arr[50::16,::16].astype(np.float32)
  distances=((signatures-sig)**2).mean((1,2,3));match=int(distances.argmin());nearest.append(match);assert match==i,(tag,i,match)
  with Image.open(folder/f'{i:05d}.png') as im:target=np.array(im.convert('RGB'))
  mae.append(float(np.abs(arr.astype(np.float32)-target).mean()))
  assert mae[-1]<5
  if i in [0,21,49]:preview.append(Image.fromarray(arr).resize((1600,475)))
 assert len(nearest)==50
 video_checks[tag]={'decoded_frames':len(nearest),'dimensions':meta['size'],'fps':meta['fps'],'duration':meta['duration'],
  'all_frames_nearest_match_expected_PNG_index':True,'maximum_RGB_MAE_after_lossy_video_compression':max(mae)}
 contact=Image.new('RGB',(1600,1425),'white')
 for i,im in enumerate(preview):contact.paste(im,(0,475*i))
 contact.save(QA/f'decoded_video_contact_{tag}.png')
 print('VERIFIED',tag,video_checks[tag],flush=True)
archive=OUT.parent/'J1_monotonic_kinematic_GT_Render.zip'
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 expected_entries={p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.encode.log')}
 assert set(z.namelist())==expected_entries
 for name in z.namelist():assert z.read(name)==(OUT/name).read_bytes(),name
archive_check={'CRC_passed':True,'all_entries_identical_to_exported_files':True,'entries':len(expected_entries),'bytes':archive.stat().st_size}
value={'verification_date':'2026-11-02','full_training_iteration':50000,'checkpoint_sha256_matches':True,'original_inputs_unchanged':True,
 'runtime_source_freeze_matches':True,'PNG_pairs_verified':len(rows),'PNG_image_regions_pixel_identical':True,
 'maximum_recomputed_PSNR_difference':max(abs(r['stored_difference']) for r in rows),'videos':video_checks,'archive':archive_check,
 'independent_mean_PSNR':{tag:statistics.mean(r['PSNR'] for r in rows if r['tag']==tag) for tag in ['best','50000']}}
(QA/'verification.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
(REPORT/'verification_20261102.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(value,ensure_ascii=False),flush=True)
