"""CPU-only ffprobe metadata of already produced videos; no rendering or inference."""
import os,glob,json,subprocess
from pathlib import Path
assert os.getuid()!=0 and not glob.glob('/dev/nvidia*')
BASE=Path('/workspace/artifacts/step05A_n_action_steps/naction20000_r02_20261101_01')
entries=[]
for d in sorted(BASE.glob('trial_*')):
    if not (d/'result.json').exists():continue
    result=json.loads((d/'result.json').read_text());prior=json.loads((d/'video_check.json').read_text())
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,codec_long_name,pix_fmt,width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(d/'diagnostic_10fps.mp4')],timeout=30))
    s=probe['streams'][0];assert s['codec_name']=='h264' and s['pix_fmt']=='yuv420p' and (s['width'],s['height'])==(640,480) and s['r_frame_rate']=='10/1'
    assert int(s['nb_read_frames'])==result['video_frames']==len(prior['frame_sim_times'])
    assert abs(float(s['duration'])-result['video_frames']/10)<1e-6 and prior['all_frames_decode']
    entries.append(dict(trial=result['trial'],pass_=True,probe=probe,simulation_seconds=result['sim_seconds'],display_seconds=float(s['duration']),all_frames_prior_decode_checked=True,representative_review_exists=(d/'review.json').exists()))
assert entries
with (BASE/'video_codec_validation.json').open('x') as f:json.dump(dict(pass_=True,videos=entries,forwards=0,physics_steps=0),f,indent=2)
print(json.dumps(dict(video_metadata_pass=True,videos=len(entries))),flush=True)
