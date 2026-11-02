"""Read-only saved-trajectory analysis. Never imports simulator or model modules."""
import sys,json,os,glob,importlib.util,shutil
from pathlib import Path
ROOT=Path('/workspace'); OUT=ROOT/'artifacts/step05A_grasp_trace_review/grasp_trace_20261031_01'
if sys.argv[1]=='inspect':
    import pyarrow.parquet as pq
    result={'modules':{n:importlib.util.find_spec(n) is not None for n in ['numpy','PIL','matplotlib','pyarrow']},'ffmpeg':shutil.which('ffmpeg'),'ffprobe':shutil.which('ffprobe'),'nvidia_devices':glob.glob('/dev/nvidia*'),'uid':os.getuid(),'gpu_visible':os.getenv('NVIDIA_VISIBLE_DEVICES'),'schema':str(pq.read_schema(ROOT/'data/step04/dataset_20261029_01/data/chunk-000/file-000.parquet'))}
    assert result['uid']!=0 and not result['nvidia_devices'] and result['gpu_visible']=='void'
    (OUT/'inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));sys.exit(0)
import gzip,csv,math,time,hashlib,subprocess,re,resource,html
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw,ImageFont
START=time.monotonic();names=['Rotation_R','Pitch_R','Elbow_R','Wrist_Pitch_R','Wrist_Roll_R','Jaw_R']
assert sys.argv[1]=='analyze' and not (OUT/'analysis_started.json').exists()
def dump(p,x):
    (OUT/p).write_text(json.dumps(x,indent=2,allow_nan=False))
def load(p):return json.loads((ROOT/p).read_text())
def finite(x):assert np.isfinite(x).all()
def csvwrite(p,rows):
    assert rows
    with (OUT/p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def guard():
    assert time.monotonic()-START<810,'analysis deadline'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<500*1024**2
    assert shutil.disk_usage(OUT).free>=10*1024**3
assert os.getuid()!=0 and not glob.glob('/dev/nvidia*') and os.getenv('NVIDIA_VISIBLE_DEVICES')=='void'
dump('analysis_started.json',{'time':time.time(),'uid':os.getuid(),'nvidia_devices':glob.glob('/dev/nvidia*')})
inputs=load(OUT.relative_to(ROOT)/'inputs.json')
for x in inputs['files']:
    p=ROOT/x['path'];assert p.stat().st_size==x['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
std=np.array(load('models/step05A/act_20261029_01/continued/statistics.json')['stats']['observation.state']['std']);assert (std>0).all()
roots={'600':'artifacts/step05A_warn_only/warn_only_20261031_01','5000':'artifacts/step05A_train_compare/train5000_20261031_01'}
D='artifacts/step04/dataset_20261029_01';DATA='data/step04/dataset_20261029_01'
all_events=[];summaries=[];correspondence=[];scenes=[];decode_total=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists() else ImageFont.load_default()
def contact(row,finger=None):
    return any('block_geom' in c['geoms'] and any(g.startswith('robot_geom') and (finger is None or g==finger) for g in c['geoms']) for c in row['contacts'])
def first(mask):
    a=np.flatnonzero(mask);return int(a[0]) if len(a) else None
def svgplot(trial,series,metric):
    W,H=1400,1100;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>',f'<text x="25" y="25" font-size="19">Trial {trial}: {metric}; seconds, rad (delta per 20 ms)</text>']
    colors={'teacher':'#145db3','600':'#c36b00','5000':'#168549'}
    for j,name in enumerate(names):
        y0=65+j*170;low=1e9;high=-1e9
        vals=[]
        for label,a in series.items():
            if metric=='state_action':items=[('state',a['q']),('action',a['a'])]
            elif metric=='command_change':items=[('delta',a['delta'])]
            else:items=[('action_minus_prestate',a['a']-a['q']),('ctrl_minus_poststate',a['a']-a['post'])]
            for kind,v in items:
                low=min(low,float(v[:,j].min()));high=max(high,float(v[:,j].max()));vals.append((label,kind,a['t'],v[:,j]))
        pad=max((high-low)*.08,.002);low-=pad;high+=pad
        parts.append(f'<text x="15" y="{y0}" font-size="15">{name}</text>')
        for k in range(5):
            y=y0+15+125*k/4;v=high-(high-low)*k/4
            parts.append(f'<path d="M100 {y} H1370" stroke="#ddd"/><text x="10" y="{y+5}" font-size="12">{v:.4f}</text>')
        for label,kind,t,v in vals:
            points=' '.join(f'{100+float(tt)/50*1270:.2f},{y0+15+(high-float(vv))/(high-low)*125:.2f}' for tt,vv in zip(t,v))
            dash=' stroke-dasharray="5,3"' if kind in ['action','ctrl_minus_poststate'] else ''
            parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[label]}" stroke-width="1.1"{dash}/>')
        parts.append(f'<text x="100" y="{y0+157}" font-size="12">0            10            20            30            40            50 s; teacher blue / 600 orange / 5000 green; dashed=action or post tracking</text>')
    parts.append('</svg>');(OUT/f'trial_{trial:02d}_{metric}.svg').write_text('\n'.join(parts))
def decode(video,indices):
    global decode_total
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate','-of','json',str(ROOT/video)]))['streams'][0]
    assert (int(meta['width']),int(meta['height']))==(640,480)
    n=int(meta['nb_frames']);assert decode_total+n<=20000
    expr='+'.join(f'eq(n\\,{i})' for i in indices)
    p=subprocess.run(['ffmpeg','-v','info','-threads','2','-filter_threads','1','-i',str(ROOT/video),'-vf',f'showinfo,select={expr}','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=60)
    seen=re.findall(rb'\bn:\s*(\d+)\s+pts:',p.stderr);assert seen and int(seen[-1])+1==n,(video,len(seen),n)
    decode_total+=n
    assert len(p.stdout)==len(indices)*640*480*3
    frames={i:Image.frombytes('RGB',(640,480),p.stdout[k*640*480*3:(k+1)*640*480*3]) for k,i in enumerate(indices)}
    return frames,n,meta
for trial in range(5):
    guard();ix=f'{trial:02d}';attempt=load(f'{D}/attempt_{ix}.json')
    with gzip.open(ROOT/f'{D}/trace_{ix}.jsonl.gz','rt') as f:teacher=[json.loads(l) for l in f]
    tab=pq.read_table(ROOT/f'{DATA}/data/chunk-000/file-{trial:03d}.parquet').to_pydict()
    assert len(teacher)==len(tab['action'])==attempt['frames']
    assert set(tab['episode_index'])=={trial} and tab['frame_index']==list(range(len(teacher)))
    tq=np.array(tab['observation.state']);ta=np.array(tab['action']);tt=np.arange(len(teacher))*.02
    assert np.max(np.abs(tq-np.array([r['state_before'] for r in teacher])))==0
    assert np.max(np.abs(ta-np.array([r['sent'] for r in teacher]).astype(np.float32)))==0
    assert np.max(np.abs(np.array(tab['timestamp'])-tt))<3e-6
    assert all(abs(r['time_before']-i*.02)<1e-8 and abs(r['time']-(i+1)*.02)<1e-8 for i,r in enumerate(teacher))
    teacher_initial=attempt['initial'];series={};raws={'teacher':teacher};results={'teacher':attempt['result']}
    series['teacher']={'q':tq,'a':ta,'t':tt,'post':np.array([r['measured'] for r in teacher]),'post_t':tt+.02,'phase':[r['phase'] for r in teacher]}
    for label,root in roots.items():
        initial=load(f'{root}/trial_{ix}/initial.json');assert initial['integration']==teacher_initial['integration'] and initial['seed']==attempt['seed']==42
        assert initial['position']==teacher_initial['block_position'] and initial['quaternion']==teacher_initial['block_quaternion']
        with gzip.open(ROOT/f'{root}/trial_{ix}/trace.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
        result=load(f'{root}/trial_{ix}/result.json');assert len(rows)==result['ticks_applied']==2500 and result['valid_observation_end']
        for i,r in enumerate(rows):
            assert r['tick']==i and r['applied'] and abs(r['time_before']-i*.02)<1e-8 and abs(r['time_after']-(i+1)*.02)<1e-8
            assert np.array_equal(r['ctrl'],r['action_rad'])
        a={'q':np.array([r['state_before'] for r in rows]),'a':np.array([r['action_rad'] for r in rows]),'t':np.array([r['time_before'] for r in rows]),'post':np.array([r['state_after'] for r in rows]),'post_t':np.array([r['time_after'] for r in rows]),'qvel':np.array([r['qvel_before'] for r in rows])}
        series[label]=a;raws[label]=rows;results[label]=result
        correspondence.append({'trial':trial,'series':label,'episode':trial,'seed':42,'initial_position_m':initial['position'],'initial_integration_exact':True,'rows':len(rows),'state_time':'pre-action','contact_time':'post-step','joint_order':names,'unit':'rad_absolute'})
    phase_starts={};phase_ends={}
    for i,r in enumerate(teacher):
        phase_starts.setdefault(r['phase'],i);phase_ends[r['phase']]=i
    events_trial=[]
    for phase,i in phase_starts.items():events_trial.append({'trial':trial,'series':'teacher','event':'phase_start_'+phase,'index':i,'pre_time_s':float(tt[i]),'post_time_s':float(tt[i]+.02),'basis':'saved phase'})
    stats={}
    for label,a in series.items():
        for k in ['q','a','post']:finite(a[k]);assert a[k].shape[1]==6
        previous=np.vstack([teacher_initial['ctrl'],a['a'][:-1]])
        a['delta']=a['a']-previous
        rows=raws[label];both=np.array([contact(r,'robot_geom_41') and contact(r,'robot_geom_44') for r in rows]);robot=np.array([contact(r) for r in rows]);block=np.array([r['block'] if label=='teacher' else r['diagnostic']['block'] for r in rows])
        baseline=float(min(teacher_initial['block_position'][2],block[a['post_t']<=1,2].min()))
        lift=block[:,2]-baseline
        no_support=np.array([not any('block_geom' in c['geoms'] and any(g=='table' or g.startswith('tray_') for g in c['geoms']) for c in r['contacts']) for r in rows])
        for ev,mask in [('first_robot_contact',robot),('first_bilateral',both),('first_40mm_lift',lift>=.04),('first_jaw_command_below_0.55',a['a'][:,5]<.55)]:
            j=first(mask);events_trial.append({'trial':trial,'series':label,'event':ev,'index':j,'pre_time_s':None if j is None else float(a['t'][j]),'post_time_s':None if j is None else float(a['post_t'][j]),'basis':'saved post contact/lift or diagnostic jaw crossing; not grasp declaration'})
        if label!='teacher':
            assert np.array_equal(both,np.array([r['diagnostic']['bilateral_contact'] for r in rows]))
            assert np.max(np.abs(lift-np.array([r['diagnostic']['lift_m'] for r in rows])))<1e-12
        late=a['t']>=40
        stats[label]={'rows':len(rows),'joint_state_min':a['q'].min(axis=0).tolist(),'joint_state_max':a['q'].max(axis=0).tolist(),'command_change_abs_max_rad':np.abs(a['delta']).max(axis=0).tolist(),'tracking_pre_abs_max_rad':np.abs(a['a']-a['q']).max(axis=0).tolist(),'tracking_post_abs_max_rad':np.abs(a['a']-a['post']).max(axis=0).tolist(),'tracking_pre_mae_rad':np.abs(a['a']-a['q']).mean(axis=0).tolist(),'robot_contact_ticks':int(robot.sum()),'bilateral_ticks':int(both.sum()),'max_lift_m':float(lift.max()),'jaw_command_range_rad':[float(a['a'][:,5].min()),float(a['a'][:,5].max())],'jaw_state_range_rad':[float(a['q'][:,5].min()),float(a['q'][:,5].max())],'late_40_50_mean_state_rad':a['q'][late].mean(axis=0).tolist() if late.any() else None,'late_40_50_mean_command_rad':a['a'][late].mean(axis=0).tolist() if late.any() else None,'late_40_50_state_range_rad':np.ptp(a['q'][late],axis=0).tolist() if late.any() else None,'qvel_source':'saved qvel_before, rad/s' if label!='teacher' else 'pre-state finite difference / .02, estimated, not measured qvel','max_abs_velocity_rad_s':np.abs(a['qvel'] if label!='teacher' else np.diff(a['q'],axis=0)/.02).max(axis=0).tolist()}
        output=[]
        for i,r in enumerate(rows):
            row={'index':i,'pre_time_s':float(a['t'][i]),'post_time_s':float(a['post_t'][i]),'phase':r['phase'] if label=='teacher' else 'not_recorded','post_robot_contact':bool(robot[i]),'post_bilateral':bool(both[i]),'post_no_support':bool(no_support[i]),'post_lift_m':float(lift[i])}
            for j,n in enumerate(names):
                for k,v in [('pre_state',a['q']),('action',a['a']),('post_state',a['post']),('command_delta',a['delta']),('action_minus_prestate',a['a']-a['q']),('ctrl_minus_poststate',a['a']-a['post'])]:row[n+'_'+k]=float(v[i,j])
                row[n+'_velocity']=float(a['qvel'][i,j]) if label!='teacher' else (None if i==0 else float((a['q'][i,j]-a['q'][i-1,j])/.02))
            output.append(row)
        csvwrite(f'trial_{ix}_{label}_trajectory.csv',output)
    for label in roots:
        a=series[label];indices=[];distances=[]
        for start in range(0,len(a['q']),64):
            guard();delta=(a['q'][start:start+64,None,:5]-tq[None,:,:5])/std[None,None,:5]
            ds=np.sum(delta*delta,axis=2);nn=ds.argmin(axis=1);indices.extend(nn.tolist());distances.extend(np.sqrt(ds[np.arange(len(nn)),nn]).tolist())
        match=[];same=[]
        for i,(j,d) in enumerate(zip(indices,distances)):
            p=max(0,j-1);n=min(len(tq)-1,j+1)
            row={'act_index':i,'act_pre_time_s':float(a['t'][i]),'teacher_index':j,'teacher_pre_time_s':float(tt[j]),'distance_l2_std_arm5':d,'teacher_phase_reference_only':teacher[j]['phase'],'teacher_prev_index':p,'teacher_next_index':n}
            for k,name in enumerate(names):
                for key,val in [('act_pre',a['q'][i,k]),('teacher_pre',tq[j,k]),('teacher_prev_pre',tq[p,k]),('teacher_next_pre',tq[n,k]),('act_command',a['a'][i,k]),('teacher_command',ta[j,k])]:row[name+'_'+key]=float(val)
            match.append(row)
            if i<len(tq):
                row2={'act_index':i,'teacher_index':i,'time_s':float(tt[i]),'teacher_phase_reference_only':teacher[i]['phase']}
                for k,name in enumerate(names):row2[name+'_command_difference']=float(a['a'][i,k]-ta[i,k]);row2[name+'_state_difference']=float(a['q'][i,k]-tq[i,k])
                same.append(row2)
        csvwrite(f'trial_{ix}_{label}_nearest.csv',match);csvwrite(f'trial_{ix}_{label}_same_time.csv',same)
        nn=np.array(indices);ds=np.array(distances);jawdiff=a['a'][:,5]-ta[nn,5]
        stats[label]['nearest']={'distance_percentiles_0_25_50_75_100':np.percentile(ds,[0,25,50,75,100]).tolist(),'late_distance_percentiles_0_50_100':np.percentile(ds[a['t']>=40],[0,50,100]).tolist(),'late_teacher_phase_counts':{p:int(sum(teacher[j]['phase']==p for j in nn[a['t']>=40])) for p in phase_starts},'late_jaw_command_minus_matched_teacher_mean_rad':float(jawdiff[a['t']>=40].mean()),'same_time_command_mae_by_axis':np.abs(a['a'][:len(tq)]-ta).mean(axis=0).tolist()}
    for metric in ['state_action','command_change','tracking']:svgplot(trial,series,metric)
    # Six scenes per series: five teacher landmarks, then each series' own final frame.
    selected=[0,phase_ends['approach'],phase_ends['descend'],phase_ends['close'],phase_starts['hold'],len(teacher)-1]
    reasons=['initial','teacher approach end','teacher descend end','teacher close end','teacher hold start','series final']
    sheet=Image.new('RGB',(1920,6*520),'white');draw=ImageDraw.Draw(sheet)
    for col,label in enumerate(['teacher','600','5000']):
        if label=='teacher':video=f'{DATA}/videos/observation.images.workspace/chunk-000/file-{trial:03d}.mp4';vt=tt;vidx=selected
        else:
            video=f'{roots[label]}/trial_{ix}/diagnostic_10fps.mp4';vt=np.array(results[label]['video_sim_times']);vidx=[int(np.argmin(np.abs(vt-tt[j]))) for j in selected[:5]]+[len(vt)-1]
        unique=sorted(set(vidx));frames,n,meta=decode(video,unique)
        for row,(idx,reason) in enumerate(zip(vidx,reasons)):
            labeltext=f'{label} t={vt[idx]:.2f}s frame={idx}: {reason}'
            sheet.paste(frames[idx],(col*640,row*520+40));draw.text((col*640+5,row*520+8),labeltext,fill='black',font=font)
            scenes.append({'trial':trial,'series':label,'reason':reason,'source_video':video,'video_index':idx,'sim_time_s':float(vt[idx]),'control_index':idx if label=='teacher' else int(round(vt[idx]*50)),'teacher_reference_index':selected[row] if row<5 else None,'teacher_reference_time_s':float(tt[selected[row]]) if row<5 else None,'decoded_video_frames':n,'video_rate':meta['r_frame_rate']})
        del frames
    sheet.save(OUT/f'trial_{ix}_review_sheet.jpg',quality=93);del sheet
    all_events.extend(events_trial);summaries.append({'trial':trial,'position':teacher_initial['block_position'],'teacher_phase_starts':phase_starts,'teacher_phase_ends':phase_ends,'teacher_result':attempt['result'],'series':stats})
    dump('partial_summary.json',summaries);print(json.dumps({'trial':trial,'complete':True,'decoded_frames':decode_total,'seconds':time.monotonic()-START}),flush=True)
assert len(scenes)==90 and decode_total==9531+5010
csvwrite('events.csv',all_events);dump('correspondence.json',correspondence);dump('scenes.json',scenes);dump('summary.json',{'trials':summaries,'std_rad':std.tolist(),'distance_definition':'L2 normalized arm5; not RMS; Jaw excluded','distance_threshold':'none; distances descriptive, not a contact/phase equivalence test','decoded_frames':decode_total,'representative_scenes':len(scenes),'teacher_rows':9531,'act_rows':25000,'forbidden_counts':{'model_forward':0,'optimizer':0,'physics_steps':0,'ctrl_application':0,'gpu':0},'elapsed_analysis_seconds':time.monotonic()-START,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'optional_limits':['teacher measured qvel not recorded; estimated 20ms finite difference','no geometric opening width measurement','ACT phase not recorded','10fps ACT video nearest frame may differ up to50ms from selected teacher time','near arm state does not imply same image/contact/intent']})
print('ANALYSIS_COMPLETE',flush=True)
