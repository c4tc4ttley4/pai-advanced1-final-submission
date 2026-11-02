"""Bounded saved-data diagnosis only. No simulator, rendering, optimizer, or control calls."""
import sys,json,csv,gzip,hashlib,time,os,glob,math,resource
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw
ROOT=Path('/workspace');BASE=ROOT/'artifacts/step05A_phase_visibility_diag/phase_visibility_r03_20261101_01'
sys.path.insert(0,str(ROOT))
C=json.loads((BASE/'spec.json').read_text());NAMES=['Rotation_R','Pitch_R','Elbow_R','Wrist_Pitch_R','Wrist_Roll_R','Jaw_R'];START=time.monotonic()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvwrite(p,rows):
    assert rows
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def guard():
    assert time.monotonic()-START<1100
    assert sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())<1024**3
    assert np.isfinite(time.monotonic())
def source_data():
    info=load(ROOT/'data/step04/dataset_20261029_01/meta/info.json');assert info['total_frames']==9531 and info['total_episodes']==5 and info['fps']==50
    assert info['features']['observation.state']['names']==info['features']['action']['names']==NAMES
    out=[]
    for ep in range(5):
        p=ROOT/f'data/step04/dataset_20261029_01/data/chunk-000/file-{ep:03d}.parquet';tab=pq.read_table(p).to_pydict()
        with gzip.open(ROOT/f'artifacts/step04/dataset_20261029_01/trace_{ep:02d}.jsonl.gz','rt') as f:tr=[json.loads(l) for l in f]
        attempt=load(ROOT/f'artifacts/step04/dataset_20261029_01/attempt_{ep:02d}.json')
        q=np.array(tab['observation.state'],dtype=np.float32);a=np.array(tab['action'],dtype=np.float32);n=len(q)
        assert n==len(tr)==attempt['frames'] and attempt['status']=='success' and attempt['seed']==42
        assert tab['frame_index']==list(range(n)) and set(tab['episode_index'])=={ep}
        assert np.array_equal(q,np.array([r['state_before'] for r in tr],dtype=np.float32))
        assert np.array_equal(a,np.array([r['sent'] for r in tr],dtype=np.float32))
        assert all(r['frame_index']==i and abs(r['time_before']-i*.02)<1e-8 for i,r in enumerate(tr))
        assert np.max(np.abs(np.array(tab['timestamp'])-np.arange(n)*.02))<3e-6
        assert np.isfinite(q).all() and np.isfinite(a).all()
        out.append(dict(ep=ep,q=q,a=a,tr=tr,tab=tab,attempt=attempt))
    assert sum(len(x['q']) for x in out)==9531;return out

# The official transition converter adds these documented empty/default fields.
# Source: installed converters.py batch_to_transition / transition_to_batch, saved by sources stage.
IMAGE_KEY='observation.images.workspace';STATE_KEY='observation.state'
DEFAULT_KEYS={'action','next.reward','next.done','next.truncated','info'}
def structure(x):
    import torch
    if isinstance(x,torch.Tensor):return dict(type='Tensor',shape=list(x.shape),dtype=str(x.dtype),device=x.device.type)
    if isinstance(x,dict):return dict(type='dict',keys={str(k):structure(v) for k,v in x.items()})
    return dict(type=type(x).__name__,null=x is None)
def contract_log(stage,label,batch):
    record=dict(stage=stage,label=label,keys={k:structure(v) for k,v in batch.items()},origin={k:('original RGB/state through saved official normalizer' if k in {IMAGE_KEY,STATE_KEY} else 'official converters.batch_to_transition defaults / transition_to_batch' if k in DEFAULT_KEYS else 'UNKNOWN') for k in batch})
    with (BASE/'contract_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return record['keys']
def checked_raw(batch,label):
    import torch
    contract_log('raw_external_observation',label,batch)
    assert set(batch)=={IMAGE_KEY,STATE_KEY},'raw input must be RGB+state only'
    for key,shape in [(IMAGE_KEY,(1,3,480,640)),(STATE_KEY,(1,6))]:
        x=batch[key];assert isinstance(x,torch.Tensor) and x.dtype==torch.float32 and tuple(x.shape)==shape and x.device.type=='cpu' and torch.isfinite(x).all()
    assert 0<=batch[IMAGE_KEY].min()<=batch[IMAGE_KEY].max()<=1

def checked_policy_batch(batch,device,label,stage='policy_boundary'):
    import torch
    schema=contract_log(stage,label,batch)
    assert set(batch)=={IMAGE_KEY,STATE_KEY}|DEFAULT_KEYS,'unknown/missing internal batch key: see structure log'
    assert batch['action'] is None and type(batch['next.reward']) is float and batch['next.reward']==0.0
    assert batch['next.done'] is False and batch['next.truncated'] is False and type(batch['info']) is dict and not batch['info']
    for key,shape in [(IMAGE_KEY,(1,3,480,640)),(STATE_KEY,(1,6))]:
        x=batch[key];assert isinstance(x,torch.Tensor) and x.dtype==torch.float32 and tuple(x.shape)==shape and x.device.type==device and torch.isfinite(x).all()
    if device=='cuda':
        expected=load(BASE/'contract_schema.json')['processed_keys']
        canonical=json.loads(json.dumps(schema))
        for key in [IMAGE_KEY,STATE_KEY]:canonical[key]['device']='cpu'
        assert canonical==expected,'GPU schema differs beyond documented device'
    return schema

def verify_official_sources():
    for r in load(BASE/'official_source_identities.json')['files']:assert sha(r['path'])==r['sha256'],'official source changed'

def contract_preflight():
    import torch
    from collections import deque
    from types import SimpleNamespace
    from safetensors.torch import load_file
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition,transition_to_batch
    from lerobot.policies.act.modeling_act import ACTPolicy
    verify_official_sources();directory=ROOT/C['models'][0]['path']
    pre=PolicyProcessorPipeline.from_pretrained(str(directory),config_filename='policy_preprocessor.json',local_files_only=True,overrides={'device_processor':{'device':'cpu'}},to_transition=batch_to_transition,to_output=transition_to_batch)
    fixture=load_file(directory/'reference_observations.safetensors');assert set(fixture)=={IMAGE_KEY,STATE_KEY}
    stats=load(directory/'statistics.json')['stats'];schemas=[];normalization_errors=[]
    class BeforeForward(Exception):pass
    class Probe:
        config=SimpleNamespace(temporal_ensemble_coeff=None)
        def __init__(self,label):self._action_queue=deque();self.label=label;self.intercepted=False
        def eval(self):return self
        def predict_action_chunk(self,batch):
            checked_policy_batch(batch,'cpu',self.label);self.intercepted=True;raise BeforeForward()
    for i in range(len(fixture[STATE_KEY])):
        raw={k:v[i:i+1] for k,v in fixture.items()};label=f'cpu_reference_{i}';checked_raw(raw,label);processed=pre(raw)
        schema=checked_policy_batch(processed,'cpu',label,'official_pre_output');schemas.append(schema)
        errs={}
        for key in [IMAGE_KEY,STATE_KEY]:
            mean=torch.tensor(stats[key]['mean'],dtype=torch.float32);std=torch.tensor(stats[key]['std'],dtype=torch.float32)
            expected=(raw[key]-mean)/(std+1e-8);err=float((expected-processed[key]).abs().max());assert err<=1e-6;errs[key]=err
        normalization_errors.append(errs);probe=Probe(label)
        try:ACTPolicy.select_action(probe,processed)
        except BeforeForward:pass
        assert probe.intercepted and not probe._action_queue
    assert all(s==schemas[0] for s in schemas)
    # All three checkpoints use byte-identical config, processor files and saved statistics.
    same=['config.json','policy_preprocessor.json','policy_postprocessor.json','statistics.json','policy_preprocessor_step_3_normalizer_processor.safetensors']
    for m in C['models'][1:]:
        for name in same:assert sha(ROOT/m['path']/name)==sha(directory/name)
    write(BASE/'contract_schema.json',dict(pass_=True,processed_keys=schemas[0],external_keys=[IMAGE_KEY,STATE_KEY],allowed_internal_defaults=dict(action='None only',next_reward='float0.0 only',next_done='False only',next_truncated='False only',info='empty dict only'),sources='official_sources/processor/converters.py:326-400; pipeline.py:284-311; policies/act/modeling_act.py:99-133,400-455',cpu_override='device_processor.device=cpu only; saved bytes/statistics unchanged',normalization_errors=normalization_errors,boundary_probes=len(schemas),model_loads=0,model_forwards=0,probe='official ACTPolicy.select_action unbound method intercepted immediately before predict_action_chunk; no synthetic predictions'))

def preflight():
    # Actual target syntax and exact helper import, in the GPU-hidden container.
    compile(Path(__file__).read_text(),str(Path(__file__)),'exec')
    from src.step05A_act import setup,dataset,load_bundle,digest,queue_reference,IMAGE,STATE,resources
    import torch
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    contract_preflight()
    old=ROOT/C['selection_source'];samples=load(old/'samples.json');coverage=load(old/'coverage.json')
    for ref in C['selection_references']:assert sha(ROOT/ref['path'])==ref['sha256']
    assert len(samples)==240 and sum(s['visual'] for s in samples)==90
    assert samples==sorted(samples,key=lambda s:(s['episode'],s['frame_index']))
    assert len({s['global_index'] for s in samples})==len({s['sample_id'] for s in samples})==240
    oldcsv=list(csv.DictReader((old/'phase_samples.csv').open()))
    assert len(oldcsv)==240
    for s,r in zip(samples,oldcsv):
        for k,v in s.items():
            if isinstance(v,(list,dict)):assert json.loads(r[k])==v
            elif isinstance(v,bool):assert r[k]==str(v)
            else:assert r[k]==str(v)
    d=source_data();events=list(csv.DictReader((ROOT/'artifacts/step05A_grasp_trace_review/grasp_trace_20261101_01/events.csv').open()))
    stats=load(ROOT/C['models'][0]['path']/'statistics.json')['stats'];qs=np.array(stats['observation.state']['std']);aas=np.array(stats['action']['std'])
    assert (qs>0).all() and (aas>0).all();coverage_refs=[];sample_refs=[]
    for x in d:
        ep=x['ep'];q=x['q'];a=x['a'];tr=x['tr']
        for phase in dict.fromkeys(r['phase'] for r in tr):
            ix=np.array([i for i,r in enumerate(tr) if r['phase']==phase]);assert np.all(np.diff(ix)==1)
            ev=[r for r in events if int(r['trial'])==ep and r['series']=='teacher' and r['event']=='phase_start_'+phase]
            assert len(ev)==1 and int(ev[0]['index'])==int(ix[0])==x['attempt']['phase_first_frame'][phase]
            oldrow=[r for r in coverage if r['episode']==ep and r['original_phase']==phase];assert len(oldrow)==1;r=oldrow[0]
            qdiff=np.linalg.norm(np.diff(q[ix],axis=0)/qs,axis=1);adiff=np.linalg.norm(np.diff(a[ix],axis=0)/aas,axis=1)
            assert r['frames']==len(ix) and r['first_frame']==int(ix[0]) and r['last_frame']==int(ix[-1])
            assert r['seconds']==len(ix)/50 and r['fraction_episode']==len(ix)/len(q)
            assert r['state_min_rad']==q[ix].min(0).tolist() and r['state_max_rad']==q[ix].max(0).tolist()
            assert r['action_min_rad']==a[ix].min(0).tolist() and r['action_max_rad']==a[ix].max(0).tolist()
            assert r['first_pre_s']==tr[ix[0]]['time_before'] and r['last_post_s']==tr[ix[-1]]['time']
            assert r['adjacent_pairs']==len(qdiff) and r['near_state_adjacent_pairs']==int((qdiff<.05).sum()) and r['near_action_adjacent_pairs']==int((adiff<.05).sum())
            assert r['median_adjacent_state_l2_std']==float(np.median(qdiff)) and r['median_adjacent_action_l2_std']==float(np.median(adiff))
            coverage_refs.append(dict(episode=ep,phase=phase,frames_verified=len(ix),source='read-only r01 coverage.json / teacher_phase_coverage.csv',source_sha256=sha(old/'coverage.json'),original_frame_range_verified=True))
        for phase in C['phases']:
            ix=np.array([i for i,r in enumerate(tr) if r['phase']==phase]);selected=ix[np.unique(np.linspace(0,len(ix)-1,8,dtype=int))]
            rows=[s for s in samples if s['episode']==ep and s['phase']==phase]
            assert [s['frame_index'] for s in rows]==selected.tolist()
            assert [s['frame_index'] for s in rows if s['visual']]==selected[np.linspace(0,7,3,dtype=int)].tolist()
            for s in rows:
                i=s['frame_index'];r=tr[i]
                assert s['sample_id']==f'e{ep}_{phase}_f{i:04d}' and s['global_index']==int(x['tab']['index'][i])
                assert s['pre_time_s']==r['time_before'] and s['post_time_s']==r['time'] and r['phase']==phase
                assert s['state']==q[i].tolist() and s['teacher_action']==a[i].tolist() and s['previous_teacher_action']==a[i-1].tolist()
                assert s['teacher_delta_from_state']==(a[i]-q[i]).tolist() and s['teacher_command_change']==(a[i]-a[i-1]).tolist()
                assert s['post_block']==r['block'] and s['post_point']==r['point']
                assert s['post_contact']==any('block_geom' in c['geoms'] and any(g.startswith('robot_geom') for g in c['geoms']) for c in r['contacts'])
    assert len(coverage)==len(coverage_refs)==55
    oldcov=list(csv.DictReader((old/'teacher_phase_coverage.csv').open()));assert len(oldcov)==55
    for r,v in zip(oldcov,coverage):
        for k,x in v.items():assert (json.loads(r[k])==x if isinstance(x,(list,dict)) else r[k]==str(x))
    allq=np.concatenate([x['q'] for x in d]);alla=np.concatenate([x['a'] for x in d]);allep=np.concatenate([np.full(len(x['q']),x['ep']) for x in d]);allindices=np.concatenate([np.array(x['tab']['index']) for x in d])
    nearest=list(csv.DictReader((old/'cross_episode_nearest.csv').open()));assert len(nearest)==240
    for s,r in zip(samples,nearest):
        mask=allep!=s['episode'];qd=np.linalg.norm((allq[mask]-s['state'])/qs,axis=1);ad=np.linalg.norm((alla[mask]-s['teacher_action'])/aas,axis=1)
        assert s['sample_id']==r['sample_id'] and int(r['episode'])==s['episode'] and r['phase']==s['phase']
        assert int(r['nearest_state_other_episode_index'])==int(allindices[mask][qd.argmin()]) and float(r['state_l2_std'])==float(qd.min())
        assert int(r['nearest_action_other_episode_index'])==int(allindices[mask][ad.argmin()]) and float(r['action_l2_std'])==float(ad.min())
    ds=dataset();assert len(ds)==9531 and ds.num_episodes==5 and ds.fps==50
    verified_r02={r['sample_id']:r for r in load(ROOT/C['image_source']/'verified_input_images.json')}
    r02_manifest={r['path']:r for r in load(ROOT/C['image_source']/'output_manifest.json')['files']}
    for s in samples:
        guard();item=ds[s['global_index']];image=item[IMAGE];state=item[STATE]
        assert image.shape==(3,480,640) and image.dtype==torch.float32 and torch.isfinite(image).all() and 0<=image.min()<=image.max()<=1
        assert np.array_equal(state.numpy(),np.array(s['state'],np.float32)) and np.array_equal(item['action'][0].numpy(),np.array(s['teacher_action'],np.float32))
        assert int(item['index'])==s['global_index'] and int(item['frame_index'])==s['frame_index'] and int(item['episode_index'])==s['episode']
        assert abs(float(item['timestamp'])-s['pre_time_s'])<3e-6
        imagehash=hashlib.sha256(image.numpy().tobytes()).hexdigest()
        if s['visual']:
            p=ROOT/C['image_source']/'images'/f"{s['sample_id']}.png";rgb=image.permute(1,2,0).mul(255).round().byte().numpy()
            assert Image.open(p).size==(640,480) and np.array_equal(np.array(Image.open(p).convert('RGB')),rgb)
            assert sha(p)==r02_manifest[str(p.relative_to(ROOT))]['sha256']
        assert imagehash==verified_r02[s['sample_id']]['input_image_chw_sha256']
        sample_refs.append(dict(sample_id=s['sample_id'],episode=s['episode'],phase=s['phase'],frame_index=s['frame_index'],global_index=s['global_index'],pre_time_s=s['pre_time_s'],visual=s['visual'],input_image_chw_sha256=imagehash,source_row_verified=True))
    assert not torch.cuda.is_initialized() and load(BASE/'budget.json')['forwards']==0
    csvwrite(BASE/'phase_samples.csv',sample_refs);csvwrite(BASE/'teacher_phase_coverage.csv',coverage_refs)
    write(BASE/'verified_input_images.json',sample_refs)
    write(BASE/'preflight_result.json',dict(pass_=True,syntax_checked=str(Path(__file__)),exact_helper_import=True,gpu_hidden=True,cuda_initialized=False,model_loads=0,forwards=0,physics_steps=0,source_selection_unchanged=True,observations=240,visual_observations=90,all_teacher_frames=9531,coverage_rows_verified=55,nearest_rows_verified=240,selection_references=C['selection_references'],resources=dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),seconds=time.monotonic()-START))
    print(json.dumps(dict(cpu_preflight_pass=True,observations=240,images=90,forwards=0)),flush=True)

def consume(kind,label):
    p=BASE/'budget.json';b=load(p);assert b['forwards']<1000;b['forwards']+=1;b[kind+'_forwards']+=1;b['last_forward']=label
    p.write_text(json.dumps(b,indent=2)+'\n')

def predict():
    assert load(BASE/'preflight_result.json')['pass_'] and not (BASE/'prediction_result.json').exists()
    import torch,gc
    from safetensors.torch import load_file
    from src.step05A_act import setup,dataset,load_bundle,digest,queue_reference,IMAGE,STATE,resources
    assert load(BASE/'visibility_pass.json')['pass_'];verify_official_sources()
    setup(42);assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    samples=load(ROOT/C['selection_source']/'samples.json');ds=dataset();cache=[];verified_images={s['sample_id']:s for s in load(BASE/'verified_input_images.json')}
    for s in samples:
        guard();item=ds[s['global_index']];image=item[IMAGE].clone();state=item[STATE].clone()
        assert image.shape==(3,480,640) and state.shape==(6,) and torch.isfinite(image).all() and 0<=image.min()<=image.max()<=1
        assert np.array_equal(state.numpy(),np.array(s['state'],np.float32)) and np.array_equal(item['action'][0].numpy(),np.array(s['teacher_action'],np.float32))
        assert hashlib.sha256(image.numpy().tobytes()).hexdigest()==verified_images[s['sample_id']]['input_image_chw_sha256']
        cache.append(({IMAGE:image.unsqueeze(0),STATE:state.unsqueeze(0)},hashlib.sha256(image.numpy().tobytes()).hexdigest()))
    del ds;gc.collect();references=[];predictions=[]
    for model in C['models']:
        directory=ROOT/model['path'];policy,pre,post=load_bundle(directory);before=digest(policy)
        assert before==load(directory/'training_result.json')['after_model_digest'];assert policy.config.n_action_steps==10 and policy.config.chunk_size==50 and policy.config.temporal_ensemble_coeff is None
        official_pre=pre
        def safe_pre(raw):
            checked_raw(raw,context['label']);out=official_pre(raw);checked_policy_batch(out,'cuda',context['label'],'official_pre_output');return out
        pre=safe_pre
        original=policy.predict_action_chunk;context={'kind':'reference','label':'reference'}
        def counted(batch):
            checked_policy_batch(batch,'cuda',context['label']);consume(context['kind'],context['label']);return original(batch)
        policy.predict_action_chunk=counted
        actual=queue_reference(policy,pre,post,load_file(directory/'reference_observations.safetensors'));expected=load_file(directory/'reference_actions.safetensors')['actions_rad'];error=float((actual-expected).abs().max());assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
        references.append(dict(step=model['step'],pass_=True,max_error_rad=error,digest=before,all_hashes_checked=True,strict=True,local_files_only=True))
        with torch.inference_mode():
            for s,(batch,imagehash) in zip(samples,cache):
                guard();policy.reset();context.update(kind='teacher',label=f"step{model['step']}:{s['sample_id']}");b0=load(BASE/'budget.json')['forwards'];out=post(policy.select_action(pre(batch))).squeeze(0).cpu().numpy()
                assert out.shape==(6,) and np.isfinite(out).all() and load(BASE/'budget.json')['forwards']==b0+1 and len(policy._action_queue)==9
                predictions.append(dict(sample_id=s['sample_id'],step=model['step'],episode=s['episode'],phase=s['phase'],frame_index=s['frame_index'],global_index=s['global_index'],pre_time_s=s['pre_time_s'],input_image_chw_sha256=imagehash,input_keys=[IMAGE,STATE],queue_reset_for_each_observation=True,queue_remaining=9,model_forward_count=1,predicted_action=out.tolist()))
        assert digest(policy)==before,'inference changed model tensors'
        del policy,pre,post;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(prediction_step=model['step'],samples=len(samples),reference_error_rad=error)),flush=True)
    with (BASE/'predictions.jsonl').open('x') as f:
        for r in predictions:f.write(json.dumps(r)+'\n')
    write(BASE/'prediction_result.json',dict(pass_=True,models=references,samples=len(samples),predictions=len(predictions),budget=load(BASE/'budget.json'),resources=resources(),seconds=time.monotonic()-START,inference='independent first action only at saved teacher observation; not 10tick or closed-loop reproduction'))

def red_components(rgb):
    f=rgb.astype(float);c=C['visibility_red_mask'];mask=(f[:,:,0]>=c['r_min'])&(f[:,:,0]>c['r_over_g']*f[:,:,1])&(f[:,:,0]>c['r_over_b']*f[:,:,2])&(f[:,:,1]<c['g_max'])&(f[:,:,2]<c['b_max']);points=set(np.flatnonzero(mask).tolist());result=[]
    while points:
        a=points.pop();todo=[a];component=[a]
        while todo:
            v=todo.pop();y,x=divmod(v,640)
            for yy,xx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                z=yy*640+xx
                if 0<=yy<480 and 0<=xx<640 and z in points:points.remove(z);component.append(z);todo.append(z)
        if len(component)>=3:
            yy,xx=np.divmod(component,640);result.append(dict(area=len(component),bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],width_px=int(xx.max()-xx.min()+1),height_px=int(yy.max()-yy.min()+1)))
    return sorted(result,key=lambda r:r['area'],reverse=True)


def camera():
    assert load(BASE/'preflight_result.json')['pass_'];samples=load(ROOT/C['selection_source']/'samples.json')
    candidates=[];scenes=[s for s in samples if s['visual']];assert len(scenes)<=90
    for s in scenes:
        p=ROOT/C['image_source']/'images'/f"{s['sample_id']}.png";im=Image.open(p).convert('RGB');assert im.size==(640,480);cs=red_components(np.array(im));candidate=cs[0] if cs else None
        candidates.append(dict(sample_id=s['sample_id'],episode=s['episode'],phase=s['phase'],frame_index=s['frame_index'],pre_time_s=s['pre_time_s'],image_path=str(p.relative_to(ROOT)),image_sha256=sha(p),candidate=candidate,red_components=cs,measurement='largest visible-red-surface bounding box only; not full object extent; manual validation pending'))
    write(BASE/'visibility_candidates.json',candidates)
    for ep in range(5):
        sheet=Image.new('RGB',(1920,6*450),'white');draw=ImageDraw.Draw(sheet)
        for row,phase in enumerate(C['phases']):
            selected=[c for c in candidates if c['episode']==ep and c['phase']==phase];assert len(selected)<=3
            for col,c in enumerate(selected):
                im=Image.open(ROOT/c['image_path']).convert('RGB');x=col*640;y=row*450
                draw.text((x+6,y+5),f"e{ep} {phase} f{c['frame_index']} t={c['pre_time_s']:.2f}s",fill='black')
                full=im.copy();box=c['candidate']['bbox'] if c['candidate'] else None
                if box:
                    ImageDraw.Draw(full).rectangle(box,outline='yellow',width=1);cx=(box[0]+box[2])//2;cy=(box[1]+box[3])//2
                else:cx,cy=320,270
                sheet.paste(full.resize((480,360)),(x,y+28));left=max(0,min(520,cx-60));top=max(0,min(390,cy-45));crop=im.crop((left,top,left+120,top+90)).resize((240,180),Image.Resampling.NEAREST);sheet.paste(crop,(x+400,y+220))
                label='red candidate absent' if box is None else f"red bbox={box}, visible width={c['candidate']['width_px']}px"
                draw.text((x+5,y+405),label,fill='black');draw.text((x+5,y+425),'full image + 2x original-pixel crop (derived; never inference input)',fill='black')
        sheet.save(BASE/f'visibility_episode_{ep}.png')
    write(BASE/'camera_result.json',dict(pass_=True,images=len(candidates),original_images_read_only=True,model_forwards=0,seconds=time.monotonic()-START))

def analyze():
    assert load(BASE/'prediction_result.json')['pass_'] and not (BASE/'analysis_result.json').exists();samples=load(ROOT/C['selection_source']/'samples.json');lookup={s['sample_id']:s for s in samples}
    predictions=[json.loads(l) for l in (BASE/'predictions.jsonl').read_text().splitlines()];assert len(predictions)==3*len(samples)
    records=[];groups={};eps=C['sign_epsilon_rad']
    def sign(x):return int(x>eps)-int(x<-eps)
    for r in predictions:
        s=lookup[r['sample_id']];q=np.array(s['state']);a=np.array(s['teacher_action']);p=np.array(r['predicted_action']);delta=p-q;target=a-q;error=p-a
        for j,name in enumerate(NAMES):
            row=dict(step=r['step'],episode=s['episode'],phase=s['phase'],sample_id=s['sample_id'],frame_index=s['frame_index'],pre_time_s=s['pre_time_s'],joint=name,state_rad=float(q[j]),teacher_action_rad=float(a[j]),predicted_action_rad=float(p[j]),signed_error_rad=float(error[j]),absolute_error_rad=float(abs(error[j])),teacher_delta_from_state_rad=float(target[j]),predicted_delta_from_state_rad=float(delta[j]),teacher_command_change_rad=float(s['teacher_command_change'][j]),teacher_direction=sign(target[j]),predicted_direction=sign(delta[j]),direction_agreement=sign(target[j])==sign(delta[j]),teacher_command_change_direction=sign(s['teacher_command_change'][j]))
            records.append(row)
        for ep in [s['episode'],'ALL']:
            for phase in [s['phase'],'ALL']:groups.setdefault((r['step'],str(ep),phase),[]).append((s,p,error,delta,target))
    csvwrite(BASE/'phase_action_comparison.csv',records);summaries=[];jaw=[]
    for (step,ep,phase),rs in groups.items():
        er=np.stack([r[2] for r in rs]);pp=np.stack([r[1] for r in rs]);dd=np.stack([r[3] for r in rs]);tt=np.stack([r[4] for r in rs]);aa=np.array([r[0]['teacher_action'] for r in rs]);times=[r[0]['pre_time_s'] for r in rs]
        result=dict(step=step,episode=ep,phase=phase,observations=len(rs),mae_rad=float(np.abs(er).mean()),median_absolute_error_rad=float(np.median(np.abs(er))),max_absolute_error_rad=float(np.abs(er).max()),joint_mae_rad=np.abs(er).mean(0).tolist(),joint_median_absolute_error_rad=np.median(np.abs(er),axis=0).tolist(),joint_max_absolute_error_rad=np.abs(er).max(0).tolist(),mean_predicted_delta_rad=dd.mean(0).tolist(),mean_teacher_delta_rad=tt.mean(0).tolist(),joint_direction_agreement=[float(np.mean([sign(x)==sign(y) for x,y in zip(dd[:,j],tt[:,j])])) for j in range(6)])
        summaries.append(result)
        close=[r[0]['pre_time_s'] for r in rs if r[1][5]-r[0]['state'][5]<-eps];open_=[r[0]['pre_time_s'] for r in rs if r[1][5]-r[0]['state'][5]>eps]
        jaw.append(dict(step=step,episode=ep,phase=phase,observations=len(rs),jaw_mae_rad=float(np.abs(er[:,5]).mean()),teacher_jaw_min_rad=float(aa[:,5].min()),teacher_jaw_max_rad=float(aa[:,5].max()),predicted_jaw_min_rad=float(pp[:,5].min()),predicted_jaw_max_rad=float(pp[:,5].max()),predicted_closing_direction_samples=len(close),predicted_opening_direction_samples=len(open_),first_selected_closing_pre_time_s=min(close) if close else None,first_selected_opening_pre_time_s=min(open_) if open_ else None,definition='action-state sign, epsilon1e-5rad; selected teacher frames only; times across ALL episodes are minima, not rollout event times'))
    csvwrite(BASE/'phase_mae_summary.csv',summaries);csvwrite(BASE/'jaw_direction_summary.csv',jaw);write(BASE/'phase_summary.json',summaries);write(BASE/'jaw_summary.json',jaw)
    ref=load(ROOT/'artifacts/step05A_train_compare/train20000_r02_20261101_01/report_metrics.json');write(BASE/'saved_rollout_reference.json',dict(source='saved report_metrics and traces; no new rollout',trials=ref['physics'],scope='ACT has no recorded phase labels; no teacher phase assigned to autonomous rows'))
    write(BASE/'analysis_result.json',dict(pass_=True,comparisons=len(records),phase_groups=len(summaries),images=90,first_action_only=True,independent_evaluation=False,forbidden=C['forbidden'],resources=dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),seconds=time.monotonic()-START))
    print(json.dumps(dict(analysis_pass=True,records=len(records),images=90,mae=[r for r in summaries if r['episode']=='ALL' and r['phase']!='ALL'])),flush=True)

def validate():
    assert load(BASE/'analysis_result.json')['pass_'];samples=load(ROOT/C['selection_source']/'samples.json');pred=[json.loads(l) for l in (BASE/'predictions.jsonl').read_text().splitlines()];b=load(BASE/'budget.json');assert b['forwards']==3*len(samples)+15<=1000 and b['teacher_forwards']==3*len(samples) and b['reference_forwards']==15 and b['optimizer_updates']==b['physics_steps']==0
    assert len(pred)==len(samples)*3 and len({(r['step'],r['sample_id']) for r in pred})==len(pred)
    for s in samples:assert len({r['input_image_chw_sha256'] for r in pred if r['sample_id']==s['sample_id']})==1
    with (BASE/'visibility_review.csv').open() as f:visibility=list(csv.DictReader(f))
    assert len(visibility)==sum(s['visual'] for s in samples)<=90 and {r['sample_id'] for r in visibility}=={s['sample_id'] for s in samples if s['visual']}
    for r in visibility:assert r['manual_reviewed']=='true' and r['object_visibility'] and r['finger_pair_visibility'] and r['left_tip_visibility'] and r['right_tip_visibility'] and r['wrist_visibility'] and r['tray_visibility'] and r['edge_loss'] and r['measurement_basis']
    candidates=load(BASE/'visibility_candidates.json')
    for c in candidates:assert sha(ROOT/c['image_path'])==c['image_sha256']
    assert (BASE/'hypothesis_table.md').exists() and (BASE/'next_priority.json').exists()
    write(BASE/'final_validation.json',dict(pass_=True,selected_observations=len(samples),model_forwards=b['forwards'],references=15,teacher_predictions=len(pred),visual_images=len(visibility),same_teacher_inputs_across_models=True,all_manual_visibility_rows_present=True,forbidden=C['forbidden'],resources=dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),seconds=time.monotonic()-START))
    print(json.dumps(dict(final_validation=True,budget=b,images=len(visibility))),flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode!='predict':assert os.getuid()!=0 and not glob.glob('/dev/nvidia*') and os.getenv('NVIDIA_VISIBLE_DEVICES')=='void'
    {'preflight':preflight,'camera':camera,'predict':predict,'analyze':analyze,'validate':validate}[mode]()
