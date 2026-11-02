"""CPU-only saved-result audit; never imports policy, simulator, or GPU libraries."""
import sys,json,gzip,csv,os,glob,math,time
from pathlib import Path
import numpy as np
ROOT=Path('/workspace');BASE=ROOT/'artifacts/step05A_train_compare/train20000_r02_20261101_01'
C=json.loads((BASE/'spec.json').read_text())
assert os.getuid()!=0 and not glob.glob('/dev/nvidia*')
def load(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def rows(p):
    with gzip.open(p/'trace.jsonl.gz','rt') as f:return [json.loads(l) for l in f]
def first_event(mask,times):
    ix=np.flatnonzero(mask)
    return None if len(ix)==0 else dict(index=int(ix[0]),time_s=float(times[ix[0]]))
def metrics(directory):
    result=load(directory/'result.json');rs=rows(directory);applied=[r for r in rs if r['applied']]
    assert applied,'no applied states for physical metrics'
    q=np.array([r['state_before'] for r in applied]);a=np.array([r['action_rad'] for r in applied]);post=np.array([r['state_after'] for r in applied]);t=np.array([r['time_before'] for r in applied]);after=np.array([r['time_after'] for r in applied]);d=[r['diagnostic'] for r in applied]
    robot=np.array([any('block_geom' in c['geoms'] and any(g.startswith('robot_geom') for g in c['geoms']) for c in r['contacts']) for r in applied]);both=np.array([x['bilateral_contact'] for x in d]);lift=np.array([x['lift_m'] for x in d]);hold=np.array([x['grasp_hold_seconds'] for x in d]);stable=np.array([x['stable_seconds'] for x in d]);late=t>=40
    block=np.array([x['block'] for x in d]);initial=np.array(result['position'])
    # Five simulated seconds of measured range <0.01rad across all six axes; explicit definition, no outcome gate.
    windows=[]
    for start in range(0,len(applied)-249,250):
        span=np.ptp(q[start:start+250],axis=0)
        if np.all(span<.01):windows.append(dict(start_s=float(t[start]),end_s=float(after[start+249]),state_span_rad=span.tolist()))
    return dict(result=result,rows=len(rs),robot_contact_ticks=int(robot.sum()),bilateral_contact_ticks=int(both.sum()),first_robot_contact=first_event(robot,after),first_bilateral_contact=first_event(both,after),first_jaw_command_below_055=first_event(a[:,5]<.55,t),first_40mm_lift=first_event(lift>=.04,after),first_1s_grasp_hold=first_event(hold+1e-9>=1,after),first_2s_storage_stable=first_event(stable+1e-9>=2,after),max_lift_m=float(max(0,lift.max())),max_horizontal_displacement_m=float(np.linalg.norm(block[:,:2]-initial[:2],axis=1).max()),jaw_command_range_rad=[float(a[:,5].min()),float(a[:,5].max())],jaw_state_range_rad=[float(q[:,5].min()),float(q[:,5].max())],pre_tracking_mae_rad=np.abs(a-q).mean(axis=0).tolist(),pre_tracking_abs_max_rad=np.abs(a-q).max(axis=0).tolist(),post_tracking_mae_rad=np.abs(a-post).mean(axis=0).tolist(),late_40_50_rows=int(late.sum()),late_40_50_state_range_rad=np.ptp(q[late],axis=0).tolist() if late.any() else None,late_40_50_mean_state_rad=q[late].mean(axis=0).tolist() if late.any() else None,late_40_50_mean_command_rad=a[late].mean(axis=0).tolist() if late.any() else None,stagnation_definition='nonoverlapping 5s windows, qpos range <0.01rad in every joint',stagnation_windows=windows)

def audit(checkpoint,trial):
    directory=BASE/f'step{checkpoint}'/f'trial_{trial:02d}';result=load(directory/'result.json');rs=rows(directory)
    assert not (directory/'audit.json').exists(),'no duplicate audit'
    initial=load(directory/'initial.json');old=load(ROOT/C['baseline']/f'trial_{trial:02d}'/'initial.json');teacher=load(ROOT/f'artifacts/step04/dataset_20261029_01/attempt_{trial:02d}.json')
    assert initial['integration']==old['integration']==teacher['initial']['integration'] and initial['position']==C['positions'][trial]
    assert initial['seed']==42 and initial['queue_length']==0 and initial['quaternion']==[1.,0.,0.,0.]
    assert result['checkpoint']==checkpoint and result['trial']==trial
    assert result['status'] in ['bounded_observation_finished','safety_stop_after_physics'],result['termination_reason']
    physical_stop=result['status']=='safety_stop_after_physics'
    assert result['video_encoder_status']==0 and result['ticks_applied']==len(rs)<=2500
    assert result['actions_clipped']==0 and not result['compensation'] and result['additional_training_updates']==0
    assert not result['warnings'] and not any(result['warning_counts'])
    if physical_stop:
        assert not result['valid_observation_end'] and result['termination_reason'].startswith('deep_robot_contact:')
        assert result['max_robot_penetration_m']>.001 and rs[-1]['stop_reason']==result['termination_reason']
    else:
        assert result['valid_observation_end'] and result['max_robot_penetration_m']<=.001
        assert result['termination_reason'] in ['simulation_time_limit','physical_task_success']
        if result['termination_reason']=='simulation_time_limit':assert len(rs)==2500 and result['physical_steps']==25000
    bounds=np.array(load(ROOT/'configs/step02.json')['action_range']);threshold=np.array(load(ROOT/C['old_run']/'data_audit.json')['action_jump_limit_rad'])
    warning_indices=[];maxex=np.zeros(6);maxratio=np.zeros(6);physics=0;speed=np.zeros(6)
    for i,r in enumerate(rs):
        assert r['tick']==i and r['applied']
        assert r['source']==('new_ACT_prediction' if i%10==0 else 'ACT_queue') and r['queue_remaining']==9-i%10
        assert abs(r['time_before']-i*.02)<1e-8 and r['image_sim_time']==r['time_before']
        duration=r['time_after']-r['time_before'];substeps=round(duration/.002);physics+=substeps
        assert 1<=substeps<=10 and abs(duration-substeps*.002)<1e-10
        if i<len(rs)-1 or not physical_stop:assert substeps==10
        assert r['policy_input_keys']==['observation.images.workspace','observation.state']
        a=np.array(r['action_rad'],dtype=np.float32);assert a.shape==(6,) and np.array_equal(a,np.array(r['ctrl']))
        assert np.all(a>=bounds[:,0]) and np.all(a<=bounds[:,1])
        for k in ['state_before','state_after','qvel_before','qvel_after','action_rad','ctrl']:assert np.isfinite(r[k]).all()
        previous=np.array(r['previous_ctrl'],dtype=np.float64 if i==0 else np.float32)
        if i:assert np.array_equal(previous,np.array(rs[i-1]['action_rad'],dtype=np.float32))
        delta=a-previous;ex=np.maximum(np.abs(delta)-threshold,0);ratio=np.abs(delta)/threshold;warn=bool(np.any(np.abs(delta)>threshold))
        assert np.array_equal(delta,np.array(r['signed_jump_rad'])) and np.array_equal(ex,np.array(r['jump_excess_rad'])) and np.array_equal(ratio,np.array(r['jump_ratio']))
        assert r['jump_warning']==warn and np.array_equal(threshold,np.array(r['jump_threshold_rad']))
        if warn:warning_indices.append(i)
        maxex=np.maximum(maxex,ex);maxratio=np.maximum(maxratio,ratio)
        sofar=np.array(r['max_measured_speed_rad_s_so_far']);assert np.all(sofar>=speed);speed=sofar
    assert physics==result['physical_steps'] and abs(physics*.002-result['sim_seconds'])<1e-8
    assert result['new_predictions']==math.ceil(len(rs)/10) and result['queue_actions']==len(rs)-result['new_predictions']
    assert len(warning_indices)==result['jump_warning_events'] and (warning_indices[0] if warning_indices else None)==result['old_strict_first_stop_tick']
    assert np.array_equal(maxex,np.array(result['max_jump_excess_rad'])) and np.array_equal(maxratio,np.array(result['max_jump_ratio'])) and np.array_equal(speed,np.array(result['max_measured_speed_rad_s']))
    v=load(directory/'video_check.json');assert v['pass_'] and v['all_frames_decode'] and len(v['frame_sim_times'])==result['video_frames']<=501 and result['video_fps']==10 and v['frame_sim_times']==result['video_sim_times']
    exit_text=(BASE/f'interval_{checkpoint}_{trial}.exit.txt').read_text();assert 'monitor_stop=\n' in exit_text
    assert exit_text.startswith('exit=3 ' if physical_stop else 'exit=0 ')
    comparison=dict(checkpoint=checkpoint,trial=trial,label=C['labels'][trial],current=metrics(directory),saved5000=metrics(ROOT/C['baseline']/f'trial_{trial:02d}'))
    write(directory/'comparison.json',comparison)
    write(directory/'audit.json',dict(pass_=True,checkpoint=checkpoint,trial=trial,rows=len(rs),initial_exact=True,warning_arithmetic_exact=True,action_ctrl_exact=True,queue_period=10,physics_steps=physics,video_checked=True,physical_safety_stop=physical_stop,resource_exit=exit_text.strip()))
    print(json.dumps(dict(audit_pass=True,checkpoint=checkpoint,trial=trial,physical_stop=physical_stop,contacts=comparison['current']['robot_contact_ticks'],bilateral=comparison['current']['bilateral_contact_ticks'],max_lift_m=comparison['current']['max_lift_m'],grasp=result['grasp_success'],storage=result['storage_success'])),flush=True)

def final():
    updates=[json.loads(l) for l in (BASE/'updates.jsonl').read_text().splitlines()]
    assert len(updates)==15000 and [r['step'] for r in updates]==list(range(5001,20001))
    assert all(r['additional_update']==r['step']-5000 and r['batch_size']==4 and len(r['sample_indices'])==4 and r['finite_weights_optimizer'] and all(0<=x<9531 for x in r['sample_indices']) for r in updates)
    for r in updates:
        for k in ['loss','l1_loss','kld_loss','grad_norm','update_wall_seconds','wall_elapsed_seconds']:assert math.isfinite(r[k])
    b=load(BASE/'budget.json');assert b['optimizer_updates']==b['train_forwards']==15000 and b['last_step']==20000 and b['diagnostic_forwards']==40
    assert b['live_forwards']<=2500 and sum(b[k] for k in ['train_forwards','live_forwards','diagnostic_forwards'])<=17800
    table=[];model_status=[];live=0
    for checkpoint in [10000,20000]:
        started=completed=stopped=0
        for trial in range(5):
            p=BASE/f'step{checkpoint}'/f'trial_{trial:02d}'
            if not (p/'result.json').exists():continue
            m=load(p/'comparison.json')['current'];r=m['result'];assert load(p/'audit.json')['pass_']
            started+=1;completed+=int(r['valid_observation_end']);stopped+=int(not r['valid_observation_end']);live+=r['new_predictions']
            table.append(dict(checkpoint=checkpoint,trial=trial,label=C['labels'][trial],ticks=r['ticks_applied'],sim_seconds=r['sim_seconds'],termination=r['termination_reason'],valid=r['valid_observation_end'],grasp=r['grasp_success'],lift40mm=r['lift_observed'],storage=r['storage_success'],contacts=m['robot_contact_ticks'],bilateral=m['bilateral_contact_ticks'],max_lift_m=m['max_lift_m'],jaw_below055_time_s=None if m['first_jaw_command_below_055'] is None else m['first_jaw_command_below_055']['time_s'],jaw_command_min_rad=m['jaw_command_range_rad'][0],jaw_measured_min_rad=m['jaw_state_range_rad'][0],strict_first_tick=r['old_strict_first_stop_tick'],jump_warnings=r['jump_warning_events'],max_jump_ratio=max(r['max_jump_ratio']),max_speed_rad_s=max(r['max_measured_speed_rad_s']),max_tracking_error_rad=max(m['pre_tracking_abs_max_rad']),max_penetration_m=r['max_robot_penetration_m'],late_qpos_max_range_rad=None if m['late_40_50_state_range_rad'] is None else max(m['late_40_50_state_range_rad'])))
        model_status.append(dict(checkpoint=checkpoint,planned=5,started=started,completed=completed,safety_stopped=stopped,not_started=5-started))
    assert b['live_forwards']==live
    diagnostics=[load(BASE/f'diagnostic_{step}.json') for step in [5000,10000,20000]]
    summary=dict(run_id=C['run_id'],model_status=model_status,trials=table,diagnostics=diagnostics,budget=b,learning_updates=15000,additional_series=1,comparison='saved5000 n10 versus new10000/20000 n10; fixed training inputs and known training placements, not generalization')
    write(BASE/'summary.json',summary)
    with (BASE/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    from PIL import Image,ImageDraw
    im=Image.new('RGB',(1400,900),'white');draw=ImageDraw.Draw(im)
    panels=[('loss',60,350),('l1_loss',480,770)]
    for key,top,bottom in panels:
        means=[(np.mean([r['step'] for r in updates[i:i+100]]),np.mean([r[key] for r in updates[i:i+100]])) for i in range(0,15000,100)]
        ymax=max(y for _,y in means)*1.05
        draw.text((90,top-25),f'{key}: 100-update means; first=5001, last=20000; max y={ymax:.6f}',fill='black')
        draw.line([(90,top),(90,bottom),(1320,bottom)],fill='black',width=2)
        pts=[(90+(x-5000)/15000*1230,bottom-y/ymax*(bottom-top)) for x,y in means];draw.line(pts,fill='blue',width=2)
        for step in [5000,10000,15000,20000]:
            x=90+(step-5000)/15000*1230;draw.text((x-20,bottom+10),str(step),fill='black')
    draw.text((90,830),'No independent validation set. Fixed-input action MAE: '+', '.join('{}: {:.6f}rad'.format(d['step'],d['inference_action_mae_rad']) for d in diagnostics),fill='black')
    im.save(BASE/'learning_curve.png')
    write(BASE/'loss_summary.json',dict(blocks=[dict(first_step=updates[i]['step'],last_step=updates[min(i+999,14999)]['step'],**{k:float(np.mean([r[k] for r in updates[i:i+1000]])) for k in ['loss','l1_loss','kld_loss','grad_norm','update_wall_seconds']}) for i in range(0,15000,1000)]))
    print(json.dumps(dict(final_audit=True,models=model_status,budget=b)),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='audit':audit(int(sys.argv[2]),int(sys.argv[3]))
    elif sys.argv[1]=='final':final()
    else:raise ValueError(sys.argv[1])
