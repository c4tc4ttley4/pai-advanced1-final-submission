"""One approved warn-only simulation trial. Existing sources remain read only."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from src.step05A_act import ROOT, IMAGE, STATE, CFG, setup, load_bundle, digest, sha, write_json, resources
from src.step05A_rollout import PhysicalObserver
from src.step03_expert import ExpertEnv, load_config, contact_records
from src.step04_dataset import integration_state

BASE=ROOT/'artifacts/step05A_train_compare/train5000_20261031_01'
C=json.loads((BASE/'spec.json').read_text())

def gate(action,previous,bounds,threshold):
    if action.shape!=(6,) or not np.isfinite(action).all():
        return 'nonfinite_or_wrong_shape_action',False
    if np.any(action<bounds[:,0]) or np.any(action>bounds[:,1]):
        return 'action_outside_joint_actuator_range',False
    return None,bool(np.any(np.abs(action-previous)>threshold))

def gate_check(bounds,threshold):
    previous=np.array(json.loads((ROOT/'configs/step02.json').read_text())['initial_action'])
    cases=[]
    for label,action,reason,warn in [
        ('unchanged',previous.copy(),None,False),
        ('jump_only',previous+np.array([0,0,0,.2,0,0]),None,True),
        ('nan',np.full(6,np.nan),'nonfinite_or_wrong_shape_action',False),
        ('inf',np.full(6,np.inf),'nonfinite_or_wrong_shape_action',False),
        ('shape',np.zeros(5),'nonfinite_or_wrong_shape_action',False),
        ('range',bounds[:,1]+1,'action_outside_joint_actuator_range',False)]:
        result=gate(action,previous,bounds,threshold)
        assert result==(reason,warn)
        cases.append(dict(case=label,stop=result[0],jump_warning=result[1]))
    write_json(BASE/'gate_check.json',dict(pass_=True,physics_steps=0,cases=cases,
        difference='Jump-only case is warned and retained; all original action rejection branches preserved.',
        per_step_checks=['finite qpos/qvel/qacc','MuJoCo warning','deep robot contact > original 0.001m']))

def forward_counter(policy,trial):
    from scripts.step05A_train5000 import consume
    original=policy.predict_action_chunk
    def counted(batch):
        assert batch.get('action') is None
        consume('nontrain',f'rollout_{trial}')
        return original(batch)
    policy.predict_action_chunk=counted


def video_check(directory,timestamps):
    video=directory/'diagnostic_10fps.mp4'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(video)],timeout=20))
    stream=probe['streams'][0]
    assert int(stream['nb_read_frames'])==len(timestamps)<=501
    assert stream['r_frame_rate']=='10/1' and (stream['width'],stream['height'])==(640,480)
    check=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(video),'-f','null','-'],
                         capture_output=True,timeout=30)
    assert check.returncode==0 and not check.stderr
    indices=sorted(set([0,len(timestamps)//4,len(timestamps)//2,3*len(timestamps)//4,len(timestamps)-1]))
    sheet=Image.new('RGB',(640*len(indices),510),'white')
    draw=ImageDraw.Draw(sheet)
    for col,i in enumerate(indices):
        raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(video),
            '-vf',f'select=eq(n\\,{i})','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],timeout=20)
        assert len(raw)==640*480*3
        sheet.paste(Image.frombytes('RGB',(640,480),raw),(640*col,30))
        draw.text((640*col+8,8),f'frame {i}, simulation t={timestamps[i]:.3f}s',fill='black')
    sheet.save(directory/'video_review_sheet.png')
    write_json(directory/'video_check.json',dict(pass_=True,probe=probe,frames_decoded=len(timestamps),
         reviewed_frame_indices=indices,frame_sim_times=timestamps,all_frames_decode=True))

def run(trial):
    assert 0<=trial<5
    if trial:
        prior=json.loads((BASE/f'trial_{trial-1:02d}'/'review.json').read_text())
        assert prior['pass_'], 'prior trial not reviewed'
    directory=BASE/f'trial_{trial:02d}';directory.mkdir(exist_ok=False)
    start=time.monotonic();setup(42)
    assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    bundle=ROOT/C['bundle'];policy,pre,post=load_bundle(bundle)
    learned=json.loads((bundle/'training_result.json').read_text())
    assert digest(policy)==learned['after_model_digest']
    assert policy.config.chunk_size==50 and policy.config.n_action_steps==10
    assert policy.config.n_obs_steps==1 and policy.config.temporal_ensemble_coeff is None
    forward_counter(policy,trial)
    threshold=np.array(json.loads((ROOT/C['old_run']/'data_audit.json').read_text())['action_jump_limit_rad'])
    config=load_config()
    bounds=np.array(json.loads((ROOT/'configs/step02.json').read_text())['action_range'])
    if trial==0:
        gate_check(bounds,threshold)
        fixture=load_file(bundle/'reference_observations.safetensors')
        expected=load_file(bundle/'reference_actions.safetensors')['actions_rad']
        actual=[]
        with torch.inference_mode():
            for i in range(5):
                policy.reset();obs={k:v[i:i+1] for k,v in fixture.items()}
                actual.append(torch.stack([post(policy.select_action(pre(obs))).squeeze(0).cpu() for _ in range(10)]))
        actual=torch.stack(actual);error=float((actual-expected).abs().max())
        passed=torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
        write_json(BASE/'restoration.json',dict(pass_=passed,max_error_rad=error,strict=True,local_files_only=True,observations=5,forwards=5))
        assert passed
    policy.reset()
    e=ExpertEnv(config)
    assert e.names==CFG['joint_names'] and e.substeps==10 and e.model.opt.timestep==.002
    e.reset(seed=42,block_position=np.array(C['positions'][trial]))
    attempt=json.loads((ROOT/f'artifacts/step04/dataset_20261029_01/attempt_{trial:02d}.json').read_text())
    assert attempt['seed']==42 and attempt['initial']['block_position']==C['positions'][trial]
    assert np.array_equal(integration_state(e),np.array(attempt['initial']['integration']))
    baseline_initial=json.loads((ROOT/C['baseline']/f'trial_{trial:02d}'/'initial.json').read_text())
    assert np.array_equal(integration_state(e),np.array(baseline_initial['integration']))
    assert json.loads((BASE/'reload_pass.json').read_text())['pass_']
    assert e.data.time==0 and not np.any(e.data.qvel) and not np.any(e.data.qacc_warmstart)
    assert np.array_equal(e.data.body('block').xquat,np.array([1.,0.,0.,0.]))
    assert len(policy._action_queue)==0
    write_json(directory/'initial.json',dict(integration=integration_state(e).tolist(),matches_step04_exactly=True,
        position=C['positions'][trial],quaternion=e.data.body('block').xquat.tolist(),seed=42,queue_length=0))
    old=[]
    if False:  # 600 trace actions/images are not expected for a new model
        with gzip.open(ROOT/C['old_run']/'rollout_01/trace.jsonl.gz','rt') as f:old=[json.loads(l) for l in f]
    previous=e.data.ctrl[e.aids].copy();initial_q=e.data.qpos[e.qids].copy()
    observer=PhysicalObserver(e,config);warnings=[]
    mujoco.set_mju_user_warning(lambda message:warnings.append(str(message)))
    encoder=None;ticks=physical_steps=observations=predictions=0
    frames=[];jump_events=[];strict_first=None;milestones=[];prefix=[]
    max_speed=np.zeros(6);max_movement=max_ctrl_error=0.;inference=[];last=None;gl=None
    max_penetration=0.;max_excess=np.zeros(6);max_ratio=np.zeros(6);jump_axis_counts=np.zeros(6,dtype=int)
    reason='simulation_time_limit';status='bounded_observation_finished';exception=None
    try:
        encoder=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-f','rawvideo',
            '-pixel_format','rgb24','-video_size','640x480','-framerate','10','-i','-','-an','-c:v','libx264',
            '-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p',str(directory/'diagnostic_10fps.mp4')],
            stdin=subprocess.PIPE,stderr=(directory/'encoder_stderr.txt').open('wb'))
        with gzip.open(directory/'trace.jsonl.gz','xt') as trace,torch.inference_mode():
            for tick in range(2500):
                assert time.monotonic()-start<540,'time reserve for finalization'
                before=float(e.data.time);assert abs(before-tick*.02)<1e-8
                raw=e.observe(image=True);observations+=1;rgb=raw[IMAGE]
                assert e.data.time==before
                assert raw[STATE].shape==(6,) and np.isfinite(raw[STATE]).all()
                assert all(np.isfinite(x).all() for x in [e.data.qpos,e.data.qvel,e.data.qacc])
                if gl is None:
                    from OpenGL import GL
                    gl={n:(GL.glGetString(k) or b'').decode() for n,k in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER)]}
                    assert 'NVIDIA' in gl['vendor'] and os.environ.get('MUJOCO_EGL_DEVICE_ID','0')=='0'
                if tick%5==0:encoder.stdin.write(rgb.tobytes());frames.append(before)
                if tick in [0,4,20,99,249,499,999,1499,1999,2499]:Image.fromarray(rgb).save(directory/f'observation_{tick:04d}.png')
                live={IMAGE:torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).unsqueeze(0),STATE:torch.from_numpy(raw[STATE]).unsqueeze(0)}
                assert set(live)=={IMAGE,STATE}
                new=len(policy._action_queue)==0;t=time.monotonic()
                normalized=policy.select_action(pre(live));rad=post(normalized).squeeze(0).numpy().copy()
                ms=(time.monotonic()-t)*1000;inference.append(ms);predictions+=int(new)
                bad,jump=gate(rad,previous,e.bounds,threshold)
                row=dict(tick=tick,time_before=before,image_sim_time=before,image_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                    state_before=raw[STATE].tolist(),qvel_before=e.data.qvel[e.dids].tolist(),action_rad=rad.tolist(),
                    action_normalized=normalized.squeeze(0).cpu().tolist(),previous_ctrl=previous.tolist(),
                    source='new_ACT_prediction' if new else 'ACT_queue',queue_remaining=len(policy._action_queue),
                    policy_input_keys=[IMAGE,STATE],inference_ms=ms)
                if False:  # model5000 has different predictions; initial state is checked above
                    o=old[tick];matched=all(row[k]==o[k] for k in ['image_sha256','state_before','action_rad','previous_ctrl','source','queue_remaining'])
                    prefix.append(dict(tick=tick,exact_match=matched));assert matched,'central old trace mismatch'
                if bad:
                    row.update(applied=False,stop_reason=bad,ctrl=e.data.ctrl[e.aids].tolist());trace.write(json.dumps(row)+'\n')
                    reason=bad;status='safety_stop_before_action';break
                delta=rad-previous;excess=np.maximum(np.abs(delta)-threshold,0);ratio=np.abs(delta)/threshold
                row.update(jump_warning=jump,signed_jump_rad=delta.tolist(),jump_threshold_rad=threshold.tolist(),
                    jump_excess_rad=excess.tolist(),jump_ratio=ratio.tolist())
                max_excess=np.maximum(max_excess,excess);max_ratio=np.maximum(max_ratio,ratio)
                if jump:
                    strict_first=tick if strict_first is None else strict_first
                    jump_axis_counts+=(excess>0)
                    jump_events.append(dict(tick=tick,sim_seconds=before,axes=np.flatnonzero(excess>0).tolist(),max_excess_rad=float(excess.max()),max_ratio=float(ratio.max())))
                e.target[:]=rad;e.data.ctrl[e.aids]=rad
                max_ctrl_error=max(max_ctrl_error,float(np.abs(e.data.ctrl[e.aids]-rad).max()));previous=rad.copy()
                physics_bad=None
                for _ in range(e.substeps):
                    mujoco.mj_step(e.model,e.data);physical_steps+=1
                    max_speed=np.maximum(max_speed,np.abs(e.data.qvel[e.dids]))
                    if not all(np.isfinite(x).all() for x in [e.data.qpos,e.data.qvel,e.data.qacc]):physics_bad='nonfinite_physics_state'
                    elif warnings or any(w.number for w in e.data.warning):physics_bad='mujoco_warning'
                    for contact in e.data.contact:
                        ns=[e.model.geom(contact.geom1).name,e.model.geom(contact.geom2).name]
                        if any(n.startswith('robot_geom') for n in ns):
                            max_penetration=max(max_penetration,-float(contact.dist))
                            if contact.dist < -config['max_contact_penetration_m']:physics_bad='deep_robot_contact:'+str(ns)
                    if physics_bad:break
                mujoco.mj_forward(e.model,e.data);last=observer.observe()
                max_movement=max(max_movement,float(np.abs(e.data.qpos[e.qids]-initial_q).max()))
                ticks+=1
                row.update(applied=True,ctrl=e.data.ctrl[e.aids].tolist(),time_after=float(e.data.time),
                    state_after=e.data.qpos[e.qids].tolist(),qvel_after=e.data.qvel[e.dids].tolist(),
                    diagnostic=last,contacts=contact_records(e),max_measured_speed_rad_s_so_far=max_speed.tolist())
                if physics_bad:status='safety_stop_after_physics';reason=physics_bad;row['stop_reason']=physics_bad
                else:assert abs(e.data.time-before-.02)<1e-10
                trace.write(json.dumps(row)+'\n')
                if ticks in [5,100,2500]:milestones.append(dict(ticks=ticks,sim_seconds=float(e.data.time),mode='warn_only_not_strict'))
                if physics_bad:break
                if observer.grasped and observer.stored:reason='physical_task_success';break
                if tick%100==0:print(json.dumps(dict(trial=trial,tick=tick,sim_seconds=float(e.data.time),jump_events=len(jump_events))),flush=True)
        final=e.observe(image=True)[IMAGE];observations+=1
        assert observations<=2501
        Image.fromarray(final).save(directory/'final_observation.png')
        if len(frames)<501:encoder.stdin.write(final.tobytes());frames.append(float(e.data.time))
    except BaseException as exc:
        exception=exc;status='error_or_interrupted';reason=repr(exc)
    finally:
        if encoder is not None:
            encoder.stdin.close()
            try:encoder_status=encoder.wait(timeout=20)
            except subprocess.TimeoutExpired:encoder.terminate();encoder_status=encoder.wait(timeout=5)
        else:encoder_status=None
        result=dict(trial=trial,label=C['labels'][trial],position=C['positions'][trial],status=status,termination_reason=reason,
            valid_observation_end=bool(status=='bounded_observation_finished' and reason in ['simulation_time_limit','physical_task_success']),
            ticks_applied=ticks,physical_steps=physical_steps,sim_seconds=float(e.data.time),observations=observations,
            new_predictions=predictions,queue_actions=len(inference)-predictions,official_queue=True,
            jump_warning_events=len(jump_events),jump_warning_axis_counts=jump_axis_counts.tolist(),old_strict_first_stop_tick=strict_first,
            max_jump_excess_rad=max_excess.tolist(),max_jump_ratio=max_ratio.tolist(),max_measured_speed_rad_s=max_speed.tolist(),
            max_robot_penetration_m=max_penetration,max_action_ctrl_difference_rad=max_ctrl_error,max_measured_movement_rad=max_movement,
            grasp_success=bool(observer.grasped),lift_observed=bool(observer.lifted),storage_success=bool(observer.stored),
            max_lift_m=observer.max_lift,max_valid_grasp_hold_seconds=observer.max_hold,max_storage_stable_seconds=observer.max_stable,
            final_diagnostic=last,milestones=milestones,central_old_prefix=prefix,model_digest=learned['after_model_digest'],
            jump_threshold_rad=threshold.tolist(),warnings=warnings,warning_counts=[int(w.number) for w in e.data.warning],
            video_frames=len(frames),video_fps=10,video_encoder_status=encoder_status,video_sim_times=frames,
            max_inference_ms=max(inference,default=0),mean_inference_ms=float(np.mean(inference)) if inference else None,
            opengl=gl,resources=resources(),wall_seconds=time.monotonic()-start,
            action_applications=ticks,actions_clipped=0,compensation=False,additional_training_updates=0,
            mode='warn_only_simulation_known_training_placements',strict_connection_pass=False)
        write_json(directory/'result.json',result);write_json(directory/'jump_events.json',jump_events)
        e.close();mujoco.set_mju_user_warning(None)
    assert encoder_status==0,'video encoder failed'
    video_check(directory,frames)
    print(json.dumps(result),flush=True)
    if exception is not None:raise exception
    if not result['valid_observation_end']:sys.exit(3)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('trial',type=int);run(parser.parse_args().trial)
