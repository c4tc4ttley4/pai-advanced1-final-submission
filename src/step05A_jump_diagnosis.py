"""Read-only fixed ACT diagnostics; no training entry point or optimizer."""
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from src.step05A_act import ROOT, IMAGE, STATE, sha, write_json, setup, load_bundle, digest, resources
from src.step03_expert import ExpertEnv, load_config
from src.step04_dataset import integration_state

C = json.loads((ROOT / 'configs/step05A_jump_diagnosis.json').read_text())
OUT = ROOT / 'artifacts/step05A_jump_diagnosis' / C['run_id']
OLD = ROOT / 'artifacts/step05A' / C['source_run']
BUNDLE = ROOT / C['bundle']
NAMES = json.loads((ROOT / 'configs/step05A_act.json').read_text())['joint_names']


def read_json(path):
    return json.loads(Path(path).read_text())


def read_trace(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def limits():
    return np.array(read_json(OLD / 'data_audit.json')['action_jump_limit_rad'])


def gate(rad, previous, bounds, threshold):
    # Exact same comparison/order as the unchanged production rollout lines155-160.
    if rad.shape != (6,) or not np.isfinite(rad).all():
        return 'nonfinite_or_wrong_shape_action'
    if np.any(rad < bounds[:,0]) or np.any(rad > bounds[:,1]):
        return 'action_outside_joint_actuator_range'
    if np.any(np.abs(rad-previous) > threshold):
        return 'action_jump_exceeds_predeclared_limit'
    return None


def audit():
    assert os.environ['MUJOCO_GL'] == 'disable' and not torch.cuda.is_available()
    assert not (OUT / 'gate_audit.json').exists(), 'no overwrite'
    inputs = {}
    for line in (OUT / 'audit/protected.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert sha(ROOT/name) == expected, name
        inputs[name] = expected
    assert sha(BUNDLE/'model.safetensors') == C['model_sha256']
    assert sha(BUNDLE/'bundle.sha256.json') == C['bundle_manifest_sha256']
    assert read_json(BUNDLE/'training_result.json')['cumulative_updates'] == 600
    rows = read_trace(OLD/'rollout_01/trace.jsonl.gz')
    result = read_json(OLD/'rollout_01/result.json')
    threshold, bounds = limits(), np.array(read_json(ROOT/'configs/step02.json')['action_range'])
    table, decisions = [], []
    previous = np.array(read_json(ROOT/'configs/step02.json')['initial_action'], dtype=np.float64)
    for row in rows:
        tick = row['tick']
        rad = np.array(row['action_rad'], dtype=np.float32)
        assert previous.tolist() == row['previous_ctrl']
        actual_delta = rad-previous
        signed64 = rad.astype(np.float64)-previous.astype(np.float64)
        reason = gate(rad, previous, bounds, threshold)
        assert (reason is None) == row['applied']
        assert row['source'] == ('new_ACT_prediction' if tick % 10 == 0 else 'ACT_queue')
        assert row['queue_remaining'] == 9-tick % 10
        if row['applied']:
            assert row['ctrl'] == rad.tolist()
            assert abs(row['time_after']-row['time_before']-.02) < 1e-10
        else:
            assert reason == row['stop_reason']
            assert row['ctrl'] == previous.tolist()
            assert 'time_after' not in row
        decisions.append(dict(tick=tick, applied=row['applied'], reason=reason,
                              delta_dtype=str(actual_delta.dtype), signed_delta=actual_delta.tolist(),
                              signed_delta_float64=signed64.tolist(), over=(np.abs(actual_delta)>threshold).tolist()))
        if tick in C['capture_ticks']:
            for j, name in enumerate(NAMES):
                table.append(dict(tick=tick, sim_time=row['time_before'], joint=name,
                    qpos=row['state_before'][j], previous_applied_ctrl=float(previous[j]),
                    proposed_action_rad=float(rad[j]), reference_used_by_gate=float(previous[j]),
                    signed_delta=float(actual_delta[j]), signed_delta_float64=float(signed64[j]),
                    abs_delta=float(np.abs(actual_delta[j])), threshold=float(threshold[j]),
                    over_threshold=bool(np.abs(actual_delta[j])>threshold[j]),
                    applied_or_rejected='applied' if row['applied'] else 'rejected',
                    prediction_origin_tick=(tick//10)*10, chunk_index=tick % 10,
                    subtraction_dtype=str(actual_delta.dtype)))
        if row['applied']:
            previous = rad.copy()
    assert rows[-1]['tick'] == 20 and not rows[-1]['applied']
    assert sum(r['applied'] for r in rows) == result['ticks_applied'] == 20
    assert result['physical_steps'] == 200 and abs(result['sim_seconds']-.4)<1e-10
    source = ROOT/'src/step05A_rollout.py'
    library = {}
    for name, obj in [('ACT.select_action', ACTPolicy.select_action),
                      ('ACT.predict_action_chunk', ACTPolicy.predict_action_chunk),
                      ('ACT.reset', ACTPolicy.reset),
                      ('Normalizer', NormalizerProcessorStep),
                      ('Unnormalizer', UnnormalizerProcessorStep),
                      ('normalization_mixin', NormalizerProcessorStep.__mro__[1])]:
        filename = Path(inspect.getfile(obj))
        lines, start = inspect.getsourcelines(obj)
        library[name] = dict(file=str(filename), first_line=start, last_line=start+len(lines)-1,
                             file_sha256=sha(filename))
        (OUT/'audit'/f'{name.replace(".","_")}.txt').write_text(''.join(lines))
    missing = dict(full_chunks='not in old trace; index10..49 discarded by executable queue',
                   images='old PNGs only tick0,4 and final; old video10fps is lossy, cannot reconstruct 50Hz input',
                   queue='queue_remaining present for every tick; full chunk not present',
                   normalized_inputs='not in old trace; normalized action present',
                   required_capture='single same-conditions live run: chunks at0/10/20, raw PNG and state at0/10/19/20')
    output = dict(pass_=True, source=dict(file='src/step05A_rollout.py', function='rollout',
                    initialization_lines=[89,95], observation_lines=[115,153],
                    gate_lines=[154,165], application_lines=[166,185], sha256=sha(source)),
                  comparison='np.any(np.abs(rad-previous)>jump_limit)', initial_reference='initial MuJoCo ctrl float64',
                  later_reference='last applied rad.copy float32, equal to ctrl after float64 assignment',
                  coordinate='postprocessor rad absolute position; no angular wrap',
                  order=['shape/finite','bounds','jump strict >','stop before assignment','ctrl assignment','10 physical steps'],
                  thresholds=threshold.tolist(), decisions=decisions, missing_evidence=missing,
                  final_ctrl_unchanged_on_rejection=True, no_extra_physics_from_trace=True,
                  library=library, versions={n:importlib.metadata.version(n) for n in ['torch','lerobot','torchcodec','mujoco']})
    write_json(OUT/'input_manifest.json', dict(files=inputs, model_sha256=C['model_sha256'],
                  bundle_manifest_sha256=C['bundle_manifest_sha256'], source_run=C['source_run']))
    write_csv(OUT/'gate_audit.csv', table)
    write_json(OUT/'gate_audit.json', output)
    if not (OUT/'budget.json').exists():
        write_json(OUT/'budget.json', dict(chunk_forwards=0, live_trials=0, expert_replays=0, events=[]))
    print(json.dumps(dict(audit_pass=True, tick20=decisions[-1], missing=missing)), flush=True)


def budget_event(kind):
    budget = read_json(OUT/'budget.json')
    key = {'forward':'chunk_forwards', 'live':'live_trials', 'expert_replay':'expert_replays'}[kind]
    maximum = {'forward':C['max_chunk_forwards'], 'live':C['max_live_trials'], 'expert_replay':C['max_expert_replays']}[kind]
    assert budget[key] < maximum, 'diagnostic budget exhausted'
    budget[key] += 1
    budget['events'].append(dict(kind=kind, cumulative=budget[key]))
    write_json(OUT/'budget.json', budget)


class ForwardRecorder:
    """One call through to the unchanged method, observing returned chunks only."""
    def __init__(self, policy, post):
        self.policy, self.post = policy, post
        self.original = policy.predict_action_chunk
        self.phase, self.tick = 'restore', None
        self.chunks, self.records = {}, []
        policy.predict_action_chunk = self.call

    def call(self, batch):
        assert set(batch) == {IMAGE, STATE, 'action', 'next.reward', 'next.done', 'next.truncated', 'info'}
        assert batch['action'] is None and batch['next.reward'] == 0.
        assert batch['next.done'] is False and batch['next.truncated'] is False and batch['info'] == {}
        budget_event('forward')
        output = self.original(batch)  # exactly one original forward
        assert output.shape == (1,50,6)
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        queue_before = len(self.policy._action_queue)
        normalized = output.detach().cpu().clone()
        # Postprocess the same (1,6) shape as real select_action; no extra forward.
        rad = torch.stack([self.post(output[:,i]).squeeze(0).cpu() for i in range(50)])
        assert torch.equal(normalized, output.detach().cpu()), 'post logging mutated the prediction'
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert len(self.policy._action_queue) == queue_before
        self.records.append(dict(phase=self.phase, tick=self.tick, queue_during_predict=queue_before,
                                 logging_rng_unchanged=True, returned_original_tensor=True))
        if self.phase == 'live':
            self.chunks[self.tick] = dict(normalized=normalized.squeeze(0).contiguous(), rad=rad.contiguous())
        return output


def input_summary(batch):
    return {key: (dict(shape=list(value.shape), dtype=str(value.dtype),
                     minimum=float(value.min()), maximum=float(value.max()),
                     mean=float(value.mean()), std=float(value.std(correction=0)))
                  if isinstance(value,torch.Tensor) else dict(default_metadata=value))
            for key,value in batch.items()}


def capture():
    assert not (OUT/'capture').exists(), 'no overwrite'
    assert read_json(OUT/'gate_audit.json')['pass_']
    target = OUT/'capture'; target.mkdir()
    start = time.monotonic()
    policy, pre, post = load_bundle(BUNDLE)
    assert policy.config.chunk_size == 50 and policy.config.n_action_steps == 10
    assert policy.config.n_obs_steps == 1 and policy.config.temporal_ensemble_coeff is None
    assert digest(policy) == read_json(BUNDLE/'training_result.json')['after_model_digest']
    recorder = ForwardRecorder(policy, post)
    fixture = load_file(BUNDLE/'reference_observations.safetensors')
    expected = load_file(BUNDLE/'reference_actions.safetensors')['actions_rad']
    actual = []
    with torch.inference_mode():
        for i in range(5):
            policy.reset()
            obs = {key:value[i:i+1] for key,value in fixture.items()}
            actual.append(torch.stack([post(policy.select_action(pre(obs))).squeeze(0).cpu() for _ in range(10)]))
    actual = torch.stack(actual)
    passed = torch.allclose(actual,expected,atol=C['reload_atol_rad'],rtol=C['reload_rtol'])
    write_json(OUT/'restoration_check.json',dict(pass_=passed, strict=True, local_files_only=True,
               max_difference_rad=float((actual-expected).abs().max()),
               atol=C['reload_atol_rad'],rtol=C['reload_rtol'],observations=5,forwards=5,
               model_digest=digest(policy), bundle_manifest_sha256=sha(BUNDLE/'bundle.sha256.json')))
    assert passed
    del fixture, expected, actual
    policy.reset()
    budget_event('live')
    recorder.phase = 'live'
    from src.step05A_rollout import PhysicalObserver
    e = ExpertEnv(load_config())
    block = np.array(e.config['block']['position'])
    e.reset(seed=42,block_position=block)
    observer = PhysicalObserver(e,load_config())
    initial = integration_state(e)
    attempt = read_json(ROOT/'artifacts/step04/dataset_20261029_01/attempt_00.json')
    initial_matches = np.array_equal(initial,np.array(attempt['initial']['integration']))
    assert initial_matches
    write_json(target/'initial.json',dict(seed=42,block_position=block.tolist(),
               integration=initial.tolist(),matches_expert_attempt00_integration=initial_matches))
    previous = e.data.ctrl[e.aids].copy()
    threshold, warnings = limits(), []
    mujoco = __import__('mujoco')
    mujoco.set_mju_user_warning(lambda message:warnings.append(str(message)))
    old_rows = read_trace(OLD/'rollout_01/trace.jsonl.gz')
    observations, summaries, rows, comparisons = {}, {}, [], []
    status = '100tick_limit'; applied=physical_steps=0
    gl = None
    try:
        with gzip.open(target/'trace.jsonl.gz','xt') as stream, torch.inference_mode():
            for tick in range(C['max_live_ticks']):
                before = float(e.data.time)
                assert abs(before-tick*.02)<1e-9
                raw = e.observe(image=True)
                assert e.data.time == before
                if gl is None:
                    from OpenGL import GL
                    gl = {n:(GL.glGetString(k) or b'').decode() for n,k in
                          [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER)]}
                    assert 'NVIDIA' in gl['vendor']
                live = {IMAGE:torch.from_numpy(raw[IMAGE].copy()).permute(2,0,1).float().div(255).unsqueeze(0),
                        STATE:torch.from_numpy(raw[STATE]).unsqueeze(0)}
                if tick in C['capture_ticks']:
                    Image.fromarray(raw[IMAGE]).save(target/f'image_{tick:02d}.png')
                    observations[tick] = {key:value.clone().contiguous() for key,value in live.items()}
                recorder.tick = tick
                queue_before = len(policy._action_queue)
                processed = pre(live)
                if tick in C['capture_ticks']:
                    summaries[tick] = dict(raw=input_summary(live),pre=input_summary(processed))
                normalized = policy.select_action(processed)
                rad = post(normalized).squeeze(0).numpy().copy()
                origin = (tick//10)*10
                # Exact correspondence between recorded full chunk and consumed action.
                assert np.array_equal(rad,recorder.chunks[origin]['rad'][tick%10].numpy())
                assert np.array_equal(normalized.squeeze(0).cpu().numpy(),
                                      recorder.chunks[origin]['normalized'][tick%10].numpy())
                reason = gate(rad,previous,e.bounds,threshold)
                row = dict(tick=tick,time_before=before,image_sim_time=before,
                      image_sha256=hashlib.sha256(raw[IMAGE].tobytes()).hexdigest(),
                      state_before=raw[STATE].tolist(),previous_ctrl=previous.tolist(),
                      action_rad=rad.tolist(),action_normalized=normalized.squeeze(0).cpu().tolist(),
                      queue_before=queue_before,queue_remaining=len(policy._action_queue),
                      prediction_origin_tick=origin,chunk_index=tick%10,
                      source='new_ACT_prediction' if queue_before==0 else 'ACT_queue',
                      applied=reason is None,stop_reason=reason)
                if tick<len(old_rows):
                    old = old_rows[tick]
                    comp=dict(tick=tick,image_hash_equal=row['image_sha256']==old['image_sha256'],
                              qpos_max_error=float(np.max(np.abs(np.array(row['state_before'])-old['state_before']))),
                              action_max_error=float(np.max(np.abs(np.array(row['action_rad'])-old['action_rad']))),
                              previous_ctrl_max_error=float(np.max(np.abs(np.array(row['previous_ctrl'])-old['previous_ctrl']))),
                              applied_equal=row['applied']==old['applied'], source_equal=row['source']==old['source'])
                    comparisons.append(comp)
                if reason:
                    status = reason
                    row['ctrl'] = e.data.ctrl[e.aids].tolist()
                    stream.write(json.dumps(row)+'\n'); rows.append(row)
                    break
                e.target[:] = rad
                e.data.ctrl[e.aids] = rad
                previous = rad.copy()
                for _ in range(e.substeps):
                    mujoco.mj_step(e.model,e.data); physical_steps+=1
                    assert all(np.isfinite(v).all() for v in [e.data.qpos,e.data.qvel,e.data.qacc])
                    assert not warnings and not any(w.number for w in e.data.warning)
                    for contact in e.data.contact:
                        names=[e.model.geom(contact.geom1).name,e.model.geom(contact.geom2).name]
                        if any(name.startswith('robot_geom') for name in names):
                            assert contact.dist >= -load_config()['max_contact_penetration_m']
                mujoco.mj_forward(e.model,e.data)
                observer.observe()  # Same read-only physical diagnostics as original path.
                applied+=1
                row.update(ctrl=e.data.ctrl[e.aids].tolist(),time_after=float(e.data.time),
                           state_after=e.data.qpos[e.qids].tolist())
                assert abs(e.data.time-before-.02)<1e-10
                stream.write(json.dumps(row)+'\n'); rows.append(row)
                if applied==5:
                    print(json.dumps(dict(milestone_ticks=5,sim_seconds=float(e.data.time))),flush=True)
    finally:
        for tick, values in observations.items():
            save_file(values,target/f'observation_{tick:02d}.safetensors')
        for tick, values in recorder.chunks.items():
            save_file(values,target/f'chunk_{tick:02d}.safetensors')
        write_json(target/'input_summaries.json',summaries)
        match = (len(comparisons)==21 and all(x['image_hash_equal'] and x['qpos_max_error']==0 and
                 x['action_max_error']==0 and x['previous_ctrl_max_error']==0 and
                 x['applied_equal'] and x['source_equal'] for x in comparisons))
        write_json(target/'result.json',dict(status=status,applied_ticks=applied,physical_steps=physical_steps,
              sim_seconds=float(e.data.time),records=len(rows),new_chunk_ticks=list(recorder.chunks),
              original_trace_exact_match=match,original_comparisons=comparisons,
              recorder_calls=recorder.records,logging_did_not_change_rng_or_queue=True,
              full_chunk_source='actual original predict_action_chunk return observed at execution',
              initial_integration_equal=initial_matches,opengl=gl,warnings=warnings,
              resources=resources(),wall_seconds=time.monotonic()-start,training_updates=0))
        e.close(); mujoco.set_mju_user_warning(None)
    print(json.dumps(dict(capture_status=status,applied_ticks=applied,physical_steps=physical_steps,
                         matches_original_exactly=match,forwards=read_json(OUT/'budget.json')['chunk_forwards'])),flush=True)


def image_metrics(a, b):
    a, b = a.squeeze(0).cpu().double().numpy(), b.squeeze(0).cpu().double().numpy()
    return dict(mae_255=np.abs(a-b).mean(axis=(1,2)).__mul__(255).tolist(),
                max_abs_255=np.abs(a-b).max(axis=(1,2)).__mul__(255).tolist(),
                first_mean=a.mean(axis=(1,2)).tolist(),first_std=a.std(axis=(1,2)).tolist(),
                second_mean=b.mean(axis=(1,2)).tolist(),second_std=b.std(axis=(1,2)).tolist())


def decompose(old_chunk, new_chunk, previous, tick, label):
    old = np.asarray(old_chunk,dtype=np.float64)
    new = np.asarray(new_chunk,dtype=np.float64)
    previous = np.asarray(previous,dtype=np.float64)
    G = new[0]-previous
    W = old[10]-old[9]
    R = new[0]-old[10]
    E = old[9]-previous
    residual = G-W-R-E
    assert np.max(np.abs(residual)) < 1e-12
    rows = [dict(path=label,tick=tick,joint=name,G=float(G[j]),W=float(W[j]),R=float(R[j]),
                 E=float(E[j]),identity_residual=float(residual[j]),threshold=float(limits()[j]),
                 abs_G_exceeds_threshold=bool(abs(G[j])>limits()[j])) for j,name in enumerate(NAMES)]
    overlap = new[:40]-old[10:50]
    scopes = {}
    for key, values in [('first',overlap[:1]),('first10',overlap[:10]),('all40',overlap)]:
        scopes[key] = dict(mean_signed=values.mean(0).tolist(),mae=np.abs(values).mean(0).tolist(),
                           max_abs=np.abs(values).max(0).tolist())
    detail = dict(path=label,tick=tick,overlap_difference_rad=overlap.tolist(),overlap=scopes,
                  old_chunk_max_adjacent_delta_rad=np.abs(np.diff(old,axis=0)).max(0).tolist(),
                  new_chunk_max_adjacent_delta_rad=np.abs(np.diff(new,axis=0)).max(0).tolist(),
                  old_first10_max_adjacent_delta_rad=np.abs(np.diff(old[:10],axis=0)).max(0).tolist(),
                  new_first10_max_adjacent_delta_rad=np.abs(np.diff(new[:10],axis=0)).max(0).tolist())
    return rows,detail


def offline():
    assert not (OUT/'offline').exists(), 'no overwrite'
    captured = read_json(OUT/'capture/result.json')
    assert captured['original_trace_exact_match'], 'resolve reproduction mismatch before comparisons'
    target=OUT/'offline'; target.mkdir()
    start=time.monotonic()
    policy,pre,post=load_bundle(BUNDLE)
    assert digest(policy)==read_json(BUNDLE/'training_result.json')['after_model_digest']
    recorder=ForwardRecorder(policy,post); recorder.phase='offline'
    stats=read_json(BUNDLE/'statistics.json')['stats']
    pre_checks=[]

    def predict(raw, name):
        assert set(raw)=={IMAGE,STATE}
        assert raw[IMAGE].shape==(1,3,480,640) and raw[STATE].shape==(1,6)
        assert raw[IMAGE].dtype==raw[STATE].dtype==torch.float32
        assert 0<=raw[IMAGE].min()<=raw[IMAGE].max()<=1
        policy.reset()
        processed=pre(raw)
        errors={}
        for key in [IMAGE,STATE]:
            mu=torch.tensor(stats[key]['mean'],dtype=torch.float32,device='cuda')
            sd=torch.tensor(stats[key]['std'],dtype=torch.float32,device='cuda')
            expected=(raw[key].to('cuda')-mu)/(sd+1e-8)
            errors[key]=float((expected-processed[key]).abs().max())
        assert max(errors.values())<1e-6
        normalized=policy.predict_action_chunk(processed)
        rad=torch.stack([post(normalized[:,i]).squeeze(0).cpu() for i in range(50)])
        mu=torch.tensor(stats['action']['mean'],dtype=torch.float32)
        sd=torch.tensor(stats['action']['std'],dtype=torch.float32)
        post_error=float((rad-(normalized.squeeze(0).cpu()*sd+mu)).abs().max())
        assert post_error<1e-6
        pre_checks.append(dict(name=name,manual_pre_max_error=errors,manual_post_max_error=post_error,
                               processor_default_action_is_none=processed['action'] is None))
        save_file({'rad':rad.contiguous(),'normalized':normalized.squeeze(0).cpu().contiguous()},target/f'{name}.safetensors')
        return rad.numpy()

    live_obs={t:load_file(OUT/f'capture/observation_{t:02d}.safetensors') for t in C['capture_ticks']}
    chunks={t:load_file(OUT/f'capture/chunk_{t:02d}.safetensors')['rad'].numpy() for t in [0,10,20]}
    original=read_trace(OLD/'rollout_01/trace.jsonl.gz')
    recomputations=[]
    with torch.inference_mode():
        for t in [0,10,20]:
            calculated=predict(live_obs[t],f'live_recomputed_{t:02d}')
            err=float(np.max(np.abs(calculated-chunks[t])))
            assert err<=C['reload_atol_rad']
            recomputations.append(dict(tick=t,max_full_chunk_error_rad=err))
        decomposition,overlaps=[],[]
        for t in [10,20]:
            table,detail=decompose(chunks[t-10],chunks[t],original[t-1]['ctrl'],t,'live')
            decomposition+=table; overlaps.append(detail)
        ds=LeRobotDataset(repo_id='local/step04-cleanup',root=ROOT/C['dataset'],video_backend='torchcodec')
        assert len(ds)==9531 and ds.num_episodes==5 and ds.fps==50
        assert ds.features[STATE]['names']==ds.features['action']['names']==NAMES
        attempt=read_json(ROOT/'artifacts/step04/dataset_20261029_01/attempt_00.json')
        assert attempt['seed']==42 and attempt['offset_m']==[0.,0.]
        assert attempt['initial']['block_position']==[-.13,.38,.68]
        episode=ds.meta.episodes[0]
        assert int(episode['dataset_from_index'])==0
        expert_obs,expert_items,comparisons={},{},[]
        for t in C['capture_ticks']:
            item=ds[t]
            assert int(item['episode_index'])==0 and int(item['frame_index'])==t
            assert abs(float(item['timestamp'])-t*.02)<1e-7
            obs={key:item[key].unsqueeze(0) for key in [IMAGE,STATE]}
            expert_obs[t],expert_items[t]=obs,item
            save_file(obs,target/f'expert_observation_{t:02d}.safetensors')
            qdiff=(live_obs[t][STATE]-obs[STATE]).squeeze(0).numpy()
            action_diff=np.array(original[t]['action_rad'])-item['action'].numpy()
            previous_expert=np.array(attempt['initial']['ctrl'])[:6] if t==0 else ds.hf_dataset[t-1]['action'].numpy()
            comparisons.append(dict(tick=t,sim_time=t*.02,expert_frame_index=int(item['frame_index']),
                expert_timestamp=float(item['timestamp']),qpos_live_minus_expert=qdiff.tolist(),
                proposed_live_minus_expert_sent=action_diff.tolist(),live_proposal_applied=original[t]['applied'],
                previous_applied_ctrl_live_minus_expert=(np.array(original[t]['previous_ctrl'])-previous_expert).tolist(),
                image=image_metrics(live_obs[t][IMAGE],obs[IMAGE]),
                image_difference_interpretation='different physical states plus raw-vs-video; not a compression-only comparison'))
        experts,errors={},[]
        for t in [0,10,20]:
            pred=predict(expert_obs[t],f'expert_{t:02d}')
            experts[t]=pred
            sent=torch.stack([ds.hf_dataset[t+j]['action'] for j in range(50)]).numpy()
            err=pred-sent
            errors.append(dict(tick=t,first_action_error_rad=err[0].tolist(),
                first10_mae_by_joint_rad=np.abs(err[:10]).mean(0).tolist(),
                all50_mae_by_joint_rad=np.abs(err).mean(0).tolist(),
                note='teacher action used only for output diagnostics, not model input'))
        expert_decomposition=[]
        for t in [10,20]:
            table,detail=decompose(experts[t-10],experts[t],experts[t-10][9],t,'expert_observations_counterfactual_stitch')
            expert_decomposition+=table; overlaps.append(detail)
        crossed=[]
        for t in C['offline_artificial_input_ticks']:
            LE=predict({IMAGE:live_obs[t][IMAGE],STATE:expert_obs[t][STATE]},f'live_image_expert_state_{t:02d}')
            EL=predict({IMAGE:expert_obs[t][IMAGE],STATE:live_obs[t][STATE]},f'expert_image_live_state_{t:02d}')
            LL,EE=chunks[t],experts[t]
            crossed.append(dict(tick=t,
                first_action_state_swap_on_live_image=(LE[0]-LL[0]).tolist(),
                first_action_image_swap_on_live_state=(EL[0]-LL[0]).tolist(),
                first_action_expert_minus_live=(EE[0]-LL[0]).tolist(),
                first_action_interaction=(EE[0]-LE[0]-EL[0]+LL[0]).tolist(),
                state_swap_all50_mae=np.abs(LE-LL).mean(0).tolist(),
                image_swap_all50_mae=np.abs(EL-LL).mean(0).tolist(),
                interpretation='artificial crossed inputs, sensitivity only; no control applied and not isolated causal effects'))
        # Same original PNG and decoded frame0: exact provenance in STEP4 anchors.
        anchor_frame=attempt['phase_first_frame']['settle_open']
        assert anchor_frame==0
        anchor=np.asarray(Image.open(ROOT/'artifacts/step04/dataset_20261029_01/anchor_00_settle_open.png').convert('RGB')).copy()
        anchor_tensor=torch.from_numpy(anchor).permute(2,0,1).float().div(255).unsqueeze(0)
        anchor_pred=predict({IMAGE:anchor_tensor,STATE:expert_obs[0][STATE]},'original_raw_anchor_00')
        compression=[dict(kind='original_saved_raw_PNG_vs_original_decoded_frame',frame=0,
            same_frame_provenance='attempt00 phase_first_frame settle_open=0; stored raw anchor before action',
            raw_anchor_equals_live_initial=bool(torch.equal(anchor_tensor,live_obs[0][IMAGE])),
            state_equal=bool(torch.equal(expert_obs[0][STATE],live_obs[0][STATE])),
            pixels=image_metrics(anchor_tensor,expert_obs[0][IMAGE]),
            first_action_decoded_minus_raw=(experts[0][0]-anchor_pred[0]).tolist(),
            all50_output_mae_by_joint=np.abs(experts[0]-anchor_pred).mean(0).tolist())]
        # Four sparse snapshots form a diagnostic clip, not the original GOP context.
        video=target/'same_raw_sparse_4frame_roundtrip.mp4'
        ffmpeg=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','640x480','-framerate','50','-i','-','-an','-c:v','libx264','-threads','2',
                '-preset','fast','-g','2','-crf','23','-pix_fmt','yuv420p',str(video)]
        encoder=subprocess.Popen(ffmpeg,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        for t in C['capture_ticks']:
            rgb=np.asarray(Image.open(OUT/f'capture/image_{t:02d}.png').convert('RGB'))
            encoder.stdin.write(rgb.tobytes())
        encoder.stdin.close()
        err=encoder.stderr.read().decode(); assert encoder.wait(timeout=30)==0,err
        import av
        with av.open(str(video)) as container:
            container.streams.video[0].thread_count=2
            for i,frame in enumerate(container.decode(video=0)):
                t=C['capture_ticks'][i]
                rgb=frame.to_ndarray(format='rgb24')
                im=torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).unsqueeze(0)
                out=predict({IMAGE:im,STATE:live_obs[t][STATE]},f'roundtrip_{t:02d}')
                # tick19 has no full recorded chunk; compute its raw-input counterpart separately.
                raw_pred=chunks[t] if t in chunks else predict(live_obs[t],f'raw_reference_{t:02d}')
                compression.append(dict(kind='same_raw_diagnostic_short_codec_roundtrip',tick=t,
                    same_state=True,pixels=image_metrics(live_obs[t][IMAGE],im),
                    first_action_decoded_minus_raw=(out[0]-raw_pred[0]).tolist(),
                    all50_output_mae_by_joint=np.abs(out-raw_pred).mean(0).tolist(),
                    limitation='sparse snapshots at50fps, original GOP/neighbouring frames not reproduced; sensitivity only'))
        assert len(compression)<=C['maximum_image_pairs']
    write_csv(OUT/'boundary_decomposition.csv',decomposition+expert_decomposition)
    write_json(OUT/'boundary_decomposition.json',dict(live=decomposition,expert=expert_decomposition,
               computation='float64 differences of saved float32 rad values; G=W+R+E signed, no absolute-value contribution percentages'))
    write_json(OUT/'overlapping_predictions.json',dict(comparisons=overlaps,recomputed_live_chunks=recomputations))
    write_json(OUT/'expert_live_comparison.json',dict(central_episode=0,verified_seed=42,
         initial_integration_equal=captured['initial_integration_equal'],same_time_comparisons=comparisons,
         expert_prediction_errors=errors,crossed_input_sensitivity=crossed,
         expert_counterfactual_note='No teacher action applied in this investigation; stitched prediction stream is hypothetical'))
    write_json(OUT/'image_comparison.json',dict(pairs=compression,expert_action_replays=0,
         original_same_frame_pairs=1,synthetic_short_codec_pairs=4,
         reason_no_expert_replay='same-frame original PNG0 exists and initial integration/live RGB match; raw live snapshots support isolated short-codec sensitivity'))
    write_json(OUT/'preprocessing_check.json',dict(checks=pre_checks,
         image='RGB CHW480x640 float32[0,1]; once /255 from raw uint8; dataset already[0,1]',
         pre_formula='(x-mean)/(std+1e-8)',post_formula='normalized_action*std+mean',
         default_metadata='official processor adds action=None, reward0, doneFalse, truncatedFalse, info{}; no privileged or teacher feature',
         dedicated_statistics_unchanged=True,statistics_sha256=sha(BUNDLE/'statistics.json')))
    write_json(target/'result.json',dict(pass_=True,wall_seconds=time.monotonic()-start,
         resources=resources(),forward_count_cumulative=read_json(OUT/'budget.json')['chunk_forwards'],
         training_updates=0,model_digest_after=digest(policy)))
    print(json.dumps(dict(offline_pass=True,forwards=read_json(OUT/'budget.json')['chunk_forwards'],
               wrist_pitch_tick20=[x for x in decomposition if x['tick']==20 and x['joint']=='Wrist_Pitch_R'][0])),flush=True)


def draw_boundary_plot(chunks, data, axes):
    """Dependency-free engineering trace plot, identical PNG and SVG geometry."""
    width,height=1220,390*len(axes)
    im=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(im)
    font=ImageFont.load_default(size=16)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>']
    def line(points,color='#444444',w=1):
        draw.line(points,fill=color,width=w)
        coords=' '.join(f'{x:.3f},{y:.3f}' for x,y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{w}"/>')
    def text(x,y,value,color='#222222'):
        draw.text((x,y),value,font=font,fill=color)
        svg.append(f'<text x="{x}" y="{y+16}" font-family="sans-serif" font-size="16" fill="{color}">{value}</text>')
    for row,j in enumerate(axes):
        y0=row*390
        text(65,y0+8,NAMES[j]+' | shared target time, fixed 600-update ACT')
        text(80,y0+35,'chunk at tick10', '#2878aa'); text(310,y0+35,'chunk at tick20','#ce742d')
        left,right,top,bottom=80,600,y0+75,y0+305
        values=np.concatenate([chunks[10][9:,j],chunks[20][:40,j]])
        lo=np.floor((values.min()-.02)*100)/100; hi=np.ceil((values.max()+.02)*100)/100
        X=lambda t:left+(t-.38)/(1.20-.38)*(right-left)
        Y=lambda v:bottom-(v-lo)/(hi-lo)*(bottom-top)
        for value in np.linspace(lo,hi,5):
            yy=Y(value); line([(left,yy),(right,yy)],'#dddddd'); text(4,yy-9,f'{value:.3f}')
        for t in [.4,.6,.8,1.,1.2]:
            xx=X(t); line([(xx,top),(xx,bottom)],'#eeeeee'); text(xx-10,bottom+6,f'{t:.1f}')
        line([(left,top),(left,bottom),(right,bottom)])
        line([(X(t*.02),Y(v)) for t,v in zip(range(19,60),chunks[10][9:,j])],'#2878aa',3)
        line([(X(t*.02),Y(v)) for t,v in zip(range(20,60),chunks[20][:40,j])],'#ce742d',3)
        line([(X(.4),top),(X(.4),bottom)],'#888888')
        text(80,y0+345,'Target simulation time (s). Curves are predictions, not executed motion.')
        record=next(x for x in data['live'] if x['tick']==20 and x['joint']==NAMES[j])
        left,right=790,1180
        bound=max(.16,max(abs(record[k]) for k in ['G','W','R','E'])*1.15)
        YY=lambda v:bottom-(v+bound)/(2*bound)*(bottom-top)
        text(780,y0+35,'Signed G = W + R + E at tick20')
        for v in [-.12,0.,.12]:
            line([(left,YY(v)),(right,YY(v))],'#dddddd'); text(left-66,YY(v)-9,f'{v:+.2f}')
        for value in [-record['threshold'],record['threshold']]:
            line([(left,YY(value)),(right,YY(value))],'#c23a32',2)
        for k,(name,color) in enumerate(zip(['G','W','R','E'],['#333333','#2878aa','#ce742d','#888888'])):
            x=left+45+k*92
            line([(x,YY(0)),(x,YY(record[name]))],color,26)
            text(x-8,bottom+6,name); text(x-38,top-23,f'{record[name]:+.4f}')
        line([(left,top),(left,bottom),(right,bottom)])
        text(790,y0+345,'rad; red lines = unchanged gate thresholds')
    svg.append('</svg>')
    im.save(OUT/'boundary_predictions.png')
    (OUT/'boundary_predictions.svg').write_text('\n'.join(svg)+'\n')


def summarize():
    assert not torch.cuda.is_available() and os.environ['MUJOCO_GL']=='disable'
    assert not (OUT/'protection_check.json').exists(), 'no overwrite'
    data=read_json(OUT/'boundary_decomposition.json')
    overlap=read_json(OUT/'overlapping_predictions.json')
    expert=read_json(OUT/'expert_live_comparison.json')
    images=read_json(OUT/'image_comparison.json')
    capture_result=read_json(OUT/'capture/result.json')
    offline_result=read_json(OUT/'offline/result.json')
    assert capture_result['original_trace_exact_match'] and offline_result['pass_']
    assert read_json(OUT/'restoration_check.json')['pass_']
    protected=read_json(OUT/'input_manifest.json')['files']
    mismatch=[name for name,value in protected.items() if sha(ROOT/name)!=value]
    assert not mismatch,mismatch
    budget=read_json(OUT/'budget.json')
    assert budget['chunk_forwards']<=C['max_chunk_forwards'] and budget['live_trials']<=2
    assert budget['expert_replays']<=2
    elapsed=sum(int(line.split()[1]) for line in (OUT/'active_seconds.tsv').read_text().splitlines())
    assert elapsed<=C['max_gpu_container_seconds']
    assert all(int(line.split()[1])<=300 for line in (OUT/'active_seconds.tsv').read_text().splitlines())
    assert all(x['E']==0 and abs(x['identity_residual'])<1e-12 for x in data['live'])
    assert all(x['max_full_chunk_error_rad']<=1e-6 for x in overlap['recomputed_live_chunks'])
    old_chunks={t:load_file(OUT/f'capture/chunk_{t:02d}.safetensors')['rad'].numpy() for t in [0,10,20]}
    live20=next(x for x in overlap['comparisons'] if x['path']=='live' and x['tick']==20)
    max_axis=int(np.argmax(live20['overlap']['all40']['max_abs']))
    axes=list(dict.fromkeys([3,max_axis]))
    draw_boundary_plot(old_chunks,data,axes)
    # Minimal visual provenance: four paired raw and decoded images, not model inputs.
    sheet=Image.new('RGB',(4*320,2*265),'white'); draw=ImageDraw.Draw(sheet)
    for col,t in enumerate(C['capture_ticks']):
        for row,kind in enumerate(['capture','expert']):
            if kind=='capture':
                im=Image.open(OUT/f'capture/image_{t:02d}.png').convert('RGB')
            else:
                obs=load_file(OUT/f'offline/expert_observation_{t:02d}.safetensors')[IMAGE]
                rgb=obs.squeeze(0).permute(1,2,0).numpy()
                im=Image.fromarray(np.rint(rgb*255).clip(0,255).astype(np.uint8))
            sheet.paste(im.resize((320,240)),(col*320,row*265+25))
            draw.text((col*320+5,row*265+5),f'{kind} tick {t} | sim {t*.02:.2f}s',fill='black')
    sheet.save(OUT/'expert_live_images.png')
    write_json(OUT/'protection_check.json',dict(pass_=True,files=len(protected),mismatches=mismatch,
         model_sha256=sha(BUNDLE/'model.safetensors'),bundle_manifest_sha256=sha(BUNDLE/'bundle.sha256.json'),
         allowed_document_changes=['README.md','prompts/README.md','docs/PAI_work_plan_Docker_uv_v2.md'],
         training_updates=0,control_or_threshold_or_dependency_changes=False))
    gpu_samples=[]; process_samples=[]
    for p in OUT.glob('*.monitor.txt'):
        for line in p.read_text().splitlines():
            fields=[x.strip() for x in line.split(',')]
            if len(fields)==4 and fields[0]=='0' and all(x.isdigit() for x in fields):
                gpu_samples.append([int(x) for x in fields])
            elif len(fields)==2 and all(x.isdigit() for x in fields):
                process_samples.append(int(fields[1]))
    resource_rows=[capture_result['resources'],offline_result['resources']]
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert size<C['max_output_bytes'] and shutil.disk_usage(ROOT).free>=C['min_disk_free_bytes']
    resources_summary=dict(gpu_related_container_seconds=elapsed,limit_seconds=900,
         chunk_forwards=budget['chunk_forwards'],chunk_forward_limit=256,live_trials=budget['live_trials'],
         expert_action_replays=budget['expert_replays'],training_updates=0,
         max_rss_kib=max(r['max_rss_kib'] for r in resource_rows),
         torch_peak_allocated_bytes=max(r['torch_peak_allocated_bytes'] for r in resource_rows),
         torch_peak_reserved_bytes=max(r['torch_peak_reserved_bytes'] for r in resource_rows),
         nvml_process_peak_observed_mib=max(process_samples,default=0),
         gpu_free_min_observed_mib=min((r[3] for r in gpu_samples),default=None),
         gpu_samples=len(gpu_samples),output_bytes=size,output_limit_bytes=C['max_output_bytes'],
         free_workspace_bytes=shutil.disk_usage(ROOT).free,
         monitoring_limit='discrete samples, not isolation or a guaranteed true peak')
    write_json(OUT/'resource_summary.json',resources_summary)
    primary=next(x for x in data['live'] if x['tick']==20 and x['joint']=='Wrist_Pitch_R')
    write_json(OUT/'summary.json',dict(investigation_complete=True,step05A_complete=False,
         tick20_wrist_pitch=primary,largest_overlap_axis=NAMES[max_axis],resources=resources_summary,
         original_stop_reproduced_exactly=True,model_or_statistics_or_control_changes=False,
         conclusions='Read detailed result for confirmed facts, supported hypotheses, and untested causes.'))
    print(json.dumps(dict(summary_pass=True,primary=primary,resources=resources_summary)),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['audit','capture','offline','summarize'])
    args = parser.parse_args()
    setup(C['seed'])
    globals()[args.mode]()


if __name__ == '__main__':
    main()
