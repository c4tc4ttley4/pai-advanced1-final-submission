"""Approved fixed-input offline sensitivity only; never applies actions or steps physics."""
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from src.step05A_act import ROOT, IMAGE, STATE, setup, load_bundle, digest, sha, write_json, resources

OUT = ROOT / 'artifacts/step05A_state_sensitivity/state_sensitivity_20261031_01'
C = json.loads((OUT/'spec.json').read_text())
assert not (OUT/'budget.json').exists(), 'no re-execution'
START = time.monotonic()
count = 0
write_json(OUT/'budget.json', dict(forwards=0, maximum=32, events=[]))
for item in json.loads((OUT/'inputs.json').read_text())['files']:
    assert sha(ROOT/item['path']) == item['sha256'], item['path']
setup(C['seed'])
assert torch.cuda.is_available()
torch.cuda.reset_peak_memory_stats()
BUNDLE = ROOT/C['bundle']
policy, pre, post = load_bundle(BUNDLE)
assert policy.config.chunk_size == 50 and policy.config.n_action_steps == 10
assert policy.config.n_obs_steps == 1 and policy.config.temporal_ensemble_coeff is None
assert digest(policy) == json.loads((BUNDLE/'training_result.json').read_text())['after_model_digest']
stats = json.loads((BUNDLE/'statistics.json').read_text())['stats']
names = json.loads((ROOT/'configs/step05A_act.json').read_text())['joint_names']
events, checks = [], []

def predict(raw, label):
    global count
    assert set(raw) == {IMAGE, STATE}
    assert raw[IMAGE].shape == (1,3,480,640) and raw[STATE].shape == (1,6)
    assert raw[IMAGE].dtype == raw[STATE].dtype == torch.float32
    assert torch.isfinite(raw[IMAGE]).all() and torch.isfinite(raw[STATE]).all()
    assert 0 <= raw[IMAGE].min() <= raw[IMAGE].max() <= 1
    assert count < C['max_forwards'] and time.monotonic()-START < 110
    policy.reset()
    processed = pre({key: value.clone() for key,value in raw.items()})
    assert processed['action'] is None
    pre_errors = {}
    for key in [IMAGE, STATE]:
        mu = torch.tensor(stats[key]['mean'],dtype=torch.float32,device='cuda')
        sd = torch.tensor(stats[key]['std'],dtype=torch.float32,device='cuda')
        expected = (raw[key].to('cuda')-mu)/(sd+1e-8)
        pre_errors[key] = float((expected-processed[key]).abs().max())
    assert max(pre_errors.values()) < 1e-6
    count += 1  # consume before calling; failed forwards also count
    events.append(dict(label=label, cumulative=count))
    write_json(OUT/'budget.json',dict(forwards=count,maximum=32,events=events))
    normalized = policy.predict_action_chunk(processed)
    assert normalized.shape == (1,50,6) and torch.isfinite(normalized).all()
    rad = torch.stack([post(normalized[:,i]).squeeze(0).cpu() for i in range(50)])
    assert torch.isfinite(rad).all()
    mu = torch.tensor(stats['action']['mean'],dtype=torch.float32)
    sd = torch.tensor(stats['action']['std'],dtype=torch.float32)
    post_error = float((rad-(normalized.squeeze(0).cpu()*sd+mu)).abs().max())
    assert post_error < 1e-6
    checks.append(dict(label=label,pre_errors=pre_errors,post_error=post_error))
    return rad

with torch.inference_mode():
    fixture = load_file(BUNDLE/'reference_observations.safetensors')
    expected = load_file(BUNDLE/'reference_actions.safetensors')['actions_rad']
    restoration = []
    for i in range(5):
        actual = predict({key:val[i:i+1] for key,val in fixture.items()},f'restore_{i}')
        err = float((actual[:10]-expected[i]).abs().max())
        restoration.append(dict(index=i,max_error_rad=err))
        assert torch.allclose(actual[:10],expected[i],atol=1e-6,rtol=1e-6)
    write_json(OUT/'restoration.json',dict(pass_=True,strict=True,local_files_only=True,checks=restoration,forwards=count))
    rows, comparisons, tensors, baselines = [], [], {}, {}
    for tick in C['ticks']:
        raw = load_file(ROOT/C['capture']/f'observation_{tick:02d}.safetensors')
        original = {k:v.clone() for k,v in raw.items()}
        baseline = predict(raw,f'tick{tick}_base')
        old = load_file(ROOT/C['capture']/f'chunk_{tick:02d}.safetensors')['rad']
        err = float((baseline-old).abs().max())
        assert err <= 1e-6, 'saved chunk mismatch'
        baselines[tick] = baseline.double().numpy()
        tensors[f'tick{tick}_base'] = baseline.contiguous()
        for axis in C['axes']:
            perturbed = {}
            steps = {}
            for sign in [-1,1]:
                obs = {k:v.clone() for k,v in raw.items()}
                obs[STATE][0,axis] += sign*C['delta_rad']
                steps[sign] = float(obs[STATE][0,axis].double()-raw[STATE][0,axis].double())
                assert torch.equal(obs[IMAGE],raw[IMAGE])
                assert torch.count_nonzero(obs[STATE]-raw[STATE]) == 1
                assert abs(steps[sign]-sign*C['delta_rad']) < 1e-7
                value = predict(obs,f'tick{tick}_axis{axis}_{sign:+d}')
                tensors[f'tick{tick}_axis{axis}_{sign:+d}'] = value.contiguous()
                perturbed[sign] = value.double().numpy()
            base = baseline.double().numpy()
            central = (perturbed[1]-perturbed[-1])/(steps[1]-steps[-1])
            left = (perturbed[-1]-base)/steps[-1]
            right = (perturbed[1]-base)/steps[1]
            for horizon in range(50):
                for j,name in enumerate(names):
                    rows.append(dict(tick=tick,input_joint=names[axis],horizon=horizon,output_joint=name,
                        baseline_rad=float(base[horizon,j]),minus_delta_rad=float(perturbed[-1][horizon,j]-base[horizon,j]),
                        plus_delta_rad=float(perturbed[1][horizon,j]-base[horizon,j]),central_rad_per_rad=float(central[horizon,j]),
                        minus_slope_rad_per_rad=float(left[horizon,j]),plus_slope_rad_per_rad=float(right[horizon,j])))
            comparisons.append(dict(tick=tick,input_joint=names[axis],saved_chunk_max_error_rad=err,
                actual_state_steps_rad={str(k):v for k,v in steps.items()},
                first_minus_delta_rad=(perturbed[-1][0]-base[0]).tolist(),first_plus_delta_rad=(perturbed[1][0]-base[0]).tolist(),
                first_central_rad_per_rad=central[0].tolist(),chunk_central_mean_rad_per_rad=central.mean(0).tolist(),
                chunk_central_abs_max_rad_per_rad=np.abs(central).max(0).tolist(),
                chunk_max_slope_asymmetry_rad_per_rad=np.abs(left-right).max(0).tolist(),
                chunk_minus_delta_range_rad=[(perturbed[-1]-base).min(0).tolist(),(perturbed[-1]-base).max(0).tolist()],
                chunk_plus_delta_range_rad=[(perturbed[1]-base).min(0).tolist(),(perturbed[1]-base).max(0).tolist()]))
        assert all(torch.equal(raw[k],original[k]) for k in raw), 'raw input mutated'
    save_file(tensors,OUT/'predictions.safetensors')
    with (OUT/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tick_differences=[]
    for axis in C['axes']:
        group=[x for x in comparisons if x['input_joint']==names[axis]]
        for a,b in zip(group,group[1:]):
            tick_differences.append(dict(input_joint=names[axis],ticks=[a['tick'],b['tick']],
                first_central_change_rad_per_rad=(np.array(b['first_central_rad_per_rad'])-np.array(a['first_central_rad_per_rad'])).tolist(),
                chunk_central_mean_change_rad_per_rad=(np.array(b['chunk_central_mean_rad_per_rad'])-np.array(a['chunk_central_mean_rad_per_rad'])).tolist()))
    assert count == 20
    write_json(OUT/'result.json',dict(pass_=True,run_id=C['run_id'],joint_order=names,comparisons=comparisons,
        tick_differences=tick_differences,preprocessing_checks=checks,forward_count=count,
        additional_training_updates=0,physics_steps=0,action_applications=0,
        model_digest=digest(policy),resources=resources(),elapsed_seconds=time.monotonic()-START,
        limitation='Artificial local input sensitivity at three stored observations only; no causal closed-loop or stability claim.'))
    print(json.dumps(dict(pass_=True,forwards=count,seconds=time.monotonic()-START)),flush=True)
