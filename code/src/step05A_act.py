"""Bounded, offline ACT connection checks. Original STEP 2--4 code is unchanged."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline, UnnormalizerProcessorStep

ROOT = Path(__file__).resolve().parents[1]
IMAGE, STATE = 'observation.images.workspace', 'observation.state'
CFG = json.loads((ROOT / 'configs/step05A_act.json').read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paths(run):
    return ROOT / 'artifacts/step05A' / run, ROOT / 'models/step05A' / run


def setup(seed=42):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Same deterministic execution settings for both containers.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def dataset():
    return LeRobotDataset(repo_id=CFG['repo_id'], root=ROOT / CFG['dataset_root'],
                          video_backend='torchcodec',
                          delta_timestamps={'action': [i / CFG['fps'] for i in range(CFG['chunk_size'])]})


def policy_config(device):
    return ACTConfig(device=device, input_features={
        IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(CFG['image_shape'])),
        STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=CFG['chunk_size'], n_action_steps=CFG['n_action_steps'],
        n_obs_steps=1, pretrained_backbone_weights=None,
        optimizer_lr=CFG['learning_rate'], optimizer_lr_backbone=CFG['backbone_learning_rate'],
        optimizer_weight_decay=CFG['weight_decay'], push_to_hub=False)


def stats_tensors(stats):
    return {k: {n: torch.tensor(v, dtype=torch.float32) for n, v in s.items()}
            for k, s in stats.items()}


def raw_batch(ds, indices):
    # Explicit whitelist; indices/tasks/phase/privileged geometry never enter ACT.
    items = [ds[int(i)] for i in indices]
    return {key: torch.stack([x[key] for x in items])
            for key in [IMAGE, STATE, 'action', 'action_is_pad']}


def observations(batch):
    return {key: batch[key] for key in (IMAGE, STATE)}


def digest(policy):
    h = hashlib.sha256()
    for k, v in policy.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit(run):
    out, _ = paths(run)
    assert not (out / 'data_audit.json').exists(), 'no overwrite'
    ds = dataset()
    assert ds.num_episodes == 5 and len(ds) == 9531 and ds.fps == 50
    assert ds.features[STATE]['names'] == ds.features['action']['names'] == CFG['joint_names']
    assert ds.meta.info['codebase_version'] == 'v3.0'
    stats = {k: {n: np.asarray(ds.meta.stats[k][n]).tolist() for n in ['mean', 'std', 'min', 'max']}
             for k in [IMAGE, STATE, 'action']}
    original_stats = copy.deepcopy(stats)
    boundaries, checks, image_ids = [], [], []
    max_delta = np.zeros(6)
    all_numeric = {k: [] for k in [STATE, 'action']}
    for ep in CFG['episodes']:
        meta = ds.meta.episodes[ep]
        start, end = int(meta['dataset_from_index']), int(meta['dataset_to_index'])
        boundaries.append(dict(episode=ep, start=start, end=end, frames=end-start))
        for key in all_numeric:
            values = torch.stack(list(ds.hf_dataset.select(range(start, end))[key])).numpy()
            assert np.isfinite(values).all()
            all_numeric[key].append(values)
            if key == 'action':
                max_delta = np.maximum(max_delta, np.abs(np.diff(values, axis=0)).max(axis=0))
        for idx in [start, (start+end)//2, end-1]:
            item = ds[idx]
            ids = np.minimum(np.arange(idx, idx+CFG['chunk_size']), end-1)
            expected = torch.stack([ds.hf_dataset[int(j)]['action'] for j in ids])
            pad = torch.tensor(np.arange(idx, idx+CFG['chunk_size']) >= end)
            assert torch.equal(item['action'], expected)
            assert torch.equal(item['action_is_pad'], pad)
            assert tuple(item[IMAGE].shape) == tuple(CFG['image_shape'])
            assert item[IMAGE].dtype == torch.float32
            assert 0 <= item[IMAGE].min() <= item[IMAGE].max() <= 1
            checks.append(dict(index=idx, episode=ep, pad_count=int(pad.sum()),
                               action_error_rad=float((item['action']-expected).abs().max())))
        image_ids.extend(np.linspace(start, end-1, CFG['image_stats_samples_per_episode'], dtype=int).tolist())
    numeric_check = {}
    for key, chunks in all_numeric.items():
        values = np.concatenate(chunks).astype(np.float64)
        mu, sd = values.mean(axis=0), values.std(axis=0)
        numeric_check[key] = dict(mean_error=float(np.max(np.abs(mu-np.array(stats[key]['mean'])))),
                                 std_error=float(np.max(np.abs(sd-np.array(stats[key]['std'])))))
        assert numeric_check[key]['mean_error'] < 1e-4 and numeric_check[key]['std_error'] < 1e-4, 'unexpected numerical metadata discrepancy'
        # Preserve original metadata; use exact full-training-set population stats.
        stats[key] = {n: v.tolist() for n,v in [('mean',mu), ('std',sd),
                      ('min',values.min(axis=0)), ('max',values.max(axis=0))]}
    # Decoded pixels, streaming one image at a time, all five episodes.
    sums, squares, minimum, maximum = np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3)
    pixels = 0
    for idx in image_ids:
        rgb = ds[idx][IMAGE].double().numpy().reshape(3, -1)
        sums += rgb.sum(axis=1)
        squares += (rgb*rgb).sum(axis=1)
        pixels += rgb.shape[1]
        minimum, maximum = np.minimum(minimum, rgb.min(axis=1)), np.maximum(maximum, rgb.max(axis=1))
    mu = sums / pixels
    sd = np.sqrt(squares / pixels - mu*mu)
    assert np.all(sd > .01), 'unexpected decoded image distribution'
    stats[IMAGE] = {n: v.reshape(3,1,1).tolist()
                    for n,v in [('mean',mu), ('std',sd), ('min',minimum), ('max',maximum)]}
    config = policy_config('cpu')
    pre, post = make_act_pre_post_processors(config, stats_tensors(stats))
    inverse = PolicyProcessorPipeline(steps=[UnnormalizerProcessorStep(
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping, stats=stats_tensors(stats))])
    errors = []
    for idx in [0, 1000, 4000, 9530]:
        raw = raw_batch(ds, [idx])
        processed = pre(raw)
        restored = inverse(processed)
        err = {k: float((raw[k]-restored[k]).abs().max()) for k in [IMAGE, STATE, 'action']}
        err['post_action'] = float((post(processed['action'])-raw['action']).abs().max())
        assert all(v <= CFG['normalization_roundtrip_atol'] for v in err.values())
        errors.append(dict(index=idx, errors=err))
    data_hash = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / CFG['dataset_root']).rglob('*')) if p.is_file()}
    stats_info = dict(stats=stats, original_stats=original_stats, image_sample_indices=image_ids,
                      image_pixels_per_channel=pixels,
                      image_stats_reason='Saved image std ~1e-5 disagrees with pooled decoded pixels. Derived STEP 5A stats only; original meta unchanged.',
                      numeric_scope='exact population statistics over all 9531 frames of episodes 0..4; originals retained separately',
                      image_scope='50 deterministic evenly spaced decoded images, full pixels, population std')
    write_json(out / 'stats.json', stats_info)
    result = dict(pass_=True, boundaries=boundaries, chunk_checks=checks,
                  numeric_stats_check=numeric_check, roundtrip=errors,
                  max_saved_sent_delta_rad=max_delta.tolist(),
                  action_jump_limit_rad=(max_delta * CFG['n_action_steps'] + 1e-6).tolist(),
                  jump_limit_note='Conservative stop gate for up to 10-tick replanning displacement; not a new per-tick rate limiter. All raw predictions logged; no clipping.',
                  dataset_files_sha256=data_hash, total_frames=len(ds), episodes=5,
                  input_keys=[IMAGE, STATE], training_target='action: actual sent rad absolute position',
                  metadata_used_as_model_inputs=False)
    write_json(out / 'data_audit.json', result)
    print(json.dumps(dict(audit_pass=True, image_original_std=original_stats[IMAGE]['std'],
                          image_derived_std=stats[IMAGE]['std'], max_delta_rad=max_delta.tolist())), flush=True)


def diagnostics(policy, pre, post, batch):
    with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(12345)
        processed = pre(batch)
        # ACT's VAE loss requires train mode; freeze gradients, fix dropout/VAE RNG.
        policy.train()
        loss, components = policy(processed)
        policy.eval()
        predicted = post(policy.predict_action_chunk(pre(observations(batch))))
        valid = (~batch['action_is_pad']).unsqueeze(-1).expand_as(batch['action'])
        mae = float((predicted-batch['action']).abs()[valid].mean())
        policy.reset()
        return dict(loss=float(loss), **components, inference_action_mae_rad=mae,
                    inference_input_keys=[IMAGE, STATE])


def queue_reference(policy, pre, post, fixture):
    output = []
    policy.eval()
    with torch.inference_mode():
        for i in range(len(fixture[STATE])):
            policy.reset()
            obs = {k: v[i:i+1] for k,v in fixture.items()}
            values = [post(policy.select_action(pre(obs))).cpu() for _ in range(CFG['n_action_steps'])]
            output.append(torch.stack(values).squeeze(1))
    policy.reset()
    return torch.stack(output)


def resources():
    result = dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if torch.cuda.is_available():
        result.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gpu_name=torch.cuda.get_device_name(0),
                      gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', 'not exposed')),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return result


def bundle_hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != 'bundle.sha256.json'}


def load_bundle(directory):
    hashes = json.loads((directory / 'bundle.sha256.json').read_text())
    assert bundle_hashes(directory) == hashes, 'bundle hashes changed'
    # strict defaults False in this release: explicitly require True.
    policy = ACTPolicy.from_pretrained(directory, local_files_only=True, strict=True)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(directory))
    policy.eval()
    policy.reset()
    assert policy.config.pretrained_backbone_weights is None
    return policy, pre, post


def train(run, stage):
    out, model = paths(run)
    model.mkdir(parents=True, exist_ok=True)
    target = model / stage
    assert not target.exists(), 'no overwrite'
    target.mkdir()
    start = time.monotonic()
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    ds = dataset()
    stats_info = json.loads((out / 'stats.json').read_text())
    sampler = torch.Generator().manual_seed(CFG['seed'])
    if stage == 'smoke':
        config = policy_config('cuda')
        policy = ACTPolicy(config).to('cuda')
        pre, post = make_act_pre_post_processors(config, stats_tensors(stats_info['stats']))
        optimizer_cfg = config.get_optimizer_preset()
        optimizer = optimizer_cfg.build(policy.get_optim_params())
        first_step, updates, batch_size = 0, CFG['smoke_updates'], CFG['smoke_batch_size']
        initial_digest = digest(policy)
    else:
        source = model / ('final' if stage == 'continued' else 'smoke')
        policy, pre, post = load_bundle(source)
        config = policy.config
        optimizer_cfg = config.get_optimizer_preset()
        optimizer = optimizer_cfg.build(policy.get_optim_params())
        state = torch.load(source / 'resume.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        sampler.set_state(state['sampler_rng'])
        first_step = state['step']
        total = CFG['corrective_total_updates'] if stage == 'continued' else CFG['planned_total_updates']
        updates = total-first_step
        batch_size = CFG['train_batch_size']
        initial_digest = state['initial_model_digest']
    assert first_step + updates <= CFG['max_total_updates']
    selected = [0, 600, 1200, 1800, 7500]
    diagnostic_batch = raw_batch(ds, selected)
    before_diag = diagnostics(policy, pre, post, diagnostic_batch)
    before_hash = digest(policy)
    tracked_names = [n for n, _ in policy.named_parameters()
                     if n == 'model.action_head.weight' or (n.startswith('model.backbone') and n.endswith('conv1.weight') and 'layer' not in n)]
    tracked = {n: p.detach().cpu().clone() for n,p in policy.named_parameters() if n in tracked_names}
    assert len(tracked) == 2
    metrics = []
    with (out / f'{stage}_loss.jsonl').open('x') as log:
        for step in range(first_step+1, first_step+updates+1):
            if time.monotonic()-start > 1800:
                raise RuntimeError('training invocation wall cap')
            t = time.monotonic()
            ids = torch.randint(len(ds), (batch_size,), generator=sampler).tolist()
            batch = pre(raw_batch(ds, ids))
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, parts = policy(batch)
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), optimizer_cfg.grad_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            row = dict(step=step, batch_size=batch_size, sample_indices=ids,
                       loss=float(loss.detach()), **parts, grad_norm=float(grad),
                       seconds=time.monotonic()-t)
            log.write(json.dumps(row)+'\n')
            log.flush()
            metrics.append(row)
            if step <= 3 or step % 20 == 0:
                print(json.dumps(row), flush=True)
            if step % 20 == 0:
                assert shutil.disk_usage(ROOT).free >= CFG['min_disk_free_bytes']
    after_diag = diagnostics(policy, pre, post, diagnostic_batch)
    after_hash = digest(policy)
    assert before_hash != after_hash
    change = {n: float((p.detach().cpu()-tracked[n]).abs().max())
              for n,p in policy.named_parameters() if n in tracked}
    assert all(v > 0 for v in change.values())
    fixture = {k: v.contiguous() for k,v in observations(diagnostic_batch).items()}
    reference = queue_reference(policy, pre, post, fixture)
    save_file(fixture, target / 'reference_observations.safetensors')
    save_file({'actions_rad': reference.contiguous()}, target / 'reference_actions.safetensors')
    policy.save_pretrained(target)
    pre.save_pretrained(target)
    post.save_pretrained(target)
    write_json(target / 'statistics.json', stats_info)
    write_json(target / 'training_spec.json', CFG)
    shutil.copyfile(out / 'data_audit.json', target / 'data_audit.json')
    torch.save(dict(optimizer=optimizer.state_dict(), step=first_step+updates,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                    sampler_rng=sampler.get_state(), initial_model_digest=initial_digest), target / 'resume.pt')
    result = dict(stage=stage, actual_updates_in_stage=updates, cumulative_updates=first_step+updates,
                  batch_size=batch_size, original_initial_model_digest=initial_digest,
                  before_model_digest=before_hash, after_model_digest=after_hash,
                  tracked_parameter_max_change=change, diagnostic_indices=selected,
                  before=before_diag, after=after_diag, wall_seconds=time.monotonic()-start,
                  mean_update_seconds=float(np.mean([x['seconds'] for x in metrics])),
                  first_loss=metrics[0], last_loss=metrics[-1], resources=resources(),
                  pretrained_backbone=False, policy_config=str(target / 'config.json'),
                  resume='optimizer, cumulative step, Python/NumPy/torch/CUDA/sampler RNG; fp32, no scheduler/scaler; diagnostics have consumed RNG before capture')
    write_json(target / 'training_result.json', result)
    write_json(out / f'{stage}_result.json', result)
    write_json(target / 'bundle.sha256.json', bundle_hashes(target))
    print(json.dumps(result), flush=True)


def reload_check(run, checkpoint='final'):
    out, model = paths(run)
    result_file = out / ('reload_result.json' if checkpoint == 'final' else f'reload_{checkpoint}_result.json')
    assert not result_file.exists()
    start = time.monotonic()
    directory = model / checkpoint
    policy, pre, post = load_bundle(directory)
    assert digest(policy) == json.loads((directory / 'training_result.json').read_text())['after_model_digest']
    actual = queue_reference(policy, pre, post, load_file(directory / 'reference_observations.safetensors'))
    expected = load_file(directory / 'reference_actions.safetensors')['actions_rad']
    passed = torch.allclose(actual, expected, atol=CFG['reload_atol_rad'], rtol=CFG['reload_rtol'])
    result = dict(pass_=passed, max_action_difference_rad=float((actual-expected).abs().max()),
                  atol_rad=CFG['reload_atol_rad'], rtol=CFG['reload_rtol'],
                  observations=5, actions_per_reset=CFG['n_action_steps'],
                  weights_digest=digest(policy), strict_load=True,
                  bundle_sha256=sha(directory / 'bundle.sha256.json'),
                  inputs_from_saved_bundle_only=True, reset_before_each_comparison=True,
                  wall_seconds=time.monotonic()-start, resources=resources())
    result['checkpoint'] = checkpoint
    write_json(result_file, result)
    print(json.dumps(result), flush=True)
    assert passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['audit', 'smoke', 'train', 'continue', 'reload', 'rollout'])
    parser.add_argument('--run', default=CFG['run_id'])
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--checkpoint', choices=['final', 'continued'], default='final')
    args = parser.parse_args()
    assert args.run.replace('_','').replace('-','').isalnum()
    out, _ = paths(args.run)
    out.mkdir(parents=True, exist_ok=True)
    setup(CFG['seed'])
    if args.mode == 'audit':
        audit(args.run)
    elif args.mode in ['smoke', 'train', 'continue']:
        train(args.run, {'smoke':'smoke', 'train':'final', 'continue':'continued'}[args.mode])
    elif args.mode == 'reload':
        reload_check(args.run, args.checkpoint)
    else:
        from src.step05A_rollout import rollout
        rollout(args.run, args.trial, args.checkpoint)
