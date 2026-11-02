"""Read finalized local datasets and replay persisted float32 actions on CPU."""
import gzip
import json
from pathlib import Path
import resource
import time

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from src.step03_expert import ContactExpert, bilateral, contact_records, grasp_pose, storage_state
from src.step04_dataset import CFG, IMAGE, STATE, cpu_only, integration_state, open_dataset, write_json


def traces(out, trial):
    with gzip.open(out / f'trace_{trial:02d}.jsonl.gz', 'rt') as f:
        return [json.loads(line) for line in f]


def validate(root, out):
    cpu_only()
    start = time.monotonic()
    root, out = Path(root), Path(out)
    ds = open_dataset(root)
    count = ds.num_episodes
    target = out / ('validation.json' if count == 5 else f'validation_{count:02d}.json')
    assert not target.exists(), 'no overwrite'
    assert 1 <= count <= 5
    assert ds.fps == 50
    assert set(ds.features) == {STATE, IMAGE, 'action', 'timestamp', 'frame_index',
                                'episode_index', 'index', 'task_index'}
    assert ds.meta.info['codebase_version'] == 'v3.0'
    assert ds.meta.info['total_tasks'] == 1
    assert len(ds.meta.tasks) == 1 and ds.meta.tasks.index[0] == CFG['task']
    assert ds.features[STATE]['dtype'] == ds.features['action']['dtype'] == 'float32'
    expected_names = json.loads(Path('configs/step02.json').read_text())['joint_names']
    assert ds.features[STATE]['names'] == ds.features['action']['names'] == expected_names
    e = ContactExpert()
    bounds = e.env.bounds
    joint_bounds = e.env.model.jnt_range[e.env.jids]
    e.close()
    result = []
    cursor = 0
    # Representative labels are placed on this review sheet only.
    sheet = Image.new('RGB', (5*320, count*265), 'white')
    draw = ImageDraw.Draw(sheet)
    # First frame of these phases shows the preceding completed transition:
    # approach complete, grasp complete, lifted, at tray, and released/stable.
    phases = ['descend', 'lift', 'hold', 'release', 'stable']
    labels = ['approached', 'grasped', 'lifted', 'at tray', 'released']
    all_expected_video_frames = 0
    for trial in range(count):
        attempt = json.loads((out / f'attempt_{trial:02d}.json').read_text())
        assert attempt['status'] == 'success'
        rows = traces(out, trial)
        n = len(rows)
        assert n == attempt['frames']
        meta = ds.meta.episodes[trial]
        assert int(meta['dataset_from_index']) == cursor
        assert int(meta['dataset_to_index']) == cursor+n
        assert int(meta['length']) == n
        max_state_rounding = 0.
        max_action_rounding = 0.
        max_rgb_mae = 0.
        previous = -1.
        minima = {STATE: np.full(6, np.inf), 'action': np.full(6, np.inf)}
        maxima = {STATE: np.full(6, -np.inf), 'action': np.full(6, -np.inf)}
        anchors = attempt['phase_first_frame']
        selected = {0, n-1, *anchors.values()}
        for frame, row in enumerate(rows):
            item = ds[cursor+frame]  # official path decodes every required frame
            assert int(item['index']) == cursor+frame
            assert int(item['frame_index']) == frame and int(item['episode_index']) == trial
            ts = float(item['timestamp'])
            assert ts > previous and abs(ts-frame/50) < 3e-6
            if frame:
                assert abs(ts-previous-.02) < 4e-6
            previous = ts
            assert abs(row['time_before']-frame/50) < 1e-8
            assert abs(row['time']-(frame+1)/50) < 1e-8
            for key, reference in [(STATE, row['state_before']), ('action', row['sent'])]:
                value = item[key].numpy()
                assert value.dtype == np.float32 and value.shape == (6,) and np.isfinite(value).all()
                limits = bounds if key == 'action' else joint_bounds
                # Physical qpos can exceed a soft joint boundary; 1e-3rad is a
                # predeclared diagnostic tolerance, never action saturation.
                margin = 0 if key == 'action' else .001
                assert np.all(value >= limits[:,0]-margin) and np.all(value <= limits[:,1]+margin)
                assert np.array_equal(value, np.array(reference, np.float32))
                err = float(np.max(np.abs(value-np.asarray(reference))))
                if key == STATE:
                    measured64 = np.asarray(row['q_measured_float64'])
                    assert np.array_equal(value, measured64.astype(np.float32))
                    max_state_rounding = max(max_state_rounding, float(np.max(np.abs(value-measured64))))
                else:
                    max_action_rounding = max(max_action_rounding, err)
                minima[key] = np.minimum(minima[key],value)
                maxima[key] = np.maximum(maxima[key],value)
            rgb = item[IMAGE]
            assert tuple(rgb.shape) == (3,480,640) and str(rgb.dtype) == 'torch.float32'
            assert bool(rgb.isfinite().all()) and 0 <= float(rgb.min()) <= float(rgb.max()) <= 1
            if frame in selected:
                im = np.rint(rgb.numpy().transpose(1,2,0)*255).astype(np.uint8)
                for phase, idx in anchors.items():
                    if idx != frame:
                        continue
                    original = np.asarray(Image.open(out/f'anchor_{trial:02d}_{phase}.png').convert('RGB'))
                    mae = float(np.abs(im.astype(float)-original).mean())
                    assert mae < 8., 'compressed image mismatch'
                    max_rgb_mae = max(max_rgb_mae,mae)
                    if phase in phases:
                        col = phases.index(phase)
                        sheet.paste(Image.fromarray(im).resize((320,240)),(col*320,trial*265+25))
                        draw.text((col*320+5,trial*265+5),f'episode {trial} | {labels[col]} | t={frame/50:.2f}s',fill='black')
            del rgb, item
        cursor += n
        all_expected_video_frames += n
        result.append(dict(episode=trial, frames=n, sim_seconds=n/50,
                           max_state_rounding_rad=max_state_rounding,
                           max_action_rounding_rad=max_action_rounding,
                           max_anchor_RGB_MAE_255=max_rgb_mae,
                           minimum={k:v.tolist() for k,v in minima.items()},
                           maximum={k:v.tolist() for k,v in maxima.items()}))
        print(json.dumps(dict(validated_episode=trial,frames=n)),flush=True)
    assert cursor == len(ds) == ds.meta.info['total_frames']
    assert len(ds.meta.episodes) == count
    assert ds.meta.info['total_episodes'] == count
    video_count = 0
    video_details = []
    for path in sorted(root.glob('videos/**/*.mp4')):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert stream.width == 640 and stream.height == 480
            assert float(stream.average_rate) == 50
            last = -1.
            frames = 0
            for image in container.decode(video=0):
                ts = float(image.pts*image.time_base)
                assert ts > last
                last = ts
                frames += 1
            video_count += frames
            video_details.append(dict(file=str(path),frames=frames,codec=stream.codec_context.name,
                                       pixel_format=stream.codec_context.format.name))
    assert video_count == all_expected_video_frames
    for key in [STATE,'action',IMAGE]:
        stats = ds.meta.stats[key]
        assert np.isfinite(np.array(stats['mean'])).all()
        assert np.isfinite(np.array(stats['std'])).all()
        if key != IMAGE:
            assert np.all(np.asarray(stats['count']) == cursor)
        else:
            # LeRobot deliberately samples PNGs for image statistics.
            assert int(np.asarray(stats['count']).item()) > 0
    sheet.save(out/('representative_review.png' if count == 5 else f'representative_review_{count:02d}.png'))
    write_json(target, dict(pass_=True, episodes=count, frames=cursor,
                             all_images_read_via_LeRobot=True, full_sequential_video_decode=True,
                             videos=video_details, results=result, privileged_features=False,
                             wall_seconds=time.monotonic()-start,
                             max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             visual_review='pending', features=ds.features))


def replay(root, out):
    cpu_only()
    root, out = Path(root), Path(out)
    assert not (out/'replay.json').exists(), 'no overwrite'
    start = time.monotonic()
    ds = open_dataset(root)
    results = []
    cursor = 0
    for trial in range(ds.num_episodes):
        attempt = json.loads((out/f'attempt_{trial:02d}.json').read_text())
        assert attempt['status'] == 'success'
        rows = traces(out,trial)
        expert = ContactExpert()
        expert.reset(seed=attempt['seed'],offset=attempt['offset_m'])
        e,c = expert.env,expert.c
        assert np.array_equal(integration_state(e),attempt['initial']['integration'])
        assert e.renderer is None and expert.generator is None
        state_max = block_max = ctrl_max = 0.
        stable = floor_gap = grip_gap = 0.
        hold = []
        transport_ok = True
        block_start = None
        warnings = []
        mujoco.set_mju_user_warning(warnings.append)
        try:
            for frame,row in enumerate(rows):
                # The numeric table belongs to the finalized LeRobot dataset.
                # Do not invoke expert.start/tick/_advance or recompute actions.
                item = ds.hf_dataset[cursor+frame]
                assert int(item['episode_index']) == trial and int(item['frame_index']) == frame
                action = item['action'].numpy().astype(np.float64)
                assert np.isfinite(action).all() and action.shape == (6,)
                assert np.all(action>=e.bounds[:,0]) and np.all(action<=e.bounds[:,1])
                assert abs(e.data.time-frame*.02) < 1e-8
                if row['phase'] == 'approach' and block_start is None:
                    block_start = e.data.body('block').xpos.copy()
                state_max = max(state_max,float(np.max(np.abs(e.data.qpos[e.qids]-item[STATE].numpy()))))
                # Direct actuator position targets, already compensated/limited.
                # Same assignment + 10 mj_step + mj_forward as the base path.
                e.target[:] = action
                e.data.ctrl[e.aids] = action
                ctrl_max = max(ctrl_max,float(np.max(np.abs(e.data.ctrl[e.aids]-action))))
                for _ in range(e.substeps):
                    mujoco.mj_step(e.model,e.data)
                    assert all(np.isfinite(x).all() for x in [e.data.qpos,e.data.qvel,e.data.qacc])
                    assert not any(w.number for w in e.data.warning)
                    assert not warnings
                    for k in e.data.contact:
                        names = [e.model.geom(k.geom1).name,e.model.geom(k.geom2).name]
                        if any(n.startswith('robot_geom') for n in names):
                            assert k.dist >= -c['max_contact_penetration_m']
                    assert np.max(np.abs(e.data.xpos[expert.fixed_ids]-expert.fixed_pos)) <= 1e-12
                mujoco.mj_forward(e.model,e.data)
                ct = contact_records(e)
                point,_ = grasp_pose(e.model,e.data,expert.local)
                block = e.data.body('block').xpos.copy()
                state_max = max(state_max,float(np.max(np.abs(e.data.qpos[e.qids]-row['measured']))))
                block_max = max(block_max,float(np.max(np.abs(block-row['block']))))
                if row['phase'] == 'hold':
                    hold.append(dict(block=block.copy(),relative=block-point,
                                     both=bilateral(ct),no_support=not any('block_geom' in k['geoms'] and any(g=='table' or g.startswith('tray_') for g in k['geoms']) for k in ct)))
                if row['phase'] in ['transport','lower']:
                    grip_gap = 0 if bilateral(ct) else grip_gap+.02
                    transport_ok &= (grip_gap <= c['grip_contact_gap_seconds'] and
                                     np.linalg.norm(block-point) <= c['max_transport_relative_distance_m'] and
                                     not any('block_geom' in k['geoms'] and any(g=='table' or g.startswith('tray_') for g in k['geoms']) for k in ct))
                state = storage_state(e,c,ct)
                if row['phase'] == 'stable':
                    floor_gap = 0 if state['floor_contact'] else floor_gap+.02
                    valid = state['contained'] and state['supported_height'] and state['released'] and state['slow'] and not state['wall_contact'] and floor_gap<=c['floor_contact_gap_seconds']
                    stable = stable+.02 if valid else 0.
            lift = min(x['block'][2]-block_start[2] for x in hold)
            slip = max(float(np.linalg.norm(x['relative']-hold[0]['relative'])) for x in hold)
            grasp_ok = (lift>=c['min_lift_m'] and len(hold)*.02>=c['hold_seconds'] and
                        slip<=c['max_hold_slip_m'] and all(x['both'] and x['no_support'] for x in hold))
            success = bool(grasp_ok and transport_ok and stable+1e-9>=c['stable_seconds'])
            tolerance_ok = state_max<=CFG['replay_state_tolerance_rad'] and block_max<=CFG['replay_block_tolerance_m']
            results.append(dict(episode=trial,success=success,tolerance_pass=tolerance_ok,
                                frames=len(rows),sim_seconds=e.data.time,
                                max_state_difference_rad=state_max,max_block_difference_m=block_max,
                                max_ctrl_action_difference_rad=ctrl_max,min_lift_m=lift,
                                hold_slip_m=slip,stable_seconds=stable,transport_ok=bool(transport_ok),
                                final_storage=state,warning_counts=[int(w.number) for w in e.data.warning],
                                renderer_created=e.renderer is not None))
            print(json.dumps(results[-1]),flush=True)
        finally:
            expert.close()
            mujoco.set_mju_user_warning(None)
        cursor += len(rows)
    passed = all(r['success'] and r['tolerance_pass'] for r in results)
    write_json(out/'replay.json',dict(pass_=passed,results=results,success_count=sum(r['success'] for r in results),
                                     episodes=len(results),frames=cursor,action_source='LeRobotDataset.hf_dataset action float32',
                                     state_tolerance_rad=CFG['replay_state_tolerance_rad'],
                                     block_tolerance_m=CFG['replay_block_tolerance_m'],
                                     wall_seconds=time.monotonic()-start,
                                     max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    assert passed
