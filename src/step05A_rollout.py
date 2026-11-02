"""Live ACT-only action generator and separate physical task observer."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
import torch
from PIL import Image
from src.step03_expert import ExpertEnv, load_config, contact_records, bilateral, grasp_pose, storage_state
from src.step05A_act import (CFG, ROOT, IMAGE, STATE, paths, load_bundle, write_json,
                            resources, digest, sha)


class PhysicalObserver:
    """Read-only diagnostics; no phase and no output used to choose actions."""
    def __init__(self, env, config):
        self.env, self.config = env, config
        self.baseline_z = float(env.data.body('block').xpos[2])
        self.hold = self.max_hold = self.stable = self.max_stable = self.floor_gap = 0.
        self.origin = None
        self.grasped = self.lifted = self.stored = False
        self.max_lift = 0.
        self.max_slip_in_valid_hold = 0.

    def observe(self):
        e, c = self.env, self.config
        contacts = contact_records(e)
        block = e.data.body('block').xpos.copy()
        point, _ = grasp_pose(e.model, e.data, np.array(c['grasp_point_local_m']))
        if e.data.time <= 1.:
            self.baseline_z = min(self.baseline_z, float(block[2]))
        lift = float(block[2]-self.baseline_z)
        self.max_lift = max(self.max_lift, lift)
        self.lifted |= lift >= c['min_lift_m']
        no_support = not any('block_geom' in x['geoms'] and
                             any(g == 'table' or g.startswith('tray_') for g in x['geoms'])
                             for x in contacts)
        both = bilateral(contacts)
        relative = block-point
        valid = both and no_support and lift >= c['min_lift_m']
        if valid:
            if self.origin is None:
                self.origin = relative.copy()
            slip = float(np.linalg.norm(relative-self.origin))
            if slip > c['max_hold_slip_m']:
                self.hold, self.origin = 0., relative.copy()
            else:
                self.hold += .02
                self.max_slip_in_valid_hold = max(self.max_slip_in_valid_hold, slip)
        else:
            self.hold, self.origin = 0., None
        self.max_hold = max(self.max_hold, self.hold)
        self.grasped |= self.hold+1e-9 >= c['hold_seconds']
        state = storage_state(e, c, contacts)
        self.floor_gap = 0. if state['floor_contact'] else self.floor_gap+.02
        storage_valid = (state['contained'] and state['supported_height'] and state['released']
                         and state['slow'] and not state['wall_contact']
                         and self.floor_gap <= c['floor_contact_gap_seconds'])
        self.stable = self.stable+.02 if storage_valid else 0.
        self.max_stable = max(self.max_stable, self.stable)
        self.stored |= self.stable+1e-9 >= c['stable_seconds']
        return dict(block=block.tolist(), bilateral_contact=both, no_environment_support=no_support,
                    lift_m=lift, grasp_hold_seconds=self.hold, stable_seconds=self.stable,
                    storage=state)


def rollout(run, trial, checkpoint='final'):
    assert 0 <= trial < CFG['max_rollouts']
    out, model = paths(run)
    directory = out / f'rollout_{trial:02d}'
    directory.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    bundle = model / checkpoint
    policy, pre, post = load_bundle(bundle)
    learned = json.loads((bundle / 'training_result.json').read_text())
    assert digest(policy) == learned['after_model_digest']
    audit = json.loads((out / 'data_audit.json').read_text())
    jump_limit = np.array(audit['action_jump_limit_rad'])
    c = load_config()
    e = ExpertEnv(c)  # No ContactExpert instance, tick, _advance, or expert action.
    assert e.names == CFG['joint_names']
    assert e.substeps == 10 and e.model.opt.timestep == .002
    offsets = [[0., 0.], [0., 0.], [.005, 0.]]
    initial_block = np.array(e.config['block']['position'])
    initial_block[:2] += offsets[trial]
    e.reset(seed=CFG['seed'], block_position=initial_block)
    policy.reset()
    initial_q = e.data.qpos[e.qids].copy()
    previous = e.data.ctrl[e.aids].copy()
    observer = PhysicalObserver(e, c)
    warning_messages = []
    mujoco.set_mju_user_warning(lambda message: warning_messages.append(str(message)))
    encoder = None
    ticks = predictions = video_frames = physical_steps = 0
    moved = max_ctrl_error = 0.
    inference_ms, new_prediction_ms = [], []
    milestones, stop_reason, status = [], 'simulation_time_limit', 'bounded_rollout_finished'
    last_diagnostic = None
    gl = None
    exception = None
    try:
        encoder = subprocess.Popen([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '640x480',
            '-framerate', '10', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
            '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
            str(directory / 'diagnostic_10fps.mp4')], stdin=subprocess.PIPE,
            stderr=(directory / 'encoder_stderr.txt').open('wb'))
        with gzip.open(directory / 'trace.jsonl.gz', 'xt') as trace, torch.inference_mode():
            for tick in range(CFG['rollout_max_ticks']):
                assert abs(e.data.time-tick/50) < 1e-8
                before = float(e.data.time)
                capture_wall = time.monotonic()
                raw = e.observe(image=True)
                assert float(e.data.time) == before
                rgb = raw[IMAGE]
                if gl is None:
                    from OpenGL import GL
                    gl = {name: (GL.glGetString(code) or b'').decode()
                          for name,code in [('vendor',GL.GL_VENDOR), ('renderer',GL.GL_RENDERER)]}
                    assert 'NVIDIA' in gl['vendor']
                    assert os.environ.get('MUJOCO_EGL_DEVICE_ID', '0') == '0'
                if tick % 5 == 0:
                    encoder.stdin.write(rgb.tobytes())
                    video_frames += 1
                if tick in [0, 4, 99, 499, 999, 1499, 1999, 2499]:
                    Image.fromarray(rgb).save(directory / f'observation_{tick:04d}.png')
                live = {IMAGE: torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).unsqueeze(0),
                        STATE: torch.from_numpy(raw[STATE]).unsqueeze(0)}
                assert set(live) == {IMAGE, STATE}
                new_prediction = len(policy._action_queue) == 0
                t = time.monotonic()
                normalized = policy.select_action(pre(live))
                rad = post(normalized).squeeze(0).numpy().copy()
                ms = (time.monotonic()-t)*1000
                inference_ms.append(ms)
                if new_prediction:
                    predictions += 1
                    new_prediction_ms.append(ms)
                row = dict(tick=tick, time_before=before, image_sim_time=before,
                           image_wall_capture_offset_s=capture_wall-start,
                           image_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest(),
                           state_before=raw[STATE].tolist(),
                           policy_input_keys=[IMAGE,STATE], action_rad=rad.tolist(),
                           action_normalized=normalized.squeeze(0).cpu().tolist(),
                           previous_ctrl=previous.tolist(),
                           source='new_ACT_prediction' if new_prediction else 'ACT_queue',
                           queue_remaining=len(policy._action_queue), inference_ms=ms)
                bad = None
                if rad.shape != (6,) or not np.isfinite(rad).all():
                    bad = 'nonfinite_or_wrong_shape_action'
                elif np.any(rad < e.bounds[:,0]) or np.any(rad > e.bounds[:,1]):
                    bad = 'action_outside_joint_actuator_range'
                elif np.any(np.abs(rad-previous) > jump_limit):
                    bad = 'action_jump_exceeds_predeclared_limit'
                if bad:
                    row.update(applied=False, stop_reason=bad, ctrl=e.data.ctrl[e.aids].tolist())
                    trace.write(json.dumps(row)+'\n')
                    status, stop_reason = 'safety_stop_before_action', bad
                    break
                # Direct sent-space command. No compensation, clipping or second queue.
                e.target[:] = rad
                e.data.ctrl[e.aids] = rad
                max_ctrl_error = max(max_ctrl_error, float(np.max(np.abs(e.data.ctrl[e.aids]-rad))))
                previous = rad.copy()
                physics_bad = None
                for _ in range(e.substeps):
                    mujoco.mj_step(e.model, e.data)
                    physical_steps += 1
                    if not all(np.isfinite(x).all() for x in [e.data.qpos,e.data.qvel,e.data.qacc]):
                        physics_bad = 'nonfinite_physics_state'
                    elif warning_messages or any(w.number for w in e.data.warning):
                        physics_bad = 'mujoco_warning'
                    for contact in e.data.contact:
                        names = [e.model.geom(contact.geom1).name, e.model.geom(contact.geom2).name]
                        if any(n.startswith('robot_geom') for n in names) and contact.dist < -c['max_contact_penetration_m']:
                            physics_bad = 'deep_robot_contact:' + str(names)
                    if physics_bad:
                        break
                mujoco.mj_forward(e.model, e.data)
                last_diagnostic = observer.observe()
                moved = max(moved, float(np.max(np.abs(e.data.qpos[e.qids]-initial_q))))
                row.update(applied=True, ctrl=e.data.ctrl[e.aids].tolist(),
                           time_after=float(e.data.time), state_after=e.data.qpos[e.qids].tolist(),
                           diagnostic=last_diagnostic)
                ticks += 1
                if physics_bad:
                    status, stop_reason = 'safety_stop_after_physics', physics_bad
                    row['stop_reason'] = physics_bad
                else:
                    assert abs(e.data.time-before-.02) < 1e-10
                trace.write(json.dumps(row)+'\n')
                if ticks in CFG['rollout_stages_ticks']:
                    milestones.append(dict(ticks=ticks, sim_seconds=float(e.data.time),
                                           max_measured_movement_rad=moved))
                    print(json.dumps(dict(milestone=milestones[-1])), flush=True)
                if physics_bad:
                    break
                if observer.grasped and observer.stored:
                    stop_reason = 'physical_task_success'
                    break
                if time.monotonic()-start > 550:
                    status, stop_reason = 'wall_limit', 'rollout_550_seconds'
                    break
                if tick % 100 == 0:
                    print(json.dumps(dict(tick=tick, sim_seconds=float(e.data.time),
                                          new_predictions=predictions, movement_rad=moved)), flush=True)
    except BaseException as exc:
        status, stop_reason = 'error_or_interrupted', repr(exc)
        exception = exc
    finally:
        if encoder is not None:
            encoder.stdin.close()
            try:
                encoder_status = encoder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                encoder.terminate()
                encoder_status = encoder.wait(timeout=5)
        else:
            encoder_status = None
        if ticks:
            final_rgb = e.observe(image=True)[IMAGE]
            Image.fromarray(final_rgb).save(directory / 'final_observation.png')
        summary = dict(trial=trial, status=status, termination_reason=stop_reason,
                       offset_m=offsets[trial], seed=CFG['seed'],
                       model_weights_digest=learned['after_model_digest'],
                       checkpoint=checkpoint, bundle_manifest_sha256=sha(bundle / 'bundle.sha256.json'),
                       ticks_applied=ticks, physical_steps=physical_steps, sim_seconds=float(e.data.time),
                       new_predictions=predictions, queue_actions=max(0,len(inference_ms)-predictions),
                       input_image_hz=50, control_hz=50, replan_every_ticks=10,
                       max_action_ctrl_difference_rad=max_ctrl_error, max_measured_movement_rad=moved,
                       connection_pass=bool(ticks >= 100 and moved > 1e-4 and max_ctrl_error == 0 and not warning_messages),
                       grasp_success=bool(observer.grasped), lift_observed=bool(observer.lifted),
                       storage_success=bool(observer.stored), max_lift_m=observer.max_lift,
                       max_valid_grasp_hold_seconds=observer.max_hold,
                       max_storage_stable_seconds=observer.max_stable,
                       final_diagnostic=last_diagnostic, milestones=milestones,
                       safety_interventions=0, actions_clipped=0, compensation_applied=False,
                       safety_stop=status.startswith('safety_stop'), jump_limit_rad=jump_limit.tolist(),
                       warnings=warning_messages, warning_counts=[int(w.number) for w in e.data.warning],
                       inference_ms={name: (dict(mean=float(np.mean(values)),p50=float(np.quantile(values,.5)),
                                                 p95=float(np.quantile(values,.95)),max=float(np.max(values))) if values else None)
                                     for name,values in [('all_ticks',inference_ms),('new_predictions',new_prediction_ms)]},
                       opengl=gl, video_frames=video_frames, video_fps=10, video_encoder_status=encoder_status,
                       resources=resources(), wall_seconds=time.monotonic()-start,
                       real_time_50hz_achievement_claimed=False, visual_review='pending')
        write_json(directory / 'result.json', summary)
        print(json.dumps(summary), flush=True)
        e.close()
        mujoco.set_mju_user_warning(None)
    if exception is not None:
        raise exception
