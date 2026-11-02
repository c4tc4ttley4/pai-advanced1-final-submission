"""Local STEP 4 recording. The STEP 3 controller/physics are called unchanged."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.step03_expert import ContactExpert, ROOT, load_config

IMAGE = 'observation.images.workspace'
STATE = 'observation.state'
CFG = json.loads((ROOT / 'configs/step04_dataset.json').read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def cpu_only():
    assert os.environ.get('MUJOCO_GL') == 'disable'
    assert not list(Path('/dev').glob('nvidia*'))


def sources():
    paths = ['src/step02_env.py', 'src/step03_expert.py', 'src/step04_dataset.py',
             'scripts/step04.py', 'src/step04_verify.py', 'scripts/step04_run.sh',
             'configs/step02.json', 'configs/step03.json',
             'configs/step04_dataset.json', 'scenes/step03/scene.xml',
             'scenes/step03/derivation.json', 'Dockerfile', 'compose.yaml',
             'compose.step02-cpu.yaml', 'pyproject.toml', 'uv.lock']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


class BoundedVideoDataset(LeRobotDataset):
    """Use official storage/statistics APIs, with a two-thread CPU encoder hook.

    v0.4.2's default AV1 encoder doesn't expose a thread count. This documented
    adapter overrides only the temporary-video encoding hook, not the format.
    The base API moves the MP4 and removes its temporary video directory.
    cleanup_episode_images handles PNGs after finalize and persisted reads.
    """
    def _encode_temporary_episode_video(self, video_key, episode_index):
        images = self._get_image_file_dir(episode_index, video_key)
        directory = Path(tempfile.mkdtemp(prefix='step04-video-', dir=self.root))
        video = directory / 'episode.mp4'
        subprocess.run([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-framerate', str(self.fps), '-i', str(images / 'frame-%06d.png'),
            '-an', '-c:v', 'libx264', '-threads', str(CFG['encoder_threads']),
            '-preset', 'fast', '-g', '2', '-crf', str(CFG['crf']),
            '-pix_fmt', 'yuv420p', str(video)
        ], check=True, timeout=90)
        return video


def open_dataset(root, create=False):
    if create:
        names = json.loads((ROOT / 'configs/step02.json').read_text())['joint_names']
        return BoundedVideoDataset.create(
            repo_id=CFG['repo_id'], root=root, fps=CFG['fps'],
            robot_type='xlerobot_fixed_left_model_R',
            features={STATE: {'dtype': 'float32', 'shape': (6,), 'names': names},
                      'action': {'dtype': 'float32', 'shape': (6,), 'names': names},
                      IMAGE: {'dtype': 'video', 'shape': (480, 640, 3),
                              'names': ['height', 'width', 'channels']}},
            image_writer_processes=0, image_writer_threads=0,
            batch_encoding_size=1, video_backend='torchcodec')
    return BoundedVideoDataset(repo_id=CFG['repo_id'], root=root, video_backend='torchcodec')


class RecordingExpert(ContactExpert):
    def _physics_step(self, request):
        # _advance has computed request but has not changed any MuJoCo state.
        # Snapshot at t_k, then let the original code apply sent and step once.
        e = self.env
        self.snapshot_time = float(e.data.time)
        self.snapshot_q64 = e.data.qpos[e.qids].copy()
        self.snapshot = e.observe(image=self.capture_images)
        assert float(e.data.time) == self.snapshot_time
        super()._physics_step(request)
        self.actual_ctrl = e.data.ctrl[e.aids].copy()


def integration_state(e):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(e.model, kind))
    mujoco.mj_getState(e.model, e.data, state, kind)
    return state


def cleanup_episode_images(root, trial):
    """Remove only this recorded episode's temporary PNGs after persisted reads.

    The stock encoder worker removes PNGs itself; our two-thread FFmpeg hook
    needs this explicit post-finalize cleanup. Video/data/meta are untouched.
    """
    ds = open_dataset(root)
    meta = ds.meta.episodes[trial]
    start, end = int(meta['dataset_from_index']), int(meta['dataset_to_index'])
    assert end > start
    for idx in [start, end-1]:
        item = ds[idx]
        assert tuple(item[IMAGE].shape) == (3,480,640)
        assert int(item['episode_index']) == trial
    path = ds._get_image_file_dir(trial, IMAGE)
    assert path.resolve().is_relative_to((Path(root)/'images').resolve())
    files = sorted(path.glob('frame-*.png'))
    assert len(files) == end-start
    assert all(p.is_file() and not p.is_symlink() for p in files)
    assert set(path.iterdir()) == set(files)
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest()
              for prefix in ['data','meta','videos']
              for p in (Path(root)/prefix).rglob('*') if p.is_file()}
    summary = dict(episode=trial, directory=str(path), frames=len(files),
                   bytes=sum(p.stat().st_size for p in files),
                   first_last_persisted_video_read=True,
                   removed_only_temporary_images=True)
    shutil.rmtree(path)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in before.items())
    summary['data_meta_video_hashes_unchanged'] = True
    return summary


def record(root, out, trial, smoke=False):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    attempt = out / f'attempt_{trial:02d}.json'
    if attempt.exists():
        raise RuntimeError('Existing attempt: choose a new run ID; no overwrite')
    assert 0 <= trial < 5
    if shutil.disk_usage(ROOT).free < CFG['min_free_bytes']:
        raise RuntimeError('disk reserve')
    ds = open_dataset(root, create=not root.exists())
    assert ds.num_episodes == trial or smoke
    expert = RecordingExpert()
    expert.capture_images = True
    e = expert.env
    offset = CFG['offsets_m'][trial]
    expert.reset(seed=CFG['seed'], offset=offset)
    initial = dict(integration=integration_state(e).tolist(),
                   qpos=e.data.qpos.tolist(), qvel=e.data.qvel.tolist(),
                   ctrl=e.data.ctrl.tolist(), time=float(e.data.time),
                   block_position=e.data.body('block').xpos.tolist(),
                   block_quaternion=e.data.body('block').xquat.tolist())
    record_info = dict(status='incomplete', trial=trial, seed=CFG['seed'],
                       offset_m=offset, initial=initial, sources=sources(),
                       smoke=smoke, dataset=str(root), frames=0)
    write_json(attempt, record_info)
    expert.start()
    start = time.monotonic()
    frames = 0
    anchors = {}
    max_rounding = 0.
    try:
        with gzip.open(out / f'trace_{trial:02d}.jsonl.gz', 'wt') as trace:
            while not expert.done:
                if time.monotonic() - start > CFG['episode_wall_limit_s']:
                    raise RuntimeError('recording wall limit')
                row = expert.tick()
                if row is None:
                    break
                if frames >= CFG['max_frames_per_episode']:
                    raise RuntimeError('frame limit')
                obs = expert.snapshot
                assert np.array_equal(obs[STATE], np.asarray(row['state_before'], np.float32))
                assert abs(row['time_before'] - frames / CFG['fps']) < 1e-8
                assert row['time_before'] == expert.snapshot_time
                assert abs(row['time'] - row['time_before'] - .02) < 1e-10
                assert np.array_equal(expert.actual_ctrl, np.asarray(row['sent']))
                action = expert.actual_ctrl.astype(np.float32)
                max_rounding = max(max_rounding, float(np.max(np.abs(action-expert.actual_ctrl))))
                if frames == 0:
                    from OpenGL import GL
                    record_info['opengl'] = {
                        name: (GL.glGetString(code) or b'').decode()
                        for name, code in [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER),
                                           ('version', GL.GL_VERSION)]}
                    assert os.environ.get('MUJOCO_GL') == 'egl'
                    assert 'NVIDIA' in record_info['opengl']['vendor']
                # v0.4.2 validates frame keys before consuming timestamp: use
                # its native frame_index/fps generation, checked against sim t.
                ds.add_frame({**obs, 'action': action, 'task': CFG['task']})
                # Small numeric diagnostics, no privileged feature in the dataset.
                row['frame_index'] = frames
                row['q_measured_float64'] = expert.snapshot_q64.tolist()
                trace.write(json.dumps(row) + '\n')
                if row['phase'] not in anchors:
                    anchors[row['phase']] = frames
                    from PIL import Image
                    path = out / f'anchor_{trial:02d}_{row["phase"]}.png'
                    Image.fromarray(obs[IMAGE]).save(path)
                frames += 1
                if frames % 100 == 0:
                    write_json(attempt, {**record_info, 'frames': frames})
                    print(json.dumps(dict(trial=trial, frames=frames, phase=row['phase'],
                                          wall_s=round(time.monotonic()-start, 2))), flush=True)
                    size = sum(p.stat().st_size for base in [root, out]
                               for p in base.rglob('*') if p.is_file())
                    if size > CFG['output_limit_bytes'] or shutil.disk_usage(ROOT).free < CFG['min_free_bytes']:
                        raise RuntimeError('disk/output limit')
                if smoke and frames == 5:
                    break
        success = expert.result['status'] == 'success'
        if not success and not smoke:
            raise RuntimeError('expert failed: ' + str(expert.result))
        capture_seconds = time.monotonic()-start
        expert.close()  # release EGL before CPU encoding
        encode_start = time.monotonic()
        ds.save_episode(parallel_encoding=False)
        ds.finalize()
        cleanup_info = cleanup_episode_images(root,trial)
        record_info.update(status='smoke_only' if smoke else 'success', frames=frames,
                           result=expert.result, phase_first_frame=anchors,
                           max_sent_float32_rounding_rad=max_rounding,
                           actual_ctrl_equals_sent_all_frames=True,
                           temporary_image_cleanup=cleanup_info,
                           capture_seconds=capture_seconds,
                           save_finalize_seconds=time.monotonic()-encode_start,
                           wall_seconds=time.monotonic()-start,
                           max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(attempt, record_info)
        print(json.dumps({k: record_info[k] for k in ['status','trial','frames','wall_seconds','max_rss_kib']}), flush=True)
    except BaseException as exc:
        record_info.update(status='failed_or_interrupted', frames=frames,
                           error=str(exc), result=expert.result)
        write_json(attempt, record_info)
        ds.finalize()
        raise
    finally:
        expert.close()


def numeric_check(out):
    cpu_only()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    e = RecordingExpert()
    e.capture_images = False
    e.start()
    checked = 0
    while checked < 10:
        row = e.tick()
        assert row is not None
        assert e.snapshot_time == row['time_before']
        assert np.array_equal(e.snapshot[STATE], np.array(row['state_before'], np.float32))
        assert np.array_equal(e.actual_ctrl, row['sent'])
        assert abs(e.env.data.time-(checked+1)*.02) < 1e-12
        checked += 1
    assert e.env.renderer is None
    names = e.env.names
    addresses = dict(qpos=e.env.qids.tolist(), dof=e.env.dids.tolist(), ctrl=e.env.aids.tolist())
    assert all(v == list(range(6)) for v in addresses.values())
    e.reset(seed=CFG['seed'], offset=(0.,0.))
    assert e.env.data.time == 0 and np.all(e.bias == 0) and not e.log
    assert e.stable_run == 0 and e.floor_gap == 0 and e.grip_gap == 0
    assert np.all(e.env.data.qvel == 0) and np.all(e.env.data.qacc_warmstart == 0)
    assert np.array_equal(e.env.data.ctrl, e.env.target)
    write_json(out/'numeric.json', dict(pass_=True, ticks=checked, physical_steps=checked*10,
                                       names=names, addresses=addresses, reset=True,
                                       sources=sources(), renderer_created=False))
    e.close()
