"""Minimal right-arm absolute-position control. All application Python runs in pai."""
import json
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class CleanupEnv:
    def __init__(self):
        self.config=json.loads((ROOT/'configs/step02.json').read_text())
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/step02/scene.xml'))
        self.data=mujoco.MjData(self.model)
        self.names=self.config['joint_names']
        self.jids=np.array([self.model.joint(n).id for n in self.names])
        self.qids=self.model.jnt_qposadr[self.jids]
        self.dids=self.model.jnt_dofadr[self.jids]
        self.aids=np.array([self.model.actuator(n).id for n in self.names])
        assert np.array_equal(self.model.actuator_trnid[self.aids,0],self.jids)
        self.bounds=np.array(self.config['action_range'])
        assert np.all(self.bounds[:,0]>=self.model.jnt_range[self.jids,0])
        assert np.all(self.bounds[:,1]<=self.model.jnt_range[self.jids,1])
        self.substeps=round(self.config['control_dt']/self.model.opt.timestep)
        self.renderer=None
        self.reset()

    def reset(self,seed=0,*,block_position=None):
        # No randomization yet; seed retained for the later API contract.
        self.seed=seed
        mujoco.mj_resetData(self.model,self.data)
        self.target=np.array(self.config['initial_action'],dtype=float)
        self.data.qpos[self.qids]=self.target
        self.data.ctrl[self.aids]=self.target
        if block_position is not None:
            # Explicit initial-condition override for contact testing only.
            adr=self.model.joint('block_free').qposadr[0]
            self.data.qpos[adr:adr+3]=block_position
        mujoco.mj_forward(self.model,self.data)
        return self.observe(image=False)

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(6,) or not np.isfinite(action).all(): raise ValueError('Expected 6 finite radians')
        if np.any(action<self.bounds[:,0]) or np.any(action>self.bounds[:,1]): raise ValueError('Action outside range')
        delta=self.config['max_target_speed_rad_s']*self.config['control_dt']
        self.target+=np.clip(action-self.target,-delta,delta)
        self.data.ctrl[self.aids]=self.target
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data)
            if not all(np.isfinite(v).all() for v in (self.data.qpos,self.data.qvel,self.data.qacc)):
                raise RuntimeError('Nonfinite state')
        mujoco.mj_forward(self.model,self.data)
        return self.observe(image=False)

    def observe(self,*,image=True):
        result={'observation.state':self.data.qpos[self.qids].astype(np.float32).copy()}
        if image:
            if self.renderer is None: self.renderer=mujoco.Renderer(self.model,height=480,width=640)
            self.renderer.update_scene(self.data,camera=self.config['camera']['name'])
            result['observation.images.workspace']=self.renderer.render().copy()
        return result

    def evaluation_info(self):
        return {'block_position':self.data.body('block').xpos.copy(),
                'tcp_position':self.data.site('tcp').xpos.copy(),
                'target_action':self.target.copy(),'time':float(self.data.time)}

    def close(self):
        if self.renderer is not None: self.renderer.close(); self.renderer=None
