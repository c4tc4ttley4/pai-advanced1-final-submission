"""Contact-only expert. Privileged geometry stays outside CleanupEnv.observe()."""
import json
from pathlib import Path
import numpy as np
import mujoco
import time
from src.step02_env import CleanupEnv, ROOT


def load_config():
    return json.loads((ROOT/'configs/step03.json').read_text())


class ExpertEnv(CleanupEnv):
    def __init__(self,c):
        super().__init__()
        if c.get('scene'):
            model=mujoco.MjModel.from_xml_path(str(ROOT/c['scene']))
            assert (model.nq,model.nv,model.nu)==(self.model.nq,self.model.nv,self.model.nu)
            assert all(model.joint(n).id==j for n,j in zip(self.names,self.jids))
            self.model=model; self.data=mujoco.MjData(model); self.reset()


def contact_records(e):
    records=[]
    for i,k in enumerate(e.data.contact):
        force=np.zeros(6); mujoco.mj_contactForce(e.model,e.data,i,force)
        records.append(dict(geoms=[e.model.geom(k.geom1).name,e.model.geom(k.geom2).name],
                            distance_m=float(k.dist),force_contact_frame=force.tolist()))
    return records


def bilateral(contacts):
    return all(any('block_geom' in k['geoms'] and finger in k['geoms'] for k in contacts)
               for finger in ['robot_geom_41','robot_geom_44'])


def storage_state(e,c,contacts):
    """Oriented box containment, floor support, release and velocity, at 50Hz."""
    d=e.data; tr=e.config['tray']; half=np.array(e.config['block']['half_size'])
    center=d.body('block').xpos; R=d.body('block').xmat.reshape(3,3)
    extent=np.abs(R)@half
    inner=np.array(tr['half_size'][:2])-2*tr['wall_half_thickness']
    contained=bool(np.all(np.abs(center[:2]-tr['center'][:2])+extent[:2]<=inner))
    bottom=float(center[2]-extent[2]); floor=tr['center'][2]+tr['half_size'][2]
    floor_contact=any(set(k['geoms'])=={'tray_bottom','block_geom'} for k in contacts)
    wall_contact=any('block_geom' in k['geoms'] and any(g in ['tray_left','tray_right','tray_front','tray_back'] for g in k['geoms']) for k in contacts)
    robot_contact=any('block_geom' in k['geoms'] and any(g.startswith('robot_geom') for g in k['geoms']) for k in contacts)
    adr=int(e.model.joint('block_free').dofadr[0]); v=d.qvel[adr:adr+6]
    linear=float(np.linalg.norm(v[:3])); angular=float(np.linalg.norm(v[3:]))
    return dict(contained=contained,floor_contact=floor_contact,wall_contact=wall_contact,robot_contact=robot_contact,
                supported_height=bool(abs(bottom-floor)<=c['support_height_tolerance_m']),bottom_z_m=bottom,
                oriented_extent_m=extent.tolist(),inner_half_size_m=inner.tolist(),
                released=bool(not robot_contact and d.qpos[e.qids[-1]]>c['open_rad']-.05),
                linear_speed_m_s=linear,angular_speed_rad_s=angular,
                slow=bool(linear<=c['linear_speed_threshold_m_s'] and angular<=c['angular_speed_threshold_rad_s']))


class ContactExpert:
    """reset -> start -> tick (one 20ms action) -> done/result. No learned policy."""
    def __init__(self,c=None):
        self.c=load_config() if c is None else c
        self.env=ExpertEnv(self.c)
        self.reset()

    def reset(self,seed=None,offset=(0.,0.)):
        e=self.env; self.offset=np.array(offset,dtype=float)
        initial=np.array(e.config['block']['position']); initial[:2]+=self.offset
        obs=e.reset(seed=self.c['seed'] if seed is None else seed,block_position=initial)
        self.bias=np.zeros(6); self.log=[]; self.done=False; self.generator=None; self.phase='reset'
        self.result=dict(status='ready',offset_m=self.offset.tolist(),seed=e.seed,phases=[])
        self.warnings=[]; self.wall_start=time.monotonic(); self.stable_run=0.; self.floor_gap=0.; self.grip_gap=0.
        self.dt=e.config['control_dt']; self.local=np.array(self.c['grasp_point_local_m'])
        self.q=e.target.copy(); self.fixed_ids=[i for i in range(1,e.model.nbody) if e.model.body_dofnum[i]==0 and not e.model.body(i).name.endswith('_2')]
        # Select all bodies not descending from the movable right arm or free block.
        right=e.model.body('Rotation_Pitch_R').id; block=e.model.body('block').id
        def moving(i):
            while i:
                if i in (right,block): return True
                i=e.model.body_parentid[i]
            return False
        self.fixed_ids=[i for i in range(1,e.model.nbody) if not moving(i)]
        self.fixed_pos=e.data.xpos[self.fixed_ids].copy(); self.fixed_quat=e.data.xquat[self.fixed_ids].copy()
        self.audit=dict(physics_samples=0,max_fixed_position_drift_m=0.,max_fixed_quaternion_drift=0.,
                        min_contact_distance_m=0.,robot_environment_contacts=0,max_target_speed_rad_s=0.,
                        max_measured_speed_rad_s=0.,max_tracking_error_rad=0.,torso_distance_by_geom_m={},
                        max_finger_normal_force_N=0.,min_pre_release_block_z_m=None)
        return obs

    def start(self,*,lift_only=False):
        if self.generator is not None: raise RuntimeError('reset before restarting')
        self.result['status']='running'; self.generator=self._run(lift_only)

    def _physics_step(self,request):
        # Same saturation and substeps as STEP 2; audit every 2ms, without rendering.
        e=self.env; m=e.model; d=e.data
        if request.shape!=(6,) or not np.isfinite(request).all() or np.any(request<e.bounds[:,0]) or np.any(request>e.bounds[:,1]):
            raise ValueError('invalid_action')
        previous=e.target.copy(); delta=e.config['max_target_speed_rad_s']*self.dt
        e.target+=np.clip(request-e.target,-delta,delta); d.ctrl[e.aids]=e.target
        self.audit['max_target_speed_rad_s']=max(self.audit['max_target_speed_rad_s'],float(np.max(np.abs(e.target-previous))/self.dt))
        for _ in range(e.substeps):
            mujoco.mj_step(m,d); self.audit['physics_samples']+=1
            if not all(np.isfinite(v).all() for v in (d.qpos,d.qvel,d.qacc)): raise RuntimeError('nonfinite_state')
            if any(w.number for w in d.warning): raise RuntimeError('physics_warning')
            self.audit['max_fixed_position_drift_m']=max(self.audit['max_fixed_position_drift_m'],float(np.max(np.abs(d.xpos[self.fixed_ids]-self.fixed_pos))))
            self.audit['max_fixed_quaternion_drift']=max(self.audit['max_fixed_quaternion_drift'],float(np.max(np.abs(d.xquat[self.fixed_ids]-self.fixed_quat))))
            for k in d.contact:
                self.audit['min_contact_distance_m']=min(self.audit['min_contact_distance_m'],float(k.dist))
                names=[m.geom(k.geom1).name,m.geom(k.geom2).name]
                if any(n.startswith('robot_geom') for n in names):
                    if any(n=='table' or n.startswith('tray_') for n in names): self.audit['robot_environment_contacts']+=1
                    if k.dist < -self.c['max_contact_penetration_m']: raise RuntimeError('deep_robot_contact:'+str(names))
        mujoco.mj_forward(m,d)
        if self.audit['max_fixed_position_drift_m']>1e-12 or self.audit['max_fixed_quaternion_drift']>1e-12: raise RuntimeError('fixed_body_drift')

    def _advance(self,q,phase):
        e=self.env; d=e.data; c=self.c; self.phase=phase
        if d.time>=c['trial_timeout_sim_seconds']: raise RuntimeError('trial_timeout')
        if time.monotonic()-self.wall_start>c['batch_timeout_wall_seconds']: raise RuntimeError('wall_timeout')
        before=e.observe(image=False)['observation.state'].tolist(); t=float(d.time)
        self.bias[:5]=np.clip(self.bias[:5]+.03*(q[:5]-d.qpos[e.qids[:5]]),-.05,.05)
        request=np.clip(q+self.bias,e.bounds[:,0],e.bounds[:,1])
        if phase in ['close','lift','hold','transport','lower']:
            delta=c['contact_squeeze_rad'] if bilateral(contact_records(e)) else c['free_close_increment_rad']
            request[-1]=max(c['close_rad'],float(d.qpos[e.qids[-1]])-delta)
        self._physics_step(request)
        ct=contact_records(e); point,axis=grasp_pose(e.model,d,self.local)
        state=storage_state(e,c,ct)
        supported=state['floor_contact']
        self.floor_gap=0. if supported else self.floor_gap+self.dt
        valid=state['contained'] and state['supported_height'] and state['released'] and state['slow'] and not state['wall_contact'] and self.floor_gap<=c['floor_contact_gap_seconds']
        self.stable_run=self.stable_run+self.dt if valid and phase=='stable' else 0.
        if phase in ['transport','lower']:
            self.grip_gap=0. if bilateral(ct) else self.grip_gap+self.dt
            if self.grip_gap>c['grip_contact_gap_seconds'] or np.linalg.norm(d.body('block').xpos-point)>c['max_transport_relative_distance_m']:
                raise RuntimeError('dropped_before_release')
            z=float(d.body('block').xpos[2]); old=self.audit['min_pre_release_block_z_m']
            self.audit['min_pre_release_block_z_m']=z if old is None else min(old,z)
            if any('block_geom' in k['geoms'] and any(g=='table' or g.startswith('tray_') for g in k['geoms']) for k in ct):
                raise RuntimeError('premature_environment_support')
        for k in ct:
            if 'block_geom' in k['geoms'] and any(n in k['geoms'] for n in ['robot_geom_41','robot_geom_44']):
                self.audit['max_finger_normal_force_N']=max(self.audit['max_finger_normal_force_N'],k['force_contact_frame'][0])
        for i in range(32,45):
            name=f'robot_geom_{i}'; dist=float(mujoco.mj_geomDistance(e.model,d,e.model.geom('robot_geom_0').id,e.model.geom(name).id,2.,None))
            self.audit['torso_distance_by_geom_m'][name]=min(self.audit['torso_distance_by_geom_m'].get(name,2.),dist)
        self.audit['max_measured_speed_rad_s']=max(self.audit['max_measured_speed_rad_s'],float(np.max(np.abs(d.qvel[e.dids]))))
        self.audit['max_tracking_error_rad']=max(self.audit['max_tracking_error_rad'],float(np.max(np.abs(e.target-d.qpos[e.qids]))))
        adr=int(e.model.joint('block_free').dofadr[0])
        row=dict(time_before=t,time=float(d.time),phase=phase,state_before=before,expert_goal=q.tolist(),
                 request=request.tolist(),sent=e.target.tolist(),measured=d.qpos[e.qids].tolist(),block=d.body('block').xpos.tolist(),
                 block_quat=d.body('block').xquat.tolist(),point=point.tolist(),axis=axis.tolist(),block_velocity=d.qvel[adr:adr+6].tolist(),
                 contacts=ct,storage=state,stable_seconds=self.stable_run)
        self.log.append(row)
        yield row

    def _move(self,name,xyz=None,jaw=None,seconds=1.,axis=None):
        e=self.env; d=e.data; c=self.c; q=self.q.copy(); info={}; phase_start=float(d.time)
        if xyz is not None:
            start_point,start_axis=grasp_pose(e.model,d,self.local)
            final_axis=np.array([0,0,1] if axis is None else axis,dtype=float); final_axis/=np.linalg.norm(final_axis)
            duration=max(np.linalg.norm(np.array(xyz)-start_point)/c['cartesian_speed_m_s'],np.linalg.norm(final_axis-start_axis)/c['axis_speed_per_s'],self.dt)
            for fraction in np.linspace(0,1,max(2,int(duration/self.dt))):
                target=start_point+fraction*(np.array(xyz)-start_point)
                direction=start_axis+fraction*(final_axis-start_axis)
                if axis is not None:
                    radial=d.body('Rotation_Pitch_R').xpos[:2]-target[:2]; radial/=np.linalg.norm(radial)
                    direction[:2]=radial*np.linalg.norm(direction[:2])
                q,info=solve_grasp(e,target,q,c,axis=direction)
                if info['position_residual_m']>c['ik_position_tolerance_m'] or info['axis_residual']>c['ik_axis_tolerance']:
                    raise RuntimeError('ik_infeasible:'+name+':'+str(info))
                if d.time-phase_start>=c['phase_timeout_seconds']: raise RuntimeError('phase_timeout:'+name)
                yield from self._advance(q,name)
        if jaw is not None: q[-1]=jaw
        start=float(d.time); reached=False
        for _ in range(round(c['phase_timeout_seconds']/self.dt)):
            if d.time-phase_start>=c['phase_timeout_seconds']: raise RuntimeError('phase_timeout:'+name)
            yield from self._advance(q,name)
            reached=np.max(np.abs(d.qpos[e.qids[:5]]-q[:5]))<c['arm_tracking_tolerance_rad'] and np.max(np.abs(e.target[:5]-(q+self.bias)[:5]))<.001
            if reached and d.time-start>=seconds: break
        if not reached: raise RuntimeError('phase_timeout:'+name)
        for _ in range(round(.4/self.dt)):
            if d.time-phase_start>=c['phase_timeout_seconds']: raise RuntimeError('phase_timeout:'+name)
            yield from self._advance(q,name)
        self.q=q
        point,_=grasp_pose(e.model,d,self.local)
        self.result['phases'].append(dict(name=name,q_rad=q.tolist(),ik=info,point=point.tolist(),block=d.body('block').xpos.tolist(),time=float(d.time)))

    def _run(self,lift_only):
        e=self.env; c=self.c
        yield from self._move('settle_open',jaw=c['open_rad'])
        block=e.data.body('block').xpos.copy(); self.result['settled_block']=block.tolist()
        yield from self._move('approach',block+[0,0,c['approach_height_m']])
        yield from self._move('descend',block+[0,0,c['grasp_height_offset_m']])
        yield from self._move('close',jaw=c['close_rad'],seconds=3.)
        yield from self._move('lift',block+[0,0,c['lift_height_m']],seconds=2.)
        yield from self._move('hold',seconds=c['hold_seconds'])
        h=[r for r in self.log if r['phase']=='hold']
        lift=min(r['block'][2]-block[2] for r in h)
        relative=np.array([np.array(r['block'])-r['point'] for r in h]); slip=float(np.linalg.norm(relative-relative[0],axis=1).max())
        both=all(bilateral(r['contacts']) for r in h)
        no_support=all(not any('block_geom' in k['geoms'] and any(g=='table' or g.startswith('tray_') for g in k['geoms']) for k in r['contacts']) for r in h)
        self.result['grasp']=dict(min_lift_m=lift,hold_seconds=len(h)*self.dt,max_relative_slip_m=slip,bilateral_contact_all_samples=both,no_environment_support=no_support)
        if lift<c['min_lift_m'] or slip>c['max_hold_slip_m'] or not both or not no_support: raise RuntimeError('grasp_not_held')
        if lift_only: self.result['status']='lift_pass'; return
        pos=np.array(c['tray_target_m']); axis=c['transport_axis']
        yield from self._move('transport',pos+[0,0,.04],axis=axis,seconds=2.)
        yield from self._move('lower',pos,axis=axis)
        yield from self._move('release',jaw=c['open_rad'],seconds=1.5)
        yield from self._move('retreat',pos+[0,0,.08],axis=axis)
        self.floor_gap=0.; self.stable_run=0.
        yield from self._move('stable',seconds=c['stable_seconds']+.5)
        self.result['stable_seconds']=self.stable_run
        self.result['final_storage']=storage_state(e,c,contact_records(e))
        if self.stable_run+1e-9<c['stable_seconds']: raise RuntimeError('storage_not_stable')
        self.result['status']='success'

    def tick(self):
        if self.generator is None or self.done: raise RuntimeError('start a reset episode first')
        try: return next(self.generator)
        except StopIteration: self.done=True
        except Exception as exc:
            self.done=True; self.result.update(status='fail',reason=str(exc),failed_phase=self.phase)
        if self.done:
            self.result.update(sim_seconds=float(self.env.data.time),wall_seconds=time.monotonic()-self.wall_start,
                               audit=self.audit,warning_counts=[int(w.number) for w in self.env.data.warning])
        return None

    def close(self):
        self.env.close()


def grasp_pose(model, data, local):
    body=data.body('Fixed_Jaw_2')
    R=body.xmat.reshape(3,3)
    return body.xpos+R@local, R[:,1].copy()


def solve_grasp(e, xyz, seed, c, axis=None):
    """Scratch-data damped IK: position and local +Y pointing up; roll is free."""
    m=e.model; s=mujoco.MjData(m); s.qpos[:]=e.data.qpos
    q=np.array(seed,dtype=float).copy(); local=np.array(c['grasp_point_local_m'])
    desired_axis=np.array([0,0,1] if axis is None else axis,dtype=float)
    desired_axis/=np.linalg.norm(desired_axis)
    # At vertical inclination, roll is redundant. Select a consistent branch
    # before grasping instead of letting numerical drift choose the joint limit.
    preferred=c.get('ik_preferred_topdown_roll_rad')
    choose_roll=preferred is not None and desired_axis[2]>.9999
    def residual(q):
        s.qpos[e.qids]=q; mujoco.mj_forward(m,s)
        p,axis=grasp_pose(m,s,local)
        err=np.r_[np.asarray(xyz)-p, .08*(desired_axis-axis)]
        return np.r_[err,.003*(preferred-q[4])] if choose_roll else err
    for iteration in range(c['ik_iterations']):
        err=residual(q)
        if np.linalg.norm(err[:3])<c['ik_position_tolerance_m'] and np.linalg.norm(err[3:6])/.08<c['ik_axis_tolerance'] and (not choose_roll or abs(q[4]-preferred)<.02): break
        J=np.empty((len(err),5))
        for k in range(5):
            z=q.copy(); z[k]+=1e-5; J[:,k]=(err-residual(z))/1e-5
        dq=np.linalg.solve(J.T@J+1e-5*np.eye(5),J.T@err)
        dq*=min(1.,.15/max(np.max(np.abs(dq)),1e-12))
        for scale in [1.,.5,.25,.125,.0625]:
            candidate=q.copy(); candidate[:5]+=scale*dq
            candidate[:5]=np.clip(candidate[:5],e.bounds[:5,0]+.001,e.bounds[:5,1]-.001)
            if np.linalg.norm(residual(candidate))<np.linalg.norm(err): q=candidate; break
        else: break
    err=residual(q)
    info={'iterations':iteration+1,'position_residual_m':float(np.linalg.norm(err[:3])), 'axis_residual':float(np.linalg.norm(err[3:6])/.08),'q_rad':q.tolist()}
    return q,info
