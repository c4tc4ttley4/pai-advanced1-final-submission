"""Deterministically derive MJCF; never edit or copy the upstream meshes."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build():
    c = json.loads((ROOT/'configs/step02.json').read_text())
    source = ROOT/'third_party/XLeRobot/simulation/mujoco/xlerobot.xml'
    tree = ET.parse(source)
    r = tree.getroot()
    r.set('model', 'step02_fixed_right_arm')
    r.find('compiler').set('meshdir', '../../third_party/XLeRobot/simulation/mujoco/assets')
    ET.SubElement(r, 'option', timestep=str(c['physics_dt']), gravity='0 0 -9.80665', integrator='implicitfast')
    ET.SubElement(r, 'visual')
    ET.SubElement(r.find('visual'), 'global', offwidth='640', offheight='480')
    removed=[]
    for parent in r.find('worldbody').iter():
        for j in list(parent):
            if j.tag == 'joint':
                if j.get('name') not in c['joint_names']:
                    removed.append(dict(j.attrib)); parent.remove(j)
                else:
                    j.set('damping', '1'); j.set('armature', '0.01')
    r.remove(r.find('tendon'))
    act = r.find('actuator')
    for a in list(act):
        if a.get('joint') not in c['joint_names']:
            act.remove(a)
        else:
            i = c['joint_names'].index(a.get('joint'))
            a.attrib.clear()
            a.attrib.update(name=c['joint_names'][i], joint=c['joint_names'][i], kp='80', kv='5',
                            ctrlrange=' '.join(map(str,c['action_range'][i])), forcerange='-35 35')
    # Explicitly name inherited collision meshes; preserve their contact masks.
    for i,g in enumerate(r.find('worldbody').iter('geom')):
        if not g.get('name'): g.set('name', f'robot_geom_{i}')
    torso = r.find(".//body[@name='chassis']/geom")
    torso.set('rgba', '0.2 0.2 0.6 0')  # Visibility change; contact decision follows separately.
    # The upstream coarse box overlaps a shoulder by 33.5 mm. Explicitly
    # disable this proxy contact separately from the visual alpha change.
    torso.set('contype','0'); torso.set('conaffinity','0')
    fixed = r.find(".//body[@name='Fixed_Jaw_2']")
    ET.SubElement(fixed, 'site', name='tcp', pos='0 -0.065 0', size='0.004', rgba='0 1 0 1')
    w=r.find('worldbody')
    ET.SubElement(w,'geom',name='floor',type='plane',size='2 2 0.05',rgba='0.15 0.18 0.20 1')
    def fmt(v): return ' '.join(map(str,v))
    def box(parent,name,pos,size,rgba,friction,**kw):
        return ET.SubElement(parent,'geom',name=name,type='box',pos=fmt(pos),size=fmt(size),rgba=rgba,
                             friction=fmt(friction),solref='0.01 1',**kw)
    t=c['table']; box(w,'table',t['center'],t['half_size'],'0.55 0.43 0.29 1',t['friction'])
    tr=c['tray']; tray=ET.SubElement(w,'body',name='tray',pos=fmt(tr['center']))
    hx,hy,hz=tr['half_size']; wt=tr['wall_half_thickness']; wh=tr['wall_height']
    box(tray,'tray_bottom',[0,0,0],[hx,hy,hz],'0.1 0.4 0.75 1',tr['friction'])
    for axis,sign,label in [(0,-1,'left'),(0,1,'right'),(1,-1,'front'),(1,1,'back')]:
        pos=[0,0,hz+wh/2]; size=[hx,hy,wh/2]
        pos[axis]=sign*([hx,hy][axis]-wt); size[axis]=wt
        box(tray,'tray_'+label,pos,size,'0.12 0.45 0.8 1',tr['friction'])
    b=c['block']; block=ET.SubElement(w,'body',name='block',pos=fmt(b['position']))
    ET.SubElement(block,'freejoint',name='block_free')
    box(block,'block_geom',[0,0,0],b['half_size'],'0.95 0.22 0.08 1',b['friction'],mass=str(b['mass']))
    cam=c['camera']; z=np.array(cam['position'])-cam['lookat']; z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    ET.SubElement(w,'camera',name=cam['name'],pos=fmt(cam['position']),xyaxes=fmt([*x,*y]),fovy=str(cam['fovy']))
    out=ROOT/'scenes/step02'; out.mkdir(parents=True,exist_ok=True)
    ET.indent(tree,space='  '); tree.write(out/'scene.xml',encoding='unicode')
    manifest={'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'removed_joints':removed,'fixed_pose':'all removed joint angles zero; rigid body tree',
              'retained_joints':c['joint_names'],'changes':['remove inactive actuators/tendons','name geoms',
              'right-arm damping/armature and position gains/ranges','torso alpha=0',
              'coarse torso proxy contact disabled due to 33.5mm shoulder overlap; other robot mesh contacts retained','tcp site',
              'task geometry, floor, fixed camera'],'mesh_copy':False}
    (out/'derivation.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': build()
