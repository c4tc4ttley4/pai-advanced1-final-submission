"""Derive contact-only finger pairs, preserving STEP 2 and upstream originals."""
from pathlib import Path
import json, hashlib, xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'scenes/step02/scene.xml'
tree=ET.parse(source); r=tree.getroot(); r.set('model','step03_contact_expert')
r.find('option').set('cone','elliptic')
r.find('option').set('noslip_iterations','5')
contact=ET.SubElement(r,'contact')
for finger in ['robot_geom_41','robot_geom_44']:
    ET.SubElement(contact,'pair',geom1='block_geom',geom2=finger,condim='4',
                  solref='0.004 1',solimp='0.95 0.99 0.001',friction='1 1 0.005 0.0001 0.0001')
out=ROOT/'scenes/step03'; out.mkdir(exist_ok=True)
ET.indent(tree,space='  '); tree.write(out/'scene.xml',encoding='unicode')
(out/'derivation.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    changes=[dict(geoms=['block_geom',f],solref_before='mixed: 0.015 1',solref_after='0.004 1',solimp_before='0.9 0.95 0.001 0.5 2',solimp_after='0.95 0.99 0.001 0.5 2',condim_before=3,condim_after=4,friction='sliding 1 unchanged; existing torsion coefficient 0.005 m enabled') for f in ['robot_geom_41','robot_geom_44']],
    solver_changes=dict(cone_before='pyramidal',cone_after='elliptic',noslip_iterations_before=0,noslip_iterations_after=5),
    reason='dev03: closing to zero produced 13mm penetration; dev05: bounded squeezing lifted then slipped/rotated out; dev08/09: soft tangential drift caused early drop. Stiffer finite-pad contacts plus torsional friction and no-slip postprocessing; no geometry, mass, gravity, force gains, contact mask or sliding friction change.',mesh_copy=False),indent=2)+'\n')
