"""CPU-only import/syntax and immutable-source check, no model load/forward/physics."""
import os,glob,json,sys,hashlib
from pathlib import Path
assert os.getuid()!=0 and not glob.glob('/dev/nvidia*') and os.getenv('NVIDIA_VISIBLE_DEVICES')=='void'
ROOT=Path('/workspace');sys.path.insert(0,str(ROOT));BASE=ROOT/'artifacts/step05A_n_action_steps/naction20000_r02_20261101_01'
for name in ['step05A_naction20000_r02.py','step05A_naction20000_r02_review.py','step05A_naction20000_r02_preflight.py']:
    p=ROOT/'scripts'/name;compile(p.read_text(),str(p),'exec')
from scripts import step05A_naction20000_r02 as target
from src.step05A_act import setup,load_bundle,digest,IMAGE,STATE,resources
import torch
assert not torch.cuda.is_initialized()
C=json.loads((BASE/'spec.json').read_text());assert target.C['bundle']==C['bundle']
original=(ROOT/'scripts/step05A_naction1_naction1_20261101_01.py').read_text();current=(ROOT/'scripts/step05A_naction20000_r02.py').read_text()
assert original[original.index('def gate('):original.index("if __name__")].replace('result=dict(trial=trial,','result=dict(checkpoint=20000,trial=trial,')==current[current.index('def gate('):current.index("if __name__")]
with (BASE/'cpu_preflight.json').open('x') as f:json.dump(dict(pass_=True,actual_scripts_compile=True,actual_helper_import=True,cuda_initialized=False,model_loads=0,forwards=0,physics_steps=0,physical_and_control_functions_unchanged_from_protected_n1_adapter=True),f,indent=2)
print(json.dumps(dict(cpu_preflight=True,physics_steps=0,forwards=0)),flush=True)
