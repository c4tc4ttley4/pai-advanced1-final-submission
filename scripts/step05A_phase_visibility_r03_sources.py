"""Read only installed official source/config identities before implementing the r03 contract."""
import os,glob,json,hashlib,importlib.util
from pathlib import Path
assert os.getuid()!=0 and not glob.glob('/dev/nvidia*') and os.getenv('NVIDIA_VISIBLE_DEVICES')=='void'
ROOT=Path('/workspace');BASE=ROOT/'artifacts/step05A_phase_visibility_diag/phase_visibility_r03_20261101_01'
pkg=Path(importlib.util.find_spec('lerobot').origin).parent
out=BASE/'official_sources';out.mkdir(exist_ok=False)
files=sorted(set((pkg/'processor').glob('*.py'))|{pkg/'policies/act/modeling_act.py',pkg/'policies/act/configuration_act.py',pkg/'policies/factory.py',pkg/'configs/types.py',pkg/'utils/constants.py'})
records=[]
for p in files:
    b=p.read_bytes();rel=p.relative_to(pkg);q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
    records.append(dict(path=str(p),relative=str(rel),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
with (BASE/'official_source_identities.json').open('x') as f:json.dump(dict(files=records,forwards=0,model_loads=0,physics_steps=0),f,indent=2)
print(json.dumps(dict(source_files=len(records),bytes=sum(r['bytes'] for r in records),forwards=0)),flush=True)
