"""Run inside the existing Docker/uv environment; no dependency synchronization."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.step04_dataset import numeric_check, record

if __name__ == '__main__':
    assert sys.prefix == '/opt/venv' and Path('/.dockerenv').exists()
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['numeric', 'smoke', 'record', 'validate', 'replay'])
    p.add_argument('--root')
    p.add_argument('--output', required=True)
    p.add_argument('--trial', type=int, default=0)
    a = p.parse_args()
    if a.mode == 'numeric':
        numeric_check(a.output)
    elif a.mode in ['smoke','record']:
        record(a.root, a.output, a.trial, smoke=a.mode == 'smoke')
    else:
        from src.step04_verify import validate, replay
        (validate if a.mode == 'validate' else replay)(a.root, a.output)
