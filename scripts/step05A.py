"""Run only inside the existing Docker + uv environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.step05A_act import main

if __name__ == '__main__':
    main()
