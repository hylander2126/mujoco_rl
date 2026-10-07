#!/usr/bin/env python3
"""Monte Carlo / sensitivity / NLS-covariance analysis of the estimator and contact selection."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

if __name__ == '__main__':
    from uncertainty.run import main
    raise SystemExit(main())
