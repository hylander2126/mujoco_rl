"""Simulation-supervised selection of a contact for a fixed press-pull controller."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'glfw' if '--show-viewer' in sys.argv else 'egl')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'contact-selection-mpl'))
