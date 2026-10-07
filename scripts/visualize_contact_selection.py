"""Plot contact outcomes or replay a candidate in MuJoCo."""
import os
import sys
import tempfile
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'glfw' if '--show-viewer' in sys.argv else 'egl')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'contact-selection-mpl'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_selection.commands.visualize import main

if __name__ == '__main__':
    main()
