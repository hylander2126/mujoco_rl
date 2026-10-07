"""Run with Python from any working directory."""
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_selection.commands.analyze_off_axis import main

if __name__ == '__main__':
    main()
