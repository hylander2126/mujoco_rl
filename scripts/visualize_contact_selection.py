"""Plot the first-milestone labels or replay a candidate in MuJoCo."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_selection.visualize import main

if __name__ == '__main__':
    main()
