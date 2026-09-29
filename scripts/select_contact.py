"""Select a contact from a saved pre-action scene manifest without labels."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_selection.candidate_generator import Candidate
from contact_selection.selector import predict_and_select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='Trained model.json')
    parser.add_argument('scene', type=Path, help='Pre-action scene.json')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    model = json.loads(args.model.read_text())
    scene = json.loads(args.scene.read_text())
    candidates = [Candidate(**row) for row in scene['candidates']]
    result = predict_and_select(model, candidates, scene['geometry'], args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
