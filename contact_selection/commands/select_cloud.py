"""Produce a press proposal from segmented Z-up point clouds, without moving a robot."""
import argparse
import json
from pathlib import Path
import numpy as np

from contact_selection.selection.robust_press import select_press
from contact_selection.sim.dataset import json_value, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('clouds', type=Path, nargs='+', help='Registered (N,3) .npy arrays in metres')
    p.add_argument('--table-z', type=float, required=True)
    p.add_argument('--com-xy', type=float, nargs=2, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    points = np.concatenate([np.load(path, allow_pickle=False) for path in args.clouds])
    result = select_press(points, table_z=args.table_z, com_xy=args.com_xy)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(json_value(result), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
