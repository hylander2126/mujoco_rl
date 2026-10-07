"""Contact selection: record a point, generate experiments, or fit the selector."""
import argparse
import importlib
import sys

COMMANDS = {
    'audit': 'audit', 'download-objects': 'download_objects', 'benchmark-objects': 'benchmark_objects',
    'select-cloud': 'select_cloud',
    'replay': 'replay', 'generate': 'generate', 'plot': 'visualize',
    'train': 'train', 'select': 'select', 'rerun': 'rerun',
    'probes': 'probes', 'off-axis': 'analyze_off_axis', 'demo': 'demo',
    'compare': 'compare', 'refeature': 'refeature', 'cloud-parity': 'cloud_parity', 'figures': 'figures', 'sim-compare': 'sim_compare', 'annotate': 'annotate',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    # Delegate command-specific help and validation to its own parser.
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f'{sys.argv[0]} {args.command}', *sys.argv[2:]]
    return importlib.import_module(f'contact_selection.commands.{COMMANDS[args.command]}').main()


if __name__ == '__main__':
    raise SystemExit(main())
