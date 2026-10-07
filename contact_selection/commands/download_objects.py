"""Download pinned YCB scan meshes with a source/checksum manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '04d976499c328d621a312081acbe1dcd01b1eb6b'
REPOSITORY = 'https://github.com/kyouma9s/ycb_gazebo_sdf'
OBJECTS = ['003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
           '006_mustard_bottle', '008_pudding_box', '009_gelatin_box',
           '021_bleach_cleanser']


def download(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in OBJECTS:
        url = f'https://raw.githubusercontent.com/kyouma9s/ycb_gazebo_sdf/{REVISION}/{name}/nontextured.stl'
        path = output / f'{name}.stl'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            temporary = path.with_suffix('.partial')
            temporary.write_bytes(content)
            temporary.replace(path)
        entries.append(dict(name=name, file=path.name, url=url,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(name, path.stat().st_size, flush=True)
    (output / 'sources.json').write_text(json.dumps(dict(
        repository=REPOSITORY, revision=REVISION,
        upstream='https://www.ycbbenchmarks.com/',
        note='Scanned geometry only. Rigid-body mass/friction in validation are assumptions, not measured YCB properties.',
        objects=entries), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/contact_selection/assets/ycb'))
    args = p.parse_args()
    download(args.output)


if __name__ == '__main__':
    main()
