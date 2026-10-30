"""Download the released TDTL checkpoints / features and verify their checksums.

Artifacts are hosted as GitHub Release assets of this repository (files are
larger than the 100 MB git limit), so they are fetched over the GitHub API.

Usage (from the repository root):
    python scripts/download_weights.py             # everything below
    python scripts/download_weights.py iemocap     # IEMOCAP fusion checkpoint
    python scripts/download_weights.py meld        # MELD fusion checkpoint
    python scripts/download_weights.py features   # stage-1 IEMOCAP features

Files land in:
    IEMOCAP/IEMOCAP/save_model/multimodal_fusion_best.bin   (iemocap)
    MELD/MELD/save_model/multimodal_fusion_best.bin         (meld)
    IEMOCAP/feature/{train,dev,test}_features.pkl          (features)
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

REPO = 'wsfwb/tdtl-anonymous'
API = f'https://api.github.com/repos/{REPO}/releases'

# md5 -> destination (relative to repo root)
ASSETS = {
    'iemocap_fusion.bin': (
        '605afd8f73a356b09386fe40adbe0be5',
        'IEMOCAP/IEMOCAP/save_model/multimodal_fusion_best.bin',
        'IEMOCAP stage-2 fusion checkpoint (paper: 73.81 Acc / 73.83 WF1)',
    ),
    'iemocap_train_features.pkl': (
        'b3c6f750739eec143f6c05cb1b8dfad1',
        'IEMOCAP/feature/train_features.pkl',
        'IEMOCAP stage-1 features (train split)',
    ),
    'iemocap_dev_features.pkl': (
        '6ba84dbc8376b363fb0227e0a25f7273',
        'IEMOCAP/feature/dev_features.pkl',
        'IEMOCAP stage-1 features (dev split)',
    ),
    'iemocap_test_features.pkl': (
        'b6ae82a6f68f23abc99ff727c258fe12',
        'IEMOCAP/feature/test_features.pkl',
        'IEMOCAP stage-1 features (test split)',
    ),
    'meld_fusion.bin': (
        '1fd9e6c7fbec56d67bbab82443d55d8d',
        'MELD/MELD/save_model/multimodal_fusion_best.bin',
        'MELD stage-2 fusion checkpoint',
    ),
}

GROUPS = {
    'iemocap': ['iemocap_fusion.bin'],
    'meld': ['meld_fusion.bin'],
    'features': ['iemocap_train_features.pkl', 'iemocap_dev_features.pkl', 'iemocap_test_features.pkl'],
}


def release_assets():
    """Return {asset_name: asset_dict} of the weights release (no network if all
    requested files are already present and verified)."""
    for url in (API, f'{API}/latest'):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'tdtl-download'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
        except urllib.error.HTTPError:
            continue
        releases = data if isinstance(data, list) else [data]
        releases = [rel for rel in releases if rel.get('assets')]
        if releases:
            return {a['name']: a for a in releases[0]['assets']}
    sys.exit(
        'No GitHub release with weights found. If you cloned from a mirror,\n'
        'fetch the assets from https://github.com/' + REPO + '/releases'
    )


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f'-> {dest}')
    req = urllib.request.Request(url, headers={'User-Agent': 'tdtl-download'})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, 'wb') as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)


def fetch(names):
    # check local copies first; only hit the API if something is missing
    pending = []
    for name in names:
        expected_md5, rel_path, desc = ASSETS[name]
        if os.path.exists(rel_path) and md5sum(rel_path) == expected_md5:
            print(f'[skip] {rel_path} already present (md5 ok)')
        else:
            pending.append(name)
    if not pending:
        return
    assets = release_assets()
    for name in pending:
        expected_md5, rel_path, desc = ASSETS[name]
        if name not in assets:
            sys.exit(f'Asset {name} not found in the latest release.')
        print(f'[get ] {name} -- {desc}')
        download(assets[name]['browser_download_url'], rel_path)
        got = md5sum(rel_path)
        if got != expected_md5:
            os.remove(rel_path)
            sys.exit(f'md5 mismatch for {rel_path}: {got} != {expected_md5}')
        print(f'[ok  ] {rel_path} md5 verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', nargs='?', default='all',
                        choices=['all', 'iemocap', 'meld', 'features'],
                        help='which artifacts to fetch (default: all)')
    args = parser.parse_args()
    names = sum(GROUPS.values(), []) if args.group == 'all' else GROUPS[args.group]
    fetch(names)
    print('\nDone. Evaluate with:  cd <IEMOCAP|MELD> && python multimodel_fusion.py --train False')


if __name__ == '__main__':
    main()
