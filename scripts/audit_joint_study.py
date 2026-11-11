"""Cross-stage pairing, duplicate-run reproducibility and exact study accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('roots', type=Path, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    pairs, records = {}, []
    for root in a.roots:
        manifest = json.loads((root/'manifest.json').read_text())
        assert json.loads((root/'complete.json').read_text())['failed'] == 0
        for seed in manifest['seeds']:
            for arm in manifest['arms']:
                path = root/f"{arm['name']}-seed{seed}"
                r = json.loads((path/'result.json').read_text())
                with np.load(path/'replay.npz') as replay:
                    hashes = (r['initial_parameters_sha256'], digest(replay['actions']),
                              digest(replay['next_observations'][:, 2:]))
                key = (seed, r['layout'], r['step_size'], r['nuisance_dims'])
                if key in pairs:
                    assert hashes == pairs[key], (path, 'unmatched initialization/action/noise stream')
                else:
                    pairs[key] = hashes
                records.append({'run':str(path), 'actions':r['interactions'], 'updates':r['updates'],
                                'samples_per_head':r['samples_per_head']})
    duplicates = []
    root = Path('runs/joint-loss-scale-20261111')
    for left, right in [('joint-07','dynamics_weight-10'),('joint-09','repulsion_weight-0.3')]:
        for seed in [42,43]:
            paths = [root/f'{name}-seed{seed}' for name in [left,right]]
            arrays = [np.load(path/'replay.npz') for path in paths]
            assert set(arrays[0].files) == set(arrays[1].files)
            for name in arrays[0].files:
                assert np.array_equal(arrays[0][name],arrays[1][name]), (paths, name)
            checkpoints = [torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False) for path in paths]
            for name in ['model','dynamics']:
                assert checkpoints[0][name].keys() == checkpoints[1][name].keys()
                for key in checkpoints[0][name]:
                    assert torch.equal(checkpoints[0][name][key],checkpoints[1][name][key]),(paths,name,key)
            duplicates.append({'runs':[str(path) for path in paths],'replay_and_weights_exact':True})
    result = {'jobs':len(records), 'collection_actions':sum(r['actions'] for r in records),
              'updates':sum(r['updates'] for r in records),
              'samples_per_head':sum(r['samples_per_head'] for r in records),
              'matched_seed_layout_scale_groups':len(pairs),
              'paired_initialization_controls_successor_noise':True,
              'accidental_duplicates':duplicates, 'runs':records}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__ == '__main__':
    main()
