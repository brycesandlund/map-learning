"""Remove inactive distribution modules from a legacy deterministic checkpoint."""
import argparse
from pathlib import Path
from map_learning.training import Trainer
from map_learning.deterministic import without_distributions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    if a.destination.exists():
        p.error('Destination already exists; refusing to overwrite a checkpoint')
    converted = without_distributions(Trainer.load(a.source))
    a.destination.parent.mkdir(exist_ok=True, parents=True)
    converted.save(a.destination)
    print(f'Saved {a.destination}; retained {converted.updates} updates and Adam state')


if __name__ == '__main__':
    main()
