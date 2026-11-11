# All KL mechanism results

Prior mismatch is inactive in β=0 arms. In deterministic-input arms, q is an auxiliary distribution: its fitted KL does not bound information passed to dynamics through the noiseless means. See [the report](kl-mechanisms.md) for definitions and interpretation.

## Shared winner data, models at 35k transitions

| Arm | Top100 productive (%) | Mean new states /640 actions (seed9 /10 /11) | Mean fitted KL at100k | Prior mismatch at100k |
|---|---:|---|---:|---:|
| Mean / no KL | 5.00 | 9.0 / 3.3 / 15.0 | 40.61 | 4630.91 |
| Fixed noise / no KL | 4.33 | 15.0 / 3.0 / 11.0 | 41.50 | 5429.55 |
| Mean / KL | 8.00 | 18.0 / 6.3 / 6.7 | 5.00 | 81.83 |
| Fixed noise / KL | 6.33 | 13.7 / 4.3 / 8.0 | 5.60 | 81.33 |
| Learned noise / no KL | 4.67 | 18.0 / 3.3 / 11.3 | 61.44 | 4266.43 |
| Current winner | 6.67 | 15.7 / 5.3 / 10.0 | 13.34 | 81.95 |
| Prior LR ×3 | 5.33 | 19.7 / 3.7 / 8.7 | 17.17 | 12.55 |
| Prior LR ×10 | 6.00 | 21.3 / 1.7 / 4.3 | 17.34 | 1.39 |
| 5 prior steps | 5.00 | 34.3 / 3.0 / 7.3 | 21.92 | 0.08 |

## Shared zero data, models at 35k transitions

| Arm | Top100 productive (%) | Mean new states /640 actions (seed9 /10 /11) | Mean fitted KL at100k | Prior mismatch at100k |
|---|---:|---|---:|---:|
| Mean / no KL | 8.67 | 71.0 / 9.3 / 91.0 | 39.70 | 3903.20 |
| Fixed noise / no KL | 6.67 | 83.3 / 3.0 / 78.7 | 39.66 | 3814.63 |
| Mean / KL | 27.33 | 135.3 / 11.3 / 149.0 | 4.94 | 71.99 |
| Fixed noise / KL | 27.00 | 135.3 / 8.7 / 128.7 | 5.56 | 78.14 |
| Learned noise / no KL | 7.67 | 77.0 / 11.3 / 91.7 | 60.76 | 3900.61 |
| Current winner | 26.33 | 136.3 / 19.3 / 135.7 | 13.70 | 79.26 |
| Prior LR ×3 | 30.00 | 135.7 / 6.3 / 139.7 | 17.76 | 11.27 |
| Prior LR ×10 | 18.33 | 120.3 / 0.3 / 121.0 | 18.72 | 1.52 |
| 5 prior steps | 28.00 | 134.0 / 10.0 / 132.3 | 17.33 | 0.03 |

## Online, seeds9/10/11

| Arm | Mean coverage over run | Final coverage by seed | Full-coverage actions by seed |
|---|---:|---|---|
| Mean / no KL | 44.5% | 87.1% / 100.0% / 98.3% | >100k / 79067 / >100k |
| Fixed noise / no KL | 36.7% | 42.8% / 53.7% / 100.0% | >100k / >100k / 76516 |
| Mean / KL | 64.6% | 100.0% / 100.0% / 100.0% | 56036 / 53514 / 46698 |
| Fixed noise / KL | 63.7% | 100.0% / 100.0% / 100.0% | 60313 / 49573 / 42403 |
| Learned noise / no KL | 39.7% | 51.0% / 100.0% / 100.0% | >100k / 86073 / 97383 |
| Current winner | 60.7% | 100.0% / 100.0% / 100.0% | 55671 / 53631 / 54328 |
| Prior LR ×3 | 60.7% | 100.0% / 100.0% / 100.0% | 86037 / 41257 / 54018 |
| Prior LR ×10 | 42.1% | 100.0% / 59.9% / 87.2% | 91846 / >100k / >100k |
| 5 prior steps | 49.1% | 56.8% / 100.0% / 100.0% | >100k / 71832 / 67549 |

## Fresh confirmation, seeds12/13/14

| Arm | Mean coverage over run | Final coverage by seed | Full-coverage actions by seed |
|---|---:|---|---|
| Mean / no KL | 36.3% | 100.0% / 69.2% / 53.8% | 98542 / >100k / >100k |
| Mean / KL | 63.1% | 100.0% / 100.0% / 100.0% | 52379 / 54613 / 51002 |
| Current winner | 60.5% | 100.0% / 100.0% / 100.0% | 56139 / 50378 / 61592 |
