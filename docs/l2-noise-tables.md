# All L2 and noise/search results

## Screen seeds 9/10/11

| Arm | Mean coverage throughout 100k | Final coverage | Completion actions by seed | Noise / movement |
|---|---:|---:|---|---:|
| No regularizer | 44.5% | 95.1% | >100k / 79067 / >100k | 0.108 |
| L2 λ=1e-05 | 42.9% | 98.8% | >100k / 94647 / >100k | 0.113 |
| L2 λ=3e-05 | 53.4% | 100.0% | 99343 / 48801 / 95548 | 0.14 |
| L2 λ=0.0001 | 52.6% | 100.0% | 67303 / 84323 / 76126 | 0.132 |
| L2 λ=0.0003 | 56.6% | 100.0% | 63718 / 62747 / 73389 | 0.145 |
| L2 λ=0.001 | 62.7% | 100.0% | 61429 / 56912 / 42748 | 0.176 |
| L2 λ=0.003 | 62.3% | 100.0% | 54232 / 56374 / 50087 | 0.214 |
| L2 λ=0.01 | 66.7% | 100.0% | 52322 / 42003 / 46848 | 0.276 |
| L2 λ=0.03 | 69.6% | 100.0% | 43247 / 46002 / 43569 | 0.354 |
| L2 λ=0.1 | 16.4% | 24.3% | >100k / >100k / >100k | 30.6 |
| Mean + KL | 64.6% | 100.0% | 56036 / 53514 / 46698 | 0.132 |

## Confirmation seeds 15/16/17

| Arm | Mean coverage throughout 100k | Final coverage | Completion actions by seed | Noise / movement |
|---|---:|---:|---|---:|
| No regularizer | 37.8% | 85.6% | >100k / >100k / >100k | 0.114 |
| L2 λ=0.03 | 68.1% | 100.0% | 50856 / 36289 / 51397 | 0.378 |
| Mean + KL | 63.3% | 100.0% | 53929 / 52243 / 55568 | 0.161 |

## Common archive at 50k

| Model | Raw new states / 640 actions | Redraw queue | Redraw both | Nuisance removed | Radius clipped | Top-100 overlap after both redraw | Physical Jaccard | Top-100 norm percentile | Top-100 nuisance-residual percentile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No regularizer | 25.8 | 22.6 | 14.7 | 142.0 | 27.2 | 11.1% | 0.154 | 50.9 | 68.3 |
| Mean + KL | 140.2 | 125.3 | 136.0 | 186.3 | 132.4 | 43.2% | 0.385 | 43.3 | 53.7 |
| Selected L2 | 154.9 | 158.8 | 156.4 | 143.7 | 164.9 | 42.1% | 0.442 | 42.5 | 52.1 |

## Common archive at 75k

| Model | Raw new states / 640 actions | Redraw queue | Redraw both | Nuisance removed | Radius clipped | Top-100 overlap after both redraw | Physical Jaccard | Top-100 norm percentile | Top-100 nuisance-residual percentile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No regularizer | 61.2 | 61.7 | 54.6 | 134.0 | 59.7 | 19.0% | 0.214 | 46.7 | 57.4 |
| Mean + KL | 146.0 | 144.0 | 143.8 | 174.3 | 144.6 | 49.3% | 0.497 | 40.1 | 49.6 |
| Selected L2 | 127.2 | 131.7 | 127.7 | 159.3 | 129.6 | 38.1% | 0.395 | 41.9 | 54.4 |
