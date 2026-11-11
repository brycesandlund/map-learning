# Walk study: every seed

Coverage and AUC are percentages. Time to95 is censored at100k; a dash means the target was not reached.

## dynamics-base-s0.125

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 69.66 | 67.36 | 25.21 | — | 0 | 4112 |
| 93 | 79.95 | 76.81 | 29.95 | — | 0 | 4112 |
| 94 | 69.62 | 67.34 | 22.14 | — | 0 | 4112 |
| 95 | 74.09 | 69.39 | 26.59 | — | 0 | 4112 |
| 96 | 68.62 | 66.80 | 22.27 | — | 0 | 4112 |
| 97 | 75.48 | 72.75 | 24.71 | — | 0 | 4112 |

Mean final: 72.90%; mean AUC: 25.15%; mean capped time to95: 100000 actions.

## adjacency-base-s0.125

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 73.00 | 71.27 | 26.93 | — | 0 | 4112 |
| 93 | 82.81 | 82.30 | 34.37 | — | 0 | 4112 |
| 94 | 55.08 | 52.52 | 13.83 | — | 0 | 4112 |
| 95 | 78.08 | 76.62 | 28.16 | — | 0 | 4112 |
| 96 | 51.95 | 49.73 | 13.96 | — | 0 | 4112 |
| 97 | 71.57 | 68.42 | 21.10 | — | 0 | 4112 |

Mean final: 68.75%; mean AUC: 23.06%; mean capped time to95: 100000 actions.

## dynamics-base-s0.25

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.14 | 61.03 | 62753 | 0 | 4112 |
| 93 | 100.00 | 99.46 | 65.78 | 50857 | 0 | 4112 |
| 94 | 100.00 | 99.45 | 63.68 | 50568 | 0 | 4112 |
| 95 | 100.00 | 99.24 | 66.25 | 48747 | 0 | 4112 |
| 96 | 100.00 | 99.46 | 60.08 | 57021 | 0 | 4112 |
| 97 | 100.00 | 98.94 | 59.09 | 57717 | 0 | 4112 |

Mean final: 100.00%; mean AUC: 62.65%; mean capped time to95: 54610 actions.

## adjacency-base-s0.25

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.02 | 67.33 | 45657 | 0 | 4112 |
| 93 | 100.00 | 99.60 | 68.92 | 44537 | 0 | 4112 |
| 94 | 100.00 | 99.39 | 60.28 | 54006 | 0 | 4112 |
| 95 | 100.00 | 99.31 | 64.21 | 49063 | 0 | 4112 |
| 96 | 100.00 | 99.15 | 53.39 | 64308 | 0 | 4112 |
| 97 | 100.00 | 99.58 | 61.65 | 55029 | 0 | 4112 |

Mean final: 100.00%; mean AUC: 62.63%; mean capped time to95: 52100 actions.

## walk-s0.125-p0.98-stay0-turn1

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 99.48 | 97.67 | 79.25 | 74546 | 0 | 0 |
| 93 | 100.00 | 99.51 | 79.62 | 82280 | 0 | 0 |
| 94 | 100.00 | 99.10 | 87.66 | 32560 | 0 | 0 |
| 95 | 99.44 | 97.58 | 86.65 | 54082 | 0 | 0 |
| 96 | 99.65 | 98.11 | 73.39 | 74714 | 0 | 0 |
| 97 | 99.96 | 99.38 | 83.03 | 53199 | 0 | 0 |

Mean final: 99.75%; mean AUC: 81.60%; mean capped time to95: 61897 actions.

## walk-s0.125-p0.98-r10000

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 92.75 | 88.73 | 77.54 | — | 9 | 0 |
| 93 | 99.96 | 98.30 | 82.80 | 45111 | 9 | 0 |
| 94 | 92.71 | 87.76 | 67.11 | — | 9 | 0 |
| 95 | 99.91 | 97.73 | 84.95 | 47692 | 9 | 0 |
| 96 | 92.32 | 90.04 | 79.12 | — | 9 | 0 |
| 97 | 99.65 | 96.18 | 86.30 | 49816 | 9 | 0 |

Mean final: 96.22%; mean AUC: 79.64%; mean capped time to95: 73770 actions.

## walk-s0.125-p0.0-r0

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 43.62 | 41.81 | 30.99 | — | 0 | 0 |
| 93 | 50.78 | 47.61 | 29.92 | — | 0 | 0 |
| 94 | 56.81 | 53.46 | 31.64 | — | 0 | 0 |
| 95 | 22.57 | 21.09 | 17.77 | — | 0 | 0 |
| 96 | 37.50 | 35.56 | 27.64 | — | 0 | 0 |
| 97 | 38.98 | 36.84 | 27.78 | — | 0 | 0 |

Mean final: 41.71%; mean AUC: 27.62%; mean capped time to95: 100000 actions.

## random-queue-s0.125

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 1.30 | 1.09 | 1.12 | — | 0 | 0 |
| 93 | 1.22 | 1.07 | 1.10 | — | 0 | 0 |
| 94 | 1.35 | 1.20 | 1.20 | — | 0 | 0 |
| 95 | 1.35 | 1.13 | 1.12 | — | 0 | 0 |
| 96 | 1.43 | 1.16 | 1.15 | — | 0 | 0 |
| 97 | 1.52 | 1.31 | 1.34 | — | 0 | 0 |

Mean final: 1.36%; mean AUC: 1.17%; mean capped time to95: 100000 actions.

## reset-policy-noreset-s0.125

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 85.59 | 84.70 | 72.56 | — | 0 | 0 |
| 93 | 98.91 | 96.32 | 83.22 | 47143 | 0 | 0 |
| 94 | 93.75 | 86.66 | 79.39 | — | 0 | 0 |
| 95 | 99.26 | 97.37 | 81.55 | 57147 | 0 | 0 |
| 96 | 97.74 | 94.65 | 80.70 | 75563 | 0 | 0 |
| 97 | 99.83 | 97.62 | 87.84 | 32271 | 0 | 0 |

Mean final: 95.85%; mean AUC: 80.88%; mean capped time to95: 68687 actions.

## noreset-policy-reset-s0.125

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 97.74 | 93.52 | 70.92 | 88353 | 9 | 0 |
| 93 | 99.87 | 98.67 | 85.40 | 37445 | 9 | 0 |
| 94 | 99.96 | 99.02 | 87.54 | 32426 | 9 | 0 |
| 95 | 99.09 | 97.20 | 80.99 | 56914 | 9 | 0 |
| 96 | 85.76 | 85.19 | 76.19 | — | 9 | 0 |
| 97 | 99.09 | 97.44 | 85.37 | 53250 | 9 | 0 |

Mean final: 96.92%; mean AUC: 81.07%; mean capped time to95: 61398 actions.

## walk-s0.125-p0.9-stay0.2-turn0

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 93.32 | 86.73 | 57.74 | — | 0 | 0 |
| 93 | 69.79 | 65.86 | 47.53 | — | 0 | 0 |
| 94 | 79.30 | 74.73 | 52.09 | — | 0 | 0 |
| 95 | 90.93 | 83.14 | 68.20 | — | 0 | 0 |
| 96 | 95.31 | 86.30 | 66.11 | 96742 | 0 | 0 |
| 97 | 88.67 | 81.32 | 64.80 | — | 0 | 0 |

Mean final: 86.22%; mean AUC: 59.41%; mean capped time to95: 99457 actions.

## openloop-policy-blockturn-s0.125

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 83.55 | 80.22 | 67.46 | — | 0 | 0 |
| 93 | 88.89 | 85.10 | 74.32 | — | 0 | 0 |
| 94 | 82.90 | 74.80 | 37.78 | — | 0 | 0 |
| 95 | 97.74 | 93.25 | 62.42 | 82583 | 0 | 0 |
| 96 | 90.71 | 87.03 | 74.30 | — | 0 | 0 |
| 97 | 97.18 | 91.57 | 71.57 | 89201 | 0 | 0 |

Mean final: 90.16%; mean AUC: 64.64%; mean capped time to95: 95297 actions.

## walk-s0.25-p0.98-stay0-turn1

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.56 | 92.26 | 32722 | 0 | 0 |
| 93 | 100.00 | 99.72 | 85.53 | 45470 | 0 | 0 |
| 94 | 99.83 | 97.97 | 84.59 | 49640 | 0 | 0 |
| 95 | 100.00 | 99.70 | 90.64 | 21157 | 0 | 0 |
| 96 | 100.00 | 99.67 | 90.14 | 35494 | 0 | 0 |
| 97 | 100.00 | 99.28 | 91.25 | 20153 | 0 | 0 |

Mean final: 99.97%; mean AUC: 89.07%; mean capped time to95: 34106 actions.

## walk-s0.25-p0.98-r10000

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.61 | 86.65 | 34138 | 9 | 0 |
| 93 | 99.78 | 97.71 | 87.80 | 35454 | 9 | 0 |
| 94 | 99.91 | 97.53 | 90.39 | 31231 | 9 | 0 |
| 95 | 100.00 | 98.37 | 82.89 | 58307 | 9 | 0 |
| 96 | 100.00 | 98.61 | 89.85 | 35699 | 9 | 0 |
| 97 | 99.78 | 97.96 | 89.66 | 32586 | 9 | 0 |

Mean final: 99.91%; mean AUC: 87.87%; mean capped time to95: 37902 actions.

## walk-s0.25-p0.0-r0

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 82.12 | 79.83 | 65.37 | — | 0 | 0 |
| 93 | 78.82 | 75.30 | 48.03 | — | 0 | 0 |
| 94 | 85.98 | 82.04 | 55.79 | — | 0 | 0 |
| 95 | 89.32 | 85.71 | 61.78 | — | 0 | 0 |
| 96 | 73.70 | 71.78 | 54.55 | — | 0 | 0 |
| 97 | 73.74 | 71.48 | 56.17 | — | 0 | 0 |

Mean final: 80.61%; mean AUC: 56.95%; mean capped time to95: 100000 actions.

## random-queue-s0.25

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 4.25 | 3.75 | 3.54 | — | 0 | 0 |
| 93 | 3.52 | 2.93 | 2.82 | — | 0 | 0 |
| 94 | 3.73 | 3.29 | 3.16 | — | 0 | 0 |
| 95 | 3.21 | 2.84 | 2.63 | — | 0 | 0 |
| 96 | 3.56 | 3.11 | 3.03 | — | 0 | 0 |
| 97 | 3.78 | 3.39 | 3.21 | — | 0 | 0 |

Mean final: 3.67%; mean AUC: 3.07%; mean capped time to95: 100000 actions.

## reset-policy-noreset-s0.25

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 99.91 | 98.88 | 89.07 | 26326 | 0 | 0 |
| 93 | 100.00 | 97.89 | 89.74 | 27535 | 0 | 0 |
| 94 | 96.48 | 94.55 | 84.88 | 84177 | 0 | 0 |
| 95 | 100.00 | 99.22 | 89.78 | 28407 | 0 | 0 |
| 96 | 100.00 | 99.47 | 87.53 | 47657 | 0 | 0 |
| 97 | 99.91 | 98.38 | 86.23 | 59138 | 0 | 0 |

Mean final: 99.39%; mean AUC: 87.87%; mean capped time to95: 45540 actions.

## noreset-policy-reset-s0.25

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.73 | 91.40 | 38568 | 9 | 0 |
| 93 | 100.00 | 99.74 | 90.48 | 23187 | 9 | 0 |
| 94 | 100.00 | 99.77 | 88.59 | 30733 | 9 | 0 |
| 95 | 100.00 | 99.71 | 89.72 | 22790 | 9 | 0 |
| 96 | 100.00 | 99.69 | 92.49 | 23051 | 9 | 0 |
| 97 | 100.00 | 99.58 | 89.29 | 31531 | 9 | 0 |

Mean final: 100.00%; mean AUC: 90.33%; mean capped time to95: 28310 actions.

## walk-s0.25-p0.9-stay0.2-turn0

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 99.83 | 93.62 | 71.93 | 83733 | 0 | 0 |
| 93 | 96.09 | 85.57 | 69.97 | 92891 | 0 | 0 |
| 94 | 96.09 | 84.56 | 73.41 | 78540 | 0 | 0 |
| 95 | 77.00 | 58.62 | 58.60 | — | 0 | 0 |
| 96 | 98.05 | 89.45 | 69.81 | 76112 | 0 | 0 |
| 97 | 99.65 | 92.17 | 78.98 | 56428 | 0 | 0 |

Mean final: 94.45%; mean AUC: 70.45%; mean capped time to95: 81284 actions.

## openloop-policy-blockturn-s0.25

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.29 | 81.93 | 83802 | 0 | 0 |
| 93 | 100.00 | 99.45 | 81.70 | 55861 | 0 | 0 |
| 94 | 100.00 | 99.44 | 87.89 | 38867 | 0 | 0 |
| 95 | 100.00 | 99.23 | 86.75 | 26459 | 0 | 0 |
| 96 | 99.96 | 98.47 | 88.66 | 28223 | 0 | 0 |
| 97 | 98.70 | 94.18 | 72.42 | 78705 | 0 | 0 |

Mean final: 99.78%; mean AUC: 83.22%; mean capped time to95: 51986 actions.

## dynamics-s0.125-warm20000-resets2

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 98.87 | 95.77 | 79.88 | 78637 | 2 | 4112 |
| 93 | 83.98 | 82.77 | 72.02 | — | 2 | 4112 |
| 94 | 85.37 | 84.32 | 71.75 | — | 2 | 4112 |
| 95 | 99.96 | 98.72 | 83.67 | 64557 | 2 | 4112 |
| 96 | 89.71 | 88.64 | 75.05 | — | 2 | 4112 |
| 97 | 100.00 | 98.91 | 86.46 | 52078 | 2 | 4112 |

Mean final: 92.98%; mean AUC: 78.14%; mean capped time to95: 82545 actions.

## continue-s0.125-warm20000-r6667

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 99.78 | 99.06 | 86.78 | 34959 | 2 | 0 |
| 93 | 99.91 | 99.10 | 80.42 | 54512 | 2 | 0 |
| 94 | 99.09 | 95.82 | 83.99 | 44498 | 2 | 0 |
| 95 | 99.96 | 99.57 | 87.97 | 30089 | 2 | 0 |
| 96 | 99.87 | 98.63 | 85.86 | 48822 | 2 | 0 |
| 97 | 100.00 | 99.39 | 87.10 | 37336 | 2 | 0 |

Mean final: 99.77%; mean AUC: 85.35%; mean capped time to95: 41703 actions.

## adjacency-s0.125-warm20000-resets2

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 99.74 | 98.55 | 80.55 | 74719 | 2 | 4112 |
| 93 | 84.90 | 84.35 | 72.23 | — | 2 | 4112 |
| 94 | 85.07 | 84.66 | 71.83 | — | 2 | 4112 |
| 95 | 100.00 | 99.70 | 83.46 | 67059 | 2 | 4112 |
| 96 | 91.41 | 90.55 | 75.03 | — | 2 | 4112 |
| 97 | 100.00 | 99.59 | 86.58 | 50778 | 2 | 4112 |

Mean final: 93.52%; mean AUC: 78.28%; mean capped time to95: 82093 actions.

## dynamics-s0.25-warm20000-resets0

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.77 | 92.63 | 24229 | 0 | 4112 |
| 93 | 100.00 | 99.02 | 84.09 | 49351 | 0 | 4112 |
| 94 | 100.00 | 99.28 | 85.88 | 50558 | 0 | 4112 |
| 95 | 100.00 | 99.74 | 90.74 | 20974 | 0 | 4112 |
| 96 | 100.00 | 99.21 | 87.21 | 48556 | 0 | 4112 |
| 97 | 100.00 | 99.82 | 91.09 | 24226 | 0 | 4112 |

Mean final: 100.00%; mean AUC: 88.61%; mean capped time to95: 36316 actions.

## adjacency-s0.25-warm20000-resets0

| Seed | Final coverage | Fine coverage | Coverage AUC | Actions to95 | Random resets | Updates |
|---|---:|---:|---:|---:|---:|---:|
| 92 | 100.00 | 99.88 | 92.68 | 25231 | 0 | 4112 |
| 93 | 100.00 | 99.59 | 85.48 | 42522 | 0 | 4112 |
| 94 | 100.00 | 99.64 | 86.83 | 43004 | 0 | 4112 |
| 95 | 100.00 | 99.84 | 90.78 | 21504 | 0 | 4112 |
| 96 | 100.00 | 99.60 | 87.37 | 46354 | 0 | 4112 |
| 97 | 100.00 | 99.95 | 91.14 | 22787 | 0 | 4112 |

Mean final: 100.00%; mean AUC: 89.05%; mean capped time to95: 33567 actions.

