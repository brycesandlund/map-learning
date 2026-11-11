# Compression study: all seed results

Endpoint occupancy of 0.25-unit bins, percent, after 100,000 actions. Every run is retained. See [main report](compression-study.md) for coefficients, controls, interpretation and limitations.


## Fresh confirmation

| Method | 80 | 81 | 82 | 83 | 84 | 85 | Mean | Mean fine-bin occupancy | Mean occupancy AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dynamics-base | 79.04 | 62.24 | 88.28 | 81.99 | 76.30 | 69.49 | 76.22 | 72.93 | 26.20 |
| adjacency-base | 65.62 | 68.53 | 77.56 | 81.25 | 76.39 | 36.11 | 67.58 | 65.02 | 20.98 |
| dynamics-multistep | 78.26 | 76.22 | 86.55 | 80.64 | 71.74 | 65.58 | 76.50 | 74.03 | 26.92 |
| dynamics-multistep-control | 73.09 | 72.09 | 77.08 | 85.33 | 79.04 | 65.23 | 75.31 | 72.12 | 25.67 |
| adjacency-multistep | 72.96 | 66.36 | 64.93 | 91.28 | 69.05 | 71.14 | 72.62 | 70.80 | 21.84 |
| adjacency-multistep-control | 76.26 | 58.03 | 84.38 | 73.91 | 75.74 | 49.61 | 69.65 | 67.54 | 21.19 |
| dynamics-l1 | 73.48 | 48.09 | 72.35 | 83.55 | 63.41 | 78.60 | 69.91 | 66.50 | 22.93 |
| dynamics-noise | 51.52 | 59.24 | 52.69 | 63.93 | 66.54 | 66.54 | 60.08 | 57.77 | 18.58 |
| dynamics-noise-control | 54.82 | 54.21 | 52.21 | 65.58 | 66.10 | 61.02 | 58.99 | 56.82 | 18.94 |
| adjacency-l1 | 27.08 | 63.11 | 79.82 | 83.25 | 83.07 | 10.11 | 57.74 | 55.29 | 19.36 |
| adjacency-noise | 58.72 | 45.70 | 50.22 | 66.71 | 29.08 | 59.81 | 51.71 | 49.89 | 16.35 |
| adjacency-noise-control | 61.72 | 48.87 | 51.35 | 71.66 | 52.08 | 63.24 | 58.15 | 56.05 | 17.78 |

## Additional layouts

| Method | 86 | 87 | 88 | Mean | Mean fine-bin occupancy | Mean occupancy AUC |
|---|---:|---:|---:|---:|---:|---:|
| dynamics-base-layout13 | 68.49 | 45.10 | 63.80 | 59.13 | 57.42 | 18.34 |
| dynamics-base-layout23 | 76.61 | 85.37 | 69.44 | 77.14 | 73.22 | 22.65 |
| adjacency-base-layout13 | 67.06 | 50.61 | 51.87 | 56.51 | 55.26 | 16.43 |
| adjacency-base-layout23 | 85.72 | 88.80 | 77.17 | 83.90 | 81.58 | 24.89 |
| dynamics-l1-layout13 | 72.87 | 54.73 | 59.51 | 62.37 | 60.76 | 20.99 |
| dynamics-l1-layout23 | 77.08 | 79.86 | 71.66 | 76.20 | 72.88 | 23.79 |
| adjacency-multistep-layout13 | 60.07 | 42.97 | 21.92 | 41.65 | 40.15 | 10.23 |
| adjacency-multistep-layout23 | 76.87 | 81.25 | 52.39 | 70.17 | 67.82 | 19.11 |
| adjacency-multistep-control-layout13 | 62.11 | 66.10 | 48.09 | 58.77 | 57.40 | 16.04 |
| adjacency-multistep-control-layout23 | 86.68 | 84.90 | 62.67 | 78.08 | 75.69 | 23.58 |

## Tuning only

| Method | 70 | 71 | Mean | Mean fine-bin occupancy | Mean occupancy AUC |
|---|---:|---:|---:|---:|---:|
| dynamics-base | 80.25 | 71.83 | 76.04 | 72.17 | 26.99 |
| dynamics-margin1 | 60.11 | 46.05 | 53.08 | 49.83 | 16.11 |
| dynamics-l1-0.0001 | 65.93 | 71.96 | 68.95 | 67.14 | 25.53 |
| dynamics-l1-0.0003 | 8.03 | 7.07 | 7.55 | 6.58 | 5.25 |
| dynamics-l1-0.001 | 6.60 | 7.94 | 7.27 | 6.41 | 4.95 |
| dynamics-l1-0.003 | 6.29 | 9.94 | 8.12 | 7.12 | 5.43 |
| dynamics-l1-0.01 | 6.68 | 7.68 | 7.18 | 6.38 | 5.02 |
| dynamics-l1only-0.001 | 8.38 | 4.56 | 6.47 | 5.82 | 4.64 |
| dynamics-l1only-0.003 | 5.95 | 3.99 | 4.97 | 4.48 | 3.66 |
| dynamics-l1only-0.01 | 6.08 | 5.38 | 5.73 | 5.07 | 4.13 |
| dynamics-bound | 71.66 | 61.76 | 66.71 | 63.72 | 21.09 |
| dynamics-noise-0.01 | 26.65 | 10.37 | 18.51 | 16.89 | 7.77 |
| dynamics-noise-0.03 | 4.69 | 3.08 | 3.88 | 3.42 | 2.80 |
| dynamics-noise-0.1 | 3.04 | 1.91 | 2.47 | 2.14 | 1.84 |
| dynamics-noise-0.3 | 2.73 | 2.26 | 2.50 | 2.19 | 1.81 |
| dynamics-tasknoise-0.1 | 5.77 | 6.16 | 5.97 | 5.37 | 3.75 |
| adjacency-base | 71.44 | 75.56 | 73.50 | 71.55 | 27.00 |
| adjacency-margin1 | 65.58 | 78.65 | 72.11 | 70.30 | 23.95 |
| adjacency-l1-0.0001 | 55.95 | 40.36 | 48.16 | 44.82 | 13.29 |
| adjacency-l1-0.0003 | 7.34 | 4.86 | 6.10 | 5.57 | 4.54 |
| adjacency-l1-0.001 | 6.60 | 5.47 | 6.03 | 5.43 | 4.13 |
| adjacency-l1-0.003 | 7.25 | 7.12 | 7.18 | 6.35 | 5.31 |
| adjacency-l1-0.01 | 5.21 | 7.29 | 6.25 | 5.71 | 4.50 |
| adjacency-l1only-0.001 | 7.94 | 6.29 | 7.12 | 6.42 | 5.09 |
| adjacency-l1only-0.003 | 5.82 | 7.12 | 6.47 | 5.84 | 4.44 |
| adjacency-l1only-0.01 | 8.12 | 4.30 | 6.21 | 5.57 | 4.31 |
| adjacency-bound | 62.80 | 61.81 | 62.30 | 61.12 | 19.94 |
| adjacency-noise-0.01 | 38.76 | 11.11 | 24.93 | 23.90 | 8.70 |
| adjacency-noise-0.03 | 11.15 | 13.41 | 12.28 | 10.99 | 4.87 |
| adjacency-noise-0.1 | 2.30 | 5.03 | 3.67 | 3.21 | 2.10 |
| adjacency-noise-0.3 | 1.95 | 2.00 | 1.97 | 1.71 | 1.59 |
| adjacency-tasknoise-0.1 | 2.65 | 4.12 | 3.39 | 2.97 | 2.16 |
| dynamics-h1-w0.1 | 76.48 | 66.28 | 71.38 | 68.83 | 24.12 |
| dynamics-h1-w1 | 79.47 | 74.87 | 77.17 | 75.16 | 25.96 |
| dynamics-h4-w0.1 | 78.34 | 74.09 | 76.22 | 73.53 | 26.38 |
| dynamics-h4-w1 | 59.72 | 75.43 | 67.58 | 65.08 | 21.43 |
| dynamics-h16-w0.1 | 77.52 | 63.28 | 70.40 | 66.28 | 20.80 |
| dynamics-h16-w1 | 76.56 | 70.36 | 73.46 | 71.58 | 22.37 |
| dynamics-h32-w0.1 | 80.38 | 77.99 | 79.19 | 76.53 | 27.29 |
| adjacency-h1-w0.1 | 71.92 | 76.52 | 74.22 | 72.16 | 25.57 |
| adjacency-h1-w1 | 67.84 | 76.65 | 72.24 | 69.94 | 23.71 |
| adjacency-h4-w0.1 | 69.10 | 66.10 | 67.60 | 64.64 | 20.75 |
| adjacency-h4-w1 | 68.19 | 69.40 | 68.79 | 66.28 | 21.11 |
| adjacency-h16-w0.1 | 80.08 | 70.05 | 75.07 | 72.50 | 24.34 |
| adjacency-h16-w1 | 79.56 | 81.77 | 80.66 | 79.25 | 30.84 |
| adjacency-h32-w0.1 | 81.77 | 74.61 | 78.19 | 75.62 | 25.08 |
| dynamics-l1-1e-5 | 71.92 | 69.62 | 70.77 | 67.13 | 22.22 |
| dynamics-l1-3e-5 | 80.86 | 78.56 | 79.71 | 75.54 | 26.31 |
| dynamics-l1-3e-4-r0.1 | 45.31 | 73.44 | 59.38 | 56.11 | 17.43 |
| dynamics-late-l1-0.0003 | 81.60 | 71.05 | 76.32 | 72.98 | 26.71 |
| dynamics-late-l1-0.001 | 80.90 | 26.91 | 53.91 | 51.11 | 21.35 |
| dynamics-bound1 | 75.35 | 77.21 | 76.28 | 73.99 | 28.13 |
| dynamics-bound1-noise0.025 | 4.08 | 4.12 | 4.10 | 3.68 | 2.82 |
| dynamics-noise0.025-r0.1 | 2.04 | 1.82 | 1.93 | 1.65 | 1.58 |
| adjacency-l1-1e-5 | 65.49 | 79.12 | 72.31 | 69.22 | 25.85 |
| adjacency-l1-3e-5 | 59.46 | 57.38 | 58.42 | 56.02 | 18.45 |
| adjacency-l1-3e-4-r0.1 | 13.37 | 81.55 | 47.46 | 45.42 | 16.89 |
| adjacency-late-l1-0.0003 | 71.14 | 73.61 | 72.37 | 70.71 | 26.76 |
| adjacency-late-l1-0.001 | 69.84 | 34.59 | 52.21 | 50.00 | 22.53 |
| adjacency-bound1 | 77.47 | 75.82 | 76.65 | 73.91 | 26.42 |
| adjacency-bound1-noise0.025 | 10.03 | 7.60 | 8.81 | 8.07 | 5.23 |
| adjacency-noise0.025-r0.1 | 1.95 | 2.39 | 2.17 | 1.89 | 1.65 |
| dynamics-late-noise0.0025 | 70.70 | 58.25 | 64.47 | 62.07 | 20.45 |
| dynamics-late-noise0.0075 | 25.04 | 57.47 | 41.25 | 39.07 | 15.52 |
| adjacency-late-noise0.0025 | 62.59 | 58.94 | 60.76 | 59.38 | 18.66 |
| adjacency-late-noise0.0075 | 29.34 | 56.68 | 43.01 | 40.29 | 12.68 |
| dynamics-weak-noise0.00025 | 69.57 | 58.25 | 63.91 | 61.47 | 20.07 |
| dynamics-weak-noise0.00075 | 64.58 | 55.30 | 59.94 | 58.02 | 18.69 |
| adjacency-weak-noise0.00025 | 61.89 | 59.29 | 60.59 | 58.42 | 20.39 |
| adjacency-weak-noise0.00075 | 47.40 | 55.34 | 51.37 | 49.06 | 13.89 |
