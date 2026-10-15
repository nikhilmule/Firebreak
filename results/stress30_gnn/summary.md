# Stress test

Baseline: 6 drones, 18 fires, wind 8.0 m/s. 30 maps per condition. Paired difference = homes lost minus greedy_break, same maps; negative is better. 95% range = mean ± 2 standard errors.


## Fleet size (drones)

| drones | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 2 | 122.4 | 122.4 (+0.0, tie) | 155.8 (+33.4, worse) |
| 4 | 99.3 | 99.3 (+0.0, tie) | 126.4 (+27.1, worse) |
| 6 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 9 | 49.1 | 47.1 (-2.0, tie) | 90.3 (+41.2, worse) |
| 12 | 37.4 | 45.9 (+8.5, tie) | 79.5 (+42.1, worse) |

## Number of fires (fires)

| fires | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 6 | 74.8 | 74.8 (+0.0, tie) | 119.4 (+44.6, worse) |
| 12 | 78.9 | 78.9 (+0.0, tie) | 113.1 (+34.2, worse) |
| 18 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 30 | 76.9 | 76.9 (+0.0, tie) | 104.5 (+27.6, worse) |
| 40 | 74.6 | 74.6 (+0.0, tie) | 111.2 (+36.6, worse) |

## Wind speed (m/s)

| wind | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 2 | 13.4 | 13.4 (+0.0, tie) | 21.0 (+7.6, worse) |
| 5 | 41.2 | 41.2 (+0.0, tie) | 79.2 (+38.0, worse) |
| 8 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 11 | 95.4 | 95.4 (+0.0, tie) | 130.8 (+35.5, worse) |
| 14 | 107.8 | 107.8 (+0.0, tie) | 158.7 (+50.9, worse) |

## Sensor noise (sd of error)

| obs_noise | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 0 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 0.25 | 73.8 | 73.8 (+0.0, tie) | 110.0 (+36.2, worse) |
| 0.5 | 86.2 | 86.2 (+0.0, tie) | 119.8 (+33.6, worse) |
| 1 | 101.4 | 101.4 (+0.0, tie) | 133.3 (+31.8, worse) |

## Missed fires (share unseen)

| miss_rate | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 0 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 0.15 | 72.5 | 72.5 (+0.0, tie) | 113.7 (+41.3, worse) |
| 0.3 | 74.3 | 74.3 (+0.0, tie) | 118.0 (+43.8, worse) |
| 0.5 | 82.9 | 82.9 (+0.0, tie) | 123.9 (+41.0, worse) |

## Drone failures (per drone-hour)

| fail_per_hour | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 0 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 2 | 74.7 | 74.7 (+0.0, tie) | 117.7 (+43.0, worse) |
| 4 | 79.7 | 79.7 (+0.0, tie) | 122.7 (+43.0, worse) |
| 8 | 86.3 | 86.3 (+0.0, tie) | 134.9 (+48.5, worse) |

## Sensing range (m (0 = whole map))

| sense_range | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 0 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 2000 | 79.9 | 79.9 (+0.0, tie) | 111.8 (+31.9, worse) |
| 1000 | 88.8 | 88.8 (+0.0, tie) | 128.0 (+39.2, worse) |
| 600 | 118.1 | 118.1 (+0.0, tie) | 133.2 (+15.1, worse) |

## Wind shift at 10 min (degrees)

| wind_shift_deg | greedy_break | gnn_policy | greedy_c_stable |
|---|---|---|---|
| 0 | 69.8 | 69.8 (+0.0, tie) | 109.9 (+40.0, worse) |
| 45 | 70.7 | 70.7 (+0.0, tie) | 109.9 (+39.2, worse) |
| 90 | 72.2 | 72.2 (+0.0, tie) | 110.1 (+37.9, worse) |
| 150 | 74.3 | 74.3 (+0.0, tie) | 111.3 (+37.0, worse) |
