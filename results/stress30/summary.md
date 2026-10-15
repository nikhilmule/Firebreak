# Stress test

Baseline: 6 drones, 18 fires, wind 8.0 m/s. 30 maps per condition. Paired difference = homes lost minus greedy_break, same maps; negative is better. 95% range = mean ± 2 standard errors.


## Fleet size (drones)

| drones | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 2 | 122.4 | 126.0 (+3.6, tie) | 126.2 (+3.8, tie) | 165.7 (+43.3, worse) | 155.8 (+33.4, worse) | 180.5 (+58.1, worse) |
| 4 | 99.3 | 94.3 (-5.1, tie) | 93.7 (-5.6, tie) | 156.7 (+57.3, worse) | 126.4 (+27.1, worse) | 180.5 (+81.1, worse) |
| 6 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 9 | 49.1 | 54.3 (+5.2, tie) | 50.5 (+1.4, tie) | 143.8 (+94.7, worse) | 90.3 (+41.2, worse) | 180.5 (+131.4, worse) |
| 12 | 37.4 | 45.4 (+8.0, worse) | 41.6 (+4.2, tie) | 145.2 (+107.8, worse) | 79.5 (+42.1, worse) | 180.5 (+143.1, worse) |

## Number of fires (fires)

| fires | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 6 | 74.8 | 71.8 (-2.9, tie) | 71.3 (-3.5, tie) | 153.3 (+78.5, worse) | 119.4 (+44.6, worse) | 183.3 (+108.5, worse) |
| 12 | 78.9 | 74.1 (-4.8, tie) | 79.0 (+0.0, tie) | 160.7 (+81.7, worse) | 113.1 (+34.2, worse) | 181.5 (+102.6, worse) |
| 18 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 30 | 76.9 | 69.1 (-7.8, tie) | 79.0 (+2.1, tie) | 148.1 (+71.2, worse) | 104.5 (+27.6, worse) | 179.1 (+102.1, worse) |
| 40 | 74.6 | 75.5 (+0.9, tie) | 74.5 (-0.1, tie) | 141.3 (+66.7, worse) | 111.2 (+36.6, worse) | 185.4 (+110.8, worse) |

## Wind speed (m/s)

| wind | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 2 | 13.4 | 14.6 (+1.2, tie) | 11.8 (-1.6, tie) | 68.8 (+55.4, worse) | 21.0 (+7.6, worse) | 87.9 (+74.5, worse) |
| 5 | 41.2 | 47.7 (+6.5, tie) | 54.5 (+13.3, worse) | 134.8 (+93.6, worse) | 79.2 (+38.0, worse) | 161.9 (+120.7, worse) |
| 8 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 11 | 95.4 | 98.0 (+2.7, tie) | 94.5 (-0.9, tie) | 158.8 (+63.4, worse) | 130.8 (+35.5, worse) | 198.6 (+103.2, worse) |
| 14 | 107.8 | 109.1 (+1.3, tie) | 108.0 (+0.2, tie) | 166.3 (+58.5, worse) | 158.7 (+50.9, worse) | 205.6 (+97.8, worse) |

## Sensor noise (sd of error)

| obs_noise | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 0.25 | 73.8 | 80.7 (+6.9, tie) | 77.3 (+3.5, tie) | 148.9 (+75.1, worse) | 110.0 (+36.2, worse) | 180.5 (+106.7, worse) |
| 0.5 | 86.2 | 83.1 (-3.1, tie) | 82.1 (-4.1, tie) | 148.9 (+62.8, worse) | 119.8 (+33.6, worse) | 180.5 (+94.3, worse) |
| 1 | 101.4 | 104.1 (+2.7, tie) | 93.9 (-7.5, tie) | 148.9 (+47.5, worse) | 133.3 (+31.8, worse) | 180.5 (+79.0, worse) |

## Missed fires (share unseen)

| miss_rate | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 0.15 | 72.5 | 71.4 (-1.0, tie) | 76.9 (+4.5, tie) | 144.0 (+71.5, worse) | 113.7 (+41.3, worse) | 180.5 (+108.0, worse) |
| 0.3 | 74.3 | 81.3 (+7.0, tie) | 80.4 (+6.2, tie) | 154.9 (+80.6, worse) | 118.0 (+43.8, worse) | 180.5 (+106.2, worse) |
| 0.5 | 82.9 | 75.0 (-7.9, tie) | 79.6 (-3.3, tie) | 155.7 (+72.8, worse) | 123.9 (+41.0, worse) | 180.5 (+97.6, worse) |

## Drone failures (per drone-hour)

| fail_per_hour | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 2 | 74.7 | 79.4 (+4.7, tie) | 80.6 (+5.9, tie) | 150.9 (+76.2, worse) | 117.7 (+43.0, worse) | 180.5 (+105.8, worse) |
| 4 | 79.7 | 87.6 (+7.9, tie) | 89.0 (+9.3, tie) | 153.7 (+74.1, worse) | 122.7 (+43.0, worse) | 180.5 (+100.8, worse) |
| 8 | 86.3 | 97.4 (+11.1, tie) | 101.6 (+15.3, worse) | 155.9 (+69.6, worse) | 134.9 (+48.5, worse) | 180.5 (+94.1, worse) |

## Sensing range (m (0 = whole map))

| sense_range | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 2000 | 79.9 | 76.6 (-3.3, tie) | 77.0 (-2.9, tie) | 141.9 (+62.0, worse) | 111.8 (+31.9, worse) | 180.5 (+100.6, worse) |
| 1000 | 88.8 | 87.0 (-1.8, tie) | 86.9 (-1.9, tie) | 134.9 (+46.1, worse) | 128.0 (+39.2, worse) | 180.5 (+91.7, worse) |
| 600 | 118.1 | 121.8 (+3.7, tie) | 118.0 (-0.1, tie) | 156.1 (+38.0, worse) | 133.2 (+15.1, worse) | 180.5 (+62.4, worse) |

## Wind shift at 10 min (degrees)

| wind_shift_deg | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.8 | 76.2 (+6.4, tie) | 72.2 (+2.4, tie) | 148.9 (+79.1, worse) | 109.9 (+40.0, worse) | 180.5 (+110.6, worse) |
| 45 | 70.7 | 77.3 (+6.7, tie) | 72.5 (+1.8, tie) | 149.7 (+79.1, worse) | 109.9 (+39.2, worse) | 181.2 (+110.5, worse) |
| 90 | 72.2 | 77.0 (+4.8, tie) | 73.9 (+1.7, tie) | 151.2 (+79.0, worse) | 110.1 (+37.9, worse) | 186.0 (+113.8, worse) |
| 150 | 74.3 | 78.6 (+4.4, tie) | 74.0 (-0.2, tie) | 158.4 (+84.1, worse) | 111.3 (+37.0, worse) | 191.8 (+117.5, worse) |
