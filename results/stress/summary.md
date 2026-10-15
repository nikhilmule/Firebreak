# Stress test

Baseline: 6 drones, 18 fires, wind 8.0 m/s. 8 maps per condition. Paired difference = homes lost minus greedy_break, same maps; negative is better. 95% range = mean ± 2 standard errors.


## Fleet size (drones)

| drones | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 2 | 118.9 | 131.6 (+12.8, tie) | 131.6 (+12.8, tie) | 164.0 (+45.1, worse) | 153.9 (+35.0, worse) | 168.1 (+49.2, worse) |
| 4 | 91.9 | 96.1 (+4.2, tie) | 93.6 (+1.8, tie) | 143.9 (+52.0, worse) | 120.1 (+28.2, tie) | 168.1 (+76.2, worse) |
| 6 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 9 | 49.1 | 50.2 (+1.1, tie) | 49.6 (+0.5, tie) | 141.0 (+91.9, worse) | 69.1 (+20.0, worse) | 168.1 (+119.0, worse) |
| 12 | 34.0 | 51.5 (+17.5, worse) | 32.5 (-1.5, tie) | 131.8 (+97.8, worse) | 73.2 (+39.2, worse) | 168.1 (+134.1, worse) |

## Number of fires (fires)

| fires | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 6 | 73.4 | 70.9 (-2.5, tie) | 81.1 (+7.8, tie) | 137.0 (+63.6, worse) | 112.8 (+39.4, worse) | 177.2 (+103.9, worse) |
| 12 | 76.1 | 70.8 (-5.4, tie) | 83.1 (+7.0, tie) | 159.8 (+83.6, worse) | 106.0 (+29.9, worse) | 166.4 (+90.2, worse) |
| 18 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 30 | 70.9 | 63.5 (-7.4, tie) | 66.9 (-4.0, tie) | 149.9 (+79.0, worse) | 89.5 (+18.6, worse) | 173.2 (+102.4, worse) |
| 40 | 70.0 | 68.4 (-1.6, tie) | 70.6 (+0.6, tie) | 125.5 (+55.5, worse) | 104.6 (+34.6, worse) | 177.5 (+107.5, worse) |

## Wind speed (m/s)

| wind | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 2 | 2.9 | 11.6 (+8.8, tie) | 15.2 (+12.4, worse) | 81.1 (+78.2, worse) | 21.5 (+18.6, worse) | 78.0 (+75.1, worse) |
| 5 | 38.4 | 46.2 (+7.9, tie) | 52.9 (+14.5, tie) | 121.5 (+83.1, worse) | 59.2 (+20.9, tie) | 143.4 (+105.0, worse) |
| 8 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 11 | 80.0 | 87.8 (+7.8, tie) | 89.0 (+9.0, tie) | 148.9 (+68.9, worse) | 118.4 (+38.4, worse) | 196.8 (+116.8, worse) |
| 14 | 93.5 | 91.2 (-2.2, tie) | 93.6 (+0.1, tie) | 166.9 (+73.4, worse) | 134.9 (+41.4, worse) | 209.0 (+115.5, worse) |

## Sensor noise (sd of error)

| obs_noise | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 0.25 | 60.8 | 82.0 (+21.2, worse) | 80.6 (+19.9, worse) | 135.5 (+74.8, worse) | 102.9 (+42.1, worse) | 168.1 (+107.4, worse) |
| 0.5 | 83.9 | 82.5 (-1.4, tie) | 75.8 (-8.1, tie) | 135.5 (+51.6, worse) | 122.9 (+39.0, worse) | 168.1 (+84.2, worse) |
| 1 | 91.2 | 90.9 (-0.4, tie) | 89.0 (-2.2, tie) | 135.5 (+44.2, worse) | 119.4 (+28.1, worse) | 168.1 (+76.9, worse) |

## Missed fires (share unseen)

| miss_rate | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 0.15 | 75.8 | 64.6 (-11.1, tie) | 57.5 (-18.2, tie) | 123.9 (+48.1, worse) | 107.6 (+31.9, worse) | 168.1 (+92.4, worse) |
| 0.3 | 68.1 | 72.0 (+3.9, tie) | 72.0 (+3.9, tie) | 157.0 (+88.9, worse) | 108.4 (+40.2, worse) | 168.1 (+100.0, worse) |
| 0.5 | 77.4 | 69.4 (-8.0, tie) | 77.5 (+0.1, tie) | 144.1 (+66.8, worse) | 117.9 (+40.5, worse) | 168.1 (+90.8, worse) |

## Drone failures (per drone-hour)

| fail_per_hour | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 2 | 67.0 | 68.4 (+1.4, tie) | 75.9 (+8.9, tie) | 136.8 (+69.8, worse) | 104.8 (+37.8, worse) | 168.1 (+101.1, worse) |
| 4 | 67.4 | 65.2 (-2.1, tie) | 76.4 (+9.0, tie) | 137.4 (+70.0, worse) | 107.6 (+40.2, worse) | 168.1 (+100.8, worse) |
| 8 | 79.6 | 72.2 (-7.4, tie) | 91.6 (+12.0, tie) | 141.9 (+62.2, worse) | 120.6 (+41.0, worse) | 168.1 (+88.5, worse) |

## Sensing range (m (0 = whole map))

| sense_range | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 2000 | 72.1 | 82.0 (+9.9, tie) | 69.2 (-2.9, tie) | 121.8 (+49.6, worse) | 106.2 (+34.1, tie) | 168.1 (+96.0, worse) |
| 1000 | 77.5 | 74.9 (-2.6, tie) | 72.1 (-5.4, tie) | 143.0 (+65.5, worse) | 130.5 (+53.0, worse) | 168.1 (+90.6, worse) |
| 600 | 115.2 | 121.5 (+6.2, tie) | 115.2 (+0.0, tie) | 139.9 (+24.6, worse) | 136.1 (+20.9, worse) | 168.1 (+52.9, worse) |

## Wind shift at 10 min (degrees)

| wind_shift_deg | greedy_break | ga_break | pso_break | nearest_break | greedy_c_stable | none |
|---|---|---|---|---|---|---|
| 0 | 69.0 | 66.5 (-2.5, tie) | 75.9 (+6.9, tie) | 135.5 (+66.5, worse) | 92.0 (+23.0, tie) | 168.1 (+99.1, worse) |
| 45 | 70.2 | 68.6 (-1.6, tie) | 76.4 (+6.1, tie) | 135.8 (+65.5, worse) | 92.0 (+21.8, tie) | 168.8 (+98.5, worse) |
| 90 | 72.2 | 66.4 (-5.9, tie) | 79.4 (+7.1, tie) | 138.2 (+66.0, worse) | 92.5 (+20.2, tie) | 179.6 (+107.4, worse) |
| 150 | 75.0 | 69.4 (-5.6, tie) | 79.9 (+4.9, tie) | 146.1 (+71.1, worse) | 96.9 (+21.9, tie) | 183.4 (+108.4, worse) |
