# Earables & IoT Workload Suite (Pure Python)

This repository contains reference implementations of common computational workloads found in earables, wearables, and edge IoT devices. 

These workloads are implemented **from scratch in pure Python/NumPy**, designed to have zero dependencies on heavy, opaque libraries (like `scikit-learn`, `librosa`, or `scipy`). 

This suite serves as a transparent benchmark for **Computer Architecture and Hardware Accelerator design**, exposing the raw mathematical operations (Matrix Multiplications, FFTs, LogSumExps, Matrix Inversions).

## Included Workloads
1. **Speaker Authentication:** `/speaker_auth` (Audio DSP & ML)
2. **Extended Kalman Filter:** `/ekf_sensor_fusion` (Sensor Fusion & Robotics)

## Built-in Execution Profiler
Run any script with the `--timing` flag to generate a terminal-based hierarchical "Flame Graph":
`python speaker_auth/speaker_auth_scratch.py --timing`

To generate detailed `cProfile` stats for viewing in tools like `snakeviz`, use the `--cprofile` flag:
`python ekf_sensor_fusion/ekf_scratch.py --cprofile prof.out`