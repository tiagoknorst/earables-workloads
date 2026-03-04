# Extended Kalman Filter (EKF) Workload

This workload implements a 7-State EKF to track 3D orientation (Quaternions) by fusing Gyroscope and Accelerometer data.

## Data Requirements (EuRoC MAV Dataset)
This script is designed to parse the industry-standard **EuRoC MAV Dataset** (Visual-Inertial Odometry).

**How to get the data:**
1. Download the **Machine Hall 01** sequence from the ASL Datasets:
   [MH_01_easy.zip (Link)](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f)
2. Extract the `.zip` file.
3. Navigate to `mav0/imu0/` inside the extracted folder.
4. Copy the `data.csv` file and paste it directly into this directory.
5. Run the script:
   ```bash
   python ekf_profiled.py --data data.csv --timing