# Extended Kalman Filter (EKF) Workload

This workload implements a 7-State EKF to track 3D orientation (Quaternions) by fusing Gyroscope and Accelerometer data.

## Data Requirements (EuRoC MAV Dataset)
This script is designed to parse the industry-standard **EuRoC MAV Dataset** (Visual-Inertial Odometry).

**How to get the data:**
1. Download the **Machine Hall 01** sequence from the ASL Datasets:
   [MH_01_easy.zip (Link)](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip)
2. Extract the `.zip` file.
3. Navigate to `mav0/imu0/` inside the extracted folder.
4. Copy the `data.csv` file and paste it directly into this directory.
5. Run the script:
   ```bash
   python ekf_scratch.py --data data.csv --timing