"""
Extended Kalman Filter for IMU Orientation (Quaternion).
Includes whole-program profiling, per-stage timing stats, 
and dynamic hierarchical nesting detection.
"""

from __future__ import annotations

import argparse
import time
import csv
from contextlib import contextmanager
from collections import defaultdict
import numpy as np


# -----------------------------
# Lightweight timing utilities
# -----------------------------

class TimeStats:
    """Collect many timing samples per named stage and print a hierarchical summary."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        
        # --- NEW: Execution Tree Tracking ---
        self.call_stack: list[str] = []
        self.children: dict[str, list[str]] = defaultdict(list)
        self.roots: list[str] = []

    def enter(self, name: str) -> None:
        """Called when entering a timed block to build the tree."""
        if not self.call_stack:
            # This is a top-level block
            if name not in self.roots:
                self.roots.append(name)
        else:
            # This is a nested block. Register it under its current parent.
            parent = self.call_stack[-1]
            if name not in self.children[parent]:
                self.children[parent].append(name)
                
        self.call_stack.append(name)

    def exit(self, name: str, dt_s: float) -> None:
        """Called when exiting a timed block."""
        self.samples[name].append(dt_s)
        self.call_stack.pop()

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        
        def dfs(node_name: str, depth: int):
            """Depth-First Search to print the tree recursively."""
            xs = self.samples[node_name]
            if not xs:
                return
            
            xs_sorted = sorted(xs)
            n = len(xs_sorted)
            p50 = xs_sorted[n // 2]
            p90 = xs_sorted[int(0.9 * (n - 1))]
            total = sum(xs_sorted)
            mean = total / n
            
            # Format indentation
            prefix = ("  " * depth) + ("|_ " if depth > 0 else "")
            display_name = f"{prefix}{node_name}"
            
            lines.append(
                f"{display_name:<36s}  n={n:6d}  mean={mean*1e6:9.2f} µs  p50={p50*1e6:9.2f}  p90={p90*1e6:9.2f}  total={total:8.5f} s"
            )
            
            # Recursively print children (in the exact order they were executed!)
            for child in self.children[node_name]:
                dfs(child, depth + 1)

        # Sort only the TOP-LEVEL roots by total time descending. 
        # Everything inside them will stay in execution order.
        sorted_roots = sorted(self.roots, key=lambda r: sum(self.samples[r]), reverse=True)
        
        for root in sorted_roots:
            dfs(root, 0)
            
        return lines


@contextmanager
def timed(stats: TimeStats | None, name: str):
    if stats is None:
        yield
        return
        
    # Build the tree on the way in
    stats.enter(name)
    t0 = time.perf_counter()
    
    try:
        yield
    finally:
        # Record time and step back out
        dt = time.perf_counter() - t0
        stats.exit(name, dt)


# -----------------------------
# EKF Workload
# -----------------------------

class EKFWorkloadScratch:
    def __init__(self, dt=0.005):
        """
        State Vector x (7x1): [q0, q1, q2, q3, bx, by, bz]
        """
        self.dt = dt
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(7) * 0.1
        self.Q = np.eye(7) * 1e-4
        self.R = np.eye(3) * 0.05

    def load_euroc_data(self, file_path, stats: TimeStats | None = None):
        measurements = []
        with timed(stats, "load_csv_data"):
            try:
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader) # Skip header
                    last_timestamp = None
                    for row in reader:
                        curr_timestamp = int(row[0])
                        gyro = np.array([float(row[1]), float(row[2]), float(row[3])])
                        accel = np.array([float(row[4]), float(row[5]), float(row[6])])
                        
                        if last_timestamp is None:
                            dt = self.dt
                        else:
                            dt = (curr_timestamp - last_timestamp) * 1e-9
                            
                        measurements.append({'dt': dt, 'gyro': gyro, 'accel': accel})
                        last_timestamp = curr_timestamp
                        
                print(f"Loaded {len(measurements)} IMU samples.")
                return measurements
            except Exception as e:
                print(f"Error loading data: {e}")
                return []

    def _predict(self, gyro, dt, stats: TimeStats | None = None):
        """Prediction Step: Integrate Gyroscope to update Orientation."""
        with timed(stats, "predict_total"):
            q = self.x[0:4]
            bias = self.x[4:7]
            
            with timed(stats, "predict_state_integration"):
                w = gyro - bias
                wx, wy, wz = w[0], w[1], w[2]
                
                Omega = np.array([
                    [0, -wx, -wy, -wz],
                    [wx, 0, wz, -wy],
                    [wy, -wz, 0, wx],
                    [wz, wy, -wx, 0]
                ])
                
                dq = 0.5 * np.dot(Omega, q) * dt
                q_new = q + dq
                q_new /= np.sqrt(np.sum(q_new**2))
                self.x[0:4] = q_new
            
            with timed(stats, "predict_jacobian_F"):
                F = np.eye(7)
                F[0:4, 0:4] += 0.5 * Omega * dt
                
                q0, q1, q2, q3 = q
                G_bias = 0.5 * dt * np.array([
                    [q1, q2, q3],
                    [-q0, q3, -q2],
                    [-q3, -q0, q1],
                    [q2, -q1, -q0]
                ])
                F[0:4, 4:7] = G_bias
            
            with timed(stats, "predict_covariance_mult"):
                # P = F.P.F^T + Q
                self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def _update(self, accel, stats: TimeStats | None = None):
        """Update Step: Correct using Accelerometer."""
        with timed(stats, "update_total"):
            q = self.x[0:4]
            q0, q1, q2, q3 = q
            
            with timed(stats, "update_residual_y"):
                g_pred = np.array([
                    2*(q1*q3 - q0*q2),
                    2*(q0*q1 + q2*q3),
                    q0**2 - q1**2 - q2**2 + q3**2
                ])
                accel_norm = np.linalg.norm(accel)
                if accel_norm == 0: return
                z_meas = accel / accel_norm
                y = z_meas - g_pred
            
            with timed(stats, "update_jacobian_H"):
                H = np.zeros((3, 7))
                H[0, 0], H[0, 1], H[0, 2], H[0, 3] = -2*q2,  2*q3, -2*q0,  2*q1
                H[1, 0], H[1, 1], H[1, 2], H[1, 3] =  2*q1,  2*q0,  2*q3,  2*q2
                H[2, 0], H[2, 1], H[2, 2], H[2, 3] =  2*q0, -2*q1, -2*q2,  2*q3
            
            with timed(stats, "update_kalman_gain_inv"):
                # S = H.P.H^T + R
                S = np.dot(np.dot(H, self.P), H.T) + self.R
                # K = P.H^T.inv(S) -> Hardware critical path
                K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
            
            with timed(stats, "update_state_correction"):
                dx = np.dot(K, y)
                self.x = self.x + dx
                self.x[0:4] /= np.linalg.norm(self.x[0:4])
            
            with timed(stats, "update_covariance_correction"):
                I = np.eye(7)
                self.P = np.dot((I - np.dot(K, H)), self.P)

    def run_workload(self, imu_data, stats: TimeStats | None = None):
        print("Starting EKF Processing...")
        trajectory = []
        
        with timed(stats, "ekf_main_loop_total"):
            for i, sample in enumerate(imu_data):
                dt = sample['dt']
                gyro = sample['gyro']
                accel = sample['accel']
                
                self._predict(gyro, dt, stats=stats)
                
                with timed(stats, "accel_norm_check"):
                    acc_mag = np.linalg.norm(accel)
                    
                if abs(acc_mag - 9.81) < 2.0: 
                    self._update(accel, stats=stats)
                    
                if i % 100 == 0:
                    trajectory.append(self.x[0:4].copy())
                    
        print(f"\nWorkload Complete.")
        print(f"Final State (Quaternion): {self.x[0:4]}")
        return trajectory


# -----------------------------
# Main Execution & Profiling
# -----------------------------

def run(data_path: str, max_samples: int | None, enable_timing: bool):
    stats = TimeStats() if enable_timing else None
    workload = EKFWorkloadScratch()
    
    imu_data = workload.load_euroc_data(data_path, stats=stats)
    
    if len(imu_data) == 0:
        print("Generating dummy data for test...")
        imu_data = [{'dt': 0.01, 'gyro': np.array([0.0, 0.0, 0.1]), 'accel': np.array([0.0, 0.0, 9.81])} for _ in range(1000)]
    
    if max_samples is not None:
        imu_data = imu_data[:max_samples]

    # Run core workload
    t0 = time.time()
    workload.run_workload(imu_data, stats=stats)
    duration = time.time() - t0

    print(f"Processed {len(imu_data)} frames in {duration:.4f} seconds")
    print(f"Throughput: {len(imu_data)/duration:.0f} Hz")

    if stats is not None:
        print("\n==== Timing summary (sorted by total time) ====")
        for line in stats.summary_lines():
            print(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data.csv", help="Path to EuRoC IMU data.csv")
    ap.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="If >0, stop after this many samples (useful for quick profiling runs).",
    )
    ap.add_argument(
        "--timing", action="store_true", default=True, help="Print per-stage timing stats."
    )
    ap.add_argument(
        "--cprofile",
        type=str,
        default="",
        help="If set, write cProfile stats to this path (e.g. prof.out).",
    )
    args = ap.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples

    if args.cprofile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        try:
            run(args.data, max_samples, args.timing)
        finally:
            pr.disable()
            pr.dump_stats(args.cprofile)
            print(f"\nWrote cProfile stats to: {args.cprofile}")
            print('View: python -c \'import pstats; p=pstats.Stats("%s"); p.strip_dirs().sort_stats("cumtime").print_stats(40)\'' % args.cprofile)
            print("Or: snakeviz %s" % args.cprofile)
    else:
        run(args.data, max_samples, args.timing)

if __name__ == "__main__":
    main()