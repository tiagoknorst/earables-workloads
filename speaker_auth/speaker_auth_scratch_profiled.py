"""
Pure Python Speaker Authentication Workload (MFCC + GMM).
Includes whole-program profiling, per-stage timing stats, 
and dynamic hierarchical nesting detection.
"""

from __future__ import annotations

import argparse
import time
import wave
import struct
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
# Speaker Authentication Workload
# -----------------------------

class SpeakerAuthWorkloadScratch:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_fft=128, n_mels=26):
        """
        Initialize with standard speech processing parameters.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = 512  # 32ms at 16k
        self.win_length = 512  # 32ms at 16k
        
        # Pre-compute fixed DSP kernels
        self.mel_basis = self._build_mel_basis()
        self.dct_basis = self._build_dct_basis()
        
        # Initialize dummy GMM parameters
        self.n_components = 16
        self.means = np.random.randn(self.n_components, self.n_mfcc)
        self.covariances = np.abs(np.random.randn(self.n_components, self.n_mfcc))
        self.weights = np.ones(self.n_components) / self.n_components

    def load_wav(self, file_path, stats: TimeStats | None = None):
        """Manually load a .wav file into a float numpy array [-1, 1]."""
        with timed(stats, "load_wav_file"):
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    
                    if n_channels != 1:
                        print("Warning: Audio is not mono. Using first channel only.")
                    if framerate != self.sample_rate:
                        print(f"Warning: Audio rate {framerate} != model rate {self.sample_rate}.")
                    
                    raw_data = wav_file.readframes(n_frames)
                    total_samples = n_frames * n_channels
                    fmt = f"{total_samples}h" 
                    pcm_data = struct.unpack(fmt, raw_data)
                    
                    audio_array = np.array(pcm_data, dtype=np.float32) / 32768.0
                    
                    if n_channels > 1:
                        audio_array = audio_array[::n_channels]
                        
                    return audio_array
            except Exception as e:
                print(f"Error loading wav file: {e}")
                return None

    def _hz_to_mel(self, freq):
        return 2595 * np.log10(1 + freq / 700.0)

    def _mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595.0) - 1)

    def _build_mel_basis(self):
        """Manually construct Mel Filterbank with Slaney Normalization."""
        low_freq_mel = self._hz_to_mel(0)
        high_freq_mel = self._hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((self.n_mels, int(self.n_fft / 2 + 1)))
        
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

        enorm = 2.0 / (hz_points[2:self.n_mels+2] - hz_points[:self.n_mels])
        fbank *= enorm[:, np.newaxis]
        return fbank

    def _build_dct_basis(self):
        """Manually construct the Discrete Cosine Transform matrix."""
        n = self.n_mels
        dct_matrix = np.zeros((self.n_mfcc, n))
        
        for i in range(self.n_mfcc):
            if i == 0:
                alpha = np.sqrt(1.0 / n)
            else:
                alpha = np.sqrt(2.0 / n)
                
            for j in range(n):
                dct_matrix[i, j] = alpha * np.cos(np.pi * i * (2 * j + 1) / (2 * n))
                
        return dct_matrix

    def extract_mfcc(self, audio, stats: TimeStats | None = None):
        """The DSP Kernel: Raw Audio -> MFCC Features"""
        with timed(stats, "dsp_extract_mfcc_total"):
            
            with timed(stats, "dsp_padding"):
                pad_width = self.n_fft // 2
                audio = np.pad(audio, pad_width, mode='reflect')

            with timed(stats, "dsp_framing"):
                num_frames = 1 + int((len(audio) - self.win_length) / self.hop_length)
                pad_signal_length = num_frames * self.hop_length + self.win_length
                z = np.zeros((pad_signal_length - len(audio)))
                pad_audio = np.append(audio, z)
                
                indices = np.tile(np.arange(0, self.win_length), (num_frames, 1)) + \
                          np.tile(np.arange(0, num_frames * self.hop_length, self.hop_length), (self.win_length, 1)).T
                frames = pad_audio[indices.astype(np.int32, copy=False)]

            with timed(stats, "dsp_windowing"):
                frames *= np.hanning(self.win_length)

            with timed(stats, "dsp_fft_power"):
                mag_frames = np.absolute(np.fft.rfft(frames, self.n_fft))
                pow_frames = ((1.0 / self.n_fft) * ((mag_frames) ** 2))

            with timed(stats, "dsp_mel_filterbank"):
                filter_banks = np.dot(pow_frames, self.mel_basis.T)
            
            with timed(stats, "dsp_log_scale"):
                filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
                filter_banks = 10 * np.log10(filter_banks)

            with timed(stats, "dsp_dct"):
                mfcc = np.dot(filter_banks, self.dct_basis.T)
            
            return mfcc

    def gmm_score_samples(self, features, stats: TimeStats | None = None):
        """The ML Kernel: GMM Log-Likelihood Scoring."""
        with timed(stats, "ml_gmm_score_total"):
            n_samples, n_dim = features.shape
            
            with timed(stats, "ml_gmm_init"):
                log_prob = np.zeros((n_samples, self.n_components))
                log_weights = np.log(self.weights)
                log_2pi = np.log(2 * np.pi)
            
            with timed(stats, "ml_gmm_component_loop"):
                for k in range(self.n_components):
                    mu = self.means[k]
                    sig = self.covariances[k]
                    
                    diff = features - mu
                    mahalanobis = np.sum((diff ** 2) / sig, axis=1)
                    log_det = np.sum(np.log(sig))
                    
                    log_prob[:, k] = log_weights[k] - 0.5 * (mahalanobis + n_dim * log_2pi + log_det)

            with timed(stats, "ml_gmm_logsumexp"):
                max_log = np.max(log_prob, axis=1, keepdims=True)
                final_scores = max_log + np.log(np.sum(np.exp(log_prob - max_log), axis=1, keepdims=True))
            
            return final_scores.flatten()

    def authenticate(self, audio_signal, stats: TimeStats | None = None):
        with timed(stats, "authenticate_pipeline_total"):
            # 1. DSP Kernel
            mfccs = self.extract_mfcc(audio_signal, stats=stats)
            
            # 2. ML Kernel
            scores = self.gmm_score_samples(mfccs, stats=stats)
            
            return np.mean(scores)


# -----------------------------
# Main Execution & Profiling
# -----------------------------

def run(wav_path: str, enable_timing: bool):
    stats = TimeStats() if enable_timing else None
    
    print("Initializing SpeakerAuth (Scratch Implementation)...")
    workload = SpeakerAuthWorkloadScratch()
    
    audio_data = None
    if wav_path:
        audio_data = workload.load_wav(wav_path, stats=stats)
    
    # Fallback to dummy data if no file provided or loading failed
    if audio_data is None:
        if wav_path:
            print(f"Could not process file: {wav_path}. Falling back to dummy data.")
        else:
            print("No audio file provided. Generating dummy data for test...")
        # Generate 1 second of dummy audio (White Noise)
        audio_data = np.random.uniform(-1, 1, 16000)

    # Run the Workload
    score = workload.authenticate(audio_data, stats=stats)
    
    print(f"\nAuthentication Score: {score:.4f}")

    if stats is not None:
        print("\n==== Timing summary (sorted by total time) ====")
        for line in stats.summary_lines():
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, default="F00-16000.wav", help="Path to input .wav file")
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

    if args.cprofile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        try:
            run(args.wav, args.timing)
        finally:
            pr.disable()
            pr.dump_stats(args.cprofile)
            print(f"\nWrote cProfile stats to: {args.cprofile}")
            print('View: python -c \'import pstats; p=pstats.Stats("%s"); p.strip_dirs().sort_stats("cumtime").print_stats(40)\'' % args.cprofile)
            print("Or: snakeviz %s" % args.cprofile)
    else:
        run(args.wav, args.timing)

if __name__ == "__main__":
    main()