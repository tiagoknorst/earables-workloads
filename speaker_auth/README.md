# Speaker Authentication Workload

This workload extracts Mel-Frequency Cepstral Coefficients (MFCCs) from raw audio and scores them against a Gaussian Mixture Model (GMM).

## Data Requirements
To run this workload on real data, you need a standard `.wav` audio file.
* **Format:** Mono (1 channel)
* **Sample Rate:** 16,000 Hz (16 kHz)
* **Bit Depth:** 16-bit PCM

**Setup:**
1. Place your audio file in this directory (e.g., `F00-16000.wav`).
2. Run the script pointing to your file:
   ```bash
   python speaker_auth_scratch.py --wav F00-16000.wav --timing