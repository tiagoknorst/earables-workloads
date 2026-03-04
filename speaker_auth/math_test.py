import numpy as np
from sklearn.mixture import GaussianMixture
# Import your scratch class. 
# Assuming it's in the same file or imported as:
from speaker_auth_scratch import SpeakerAuthWorkloadScratch

def calibration_test():
    print("--- STARTING CALIBRATION ---")
    
    # 1. Create Dummy Data (1 sec audio)
    # We use random noise, but fixed seed so it's reproducible
    np.random.seed(42)
    dummy_audio = np.random.uniform(-1, 1, 16000)
    
    # 2. Initialize the SCRATCH workload
    # (Make sure to use the 'ortho' DCT fix I gave you!)
    scratch_workload = SpeakerAuthWorkloadScratch()
    
    # Extract features using our manual DSP
    # We need these to train the reference sklearn model
    features = scratch_workload.extract_mfcc(dummy_audio)
    print(f"Feature Shape: {features.shape}")

    # 3. Initialize and Fit the REFERENCE (Sklearn) model
    # We force it to fit OUR features so it has valid parameters
    ref_gmm = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
    ref_gmm.fit(features)
    
    # Get the "Golden" scores from the library
    ref_scores = ref_gmm.score_samples(features)
    ref_avg_score = np.mean(ref_scores)
    print(f"Reference (Sklearn) Score: {ref_avg_score:.5f}")

    # 4. SURGERY: Transplant weights from Sklearn -> Scratch
    # We overwrite the random initialization with the trained values
    print("Transplanting weights...")
    scratch_workload.means = ref_gmm.means_
    scratch_workload.covariances = ref_gmm.covariances_
    scratch_workload.weights = ref_gmm.weights_
    
    # 5. Run the SCRATCH scoring with the copied weights
    scratch_avg_score = scratch_workload.authenticate(dummy_audio)
    print(f"Scratch (Manual) Score:  {scratch_avg_score:.5f}")
    
    # 6. Verdict
    diff = abs(ref_avg_score - scratch_avg_score)
    if diff < 1e-4:
        print(f"\nSUCCESS! The models are identical (Diff: {diff:.8f})")
    else:
        print(f"\nMISMATCH. (Diff: {diff:.5f}) - Check normalization or log constants.")

if __name__ == "__main__":
    # Paste your SpeakerAuthWorkloadScratch class above or import it
    calibration_test()