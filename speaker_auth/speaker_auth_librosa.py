import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
import os

class SpeakerAuthWorkload:
    def __init__(self, sample_rate=16000, n_mfcc=13, n_components=16):
        """
        Initialize the GMM-based Speaker Authentication system.
        The paper typically uses standard MFCC counts (e.g., 13-20) 
        and GMM components suitable for short audio segments.
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        # GMM classifier as described in the paper 
        self.classifier = GaussianMixture(n_components=n_components, covariance_type='diag')
        self.is_trained = False

    def load_and_preprocess(self, file_path, duration=1.0):
        """
        Simulates the workload input size: 1 second of audio.
        """
        try:
            # Load audio, resampling to consistent Hz
            audio, _ = librosa.load(file_path, sr=self.sample_rate)
            
            # Workload constraint: Ensure exactly 1 second of input
            target_length = int(duration * self.sample_rate)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, target_length - len(audio)))
                
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def extract_features(self, audio):
        """
        Kernel: MFCC Extraction.
        The paper notes this uses 'traditional DSP techniques'[cite: 112, 120].
        """
        # Compute MFCCs (Mel-frequency cepstral coefficients)
        # Output shape: (n_mfcc, T) -> Transpose to (T, n_mfcc) for GMM
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return mfccs.T

    def train_user_model(self, audio_files):
        """
        Train the GMM on the 'legitimate' user's voice data.
        """
        features_list = []
        for f in audio_files:
            audio = self.load_and_preprocess(f)
            if audio is not None:
                feat = self.extract_features(audio)
                features_list.append(feat)
        
        if features_list:
            # Stack all features to fit the GMM density
            X = np.vstack(features_list)
            self.classifier.fit(X)
            self.is_trained = True
            print("Speaker model trained successfully.")

    def authenticate(self, file_path, threshold=-50.0):
        """
        The runtime workload: Verify if input speech is the user[cite: 119].
        Returns True (Authenticated) or False (Rejected).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        # 1. Load Data (Input Stream)
        audio = self.load_and_preprocess(file_path)
        if audio is None: return False

        # 2. Extract Features (DSP Kernel)
        features = self.extract_features(audio)

        # 3. Score (ML Kernel)
        # GMM 'score' returns the log-likelihood of the samples
        scores = self.classifier.score_samples(features)
        avg_score = np.mean(scores)

        # Threshold logic: Higher log-likelihood = closer match
        print(f"Auth Score for {file_path}: {avg_score:.2f}")
        return avg_score > threshold

# --- Mock Usage (Resembling your colleague's main block) ---
if __name__ == "__main__":
    # Simulate workload with dummy data if files don't exist
    print("Initializing Speaker Auth Workload...")
    workload = SpeakerAuthWorkload()

    # In a real scenario, you would point these to .wav files
    # For now, we generate random noise to demonstrate the data flow
    dummy_training_data = np.random.uniform(-1, 1, 16000) # 1 sec noise
    
    # 1. Feature Extraction flow
    mfcc_feats = workload.extract_features(dummy_training_data)
    print(f"MFCC Feature Matrix Shape: {mfcc_feats.shape}")
    
    # 2. Model Training flow (GMM)
    # We cheat and fit on the dummy noise for demonstration
    workload.classifier.fit(mfcc_feats) 
    workload.is_trained = True
    
    # 3. Inference flow
    result = workload.authenticate("F00-16000.wav") # Will fail on file load, handled gracefully