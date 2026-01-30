import sys
import numpy as np
import warnings

# Suppress numba deprecation warnings to keep the logs focused on the logic
warnings.filterwarnings("ignore", category=UserWarning)

def test_audio_feature_modernization():
    print("--- Starting Librosa Functional Verification (Repo #20) ---")
    
    try:
        import librosa
        import scipy
        print(f"--> Librosa v{librosa.__version__} imported.")
        print(f"--> SciPy v{scipy.__version__} detected.")

        # 1. Generate a synthetic audio signal
        # A simple 1-second 440Hz sine wave at 22.05kHz
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        print("    [✓] Test signal generated.")

        # 2. TRIGGER THE SCI-PY TRAP
        # Librosa v0.9.x feature.rms calls scipy.integrate.trapz()
        # SciPy 1.14+ removed trapz (renamed to trapezoid).
        print("--> Calculating RMS Energy (Testing SciPy Integration)...")
        rms = librosa.feature.rms(y=y)
        
        if rms.size > 0:
            print(f"    [✓] RMS Energy calculated: {np.mean(rms):.4f}")
        else:
            raise ValueError("RMS calculation returned an empty array.")

        # 3. Spectral Centroid Check
        # This often relies on the same integration logic
        print("--> Calculating Spectral Centroid...")
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        print(f"    [✓] Spectral Centroid verified: {np.mean(centroid):.1f} Hz")

        print("--- SMOKE TEST PASSED: Environment Modernized Successfully ---")

    except AttributeError as ae:
        print(f"CRITICAL VALIDATION FAILURE: {str(ae)}")
        print("REASON: A dependency (likely SciPy) removed a function Librosa needs.")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_audio_feature_modernization()