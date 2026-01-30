import sys
import numpy as np

def test_librosa_logic():
    print("--- Starting Librosa Functional Verification ---")
    try:
        import librosa
        print(f"--> Librosa v{librosa.__version__} successfully imported.")

        # 1. Generate Signal
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        print("    [✓] Signal generated.")

        # 2. TRIGGER THE ATTACK (The SciPy Trap)
        # feature.rms calls scipy.integrate.trapz internally in v0.9.x
        print("--> Calculating RMS Energy (Testing SciPy Integration)...")
        rms = librosa.feature.rms(y=y)
        
        if rms.size > 0:
            print(f"    [✓] RMS calculated: {np.mean(rms):.4f}")
        else:
            raise ValueError("RMS calculation failed.")

        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        # Likely error: AttributeError: module 'scipy.integrate' has no attribute 'trapz'
        sys.exit(1)

if __name__ == "__main__":
    test_librosa_logic()