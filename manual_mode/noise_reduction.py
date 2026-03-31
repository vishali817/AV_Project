import numpy as np
import scipy.signal
import scipy.ndimage

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import librosa
except ImportError:
    librosa = None

class AudioDenoiser:
    """
    Handles audio noise reduction using Spectral Gating (Spectral Subtraction).
    This is much more effective for background noise (jackhammers, static) than simple filtering.
    """
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def denoise(self, audio_data):
        """
        Apply Spectral Gating to remove background noise.
        """
        if audio_data is None or len(audio_data) == 0:
            return audio_data
            
        # Ensure input is float32
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        audio_data = audio_data.astype(np.float32)

        if librosa is None:
            # Fallback to the old Wiener filter if librosa isn't available
            print("[AudioDenoiser] Librosa not found, using basic Wiener filter.")
            try:
                return scipy.signal.wiener(audio_data)
            except:
                return audio_data

        # --- Spectral Gating Implementation ---
        
        # 1. Compute Short-Time Fourier Transform (STFT)
        # n_fft=2048 corresponds to ~128ms window at 16kHz, good for general audio
        # hop_length=512 is typical 25% overlap
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        
        # Calculate magnitude and phase
        magnitude, phase = librosa.magphase(stft)
        
        # 2. Estimate Noise Profile
        # Jackhammer/Industrial noise is often better estimated by the median or lower percentiles
        # We use a slightly more aggressive percentile to catch the high-energy noise floor
        noise_profile = np.percentile(magnitude, 25, axis=1, keepdims=True)
        
        # 3. Compute Threshold as a multiple of noise profile
        # std_thresh = 1.2 is a conservative starting point to avoid killing speech
        std_thresh = 1.2 
        threshold = noise_profile * std_thresh
        
        # 4. Create Soft Mask (Sigmoid-like smoothing)
        # Instead of Hard Gating (0 or 1), we use a gain curve.
        # gain = (magnitude - threshold) / magnitude, clamped at 0.
        # This is essentially spectral subtraction.
        
        # Avoid division by zero
        epsilon = 1e-10
        gain = (magnitude - threshold) / (magnitude + epsilon)
        gain = np.maximum(gain, 0.0) # Clamp negative values to 0
        
        # 5. Smooth the mask
        # Temporal smoothing (across time) and Frequency smoothing
        gain = scipy.ndimage.gaussian_filter1d(gain, sigma=1.0, axis=1) # Time
        gain = scipy.ndimage.gaussian_filter1d(gain, sigma=1.0, axis=0) # Freq
        
        # 6. Apply Mask to Magnitude
        # We also mix back a tiny bit of the original noisy signal (0.05) to prevent
        # "dead silence" unnatural artifacts which Whisper dislikes.
        magnitude_clean = magnitude * gain + magnitude * 0.05
        
        # 7. Inverse STFT
        # Recombine with original phase
        stft_clean = magnitude_clean * phase
        audio_clean = librosa.istft(stft_clean, hop_length=hop_length)
        
        # 7. Pad or Trim to match original length (istft can result in slight length diff)
        if len(audio_clean) > len(audio_data):
            audio_clean = audio_clean[:len(audio_data)]
        elif len(audio_clean) < len(audio_data):
            padding = np.zeros(len(audio_data) - len(audio_clean), dtype=np.float32)
            audio_clean = np.concatenate((audio_clean, padding))
            
        return audio_clean

class VideoDenoiser:
    """
    Handles video frame noise reduction and smoothing.
    """
    def __init__(self):
        pass

    def smooth_frame(self, frame):
        """
        Apply Gaussian Blur to remove visual noise from a frame.
        Args:
            frame (np.array): Video frame (OpenCV image).
        Returns:
            np.array: Smoothed frame.
        """
        if cv2 is None:
            return frame
            
        try:
            # Gaussian smoothing
            smoothed = cv2.GaussianBlur(frame, (3, 3), 0)
            return smoothed
        except Exception as e:
            print(f"[VideoDenoiser] Error smoothing frame: {e}")
            return frame
