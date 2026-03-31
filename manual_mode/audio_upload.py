import os
import sys
import numpy as np

# Adjust path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from whisper_asr import WhisperASR
    from text_refinement_t5 import TextRefinementT5
    from utils.performance_monitor import Timer, get_cpu_usage, get_memory_usage
except ImportError:
    # Adjust path to import from parent directory if needed
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from whisper_asr import WhisperASR
    from text_refinement_t5 import TextRefinementT5
    from utils.performance_monitor import Timer, get_cpu_usage, get_memory_usage

from noise_reduction import AudioDenoiser

try:
    import librosa
except ImportError:
    librosa = None
    import scipy.io.wavfile

class ManualAudioProcessor:
    def __init__(self, whisper_model="base", t5_model="AventIQ-AI/T5-small-grammar-correction"):
        print(f"[ManualAudioProcessor] Initializing... (Model: {whisper_model})")
        self.denoiser = AudioDenoiser()
        self.asr = WhisperASR(model_size=whisper_model)
        self.refiner = TextRefinementT5(model_name=t5_model, device="cpu")
        get_cpu_usage() # Initialize CPU counter
        
    def load_audio(self, file_path):
        """
        Load audio file to 16kHz mono float32.
        Accepts .wav, .mp3 if librosa is available.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        if librosa:
            # librosa handles mp3/wav and resampling
            # sr=16000 ensures compatibility with Whisper
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            return audio.astype(np.float32)
        else:
            # Fallback to scipy (WAV only)
            if not file_path.lower().endswith(".wav"):
                raise ValueError("librosa not installed. Only .wav files are supported in this mode.")
            sr, audio = scipy.io.wavfile.read(file_path)
            
            # Handle stereo -> mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Warning if SR mismatch
            if sr != 16000:
                print(f"[Warning] Sample rate {sr} != 16000. Timings may be off or Whisper may degrade.")
            
            # Convert int to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128.0
                
            return audio

    def process_file(self, file_path):
        """
        Full pipeline: Load -> Denoise -> Whisper -> Refine
        """
        print(f"[ManualAudioProcessor] Processing {file_path}...")
        
        timer_total = Timer()
        timer_asr = Timer()
        timer_t5 = Timer()
        
        timer_total.start()
        cpu_before = get_cpu_usage()
        
        # 1. Load
        try:
            audio = self.load_audio(file_path)
        except Exception as e:
            return {"error": f"Error loading audio: {e}"}
        
        # 2. Denoise (Spectral Gating / Wiener)
        print("[ManualAudioProcessor] Applying noise reduction...")
        audio_clean = self.denoiser.denoise(audio)
        
        # 3. Whisper ASR
        print("[ManualAudioProcessor] Running Whisper ASR...")
        timer_asr.start()
        result = self.asr.transcribe(audio_clean)
        timer_asr.stop()
        
        if not result or not result.get("text"):
            return {"error": "No speech detected (ASR returned empty)."}
        
        raw_text = result["text"]
        asr_confidence = result.get("asr_confidence", 0.0)
        
        # 4. Refine
        print("[ManualAudioProcessor] Refining text with T5...")
        timer_t5.start()
        refined_text = self.refiner.refine(raw_text)
        timer_t5.stop()
        
        timer_total.stop()
        cpu_after = get_cpu_usage()
        
        # Metrics Calculation
        total_latency = timer_total.elapsed_ms()
        asr_latency = timer_asr.elapsed_ms()
        t5_latency = timer_t5.elapsed_ms()
        avg_cpu = (cpu_before + cpu_after) / 2
        memory_mb = get_memory_usage()
        
        output = {
            "modality": "audio_manual",
            "file_path": file_path,
            "raw_text": raw_text,
            "refined_text": refined_text,
            "asr_confidence": float(f"{asr_confidence:.4f}"),
            "latency_ms": {
                "total": float(f"{total_latency:.2f}"),
                "asr": float(f"{asr_latency:.2f}"),
                "t5": float(f"{t5_latency:.2f}")
            },
            "cpu_usage_percent": float(f"{avg_cpu:.2f}"),
            "memory_usage_mb": float(f"{memory_mb:.2f}")
        }
        
        return output

if __name__ == "__main__":
    if len(sys.argv) > 1:
        processor = ManualAudioProcessor()
        print(processor.process_file(sys.argv[1]))
