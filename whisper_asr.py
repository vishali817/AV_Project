from faster_whisper import WhisperModel
import numpy as np
import time
import os

class WhisperASR:
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8", cpu_threads=4):
        """
        Initializes the Whisper ASR module optimized for CPU.
        
        Args:
            model_size (str): "tiny" or "base" (default: "tiny").
            device (str): "cpu" is required for this constraint.
            compute_type (str): "int8" for quantized CPU inference (faster, less memory).
            cpu_threads (int): Number of threads to use for inference.
        """
        print(f"[WhisperASR] Loading model '{model_size}' on {device} with {compute_type} precision...")
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type, 
            cpu_threads=cpu_threads
        )
        print("[WhisperASR] Model loaded successfully.")

    def transcribe(self, audio_data):
        """
        Transcribes audio data using Whisper.
        
        Args:
            audio_data (np.array): Float32 valid audio array (16kHz).
            
        Returns:
            dict: {
                "text": str,
                "segments": list of dicts (for timestamp detail),
                "confidence": float
            }
        """
        # Ensure we have a valid array
        if audio_data is None or len(audio_data) == 0:
            return None

        # Run inference
        # beam_size=1 for fastest greedy decoding
        # temperature=0.0 to be deterministic and robust
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=2, 
            temperature=0.0,
            language="en", # Force English
            condition_on_previous_text=False, # critical for streaming to prevent loops
            vad_filter=False,
            vad_parameters=dict(min_silence_duration_ms=500),
            repetition_penalty=1.1,
            no_speech_threshold=0.6
        )
        
        # Collect results
        full_text = []
        segment_details = []
        confidences = []
        
        for segment in segments:
            full_text.append(segment.text)
            confidences.append(segment.avg_logprob) # Note: avg_logprob is < 0. Convert to prob later if needed.
            segment_details.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "confidence_logprob": segment.avg_logprob
            })
            
        # Optimization: Early exit if nothing found
        if not full_text:
            return None
            
        text_str = " ".join(full_text).strip()
        
        # Calculate an aggregate confidence score (0.0 - 1.0)
        # faster-whisper provides logprob. exp(avg_logprob) approx probability.
        avg_prob = np.exp(np.mean(confidences)) if confidences else 0.0
        
        inference_time = time.time() - start_time
        # print(f"[WhisperASR] Transcribed in {inference_time:.3f}s: '{text_str}' ({avg_prob:.2f})")
        
        return {
            "text": text_str,
            "segments": segment_details,
            "asr_confidence": avg_prob
        }
