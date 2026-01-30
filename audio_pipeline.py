import argparse
import json
import time
import sys
import threading
from audio_capture import AudioCapture
from whisper_asr import WhisperASR
from text_refinement_t5 import TextRefinementT5

def main():
    parser = argparse.ArgumentParser(description="Audio Module Pipeline")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "base"], help="Whisper model size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    args = parser.parse_args()

    # 1. Initialize Components
    print("--- Initializing Audio Pipeline ---")
    
    # Audio Capture (16kHz, 30ms frames, -40dB threshold, 1200ms silence to commit)
    capture = AudioCapture(rate=16000, chunk_ms=30, vad_threshold_db=-40, silence_duration_ms=1200)
    
    # ASR (Whisper)
    asr = WhisperASR(model_size=args.model_size, device="cpu", compute_type="int8", cpu_threads=4)
    
    # Refinement (T5-Small Grammar Correction)
    refiner = TextRefinementT5(model_name="AventIQ-AI/T5-small-grammar-correction", device="cpu")
    
    print("--- Pipeline Ready. Listening... ---")
    
    try:
        capture.start()
        
        while True:
            # 2. Capture Audio
            # This blocks until a speech segment is committed (silence detected)
            audio_segment = capture.get_audio_segment()
            
            # Timestamp for latency check
            process_start = time.time()
            
            # 3. ASR Transcription
            transcript_result = asr.transcribe(audio_segment)
            
            if transcript_result:
                raw_text = transcript_result["text"]
                confidence = transcript_result["confidence"]
                
                # Failure Handling: Check confidence
                low_confidence = bool(confidence < 0.4) # Threshold can be tuned
                
                # 4. Text Refinement
                if not low_confidence:
                    refined_text = refiner.refine(raw_text)
                else:
                    refined_text = raw_text # Don't refine garbage
                
                # 5. Fusion-Ready Output
                output = {
                    "modality": "audio",
                    "raw_text": raw_text,
                    "refined_text": refined_text,
                    "confidence": float(f"{confidence:.4f}"),
                    "low_confidence": low_confidence,
                    "processing_latency": float(f"{time.time() - process_start:.4f}")
                }
                
                # Print JSON line for downstream integration (LipNet fusion)
                print(json.dumps(output, ensure_ascii=False), flush=True)
                
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        capture.stop()

if __name__ == "__main__":
    main()
