import argparse
import json
import time
import sys
import threading
import os
from audio_capture import AudioCapture
from whisper_asr import WhisperASR
from text_refinement_t5 import TextRefinementT5
from utils.performance_monitor import Timer, get_cpu_usage, get_memory_usage

def log_performance(data, log_file="logs/performance_log.json"):
    """Appends performance data to a JSON log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except:
            logs = []
    
    logs.append(data)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

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
    
    # Initialize CPU monitor counter
    get_cpu_usage()
    
    print("--- Pipeline Ready. Listening... ---")
    
    try:
        capture.start()
        
        while True:
            # 2. Capture Audio
            # This blocks until a speech segment is committed (silence detected)
            audio_segment = capture.get_audio_segment()
            
            # Start performance monitoring
            timer_total = Timer()
            timer_asr = Timer()
            timer_t5 = Timer()
            
            timer_total.start()
            cpu_before = get_cpu_usage()
            
            # 3. ASR Transcription
            timer_asr.start()
            transcript_result = asr.transcribe(audio_segment)
            timer_asr.stop()
            
            if transcript_result:
                raw_text = transcript_result["text"]
                asr_confidence = transcript_result["asr_confidence"]
                
                # Failure Handling: Check confidence
                low_confidence = bool(asr_confidence < 0.4) 
                
                # 4. Text Refinement
                timer_t5.start()
                if not low_confidence:
                    refined_text = refiner.refine(raw_text)
                else:
                    refined_text = raw_text
                timer_t5.stop()
                
                timer_total.stop()
                cpu_after = get_cpu_usage()
                
                # Metrics Calculation
                total_latency = timer_total.elapsed_ms()
                asr_latency = timer_asr.elapsed_ms()
                t5_latency = timer_t5.elapsed_ms()
                avg_cpu = (cpu_before + cpu_after) / 2
                memory_mb = get_memory_usage()
                
                # 5. Structured Output
                output = {
                    "modality": "audio",
                    "raw_text": raw_text,
                    "refined_text": refined_text,
                    "asr_confidence": float(f"{asr_confidence:.4f}"),
                    "low_confidence": low_confidence,
                    "latency_ms": {
                        "total": float(f"{total_latency:.2f}"),
                        "asr": float(f"{asr_latency:.2f}"),
                        "t5": float(f"{t5_latency:.2f}")
                    },
                    "cpu_usage_percent": float(f"{avg_cpu:.2f}"),
                    "memory_usage_mb": float(f"{memory_mb:.2f}")
                }
                
                # 6. Log Performance
                log_performance(output)
                
                # 7. Print Clean Output for Terminal
                print("\n" + "="*30)
                print(f"Raw Text:\n{raw_text}")
                print(f"\nRefined Text:\n{refined_text}")
                print(f"\nConfidence:\n{asr_confidence:.4f}")
                print(f"\nLatency:")
                print(f"  ASR:   {asr_latency:.2f} ms")
                print(f"  T5:    {t5_latency:.2f} ms")
                print(f"  Total: {total_latency:.2f} ms")
                print(f"\nCPU Usage:\n{avg_cpu:.2f} %")
                print("="*30 + "\n")
                
                # Still print JSON for potential downstream pipe integration
                # print(json.dumps(output, ensure_ascii=False), flush=True)
                
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        capture.stop()

if __name__ == "__main__":
    main()
