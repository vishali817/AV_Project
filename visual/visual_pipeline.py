import argparse
import json
import os
import sys
import time

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visual.video_loader import load_video
from visual.frame_extractor import extract_frames
from visual.mouth_detector import MouthDetector
from visual.cnn_lstm_model import VisualSpeechInference
from text_refinement_t5 import TextRefinementT5
from utils.performance_monitor import Timer, get_cpu_usage, get_memory_usage

def main():
    parser = argparse.ArgumentParser(description="Visual Pipeline - Lip Reading (CNN+LSTM)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    args = parser.parse_args()

    # 1. Initialize Components
    print("\n--- Initializing Visual Pipeline (CNN+LSTM) ---")
    
    # Initialize CPU monitor counter
    get_cpu_usage()
    
    try:
        visual_model = VisualSpeechInference()
        refiner = TextRefinementT5(device="cpu")
        detector = MouthDetector()
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 2. Start Processing
    print(f"--- Processing Video: {args.video} ---")
    
    timer_total = Timer()
    timer_model = Timer()
    timer_t5 = Timer()
    
    timer_total.start()
    cpu_before = get_cpu_usage()
    
    try:
        # Step A: Load Video
        cap = load_video(args.video)
        
        # Step B: Extract Frames
        print("[Visual] Extracting frames...")
        frames, count = extract_frames(cap)
        print(f"[Visual] Extracted {count} frames.")
        
        # Step C: Detect Mouth Region
        print("[Visual] Detecting mouth ROI...")
        mouth_frames, q_status = detector.detect_mouth(frames)
        detector.save_debug_video(mouth_frames) # Save for manual verification
        
        if q_status == "invalid_input":
            invalid_dict = {
                "visual_text": "",
                "visual_confidence": 0.0,
                "status": "invalid_input"
            }
            print("\n" + "="*30)
            print(json.dumps(invalid_dict, indent=4))
            print("="*30 + "\n")
            return
            
        # Step D: CNN+LSTM Inference (replaces LipNet)
        print("[Visual] Running CNN+LSTM visual speech inference...")
        timer_model.start()
        prediction = visual_model.predict_visual_text(mouth_frames)
        timer_model.stop()
        
        raw_text = prediction["visual_text"]
        confidence = prediction["visual_confidence"]
        
        # Step 13 - Handle Low Confidence
        if confidence < 0.3:
            print("[WARNING] Low confidence visual prediction — check input quality")
            
        # Step E: Text Refinement
        print("[Visual] Refining text with T5...")
        timer_t5.start()
        if raw_text:
            refined_text = refiner.refine(raw_text)
        else:
            refined_text = ""
        timer_t5.stop()
        
        timer_total.stop()
        cpu_after = get_cpu_usage()
        
        # 3. Metrics Calculation
        total_latency = timer_total.elapsed_ms()
        model_latency = timer_model.elapsed_ms()
        t5_latency = timer_t5.elapsed_ms()
        avg_cpu = (cpu_before + cpu_after) / 2
        
        # Step 15 - Final Output Format
        output = {
            "visual_text_raw": raw_text,
            "visual_text_refined": refined_text,
            "visual_confidence": float(f"{confidence:.4f}"),
            "latency_ms": {
                "cnn_lstm": float(f"{model_latency:.2f}"),
                "t5": float(f"{t5_latency:.2f}"),
                "total": float(f"{total_latency:.2f}")
            },
            "cpu_usage_percent": float(f"{avg_cpu:.2f}")
        }
        
        print("\n" + "="*30)
        print(json.dumps(output, indent=4))
        print("="*30 + "\n")

    except Exception as e:
        print(f"Pipeline Error: {e}")
    finally:
        timer_total.stop() # Ensure timer doesn't hang if exception occurs

if __name__ == "__main__":
    main()
