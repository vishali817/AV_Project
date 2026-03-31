import sys
import os
import cv2
import json

# Add parent directory to path to import utils and modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual.webcam_capture import WebcamCapture
from visual.mouth_detector import MouthDetector
from visual.frame_buffer import FrameBuffer
from visual.cnn_lstm_model import VisualSpeechInference
from text_refinement_t5 import TextRefinementT5
from utils.performance_monitor import Timer, get_cpu_usage, get_memory_usage

def main():
    print("--- Initializing Visual Pipeline (CNN+LSTM) ---")
    
    # 1. Init Webcam (STEP 1)
    capture = WebcamCapture(camera_index=0)
    
    # 2. Init Mouth Detector (STEP 2)
    detector = MouthDetector()
    
    # 3. Init Frame Buffer (STEP 3)
    buffer = FrameBuffer(sequence_length=25)
    
    # 4. Init CNN+LSTM Visual Speech Model (replaces LipNet)
    visual_model = VisualSpeechInference()
    
    # 5. Init T5 Refinement
    refiner = TextRefinementT5(model_name="AventIQ-AI/T5-small-grammar-correction", device="cpu")
    
    # Initialize CPU monitor counter
    get_cpu_usage()
    
    print("--- Pipeline Ready. Starting Webcam... Press 'q' to exit. ---")
    
    for frame in capture.start_webcam():
        timer_total = Timer()
        timer_total.start()
        
        cpu_before = get_cpu_usage()
        
        # Mouth Detection
        mouth_frame, bbox = detector.detect_and_crop(frame)
        display_frame = frame.copy()
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            buffer.add_frame(mouth_frame)
            
            # Step 5 - If buffer is full, run Inference and T5
            if buffer.is_full():
                timer_model = Timer()
                timer_t5 = Timer()
                
                # CNN+LSTM Inference (replaces LipNet)
                timer_model.start()
                seq = buffer.get_sequence()
                model_out = visual_model.predict_visual_text(seq)
                timer_model.stop()
                
                raw_text = model_out["visual_text"]
                confidence = model_out["visual_confidence"]
                
                # T5 Refinement
                timer_t5.start()
                if raw_text.strip():
                    refined_text = refiner.refine(raw_text)
                else:
                    refined_text = raw_text
                timer_t5.stop()
                
                timer_total.stop()
                cpu_after = get_cpu_usage()
                
                # Metrics (Step 6)
                total_latency = timer_total.elapsed_ms()
                model_latency = timer_model.elapsed_ms()
                t5_latency = timer_t5.elapsed_ms()
                avg_cpu = (cpu_before + cpu_after) / 2
                memory_mb = get_memory_usage()
                
                # Make JSON Output match structure
                output = {
                    "modality": "visual",
                    "raw_text": raw_text,
                    "refined_text": refined_text,
                    "visual_confidence": float(f"{confidence:.4f}"),
                    "latency_ms": {
                        "total": float(f"{total_latency:.2f}"),
                        "cnn_lstm": float(f"{model_latency:.2f}"),
                        "t5": float(f"{t5_latency:.2f}")
                    },
                    "cpu_usage_percent": float(f"{avg_cpu:.2f}"),
                    "memory_usage_mb": float(f"{memory_mb:.2f}")
                }
                
                if raw_text.strip() or True:
                    # Output Format (STEP 7)
                    print("\n" + "="*30)
                    print(f"Visual Raw Text:\n{raw_text}")
                    print(f"\nRefined Text:\n{refined_text}")
                    print(f"\nConfidence:\n{confidence:.2f}")
                    print(f"\nLatency:")
                    print(f"  CNN-LSTM: {model_latency:.0f} ms")
                    print(f"  T5:      {t5_latency:.0f} ms")
                    print(f"  Total:   {total_latency:.0f} ms")
                    print(f"\nCPU Usage:\n{avg_cpu:.0f} %")
                    print("="*30 + "\n")
                    
                    # Prevent instant repeated transcription spamming, wait or overlap
                    # We clear the buffer so we capture fresh context next
                    buffer.clear()
        
        # Display Webcam (STEP 8 & 9)
        cv2.imshow("Webcam Feed", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
