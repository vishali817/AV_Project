import argparse
import os
import sys
import json

# Ensure we can import from local package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_upload import ManualAudioProcessor
from video_upload import ManualVideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Manual Processing Mode for AV System")
    parser.add_argument("mode", choices=["audio", "video"], help="Mode to run: 'audio' or 'video'")
    parser.add_argument("filepath", type=str, help="Path to the input file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights (video mode only, optional)")
    
    args = parser.parse_args()
    
    print("="*40)
    print(f"Manual Processing Mode: {args.mode.upper()}")
    print("="*40)
    print("Loading modules... (Please wait, first run may be slow due to model downloads)")

    if args.mode == "audio":
        if not os.path.isfile(args.filepath):
            print(f"Error: File '{args.filepath}' not found.")
            return

        processor = ManualAudioProcessor()
        try:
            result = processor.process_file(args.filepath)
        except Exception as e:
            print(f"Error processing audio: {e}")
            return
            
        if isinstance(result, dict) and "error" not in result:
            print("\n" + "="*30)
            print(f"Raw Text:\n{result['raw_text']}")
            print(f"\nRefined Text:\n{result['refined_text']}")
            print(f"\nConfidence:\n{result['asr_confidence']:.4f}")
            print(f"\nLatency:")
            print(f"  ASR:   {result['latency_ms']['asr']:.2f} ms")
            print(f"  T5:    {result['latency_ms']['t5']:.2f} ms")
            print(f"  Total: {result['latency_ms']['total']:.2f} ms")
            print(f"\nCPU Usage:\n{result['cpu_usage_percent']:.2f} %")
            print("="*30 + "\n")
            
            # Print full JSON for potential research use
            print(f"Full Metrics JSON: {json.dumps(result, indent=4)}")
        else:
            print(f"\nError or Invalid Output: {result}")
        
    elif args.mode == "video":
        if not os.path.isfile(args.filepath):
            print(f"Error: File '{args.filepath}' not found.")
            return

        processor = ManualVideoProcessor(lip_model_path=args.model_path)
        try:
            result = processor.process_file(args.filepath)
        except Exception as e:
            print(f"Error processing video: {e}")
            return
            
        print("\n" + "-"*40)
        print("FINAL TRANSCRIPT:")
        print("-"*40)
        print(result)
        print("-"*40)

if __name__ == "__main__":
    main()
