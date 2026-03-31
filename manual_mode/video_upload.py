import os
import sys

# Adjust path to import from parent directory to access shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from text_refinement_t5 import TextRefinementT5
except ImportError:
    # Fallback
    print("[ManualVideoProcessor] Warning: Relative import failed.")
    pass

from lip_reader import LipReader
from noise_reduction import VideoDenoiser

class ManualVideoProcessor:
    def __init__(self, lip_model_path=None):
        print("[ManualVideoProcessor] Initializing... (This might take a moment)")
        self.lip_reader = LipReader(model_path=lip_model_path)
        self.denoiser = VideoDenoiser()
        self.refiner = TextRefinementT5(device="cpu")

    def process_file(self, file_path):
        """
        Full pipeline: Load Video -> Frame Extraction -> Visual Denoise -> CNN+LSTM -> Refine
        """
        print(f"[ManualVideoProcessor] Processing {file_path}...")
        
        if not os.path.exists(file_path):
            return "Error: File not found."

        # 1. Load Frames
        frames = self.lip_reader.load_video(file_path)
        if not frames:
            return "Error: Could not load video or no frames extracted."
            
        print(f"[ManualVideoProcessor] Extracted {len(frames)} frames.")

        # 2. Visual Denoising/Smoothing
        # Apply smoothing to each frame before lip reading
        print("[ManualVideoProcessor] Applying visual noise reduction...")
        cleaned_frames = [self.denoiser.smooth_frame(f) for f in frames]
        
        # 3. Lip Reading (CNN+LSTM)
        print("[ManualVideoProcessor] Running CNN+LSTM visual speech model...")
        raw_text = self.lip_reader.predict_frames(cleaned_frames)
        print(f"[ManualVideoProcessor] Raw Lip Transcript: {raw_text}")
        
        # 4. Refine
        # Skip refinement if prediction is empty
        if not raw_text or not raw_text.strip():
             refined_text = raw_text 
        else:
             print("[ManualVideoProcessor] Refining text with T5...")
             refined_text = self.refiner.refine(raw_text)
             
        print(f"[ManualVideoProcessor] Refined Transcript: {refined_text}")
        
        return refined_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        processor = ManualVideoProcessor()
        print(processor.process_file(sys.argv[1]))
