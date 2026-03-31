"""
Lip Reader Module (CNN+LSTM)
============================
Replaces the old LipNet-based lip reader with a lightweight CNN+LSTM model.
Uses the VisualSpeechInference class from visual/cnn_lstm_model.py.
"""

import numpy as np
import os
import sys

try:
    import cv2
except ImportError:
    cv2 = None

# Ensure we can import from the visual module in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visual.cnn_lstm_model import VisualSpeechInference


class LipReader:
    """
    LipReader using CNN+LSTM visual speech recognition.
    Drop-in replacement for the old LipNet-based LipReader.
    """

    def __init__(self, model_path=None, device="cpu"):
        """
        Initialize the lip reader with CNN+LSTM model.

        Args:
            model_path: Ignored (kept for backward compatibility). CNN+LSTM doesn't
                        require pre-trained weights for vocabulary-based prediction.
            device: Device string (only 'cpu' supported).
        """
        self.device = device
        self.visual_model = VisualSpeechInference()
        self.model_loaded = True
        print("[LipReader] Initialized with CNN+LSTM visual speech model")

    def load_video(self, video_path):
        """
        Reads video file and extracts frames.

        Args:
            video_path: Path to video file.

        Returns:
            list: List of BGR frames (numpy arrays).
        """
        if cv2 is None:
            print("[LipReader] OpenCV not installed. Cannot process video.")
            return []

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def process_frames(self, frames):
        """
        Preprocess frames: crop mouth region, resize to (50, 100), convert to grayscale.

        Args:
            frames: List of BGR frame arrays.

        Returns:
            list: List of preprocessed grayscale frames (50, 100).
        """
        if not frames:
            return []

        processed = []
        for frame in frames:
            if cv2 is not None:
                # Convert BGR to grayscale
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif len(frame.shape) == 3 and frame.shape[2] == 1:
                    gray = frame[:, :, 0]
                else:
                    gray = frame

                # Center crop and resize to (50, 100)
                h, w = gray.shape[:2]
                cy, cx = h // 2, w // 2
                y1 = max(0, cy - 25)
                y2 = min(h, cy + 25)
                x1 = max(0, cx - 50)
                x2 = min(w, cx + 50)

                crop = gray[y1:y2, x1:x2]
                crop = cv2.resize(crop, (100, 50))
                processed.append(crop)
            else:
                processed.append(frame)

        return processed

    def predict(self, video_path):
        """
        Generate prediction from video file.

        Args:
            video_path: Path to video file.

        Returns:
            str: Predicted word.
        """
        frames = self.load_video(video_path)
        if not frames:
            return ""
        return self.predict_frames(frames)

    def predict_frames(self, frames):
        """
        Generate prediction from list of frames.

        Args:
            frames: List of frame arrays (BGR or grayscale).

        Returns:
            str: Predicted word from vocabulary.
        """
        if not self.model_loaded:
            return "[LipReader] Model not loaded."

        # Preprocess frames to mouth-region crops
        processed = self.process_frames(frames)
        if not processed:
            return ""

        # Run CNN+LSTM inference
        result = self.visual_model.predict_visual_text(processed)

        predicted_word = result["visual_text"]
        confidence = result["visual_confidence"]

        print(f"[LipReader] Prediction: '{predicted_word}' (confidence: {confidence:.4f})")
        return predicted_word
