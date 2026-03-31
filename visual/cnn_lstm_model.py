"""
CNN + LSTM Visual Speech Recognition Model
==========================================
Lightweight, CPU-friendly model for visual speech recognition.
Replaces LipNet with a simpler CNN feature extractor + LSTM sequence model.

Input:  (sequence_length, 50, 100) grayscale mouth frames
Output: predicted word from a fixed vocabulary with confidence score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Fixed Vocabulary ---
VOCAB = ["hello", "yes", "no", "thanks", "help"]


class CNNFeatureExtractor(nn.Module):
    """
    2D CNN to extract spatial features from individual grayscale frames.
    Input:  (B, 1, 50, 100)  — single grayscale frame
    Output: (B, feature_dim)  — flattened feature vector
    """

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (32, 25, 50)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (64, 12, 25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> (128, 6, 12)

        # After 3x maxpool(2,2) on (50, 100): 50->25->12->6, 100->50->25->12
        self.feature_dim = 128 * 6 * 12  # = 9216

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) — grayscale frame
        Returns:
            (B, feature_dim) — flattened spatial features
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # flatten
        return x


class VisualSpeechModel(nn.Module):
    """
    CNN + LSTM model for visual speech recognition.

    Architecture:
        1. CNN extracts spatial features per frame
        2. LSTM processes the sequence of frame features
        3. FC layer maps final hidden state → vocabulary logits
        4. Softmax produces word probabilities

    Input:  (B, T, 1, 50, 100) — batch of frame sequences
    Output: (B, vocab_size) — word probabilities
    """

    def __init__(self, vocab_size=len(VOCAB), lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super(VisualSpeechModel, self).__init__()

        self.cnn = CNNFeatureExtractor()
        feature_dim = self.cnn.feature_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, vocab_size)

    def forward(self, x):
        """
        Args:
            x: (B, T, 1, H, W) — sequence of grayscale frames
        Returns:
            (B, vocab_size) — log-softmax probabilities over vocabulary
        """
        B, T, C, H, W = x.size()

        # Process each frame through CNN
        x = x.view(B * T, C, H, W)          # (B*T, 1, H, W)
        cnn_features = self.cnn(x)            # (B*T, feature_dim)
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, feature_dim)

        # LSTM sequence processing
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)  # lstm_out: (B, T, hidden)

        # Use last time-step output for classification
        last_output = lstm_out[:, -1, :]      # (B, hidden)
        last_output = self.dropout(last_output)

        logits = self.fc(last_output)         # (B, vocab_size)
        return logits


class VisualSpeechInference:
    """
    Inference wrapper for the CNN+LSTM visual speech recognition model.

    Handles:
        - Frame preprocessing (grayscale conversion, normalization, resizing)
        - Model inference with torch.no_grad()
        - Word prediction with confidence score
        - Debug output printing
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.vocab = VOCAB
        self.model = VisualSpeechModel(vocab_size=len(self.vocab)).to(self.device)
        self.model.eval()
        self.model_ready = True
        print(f"[CNN-LSTM] Visual speech model initialized on CPU")
        print(f"[CNN-LSTM] Vocabulary: {self.vocab}")

    def _preprocess_frames(self, frame_sequence):
        """
        Convert raw frames to normalized grayscale tensors of shape (50, 100).

        Args:
            frame_sequence: list of numpy arrays (BGR, grayscale, or mixed)

        Returns:
            torch.Tensor of shape (1, T, 1, 50, 100) or None if invalid
        """
        if not frame_sequence or len(frame_sequence) == 0:
            return None

        processed = []
        for frame in frame_sequence:
            # Convert to grayscale if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                gray = frame[:, :, 0]
            else:
                gray = frame

            # Resize to expected dimensions (50, 100) if needed
            if gray.shape != (50, 100):
                import cv2
                gray = cv2.resize(gray, (100, 50))

            # Normalize to [0, 1]
            normalized = gray.astype(np.float32) / 255.0
            processed.append(normalized)

        # Stack: (T, 50, 100) -> (1, T, 1, 50, 100)
        data = np.array(processed)                          # (T, 50, 100)
        data = np.expand_dims(data, axis=1)                 # (T, 1, 50, 100)
        data = np.expand_dims(data, axis=0)                 # (1, T, 1, 50, 100)

        tensor = torch.FloatTensor(data).to(self.device)
        return tensor

    def predict_visual_text(self, frames):
        """
        Main inference function. Replaces LipNet predict_sequence / predict_lip_text.

        Args:
            frames: list of numpy frame arrays (mouth ROI crops)

        Returns:
            dict: {
                "visual_text": str — predicted word,
                "visual_confidence": float — prediction confidence (0.0 - 1.0)
            }
        """
        if not self.model_ready or not frames or len(frames) == 0:
            return {"visual_text": "", "visual_confidence": 0.0}

        # Preprocess
        tensor = self._preprocess_frames(frames)
        if tensor is None:
            return {"visual_text": "", "visual_confidence": 0.0}

        # Inference (CPU-safe, no gradient tracking)
        with torch.no_grad():
            logits = self.model(tensor)                     # (1, vocab_size)
            probabilities = F.softmax(logits, dim=1)        # (1, vocab_size)

            confidence, predicted_idx = torch.max(probabilities, dim=1)
            confidence = float(confidence.item())
            predicted_idx = int(predicted_idx.item())

        predicted_word = self.vocab[predicted_idx]

        # --- Debug Output ---
        print(f"[CNN-LSTM] Predicted word : {predicted_word}")
        print(f"[CNN-LSTM] Confidence     : {confidence:.4f}")
        print(f"[CNN-LSTM] Sequence length: {tensor.size(1)} frames")
        print(f"[CNN-LSTM] All probs      : {dict(zip(self.vocab, [f'{p:.4f}' for p in probabilities.squeeze().tolist()]))}")

        return {
            "visual_text": predicted_word,
            "visual_confidence": confidence,
        }

    # --- Aliases for backward compatibility ---
    def predict_sequence(self, frames):
        """Alias for predict_visual_text (replaces LipNetInference.predict_sequence)."""
        return self.predict_visual_text(frames)

    def predict_lip_text(self, frames):
        """Alias for predict_visual_text (replaces LipNetInference.predict_lip_text)."""
        return self.predict_visual_text(frames)
