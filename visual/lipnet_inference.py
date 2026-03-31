import torch
import torch.nn as nn
import numpy as np
import os

class LipNet(nn.Module):
    def __init__(self, vocab_size=28):
        super(LipNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout1 = nn.Dropout3d(0.5)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout2 = nn.Dropout3d(0.5)
        
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout3 = nn.Dropout3d(0.5)

        self.gru1 = nn.GRU(96 * 3 * 6, 256, bidirectional=True, batch_first=True) 
        self.dropout4 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.dropout3(x)
        
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b, t, -1)
        
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        
        x, _ = self.gru1(x)
        x = self.dropout4(x)
        x, _ = self.gru2(x)
        x = self.dropout5(x)
        
        x = self.fc(x)
        return x

class LipNetInference:
    def __init__(self):
        self.device = torch.device('cpu')
        self.vocab = "abcdefghijklmnopqrstuvwxyz' "
        self.model = LipNet(vocab_size=29).to(self.device)
        self.model_loaded = False
        
    def load_lipnet_model(self, model_path="../lipnet_weights.pth"):
        if not os.path.exists(model_path):
            model_path = "lipnet_weights.pth"
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if 'fc.weight' in state_dict:
                    checkpoint_vocab_size = state_dict['fc.weight'].shape[0]
                    if checkpoint_vocab_size != 29:
                        self.model.fc = nn.Linear(512, checkpoint_vocab_size).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_loaded = True
                print(f"[LipNet] Model loaded from {model_path}")
            except Exception as e:
                print(f"[LipNet] Error loading weights: {e}")
        else:
            print(f"[LipNet] Warning: Model weights not found at {model_path}")
            
    def predict_sequence(self, frame_sequence):
        if not self.model_loaded or not frame_sequence:
            return {"visual_text": "", "visual_confidence": 0.0}
            
        processed_frames = []
        for frame in frame_sequence:
            # Check logic for converting depending on input dims
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                import cv2
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                frame_gray = frame[:, :, 0]
            else:
                frame_gray = frame
                
            norm_frame = frame_gray.astype(np.float32) / 255.0
            processed_frames.append(norm_frame)
            
        data = np.array(processed_frames) # (T, 50, 100)
        data = np.expand_dims(data, axis=-1) # (T, 50, 100, 1)
        data = np.expand_dims(data, axis=0)  # (1, T, 50, 100, 1)
        
        # PyTorch expects shape (B, C, T, H, W)
        tensor = torch.FloatTensor(data).permute(0, 4, 1, 2, 3).to(self.device)
        
        # Duplicate channels if model expects 3
        if self.model.conv1.in_channels == 3:
            tensor = tensor.repeat(1, 3, 1, 1, 1)
            
        with torch.no_grad():
            logits = self.model(tensor)
            
        probs = torch.softmax(logits, dim=2)
        max_probs, preds = torch.max(probs, dim=2)
        confidence = float(torch.mean(max_probs).item())
        
        preds = preds.squeeze(0).cpu().numpy()
        blank_idx = logits.shape[2] - 1
        decoded_chars = []
        prev = -1
        
        for p in preds:
            if p != prev and p != blank_idx:
                if p < len(self.vocab):
                    decoded_chars.append(self.vocab[p])
            prev = p
            
        text = "".join(decoded_chars).strip()
        
        return {
            "visual_text": text,
            "visual_confidence": confidence
        }
