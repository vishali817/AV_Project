import numpy as np

class FrameBuffer:
    def __init__(self, sequence_length=25):
        self.sequence_length = sequence_length
        self.buffer = []
        
    def add_frame(self, frame):
        self.buffer.append(frame)
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
            
    def is_full(self):
        return len(self.buffer) == self.sequence_length
        
    def get_sequence(self):
        return self.buffer.copy()
        
    def clear(self):
        self.buffer = []
