import pyaudio
import numpy as np
import threading
import queue
import time

class AudioCapture:
    def __init__(self, rate=16000, chunk_ms=30, vad_threshold_db=-40.0, silence_duration_ms=1200):
        """
        Initializes the AudioCapture module.
        
        Args:
            rate (int): Sampling rate in Hz (default 16000).
            chunk_ms (int): Duration of each audio chunk in milliseconds (default 30).
            vad_threshold_db (float): RMS energy threshold in Decibels for VAD (default -40).
            silence_duration_ms (int): Duration of silence to trigger a segment commit (default 1200).
        """
        self.rate = rate
        self.chunk_size = int(rate * chunk_ms / 1000)
        self.vad_threshold = 10 ** (vad_threshold_db / 20) # Convert dB to linear amplitude
        self.silence_limit_chunks = int(silence_duration_ms / chunk_ms)
        
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        
        # Queues for communicating with the pipeline
        self.audio_queue = queue.Queue() # To send complete speech segments
        
        self.frames = [] # Buffer for current speech frames
        self.silence_counter = 0
        self.is_speaking = False
        self.lock = threading.Lock()

    def _calculate_rms(self, audio_data):
        """Calculates Root Mean Square amplitude of the audio chunk."""
        # Convert to float32 for calculation
        data_float = audio_data.astype(np.float32)
        rms = np.sqrt(np.mean(data_float**2))
        return rms

    def _process_stream(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio stream."""
        if not self.running:
            return (None, pyaudio.paComplete)
        
        # Convert raw bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Energy-based VAD
        rms = self._calculate_rms(audio_data)
        
        # Determine if current frame is speech or silence
        if rms > self.vad_threshold:
            self.is_speaking = True
            self.silence_counter = 0
            self.frames.append(audio_data)
        else:
            if self.is_speaking:
                self.frames.append(audio_data) # Keep trailing silence for better context
                self.silence_counter += 1
                
                # If silence exceeds limit, commit the segment
                if self.silence_counter >= self.silence_limit_chunks:
                    self._commit_segment()
                    self.is_speaking = False
                    self.silence_counter = 0
            else:
                # Pre-buffer functionality could be added here to catch start of speech
                pass
        
        # Safety: Force commit if segment is too long (e.g. > 10 seconds) to maintain latency constraints
        # 10 seconds / 0.030s per chunk ~= 333 chunks
        if len(self.frames) > 333:
             self._commit_segment()
             self.is_speaking = False # Reset state to treat next as new segment (or continuation)
             self.silence_counter = 0

        return (in_data, pyaudio.paContinue)

    def _commit_segment(self):
        """Packages the buffered frames into a segment and puts it in the queue."""
        if not self.frames:
            return
            
        # Concatenate all frames
        full_audio = np.concatenate(self.frames)
        
        # Optional: Duration check (ignore too short segments i.e. clicks)
        duration = len(full_audio) / self.rate
        if duration > 0.3: # Ignore segments shorter than 300ms
            self.audio_queue.put(full_audio)
            
        # Clear buffer
        self.frames = []

    def start(self):
        """Starts the audio capture stream."""
        self.running = True
        self.stream = self.audio_interface.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._process_stream
        )
        self.stream.start_stream()
        print(f"[AudioCapture] Microphone streaming started at {self.rate}Hz")

    def stop(self):
        """Stops the audio capture stream."""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio_interface.terminate()
        print("[AudioCapture] Microphone streaming stopped")

    def get_audio_segment(self):
        """Blocking call to retrieve the next audio segment."""
        return self.audio_queue.get()
