"""
Mouth Detector Module
=====================
Extracts mouth ROI from video frames using Mediapipe Face Mesh.
Falls back to center-crop if Mediapipe is unavailable or fails.

Never crashes — all errors are caught and handled gracefully.
"""

import cv2
import numpy as np

# --- Safe Mediapipe Import ---
try:
    import mediapipe as mp
    _MP_AVAILABLE = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh')
    if _MP_AVAILABLE:
        print("[MouthDetector] Mediapipe solutions module available")
    else:
        print("[MouthDetector] WARNING: Mediapipe installed but 'solutions.face_mesh' not found")
        print("[MouthDetector] Installed version:", getattr(mp, '__version__', 'unknown'))
except ImportError:
    _MP_AVAILABLE = False
    print("[MouthDetector] WARNING: Mediapipe not installed")


class MouthDetector:
    """
    Detects and crops the mouth region from video frames.
    
    Primary:  Mediapipe Face Mesh landmarks
    Fallback: Center-crop (lower-center region of frame)
    """

    def __init__(self):
        self.use_mediapipe = False
        self.face_mesh = None
        self.mouth_points = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375]
        self._frame_count = 0

        # --- Try to initialize Mediapipe ---
        if _MP_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True
                )
                self.use_mediapipe = True
                print("[MouthDetector] ✓ Mediapipe Face Mesh loaded successfully")
            except Exception as e:
                print(f"[MouthDetector] WARNING: Mediapipe init failed: {e}")
                print("[MouthDetector] Using fallback center-crop detection")
                self.use_mediapipe = False
        else:
            print("[MouthDetector] Using fallback center-crop detection")

    # ---------------------------------------------------------------
    # detect_and_crop — single frame (used by realtime pipeline)
    # ---------------------------------------------------------------
    def detect_and_crop(self, frame):
        """
        Detect and crop the mouth ROI from a single frame.

        Args:
            frame: BGR numpy array (H, W, 3)

        Returns:
            tuple: (mouth_frame, mouth_bbox)
                - mouth_frame: (50, 100, 3) BGR crop of mouth region
                - mouth_bbox: (x1, y1, x2, y2) or None
        """
        if frame is None or frame.size == 0:
            print("[MouthDetector] WARNING: Invalid frame received, returning blank")
            return np.zeros((50, 100, 3), dtype=np.uint8), None

        self._frame_count += 1

        # Try Mediapipe first
        if self.use_mediapipe:
            try:
                mouth_frame, bbox = self._detect_mediapipe(frame)
                if bbox is not None:
                    return mouth_frame, bbox
                # No face detected — fall through to fallback
            except Exception as e:
                print(f"[MouthDetector] Mediapipe error on frame {self._frame_count}: {e}")
                # Fall through to fallback

        # Fallback: center crop
        mouth_frame, bbox = self._detect_fallback(frame)
        return mouth_frame, bbox

    # ---------------------------------------------------------------
    # detect_mouth — batch of frames (used by visual_pipeline.py)
    # ---------------------------------------------------------------
    def detect_mouth(self, frames):
        """
        Detect and crop mouth ROI from a list of frames.

        Args:
            frames: list of numpy arrays (BGR or grayscale)

        Returns:
            tuple: (mouth_frames, status)
                - mouth_frames: list of (50, 100) or (50, 100, 3) mouth crops
                - status: "ok" or "invalid_input"
        """
        if not frames or len(frames) == 0:
            print("[MouthDetector] WARNING: Empty frame list received")
            return [], "invalid_input"

        mouth_frames = []
        detection_method = "mediapipe" if self.use_mediapipe else "fallback"
        mediapipe_hits = 0
        fallback_hits = 0

        for i, frame in enumerate(frames):
            if frame is None or (hasattr(frame, 'size') and frame.size == 0):
                # Skip invalid frames — append a blank
                mouth_frames.append(np.zeros((50, 100), dtype=np.uint8))
                continue

            # Ensure frame is BGR (3-channel) for detection
            if len(frame.shape) == 2:
                # Grayscale frame — convert to BGR for mediapipe, keep gray for crop
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame

            mouth_crop, bbox = self.detect_and_crop(frame_bgr)

            if bbox is not None and self.use_mediapipe:
                mediapipe_hits += 1
            else:
                fallback_hits += 1

            # Convert to grayscale for CNN+LSTM input
            if len(mouth_crop.shape) == 3:
                mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
            else:
                mouth_gray = mouth_crop

            # Ensure correct size
            if mouth_gray.shape != (50, 100):
                mouth_gray = cv2.resize(mouth_gray, (100, 50))

            mouth_frames.append(mouth_gray)

        # Debug output
        total = len(frames)
        print(f"[MouthDetector] Processed {total} frames | Method: {detection_method}")
        print(f"[MouthDetector] Mediapipe detections: {mediapipe_hits} | Fallback: {fallback_hits}")
        print(f"[MouthDetector] Output ROI shape: (50, 100) grayscale")

        if len(mouth_frames) == 0:
            return [], "invalid_input"

        return mouth_frames, "ok"

    # ---------------------------------------------------------------
    # save_debug_video — save mouth crops as debug video
    # ---------------------------------------------------------------
    def save_debug_video(self, mouth_frames, output_path="debug_mouth.avi"):
        """
        Save mouth ROI frames as a debug video for manual inspection.

        Args:
            mouth_frames: list of numpy arrays (grayscale or BGR)
            output_path: path to save the debug video
        """
        if not mouth_frames or len(mouth_frames) == 0:
            print("[MouthDetector] No frames to save for debug video")
            return

        try:
            # Determine output format from first frame
            sample = mouth_frames[0]
            if len(sample.shape) == 2:
                h, w = sample.shape
                is_gray = True
            else:
                h, w = sample.shape[:2]
                is_gray = False

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h), not is_gray)

            for frame in mouth_frames:
                if frame is None:
                    continue
                if is_gray and len(frame.shape) == 2:
                    frame_write = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(frame_write)
                elif not is_gray and len(frame.shape) == 3:
                    out.write(frame)
                else:
                    # Shape mismatch — convert
                    if len(frame.shape) == 2:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                    else:
                        out.write(frame)

            out.release()
            print(f"[MouthDetector] Debug video saved: {output_path} ({len(mouth_frames)} frames)")
        except Exception as e:
            print(f"[MouthDetector] WARNING: Could not save debug video: {e}")

    # ---------------------------------------------------------------
    # Internal: Mediapipe-based mouth detection
    # ---------------------------------------------------------------
    def _detect_mediapipe(self, frame):
        """
        Use Mediapipe Face Mesh to detect mouth landmarks and crop ROI.
        
        Returns:
            tuple: (mouth_frame, bbox) or (blank_frame, None) if no face found
        """
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            xs = [int(landmarks[i].x * w) for i in self.mouth_points]
            ys = [int(landmarks[i].y * h) for i in self.mouth_points]

            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))

            # Add padding
            pad_x, pad_y = 15, 15
            x1 = max(0, x1 - pad_x)
            x2 = min(w, x2 + pad_x)
            y1 = max(0, y1 - pad_y)
            y2 = min(h, y2 + pad_y)

            mouth_roi = frame[y1:y2, x1:x2]
            if mouth_roi.size > 0:
                mouth_frame = cv2.resize(mouth_roi, (100, 50))
                return mouth_frame, (x1, y1, x2, y2)

        # No face detected
        return np.zeros((50, 100, 3), dtype=np.uint8), None

    # ---------------------------------------------------------------
    # Internal: Fallback center-crop detection
    # ---------------------------------------------------------------
    def _detect_fallback(self, frame):
        """
        Fallback: crop the lower-center region of the frame as approximate mouth ROI.
        This works reasonably when the face is roughly centered in the frame.

        Returns:
            tuple: (mouth_frame, bbox)
        """
        h, w = frame.shape[:2]

        # Mouth is typically in the lower 60-90% vertically, center 30-70% horizontally
        y1 = int(h * 0.6)
        y2 = int(h * 0.9)
        x1 = int(w * 0.3)
        x2 = int(w * 0.7)

        # Clamp
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)

        mouth_roi = frame[y1:y2, x1:x2]
        if mouth_roi.size > 0:
            mouth_frame = cv2.resize(mouth_roi, (100, 50))
        else:
            mouth_frame = np.zeros((50, 100, 3), dtype=np.uint8)

        bbox = (x1, y1, x2, y2)
        return mouth_frame, bbox
