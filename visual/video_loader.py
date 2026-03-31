import cv2
import os

def load_video(video_path):
    """
    Loads a video file and returns a cv2.VideoCapture object.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        cv2.VideoCapture: The video capture object.
        
    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video file cannot be opened.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    return cap
