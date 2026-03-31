import cv2

def extract_frames(video_capture):
    """
    Extracts frames from a cv2.VideoCapture object.
    
    Args:
        video_capture (cv2.VideoCapture): The video capture object.
        
    Returns:
        tuple: (frames_list, frame_count)
            frames_list (list): List of grayscale frames (224x224).
            frame_count (int): Total number of frames extracted.
    """
    frames_list = []
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to 224x224
        resized_frame = cv2.resize(gray_frame, (224, 224))
        
        frames_list.append(resized_frame)
        
    video_capture.release()
    frame_count = len(frames_list)
    
    return frames_list, frame_count
