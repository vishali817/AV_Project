import cv2

class WebcamCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        
    def start_webcam(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
            
    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
