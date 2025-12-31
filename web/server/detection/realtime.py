import cv2
import mediapipe as mp
import numpy as np
from django.conf import settings
from .plank import PlankDetection
from .bicep_curl import BicepCurlDetection
from .squat import SquatDetection
from .utils import rescale_frame

class VideoCamera(object):
    def __init__(self, exercise_type):
        self.video = cv2.VideoCapture(0)
        self.exercise_type = exercise_type
        
        # Load specific model
        if exercise_type == 'squat':
            self.detection = SquatDetection()
        elif exercise_type == 'plank':
            self.detection = PlankDetection()
        elif exercise_type == 'bicep_curl':
            self.detection = BicepCurlDetection()
        else:
            self.detection = None

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        if not self.video.isOpened():
            print("Error: Could not open video device.")
        else:
            print("Video device opened successfully.")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            print("Error: Failed to read frame from video device.")
            return None

        # Process frame
        if self.detection:
            # Recolor image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = self.pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # The detect method modifies the image in-place
                self.detection.detect(mp_results=results, image=image, timestamp=0)

        # Encode to JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
