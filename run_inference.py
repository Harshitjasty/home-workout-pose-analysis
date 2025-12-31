import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import glob
import os
import warnings

warnings.filterwarnings('ignore')

# --- Helper Functions ---

def calculate_angle(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)
    
    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg

def calculate_distance(pointX, pointY):
    x1, y1 = pointX
    x2, y2 = pointY
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_important_keypoints(results, important_landmarks, mp_pose):
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

# --- Bicep Analysis Class ---

class BicepPoseAnalysis:
    def __init__(self, side, stage_down_threshold, stage_up_threshold, peak_contraction_threshold, loose_upper_arm_angle_threshold, visibility_threshold):
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold
        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {"LOOSE_UPPER_ARM": 0, "PEAK_CONTRACTION": 0}
        self.loose_upper_arm = False
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None

    def get_joints(self, landmarks, mp_pose):
        side = self.side.upper()
        joints_visibility = [
            landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility,
            landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility
        ]
        is_visible = all([vis > self.visibility_threshold for vis in joints_visibility])
        self.is_visible = is_visible
        if not is_visible:
            return self.is_visible
        
        self.shoulder = [landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y]
        self.elbow = [landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y]
        self.wrist = [landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y]
        return self.is_visible

    def analyze_pose(self, landmarks, frame, mp_pose):
        self.get_joints(landmarks, mp_pose)
        if not self.is_visible:
            return (None, None)
        
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            
        shoulder_projection = [self.shoulder[0], 1]
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))
        
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                self.detected_errors["LOOSE_UPPER_ARM"] += 1
        else:
            self.loose_upper_arm = False
            
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            self.peak_contraction_angle = bicep_curl_angle
        elif self.stage == "down":
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                self.detected_errors["PEAK_CONTRACTION"] += 1
            self.peak_contraction_angle = 1000
            
        return (bicep_curl_angle, ground_upper_arm_angle)

# --- Inference Functions ---

def run_bicep_inference(video_path, model, scaler):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    
    # Thresholds
    STAGE_UP_THRESHOLD = 90
    STAGE_DOWN_THRESHOLD = 120
    PEAK_CONTRACTION_THRESHOLD = 60
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40
    VISIBILITY_THRESHOLD = 0.65
    
    left_arm = BicepPoseAnalysis("left", STAGE_DOWN_THRESHOLD, STAGE_UP_THRESHOLD, PEAK_CONTRACTION_THRESHOLD, LOOSE_UPPER_ARM_ANGLE_THRESHOLD, VISIBILITY_THRESHOLD)
    right_arm = BicepPoseAnalysis("right", STAGE_DOWN_THRESHOLD, STAGE_UP_THRESHOLD, PEAK_CONTRACTION_THRESHOLD, LOOSE_UPPER_ARM_ANGLE_THRESHOLD, VISIBILITY_THRESHOLD)
    
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            frame_count += 1
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            if not results.pose_landmarks:
                continue
                
            landmarks = results.pose_landmarks.landmark
            left_arm.analyze_pose(landmarks, image, mp_pose)
            right_arm.analyze_pose(landmarks, image, mp_pose)
            
    cap.release()
    return {
        "video": os.path.basename(video_path),
        "left_reps": left_arm.counter,
        "right_reps": right_arm.counter,
        "left_errors": left_arm.detected_errors,
        "right_errors": right_arm.detected_errors
    }

def run_plank_inference(video_path, model, scaler):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    
    IMPORTANT_LMS = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
        "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    
    HEADERS = ["label"]
    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
        
    predictions = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            if not results.pose_landmarks:
                continue
                
            try:
                row = extract_important_keypoints(results, IMPORTANT_LMS, mp_pose)
                X = pd.DataFrame([row], columns=HEADERS[1:])
                X = pd.DataFrame(scaler.transform(X))
                
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0]
                
                # 0: C, 1: H, 2: L
                label_map = {0: "Correct", 1: "High back", 2: "Low back"}
                predictions.append(label_map.get(pred, "Unknown"))
            except Exception as e:
                pass
                
    cap.release()
    
    if not predictions:
        return {"video": os.path.basename(video_path), "most_frequent_state": "No Detection"}
        
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0][0]
    return {"video": os.path.basename(video_path), "most_frequent_state": most_common}

def run_squat_inference(video_path, model, scaler):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    
    IMPORTANT_LMS = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ]
    
    HEADERS = ["label"]
    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
        
    predictions = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            if not results.pose_landmarks:
                continue
                
            try:
                row = extract_important_keypoints(results, IMPORTANT_LMS, mp_pose)
                X = pd.DataFrame([row], columns=HEADERS[1:])
                X = pd.DataFrame(scaler.transform(X))
                
                pred = model.predict(X)[0]
                # 0: down, 1: up
                label_map = {0: "down", 1: "up"}
                predictions.append(label_map.get(pred, "Unknown"))
            except Exception as e:
                pass
                
    cap.release()
    
    if not predictions:
        return {"video": os.path.basename(video_path), "most_frequent_state": "No Detection"}
        
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0][0]
    return {"video": os.path.basename(video_path), "most_frequent_state": most_common}

# --- Main Execution ---

if __name__ == "__main__":
    results = []
    
    # Load Models
    print("Loading models...")
    with open("core/bicep_model/model/LR_model.pkl", "rb") as f:
        bicep_model = pickle.load(f)
    with open("core/bicep_model/model/input_scaler.pkl", "rb") as f:
        bicep_scaler = pickle.load(f)
        
    with open("core/plank_model/model/LR_model.pkl", "rb") as f:
        plank_model = pickle.load(f)
    with open("core/plank_model/model/input_scaler.pkl", "rb") as f:
        plank_scaler = pickle.load(f)
        
    with open("core/squat_model/model/squat_model.pkl", "rb") as f:
        squat_model = pickle.load(f)
    with open("core/squat_model/model/input_scaler.pkl", "rb") as f:
        squat_scaler = pickle.load(f)
        
    print("Models loaded.")
    
    # Process Bicep Videos (Sample 3)
    bicep_videos = glob.glob("data/bicep_curl/*.mp4")[:3]
    print(f"Processing {len(bicep_videos)} Bicep videos...")
    for video in bicep_videos:
        res = run_bicep_inference(video, bicep_model, bicep_scaler)
        print(res)
        results.append({"Exercise": "Bicep", **res})
        
    # Process Plank Videos (Sample 3)
    plank_videos = glob.glob("data/plank/*.mp4")[:3]
    print(f"Processing {len(plank_videos)} Plank videos...")
    for video in plank_videos:
        res = run_plank_inference(video, plank_model, plank_scaler)
        print(res)
        results.append({"Exercise": "Plank", **res})
        
    # Process Squat Videos (Sample 3)
    squat_videos = glob.glob("data/squat/*.mp4")[:3]
    print(f"Processing {len(squat_videos)} Squat videos...")
    for video in squat_videos:
        res = run_squat_inference(video, squat_model, squat_scaler)
        print(res)
        results.append({"Exercise": "Squat", **res})
        
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv("inference_results.csv", index=False)
    print("Inference completed. Results saved to inference_results.csv")
