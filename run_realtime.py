import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import warnings
import math

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
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_important_keypoints(results, important_landmarks, mp_pose):
    landmarks = results.pose_landmarks.landmark
    data = []
    for lm in important_landmarks:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    return np.array(data).flatten().tolist()

# --- Squat Analysis Functions ---

def analyze_foot_knee_placement(results, stage, foot_shoulder_ratio_thresholds, knee_foot_ratio_thresholds, visibility_threshold):
    analyzed_results = {
        "foot_placement": -1,
        "knee_placement": -1,
    }

    landmarks = results.pose_landmarks.landmark
    mp_pose = mp.solutions.pose

    # Visibility check
    left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
    right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility
    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        return analyzed_results
    
    # Calculate widths
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    foot_width = calculate_distance(left_foot_index, right_foot_index)

    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    knee_width = calculate_distance(left_knee, right_knee)

    # Analyze Foot Placement
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)
    min_ratio_foot, max_ratio_foot = foot_shoulder_ratio_thresholds
    if min_ratio_foot <= foot_shoulder_ratio <= max_ratio_foot:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot:
        analyzed_results["foot_placement"] = 2
    
    # Analyze Knee Placement
    knee_foot_ratio = round(knee_width / foot_width, 1)
    
    # Handle unknown stage (default to middle if unknown)
    stage_key = stage if stage in ["up", "middle", "down"] else "middle"
    min_ratio_knee, max_ratio_knee = knee_foot_ratio_thresholds.get(stage_key, [0.7, 1.0])

    if min_ratio_knee <= knee_foot_ratio <= max_ratio_knee:
        analyzed_results["knee_placement"] = 0
    elif knee_foot_ratio < min_ratio_knee:
        analyzed_results["knee_placement"] = 1
    elif knee_foot_ratio > max_ratio_knee:
        analyzed_results["knee_placement"] = 2
        
    return analyzed_results

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
        self.shoulder = None
        self.elbow = None
        self.wrist = None

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

# --- Real-time Functions ---

def draw_limb(image, p1, p2, color, thickness=2):
    h, w, c = image.shape
    pt1 = (int(p1[0] * w), int(p1[1] * h))
    pt2 = (int(p2[0] * w), int(p2[1] * h))
    cv2.line(image, pt1, pt2, color, thickness)

def start_bicep_realtime():
    print("Starting Bicep Curls Real-time Analysis...")
    print("Press 'q' to quit.")
    
    with open("core/bicep_model/model/LR_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("core/bicep_model/model/input_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    # Thresholds
    STAGE_UP_THRESHOLD = 90
    STAGE_DOWN_THRESHOLD = 120
    PEAK_CONTRACTION_THRESHOLD = 60
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40
    VISIBILITY_THRESHOLD = 0.65
    
    left_arm = BicepPoseAnalysis("left", STAGE_DOWN_THRESHOLD, STAGE_UP_THRESHOLD, PEAK_CONTRACTION_THRESHOLD, LOOSE_UPPER_ARM_ANGLE_THRESHOLD, VISIBILITY_THRESHOLD)
    right_arm = BicepPoseAnalysis("right", STAGE_DOWN_THRESHOLD, STAGE_UP_THRESHOLD, PEAK_CONTRACTION_THRESHOLD, LOOSE_UPPER_ARM_ANGLE_THRESHOLD, VISIBILITY_THRESHOLD)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Analyze
                left_res = left_arm.analyze_pose(landmarks, image, mp_pose)
                right_res = right_arm.analyze_pose(landmarks, image, mp_pose)
                
                # Draw Landmarks (Standard)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Visual Feedback for Errors (RED LINES)
                if left_arm.loose_upper_arm and left_arm.is_visible:
                    draw_limb(image, left_arm.shoulder, left_arm.elbow, (0, 0, 255), 4) # Red for loose arm
                if right_arm.loose_upper_arm and right_arm.is_visible:
                    draw_limb(image, right_arm.shoulder, right_arm.elbow, (0, 0, 255), 4)

                # Display Stats
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                
                # Left Arm
                cv2.putText(image, 'L Reps', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm.counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Right Arm
                cv2.putText(image, 'R Reps', (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm.counter), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Errors Text
                if left_arm.loose_upper_arm or right_arm.loose_upper_arm:
                    cv2.putText(image, 'LOOSE ARM', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Bicep Curl Analysis', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

def start_plank_realtime():
    print("Starting Plank Real-time Analysis...")
    print("Press 'q' to quit.")
    
    with open("core/plank_model/model/LR_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("core/plank_model/model/input_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    IMPORTANT_LMS = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
        "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    
    HEADERS = ["label"]
    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
        
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            current_stage = "Unknown"
            prob = 0.0
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                try:
                    row = extract_important_keypoints(results, IMPORTANT_LMS, mp_pose)
                    X = pd.DataFrame([row], columns=HEADERS[1:])
                    X = pd.DataFrame(scaler.transform(X))
                    
                    pred = model.predict(X)[0]
                    prob_arr = model.predict_proba(X)[0]
                    prob = prob_arr[np.argmax(prob_arr)]
                    
                    label_map = {0: "Correct", 1: "High back", 2: "Low back"}
                    current_stage = label_map.get(pred, "Unknown")
                    
                    # Visual Feedback
                    if current_stage != "Correct" and current_stage != "Unknown":
                        # Draw back line in RED
                        landmarks = results.pose_landmarks.landmark
                        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        
                        draw_limb(image, l_shoulder, l_hip, (0, 0, 255), 4)
                        draw_limb(image, l_hip, l_ankle, (0, 0, 255), 4)
                        
                        cv2.putText(image, current_stage.upper(), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                except Exception as e:
                    pass
                
                # Display
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(prob, 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Plank Analysis', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

def start_squat_realtime():
    print("Starting Squat Real-time Analysis...")
    print("Press 'q' to quit.")
    
    with open("core/squat_model/model/squat_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("core/squat_model/model/input_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    IMPORTANT_LMS = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ]
    
    HEADERS = ["label"]
    for lm in IMPORTANT_LMS:
        HEADERS += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
    
    # Thresholds
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": [0.5, 1.0],
        "middle": [0.7, 1.0],
        "down": [0.7, 1.1],
    }
        
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            current_stage = "Unknown"
            foot_status = "UNK"
            knee_status = "UNK"
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                try:
                    row = extract_important_keypoints(results, IMPORTANT_LMS, mp_pose)
                    X = pd.DataFrame([row], columns=HEADERS[1:])
                    X = pd.DataFrame(scaler.transform(X))
                    
                    pred = model.predict(X)[0]
                    label_map = {0: "down", 1: "up"}
                    current_stage = label_map.get(pred, "Unknown")
                    
                    # Analyze Bad Pose
                    analysis = analyze_foot_knee_placement(results, current_stage, FOOT_SHOULDER_RATIO_THRESHOLDS, KNEE_FOOT_RATIO_THRESHOLDS, VISIBILITY_THRESHOLD)
                    
                    foot_res = analysis["foot_placement"]
                    knee_res = analysis["knee_placement"]
                    
                    status_map = {-1: "UNK", 0: "Correct", 1: "Too Tight", 2: "Too Wide"}
                    foot_status = status_map.get(foot_res, "UNK")
                    knee_status = status_map.get(knee_res, "UNK")
                    
                    # Visual Feedback
                    if foot_res in [1, 2]: # Error
                        cv2.putText(image, f"FEET: {foot_status.upper()}", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        # Draw line between feet in RED
                        landmarks = results.pose_landmarks.landmark
                        l_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                        r_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                        draw_limb(image, l_foot, r_foot, (0, 0, 255), 4)

                    if knee_res in [1, 2]: # Error
                        cv2.putText(image, f"KNEES: {knee_status.upper()}", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        # Draw line between knees in RED
                        landmarks = results.pose_landmarks.landmark
                        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        draw_limb(image, l_knee, r_knee, (0, 0, 255), 4)
                        
                except Exception as e:
                    pass
                
                # Display Stats
                cv2.rectangle(image, (0, 0), (250, 100), (245, 117, 16), -1)
                cv2.putText(image, 'STAGE', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, f'Feet: {foot_status}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, f'Knees: {knee_status}', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Squat Analysis', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Select Exercise to Track:")
    print("1. Bicep Curl")
    print("2. Plank")
    print("3. Squat")
    
    choice = input("Enter choice (1/2/3): ")
    
    if choice == '1':
        start_bicep_realtime()
    elif choice == '2':
        start_plank_realtime()
    elif choice == '3':
        start_squat_realtime()
    else:
        print("Invalid choice.")
