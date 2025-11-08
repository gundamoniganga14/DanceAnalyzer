
import cv2
import mediapipe as mp
import numpy as np
import os
import utils
class DancerPerformance:
    def __init__(self, dancer_id):
        self.id = dancer_id
        self.total_movement_score = 0
        self.total_grace_score = 0
        self.total_frames = 0
        self.previous_landmarks = None

def analyze_performance(video_path):
    
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    dancers_data = {}
    
    overall_best_moment_score = -1
    overall_best_moment_snapshot = None
    overall_best_dancer_id = None
    overall_best_landmarks = None  
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(frame_rgb)
        
        detected_poses_landmarks = []
        if results.pose_landmarks:
            detected_poses_landmarks.append(results.pose_landmarks.landmark)

        for dancer_index, landmarks in enumerate(detected_poses_landmarks):
            dancer_id = f"Dancer_{dancer_index + 1}"
            
            if dancer_id not in dancers_data:
                dancers_data[dancer_id] = DancerPerformance(dancer_id)

            dancer = dancers_data[dancer_id]
            
            current_landmarks = [(lmk.x, lmk.y, lmk.z, lmk.visibility) for lmk in landmarks]
            
            if dancer.previous_landmarks:
                movement_score = utils.calculate_movement_score(dancer.previous_landmarks, current_landmarks)
                dancer.total_movement_score += movement_score
                
                grace_score = utils.calculate_grace_score(dancer.previous_landmarks, current_landmarks)
                dancer.total_grace_score += grace_score

                current_frame_combined_score = (0.6 * movement_score) + (0.4 * grace_score)
                if current_frame_combined_score > overall_best_moment_score:
                    overall_best_moment_score = current_frame_combined_score
                    overall_best_moment_snapshot = frame.copy()
                    overall_best_dancer_id = dancer_id
                    overall_best_landmarks = landmarks 
            
            dancer.previous_landmarks = current_landmarks
            dancer.total_frames += 1

    cap.release()
    
    leaderboard = []
    
    if not os.path.exists("data/output/snapshots"):
        os.makedirs("data/output/snapshots")
    
    best_snapshot_info = {}
    if overall_best_moment_snapshot is not None and overall_best_landmarks is not None:
        h, w, _ = overall_best_moment_snapshot.shape
        x_min = w
        y_min = h
        x_max = 0
        y_max = 0

        for landmark in overall_best_landmarks:
            if landmark.visibility > 0.5: 
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
        
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        cropped_image = overall_best_moment_snapshot[y_min:y_max, x_min:x_max]
    
        snapshot_path = os.path.join("data/output/snapshots", f"{overall_best_dancer_id}_best_moment.png")
        cv2.imwrite(snapshot_path, cropped_image)
        
        best_snapshot_info = {
            "path": snapshot_path,
            "dancer_id": overall_best_dancer_id
        }

    for dancer_id, data in dancers_data.items():
        if data.total_frames > 0:
            final_movement_score = data.total_movement_score / data.total_frames
            final_grace_score = data.total_grace_score / data.total_frames
            
            overall_score = (final_movement_score * 0.6) + (final_grace_score * 0.4)

            leaderboard.append({
                "Dancer": dancer_id,
                "Timing Score": round(overall_score, 2),
                "Movement Score": round(final_movement_score, 2),
                "Grace Score": round(final_grace_score, 2),
            })

    leaderboard_sorted = sorted(leaderboard, key=lambda x: x["Timing Score"], reverse=True)
    
    for i, entry in enumerate(leaderboard_sorted):
        entry['Rank'] = i + 1
        
    return {
        "leaderboard": leaderboard_sorted,
        "best_performer_snapshot": best_snapshot_info
    }