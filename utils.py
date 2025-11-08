# src/utils.py
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle (in degrees) between three 3D points.
    
    Args:
        p1, p2, p3 (tuple): Tuples representing (x, y, z, visibility) coordinates.
    
    Returns:
        float: The angle in degrees.
    """
    p1 = np.array(p1[:2])  # Use only x, y coordinates
    p2 = np.array(p2[:2])
    p3 = np.array(p3[:2])

    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    angle = np.degrees(np.arccos(dot_product / (norm_v1 * norm_v2)))
    return angle

def calculate_movement_score(prev_landmarks, current_landmarks):
    """
    Calculates a score based on the total distance moved by all keypoints
    between two consecutive frames.
    A higher score indicates more dynamic movement.
    
    Args:
        prev_landmarks (list): List of previous frame's landmark coordinates.
        current_landmarks (list): List of current frame's landmark coordinates.
        
    Returns:
        float: The movement score.
    """
    score = 0
    if not prev_landmarks or not current_landmarks:
        return 0
    
    for i in range(len(prev_landmarks)):
        dist = np.sqrt((current_landmarks[i][0] - prev_landmarks[i][0])**2 +
                       (current_landmarks[i][1] - prev_landmarks[i][1])**2)
        score += dist
    return score

def calculate_grace_score(prev_landmarks, current_landmarks):
    """
    Calculates a conceptual grace score based on the smoothness of joint movements.
    A higher score indicates more fluid, less jerky motion.
    
    This is a simplification for a hackathon. A more advanced metric could
    track angular velocity over a larger time window.
    
    Args:
        prev_landmarks (list): Previous frame's landmark coordinates.
        current_landmarks (list): Current frame's landmark coordinates.
        
    Returns:
        float: The grace score.
    """
    # Define key joint triplets for analysis
    joint_triplets = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    ]
    
    grace_score_sum = 0
    
    if not prev_landmarks or not current_landmarks:
        return 0
    
    for p1_lm, p2_lm, p3_lm in joint_triplets:
        try:
            prev_p1 = prev_landmarks[p1_lm.value]
            prev_p2 = prev_landmarks[p2_lm.value]
            prev_p3 = prev_landmarks[p3_lm.value]

            curr_p1 = current_landmarks[p1_lm.value]
            curr_p2 = current_landmarks[p2_lm.value]
            curr_p3 = current_landmarks[p3_lm.value]

            prev_angle = calculate_angle(prev_p1, prev_p2, prev_p3)
            curr_angle = calculate_angle(curr_p1, curr_p2, curr_p3)
            
            angle_change = abs(curr_angle - prev_angle)
            
            # A smaller change means smoother movement, resulting in a higher score
            # A small constant (0.1) is added to avoid division by zero
            grace_score_sum += 1.0 / (angle_change + 0.1)
        except (IndexError, AttributeError):
            # Handle cases where a landmark is not detected (e.g., person is partially off-screen)
            continue

    # Return the average grace score for the frame
    return grace_score_sum / len(joint_triplets) if joint_triplets else 0