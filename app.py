import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_face_expression(landmarks):
    """Phân tích cảm xúc cơ mặt từ các điểm mốc khuôn mặt"""
    
    left_eye = landmarks[33]  
    right_eye = landmarks[133]  
    mouth_left = landmarks[78] 
    mouth_right = landmarks[308]  
    nose = landmarks[1]  
    
    eye_distance = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
    
    mouth_distance = np.linalg.norm([mouth_left.x - mouth_right.x, mouth_left.y - mouth_right.y])
    
    smile_threshold = 0.05  
    surprise_threshold = 0.15  
    sad_threshold = 0.07  
    
    if mouth_distance > (eye_distance + smile_threshold):
        return "Happy"
    
    elif eye_distance > (mouth_distance + surprise_threshold):
        return "Surprised"
    
    elif mouth_distance < (eye_distance - sad_threshold):
        return "Sad"
    
    elif mouth_left.y < nose.y and mouth_right.y < nose.y:
        return "Angry"
    
    return "Neutral"

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id, wrist, direction):
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    if direction == 'vertical':
        return tip.y < pip.y
    elif direction == 'horizontal':
        return tip.x > wrist.x if tip.x > pip.x else tip.x < wrist.x

def detect_hand_direction(wrist, middle_mcp):
    dx = abs(middle_mcp.x - wrist.x)
    dy = abs(middle_mcp.y - wrist.y)
    if dx > dy:
        return 'horizontal'
    else:
        return 'vertical'

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

frame_counter = 0
frame_skip = 2  

with mp_hands.Hands(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    max_num_hands=2
) as hands, mp_face_detection.FaceDetection(
    min_detection_confidence=0.7
) as face_detection, mp_face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as face_mesh, mp_pose.Pose(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_face = face_detection.process(rgb_frame)
        
        results_face_mesh = face_mesh.process(rgb_frame)
        
        results_hands = hands.process(rgb_frame)
        
        results_pose = pose.process(rgb_frame)
        
        total_fingers = 0
        frame_height, frame_width, _ = frame.shape

        if results_face.detections:
            for detection in results_face.detections:
                mp_drawing.draw_detection(frame, detection)
        
        if results_face_mesh.multi_face_landmarks:
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                expression = analyze_face_expression(face_landmarks.landmark)
                cv2.putText(frame, f'Expression: {expression}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                direction = detect_hand_direction(wrist, middle_mcp)
                count = 0
                thumb_tip = hand_landmarks.landmark[4]
                thumb_ip = hand_landmarks.landmark[3]
                if direction == 'horizontal':
                    thumb_extended = thumb_tip.x < thumb_ip.x
                else:
                    thumb_extended = thumb_tip.y < thumb_ip.y
                if thumb_extended:
                    count += 1
                for finger_tip_id, finger_pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                    if is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id, wrist, direction):
                        count += 1
                total_fingers += count

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f'Total Fingers: {total_fingers}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if results_pose.pose_landmarks:
            cv2.putText(frame, 'Pose Detected', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Hand and Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
