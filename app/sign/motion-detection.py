"""
Sign Language Inference with Motion Detection 
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


ACTIONS = ["love" , "house", "mall" , "father" , "mosque" , "me" ,
            "important", "happy" , "finish" , "thinking","baby" ,
            "worry" , "normal" , "eat" , "stop" , "sad", "thanks" ,
            "mother" , "hear" ,"good"]

mp_holistic = mp.solutions.holistic


def extract_and_normalize_keypoints(results):
    """Extract and normalize pose/hand keypoints"""
    def to_array(landmarks, count):
        if landmarks:
            return np.array([[res.x, res.y, res.z] for res in landmarks.landmark])
        else:
            return np.zeros((count, 3))

    pose_raw = to_array(results.pose_landmarks, 33)
    lh_raw = to_array(results.left_hand_landmarks, 21)
    rh_raw = to_array(results.right_hand_landmarks, 21)

    if results.pose_landmarks:
        left_shoulder = pose_raw[11]
        right_shoulder = pose_raw[12]
        center = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_width < 0.001:
            shoulder_width = 1.0
    else:
        return np.zeros(144)

    pose_norm = (pose_raw - center) / shoulder_width
    lh_norm = (lh_raw - center) / shoulder_width
    rh_norm = (rh_raw - center) / shoulder_width

    if not results.left_hand_landmarks:
        lh_norm = np.zeros((21, 3))
    if not results.right_hand_landmarks:
        rh_norm = np.zeros((21, 3))

    upper_body_indices = [11, 12, 13, 14, 15, 16]
    pose_filtered = pose_norm[upper_body_indices]

    return np.concatenate([pose_filtered.flatten(), lh_norm.flatten(), rh_norm.flatten()])


def detect_word_segments(video_path: str,
                         sample_rate: int = 1,
                         activity_threshold: float = 0.02,
                         min_length_frames: int = 5,
                         merge_gap: int = 8):
    """Detect motion segments in video"""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    motions = []
    prev_point = None
    
    with mp_holistic.Holistic(min_detection_confidence=0.4,
                              min_tracking_confidence=0.4) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            point = None
            if results.left_hand_landmarks:
                lm = results.left_hand_landmarks.landmark[9]
                point = np.array([lm.x, lm.y])
            elif results.right_hand_landmarks:
                lm = results.right_hand_landmarks.landmark[9]
                point = np.array([lm.x, lm.y])
            elif results.pose_landmarks:
                l = results.pose_landmarks.landmark[15]
                r = results.pose_landmarks.landmark[16]
                point = np.array([(l.x+r.x)/2, (l.y+r.y)/2])
                
            if point is None or prev_point is None:
                motions.append(0.0)
            else:
                motions.append(float(np.linalg.norm(point - prev_point)))
                
            prev_point = point
            frame_idx += 1
            
    cap.release()
    
    mot = np.array(motions)
    kernel = np.ones(3) / 3
    mot_smooth = np.convolve(mot, kernel, mode='same')
    active = mot_smooth > activity_threshold
    
    segments = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        if not a and start is not None:
            end = i - 1
            if (end - start + 1) >= min_length_frames:
                segments.append({'start': start * sample_rate, 'end': end * sample_rate})
            start = None
            
    if start is not None:
        end = len(active) - 1
        if (end - start + 1) >= min_length_frames:
            segments.append({'start': start * sample_rate, 'end': end * sample_rate})
    
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
        else:
            if seg['start'] - merged[-1]['end'] <= merge_gap:
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg)
                
    return merged


def process_video(video_path, start=None, end=None, sequence_length=30):
    """Extract keypoints from video segment"""
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_id = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if start is not None and frame_id < start:
                frame_id += 1
                continue
                
            if end is not None and frame_id > end:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            keypoints = extract_and_normalize_keypoints(results)
            sequence.append(keypoints)
            frame_id += 1
            
    cap.release()
    
    if len(sequence) == 0:
        return None
        
    sequence = np.array(sequence)
    
    if len(sequence) >= sequence_length:
        idx = np.linspace(0, len(sequence) - 1, sequence_length, dtype=int)
        sequence = sequence[idx]
    else:
        pad = np.zeros((sequence_length - len(sequence), sequence.shape[1]))
        sequence = np.concatenate([sequence, pad], axis=0)
        
    return sequence




def predict_sentence(video_path: str, model_path: str, verbose: bool = False):
    """
    Predict sign language sentence from video
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model 
        verbose: Print progress (default: False)
        
    Returns:
        dict: {
            'sentence': list of words,
            'segments': number of detected segments,
            'words': list of dicts with word + confidence
        }
    """
    model = load_model(model_path)
    
    segments = detect_word_segments(video_path)
    
    if len(segments) == 0:
        segments = [{'start': None, 'end': None}]
    
    words_data = []
    sentence = []
    
    for seg in segments:
        seq = process_video(video_path, seg['start'], seg['end'])
        
        if seq is None:
            continue
            
        pred = model.predict(seq[np.newaxis, ...], verbose=0)
        word = ACTIONS[np.argmax(pred)]
        confidence = float(np.max(pred))
        
        sentence.append(word)
        words_data.append({
            'word': word,
            'confidence': confidence
        })
        
        if verbose:
            print(f"Predicted: {word} ({confidence:.2%})")
    
    return {
        'sentence': sentence,
        'full_sentence': ' '.join(sentence),
        'segments_count': len(segments),
        'words': words_data
    }