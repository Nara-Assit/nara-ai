"""
Sign-to-text model service.

Handles video preprocessing, normalization, and sign language recognition.
Uses the EXACT same preprocessing as training:
  - MediaPipe Holistic (mp.solutions.holistic)
  - Shoulder-based normalization
  - Upper body (6 points) + both hands (21 each) = 144 features
  - Motion-based top-30 frame selection
"""

import os
import logging
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# Configure logging (suppress verbose TensorFlow/MediaPipe logs)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# ── MediaPipe Holistic setup ────────────────────────────────────────────
mp_holistic = mp.solutions.holistic

# ── Label map (same as training) ────────────────────────────────────────
LABEL_MAP = {
    'mall': 0,
    'good': 1,
    'mosque': 2,
    'finish': 3,
    'thinking': 4,
    'mother': 5,
    'eat': 6,
    'sad': 7,
    'house': 8,
    'love': 9,
    'normal': 10,
    'me': 11,
    'worry': 12,
    'thanks': 13,
    'baby': 14,
    'father': 15,
    'hear': 16,
    'stop': 17,
    'important': 18,
    'happy': 19,
}

# Inverted map: index → word
INDEX_TO_WORD = {v: k for k, v in LABEL_MAP.items()}

# ── Constants ───────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30      # same as training
NUM_FEATURES = 144        # 48 points × 3 coords
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sttm_demo_2.h5")

# Global model instance
_model = None


def load_model_instance():
    """Load and cache the model."""
    global _model
    if _model is None:
        logger.info(f"Loading model from: {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    return _model


# ── Preprocessing (identical to training notebook) ──────────────────────

def extract_and_normalize_keypoints(results):
    """
    Extracts pose and hand data, normalizes them relative to the body center,
    and filters out noise (legs, unnecessary face points).

    Exactly matches the training preprocessing.
    """

    # 1. Extract Raw Landmarks (Shape: N x 3)
    def to_array(landmarks, count):
        if landmarks:
            return np.array([[res.x, res.y, res.z] for res in landmarks.landmark])
        else:
            return np.zeros((count, 3))

    pose_raw = to_array(results.pose_landmarks, 33)
    lh_raw = to_array(results.left_hand_landmarks, 21)
    rh_raw = to_array(results.right_hand_landmarks, 21)

    # 2. Establish Normalization Reference (Based on Shoulders)
    if results.pose_landmarks:
        left_shoulder = pose_raw[11]
        right_shoulder = pose_raw[12]

        center = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_width < 0.001:
            shoulder_width = 1.0
    else:
        return np.zeros(NUM_FEATURES)

    # 3. Apply Normalization
    pose_norm = (pose_raw - center) / shoulder_width
    lh_norm = (lh_raw - center) / shoulder_width
    rh_norm = (rh_raw - center) / shoulder_width

    # Reset missing hands back to zero
    if not results.left_hand_landmarks:
        lh_norm = np.zeros((21, 3))
    if not results.right_hand_landmarks:
        rh_norm = np.zeros((21, 3))

    # 4. Feature Selection – upper body only
    upper_body_indices = [11, 12, 13, 14, 15, 16]
    pose_filtered = pose_norm[upper_body_indices]

    # 5. Flatten and Concatenate → 48 × 3 = 144
    return np.concatenate([pose_filtered.flatten(), lh_norm.flatten(), rh_norm.flatten()])


def process_video(video_path, sequence_length=SEQUENCE_LENGTH):
    """
    Process a video using the same motion-based frame selection as training:
      1. Extract keypoints from every frame.
      2. Score each frame by how much the keypoints moved vs. previous frame.
      3. Pick the top `sequence_length` frames by motion score.
      4. Re-sort them chronologically.
      5. Pad with zeros if the video has fewer frames than needed.
    """
    cap = cv2.VideoCapture(video_path)

    frames_data = []
    prev_keypoints = None
    frame_idx = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            keypoints = extract_and_normalize_keypoints(results)

            # Movement score (Euclidean distance between consecutive frames)
            if prev_keypoints is None:
                score = 0.0
            else:
                score = np.linalg.norm(keypoints - prev_keypoints)

            prev_keypoints = keypoints

            frames_data.append({
                'keypoints': keypoints,
                'score': score,
                'original_index': frame_idx,
            })
            frame_idx += 1

        cap.release()

    if not frames_data:
        return None

    # Motion-Based Selection (same as training)
    if len(frames_data) >= sequence_length:
        sorted_by_motion = sorted(frames_data, key=lambda x: x['score'], reverse=True)
        top_frames = sorted_by_motion[:sequence_length]
        top_frames.sort(key=lambda x: x['original_index'])
        sequence = np.array([f['keypoints'] for f in top_frames])
    else:
        raw_sequence = np.array([f['keypoints'] for f in frames_data])
        padding_size = sequence_length - len(raw_sequence)
        padding = np.zeros((padding_size, raw_sequence.shape[1]))
        sequence = np.concatenate([raw_sequence, padding], axis=0)

    return sequence


# ── Prediction helpers ──────────────────────────────────────────────────

def predict_video(video_path, model=None, top_k=3):
    """
    Run the full pipeline on a single video and return the prediction.

    Parameters
    ----------
    video_path : str
        Path to video file
    model : keras.Model, optional
        Loaded model. If None, loads the global instance.
    top_k : int
        Number of top predictions to return

    Returns
    -------
    dict with keys:
        predicted_word : str
        confidence     : float
        top_k          : list of (word, confidence) tuples
    Or None if the video could not be processed.
    """
    if model is None:
        model = load_model_instance()

    sequence = process_video(video_path, sequence_length=SEQUENCE_LENGTH)
    if sequence is None:
        logger.warning(f"Could not process video: {video_path}")
        return None

    # Add batch dimension: (30, 144) → (1, 30, 144)
    X = np.expand_dims(sequence, axis=0)

    predictions = model.predict(X, verbose=0)[0]  # shape (20,)
    predicted_idx = int(np.argmax(predictions))
    predicted_word = INDEX_TO_WORD.get(predicted_idx, f"unknown({predicted_idx})")
    confidence = float(predictions[predicted_idx])

    # Top-K predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]
    top_predictions = [
        (INDEX_TO_WORD.get(int(i), f"unknown({i})"), float(predictions[i]))
        for i in top_indices
    ]

    logger.info(f"Prediction for {video_path}: {predicted_word} (confidence: {confidence:.4f})")

    return {
        'predicted_word': predicted_word,
        'confidence': confidence,
        'top_k': top_predictions,
    }


def predict_batch(video_paths, model=None):
    """Process a list of videos and return predictions for each."""
    if model is None:
        model = load_model_instance()

    results = []
    for vp in video_paths:
        result = predict_video(vp, model)
        results.append({'video': vp, 'result': result})
    return results



