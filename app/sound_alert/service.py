import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import tempfile 
import joblib



yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
model_folder = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(model_folder, "sound_classifier.h5")
classifier = load_model(MODEL_PATH)
scaler = joblib.load(os.path.join(model_folder, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))


def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".wav":
        audio = AudioSegment.from_file(file_path)
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name
    return file_path


def predict_sound(file_path):
    file_path = convert_to_wav(file_path)

    
    audio, sr = librosa.load(file_path, sr=16000,duration=5.0, mono=True)

    
    scores, embeddings, spectrogram = yamnet_model(audio)
    emb = embeddings.numpy()
 
    feature_vector = np.mean(emb, axis=0).reshape(1, -1)
    
    feature_vector = scaler.transform(feature_vector)
    
    pred = classifier.predict(feature_vector)
    class_index = np.argmax(pred)
    class_name = label_encoder.inverse_transform([class_index])[0]

    return class_name



