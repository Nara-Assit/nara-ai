import os
import json
import tempfile
import subprocess
from vosk import Model, KaldiRecognizer
import wave
from pathlib import Path

# Model path - points to the model downloaded in Docker
MODEL_PATH = "/app/app/stt/vosk-model-ar-mgb2-0.4"

# Load model once at startup
_model = None

def get_model():
    """Lazy load the Vosk model"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Vosk model not found at {MODEL_PATH}")
        _model = Model(MODEL_PATH)
    return _model

def convert_to_wav(input_file: str) -> str:
    """Convert audio file to WAV format (16kHz, mono, 16-bit PCM)"""
    output_file = tempfile.mktemp(suffix=".wav")
    
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '16000',        # 16kHz sample rate
        '-ac', '1',            # Mono
        '-c:a', 'pcm_s16le',   # 16-bit PCM
        '-y',
        output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not installed")

async def transcribe_audio(audio_path: str) -> str:
    """Transcribe Arabic audio file using Vosk"""

    # Always convert the input audio to the required WAV format to ensure consistency
    # and handle potential issues with input WAV files not conforming to expected format.
    wav_file = convert_to_wav(audio_path)
    temp_file = wav_file # Mark for cleanup

    try:
        wf = wave.open(wav_file, "rb")

        # These checks are now redundant if convert_to_wav always produces the correct format,
        # but keeping them for an extra layer of validation or if convert_to_wav changes.
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Audio must be mono 16-bit PCM")

        model = get_model()
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        full_text = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    full_text.append(text)

        final_result = json.loads(rec.FinalResult())
        final_text = final_result.get("text", "").strip()
        if final_text:
            full_text.append(final_text)

        wf.close()

        transcription = " ".join(full_text)
        return transcription if transcription else "لم يتم اكتشاف أي كلام"

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
