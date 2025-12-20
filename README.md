# AI Services API

FastAPI backend that bundles multiple assistive AI capabilities: text-to-speech (TTS), speech-to-text (STT) powered by the klaam Arabic wav2vec models, and sign-language utilities. Everything can be shipped as a single Docker image.

## Prerequisites

- Docker (recommended) or Python 3.11 with `pip`
- At least 8 GB of RAM and ~3 GB of free disk space for the Hugging Face STT checkpoints

## Running with Docker

```bash
docker build -t ai-services .
docker run --rm -p 8000:8000 ai-services
```

The API becomes available at `http://localhost:8000`. The first STT request lazily downloads the wav2vec weights, so expect a longer cold start.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Key endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/tts/convert` | `POST` | Form fields `text` and optional `apply_tashkeel` generate an MP3 response. |
| `/api/stt/convert` | `POST` | Multipart form with `audio` (file) and `lang` (`egy`/`msa`) returns a JSON transcript. |
| `/api/sign/...` | `POST/GET` | Sign modules (see `app/sign`). |

### Speech-to-text example

```bash
curl -X POST "http://localhost:8000/api/stt/convert" \
	-F "audio=@/path/to/audio.wav" \
	-F "lang=egy"
```

Response:

```json
{
	"service": "stt",
	"language": "egy",
	"transcript": "..."
}
```

### Text-to-speech example

```bash
curl -X POST "http://localhost:8000/api/tts/convert" \
	-F "text=مرحبا" \
	-o output.mp3
```

## Notes

- STT files larger than 10 MB are rejected (`MAX_FILE_SIZE`).
- Supported STT languages: Egyptian Arabic (`egy`) and Modern Standard Arabic (`msa`).
- Temporary audio assets are stored under `/tmp/ai_services` (configurable via `TEMP_DIR`).

---

## Sound Alert 

Detects critical environmental sounds to alert deaf and hard-of-hearing users.
This service is designed for **mobile deployment (Flutter)** using TensorFlow Lite.

### Pipeline
- Input audio: Mono WAV, 16 kHz, 5 seconds
- Audio normalization to [-1, 1]
- Feature extraction using **YAMNet**
- 1024-dim audio embeddings (mean pooled over time)
- Embedding normalization using **StandardScaler**
- Dense classifier (TFLite)

### Model Details
- Input shape: (1, 1024)
- Data type: float32
- Output: Softmax probabilities (6 classes — see `labels.txt`) 

### Deployment Notes
- The classifier **does not accept raw audio**
- YAMNet must be executed first to extract embeddings
- Mobile inference uses:
  - `yamnet.tflite`
  - `sound_classifier.tflite`
  - `scaler.json`
  - `labels.txt`
  
