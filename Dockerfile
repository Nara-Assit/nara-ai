FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

    
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /app/app/stt && \
    cd /app/app/stt && \
    wget https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip && \
    unzip vosk-model-ar-mgb2-0.4.zip && \
    rm vosk-model-ar-mgb2-0.4.zip


COPY ./app ./app

RUN mkdir -p /tmp/ai_services

EXPOSE 8000

CMD ["python", "-m", "app.main"]