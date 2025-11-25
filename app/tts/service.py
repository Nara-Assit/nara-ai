from gtts import gTTS
import uuid
import os
import requests
from app.config import settings

def auto_tashkeel(text: str) -> str:
    try:
        response = requests.post(
            settings.MISHKAL_URL,
            data={'text': text, 'action': 'Tashkeel'},
            headers={
                'User-Agent': 'Mozilla/5.0',
                'Referer': 'https://tahadz.com/mishkal/'
            },
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get('result', text)
        return text
    except:
        return text

async def text_to_speech(text: str, apply_tashkeel: bool = True) -> str:
    if apply_tashkeel:
        processed_text = auto_tashkeel(text)
        print(f"Original: {text}")
        print(f"Tashkeel: {processed_text}")
    else:
        processed_text = text
    
    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(settings.TEMP_DIR, filename)
    
    tts = gTTS(text=processed_text, lang=settings.TTS_LANG)
    tts.save(output_path)
    
    return output_path