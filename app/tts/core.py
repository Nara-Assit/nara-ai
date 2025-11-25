import requests
from gtts import gTTS

def auto_tashkeel(text: str) -> str:
    """
    get the text diacritized using Mishkal API
    """
    try:
        url = "https://tahadz.com/mishkal/ajaxGet"
        payload = {'text': text, 'action': 'Tashkeel'}
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://tahadz.com/mishkal/'
        }
        
        response = requests.post(url, data=payload, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()

            return data.get('result', text)
        else:
            print("Mishkal API Error, using raw text.")
            return text
            
    except Exception as e:
        print(f"Tashkeel Connection Failed: {e}")
        return text 

def generate_tts(text: str, output_path: str):
    tts = gTTS(text=text, lang="ar")
    tts.save(output_path)