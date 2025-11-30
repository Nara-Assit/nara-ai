import freesound
import os
import urllib.request
from pydub import AudioSegment
import time

API_KEY = "Aj4hCXiilQI060G5cCYHqJpeMIxW7oW6yQsEpmtY"  
client = freesound.FreesoundClient()
client.set_token(API_KEY)


SAVE_DIR = "dataset/doorbell"
os.makedirs(SAVE_DIR, exist_ok=True)


QUERY = "doorbell"
MAX_SOUNDS = 100  
def safe_download(url, path, retries=3):
    """Download file with retry logic"""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, path)
            return True
        except Exception as e:
            print(f"  Download failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2)
    return False

def download_and_convert(sound):
    """Download MP3 and convert to WAV"""
    try:
        if not hasattr(sound, 'previews'):
            print(f"  No preview available for sound {sound.id}")
            return False
        
        mp3_url = None
        if hasattr(sound.previews, 'preview_hq_mp3'):
            mp3_url = sound.previews.preview_hq_mp3
        elif hasattr(sound.previews, 'preview_lq_mp3'):
            mp3_url = sound.previews.preview_lq_mp3
        
        if not mp3_url:
            print(f"  ⚠ No MP3 preview for sound {sound.id}")
            return False
        
        mp3_name = f"{sound.id}.mp3"
        mp3_path = os.path.join(SAVE_DIR, mp3_name)
        wav_name = f"{sound.id}.wav"
        wav_path = os.path.join(SAVE_DIR, wav_name)
        
        # Skip if already downloaded
        if os.path.exists(wav_path):
            print(f"  ✓ Already exists: {wav_name}")
            return True

        print(f"\n Downloading: {mp3_name}")
        
        # Download MP3
        if not safe_download(mp3_url, mp3_path):
            print(f"  Failed to download: {mp3_name}")
            return False
        
        print(f"  Downloaded: {mp3_name}")
        
        # Convert to WAV
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        print(f"  Converted to WAV: {wav_name}")
        
        # Remove MP3
        os.remove(mp3_path)
        
        time.sleep(0.5)
        return True
        
    except Exception as e:
        print(f"  Error processing sound {sound.id}: {e}")
        if 'mp3_path' in locals() and os.path.exists(mp3_path):
            try:
                os.remove(mp3_path)
            except:
                pass
        return False

print(f" Searching for '{QUERY}' sounds...")
print(f" Saving to: {SAVE_DIR}\n")

try:
    results = client.search(
        query=QUERY,
        fields="id,name,previews",
        page_size=40
    )
    
    print(f"Search successful! Starting download...")
    print(f"Downloading up to {MAX_SOUNDS} sounds...\n")
    
    downloaded_count = 0
    page_num = 1
    
    while downloaded_count < MAX_SOUNDS:
        print(f"\n{'='*50}")
        print(f" Processing page {page_num}")
        print(f"{'='*50}")
        
        sound_count_this_page = 0
        for sound in results:
            if downloaded_count >= MAX_SOUNDS:
                break
            
            sound_count_this_page += 1
            print(f"[{downloaded_count + 1}/{MAX_SOUNDS}] Processing sound ID: {sound.id}")
            if download_and_convert(sound):
                downloaded_count += 1
        
        if sound_count_this_page == 0:
            print("No more sounds found.")
            break
        
        print(f"\nProcessed {sound_count_this_page} sounds on this page")
        
        if downloaded_count >= MAX_SOUNDS:
            break
            
        try:
            print(f"\n Loading next page...")
            results = results.next_page()
            page_num += 1
            time.sleep(1)  
        except Exception as e:
            print(f"\n No more pages available")
            break
    
    print(f" Download complete!")
    print(f" Successfully downloaded: {downloaded_count} sounds")


except Exception as e:
    print(f"\n Fatal error: {e}")
    print("\nTroubleshooting:")
    import traceback
    traceback.print_exc()