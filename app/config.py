import os

class Settings:
    TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/ai_services")
    TTS_LANG = os.getenv("TTS_LANG", "ar")
    MISHKAL_URL = os.getenv("MISHKAL_URL", "https://tahadz.com/mishkal/ajaxGet")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))

settings = Settings()