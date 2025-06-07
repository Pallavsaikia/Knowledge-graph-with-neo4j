from gtts import gTTS
import io

class GTTSService:
    def __init__(self, lang="en"):
        self.lang = lang

    def text_to_audio_bytes(self, text: str) -> bytes:
        tts = gTTS(text=text, lang=self.lang)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        return buffer.getvalue()
