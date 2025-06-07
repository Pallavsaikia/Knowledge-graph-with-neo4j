import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
import io
import soundfile as sf
import sounddevice as sd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from scipy.signal import resample_poly
import os
import time
import uuid

from rag.neo4j import Neo4jQueryEngine
from gtts import gTTS
from text_to_audio.generator import GTTSService,TTSCache
import random

# ---------- Setup ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify origins like ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = WhisperModel("base", compute_type="int8", device="cpu")
audio_buffer = bytearray()
os.makedirs("audio", exist_ok=True)

Neo4jQueryEngine.setup_llm()
query_engine = Neo4jQueryEngine()
tts_service = GTTSService()
tts_cache = TTSCache(tts_service)
WAIT_TEXT=["Just a sec.","Uhmm give me a sec","Please be on line"]

@app.post("/join")
async def join_call(request: Request):
    data = await request.json()
    call_id = data["room_id"]
    print(f"[BOT] Received join request for call_id: {call_id}")
    asyncio.create_task(bot_join_call(call_id))
    return {"status": "bot joining"}


async def bot_join_call(call_id: str):
    uri = f"ws://localhost:8080/ws?room={call_id}"
    print(f"[BOT] Connecting to {uri}")
    async with websockets.connect(uri) as websocket:
        print("[BOT] Connected, listening for audio...")

        global audio_buffer

        silence_threshold = 0.02
        silence_duration_ms = 1500
        original_sample_rate = 48000
        target_sample_rate = 16000
        up = target_sample_rate
        down = original_sample_rate

        bytes_per_sample = 2  # int16
        channels = 1
        silent_duration = 0.0

        while True:
            try:
                data = await websocket.recv()
                if isinstance(data, bytes):
                    audio_buffer.extend(data)

                    chunk_samples = len(data) // (bytes_per_sample * channels)
                    chunk_duration = chunk_samples / original_sample_rate

                    audio_int16 = np.frombuffer(data, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    audio_16k = resample_poly(audio_float, up, down)

                    energy = np.sqrt(np.mean(audio_16k ** 2))

                    if energy < silence_threshold:
                        silent_duration += chunk_duration
                    else:
                        silent_duration = 0.0

                    min_buffer_len = target_sample_rate * 2 * 2  # 2s buffer

                    if silent_duration >= silence_duration_ms / 1000.0 and len(audio_buffer) > min_buffer_len:
                        audio_int16_full = np.frombuffer(audio_buffer, dtype=np.int16)
                        audio_float_full = audio_int16_full.astype(np.float32) / 32768.0
                        audio_16k_full = resample_poly(audio_float_full, up, down)
                        max_energy = np.max(np.abs(audio_16k_full))

                        if max_energy < silence_threshold:
                            print("[BOT] Buffer contains only silence, skipping transcription.")
                            audio_buffer = bytearray()
                            silent_duration = 0.0
                        else:
                            print(f"[BOT] {silent_duration:.2f}s silence detected. Processing audio chunk...")
                            await transcribe_and_respond(audio_buffer, target_sample_rate, up, down, websocket)
                            audio_buffer = bytearray()
                            silent_duration = 0.0
                else:
                    print(f"[BOT] Received non-bytes data: {data}")
            except Exception as e:
                print(f"[BOT] Error: {e}")
                break


async def transcribe_and_respond(buffer: bytearray, target_sample_rate, up, down, websocket):
    print("[BOT] Transcribing buffered audio...")

    audio_int16 = np.frombuffer(buffer, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    audio_16k = resample_poly(audio_float, up=up, down=down) if up != down else audio_float

    wav_io = io.BytesIO()
    sf.write(wav_io, audio_16k, target_sample_rate, format='WAV')
    wav_io.seek(0)

    segments, _ = model.transcribe(wav_io, language="en")
    text = "".join(segment.text for segment in segments).strip()

    if text:
        tts_cache = TTSCache(tts_service)
        wait_audio = tts_cache.get_audio_bytes(random.choice(WAIT_TEXT))
        await websocket.send(wait_audio)
        print(f"[BOT] Transcription result: {text}")
        response = query_engine.query(text)
        print(f"\n[BOT] Response: {response}")

        # Generate TTS response
        tts_audio = tts_service.text_to_audio_bytes(str(response))
        await websocket.send(tts_audio)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
