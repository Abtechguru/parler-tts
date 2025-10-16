from fastapi import FastAPI
from fastapi.responses import FileResponse
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import torch

app = FastAPI(title="TTS API Test")

@app.get("/")
def home():
    return {"message": "TTS API Running Successfully ✅"}

@app.get("/speak")
def speak(text: str = "Hello from Lateef Abiodun!"):
    desc = "A calm English male voice, moderate tone."
    model_id = "parler-tts/parler-tts-mini-v1"
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(desc, return_tensors="pt")
    prompt_inputs = tokenizer(text, return_tensors="pt")
    audio = model.generate(**inputs, prompt_input_ids=prompt_inputs)
    wave = audio.cpu().numpy().squeeze()
    sf.write("output.wav", wave, model.config.sampling_rate)
    return FileResponse("output.wav", media_type="audio/wav")
