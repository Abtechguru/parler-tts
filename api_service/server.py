"""Authenticated GPU inference adapter for the Parler-TTS Mini model.

This service is NOT hosted on Vercel and requires a separately configured inference host.
No model is loaded during import, which permits contract testing without GPU resources.
"""
from __future__ import annotations
import asyncio
import hmac
import io
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = "parler-tts/parler-tts-mini-v1"
VOICE_DESCRIPTIONS = {
    "host": "Jon has a clear, confident and professional male voice, speaking at a measured pace with crisp articulation. The recording is very clear and close up with no background noise.",
    "calm": "Laura has a warm, calm and reassuring female voice, speaking slowly and clearly. The recording is of very high quality, very clear and close up.",
    "energy": "Jordan has an energetic and expressive voice with a lively speaking pace, crisp pronunciation and an engaging presenter style. The recording is very clear and close up.",
}

class SynthesisRequest(BaseModel):
    text: str = Field(min_length=4, max_length=450)
    style: Literal["host", "calm", "energy"] = "host"

_engine = None
_lock = asyncio.Lock()


def require_key(key: str | None):
    expected = os.environ.get("PARLER_TTS_API_KEY")
    if not expected or not key or not hmac.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def load_engine():
    global _engine
    if _engine is not None:
        return _engine
    import torch
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and os.getenv("PARLER_ALLOW_CPU", "false").lower() != "true":
        raise RuntimeError("GPU unavailable: set PARLER_ALLOW_CPU=true to opt in to slower CPU inference")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith("cuda") else torch.float32
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _engine = (model, tokenizer, device, torch)
    return _engine


def generate_wav(text: str, style: str) -> bytes:
    import numpy as np
    import soundfile as sf
    model, tokenizer, device, torch = load_engine()
    prompt = tokenizer(text, return_tensors="pt")
    description = tokenizer(VOICE_DESCRIPTIONS[style], return_tensors="pt")
    with torch.inference_mode():
        samples = model.generate(
            input_ids=description.input_ids.to(device),
            attention_mask=description.attention_mask.to(device),
            prompt_input_ids=prompt.input_ids.to(device),
            prompt_attention_mask=prompt.attention_mask.to(device),
            do_sample=True,
        )
    arr = samples.cpu().numpy().squeeze().astype(np.float32)
    wav = io.BytesIO()
    sf.write(wav, arr, model.config.sampling_rate, format="WAV", subtype="PCM_16")
    return wav.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("PARLER_PRELOAD_ON_START", "true").lower() == "true":
        await asyncio.to_thread(load_engine)
    yield

app = FastAPI(title="DUEL LAB Parler Mini Voice Adapter", docs_url=None, redoc_url=None, lifespan=lifespan)

@app.get("/health")
def health():
    return {"service": "parler-mini", "configured": bool(os.getenv("PARLER_TTS_API_KEY")), "model": MODEL_ID, "loaded": _engine is not None}

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest, x_api_key: str | None = Header(default=None)):
    require_key(x_api_key)
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Empty input")
    if len(request.text)>450:
        raise HTTPException(status_code=413, detail="Text too long")
    async with _lock:
        try:
            wav = await asyncio.to_thread(generate_wav, request.text, request.style)
        except Exception as e:
            # Avoid leaking model install details, environment variables or upstream paths.
            raise HTTPException(status_code=503, detail="Inference unavailable") from e
    if len(wav)>4_200_000:
        raise HTTPException(status_code=413, detail="Audio exceeds the upstream proxy limit")
    return Response(content=wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})