# DUEL LAB — Parler-TTS Mini integration

This is a **separate opt-in** inference service for the DUEL LAB question narrator. The existing repo contains training/inference docs but is not itself a live API server. This adapter adds an authenticated FastAPI `/synthesize` endpoint.

## Why separate hosting?

Parler-TTS Mini v1 has 880M parameters and model weights of around 3.5 GB. It is unsuitable for execution in a standard static Vercel site. A CUDA-capable inference host is recommended. **No GPU hosting is created automatically by this repository. The existing fork is a partial upstream snapshot (no installable `parler_tts/` package directory); the reference container installs the canonical upstream toolkit from Hugging Face and uses the Mini v1 checkpoint.** The DUEL LAB frontend works for free using the Web Speech API without this service.

## Development (on a GPU machine)

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-voice-api.txt
pip install git+https://github.com/huggingface/parler-tts.git
export PARLER_TTS_API_KEY='set-a-secret-outside-the-repository'
uvicorn api_service.server:app --host 127.0.0.1 --port 8000
```

`PARLER_PRELOAD_ON_START=true` (default) attempts to load the model before serving requests; this prevents the first student from encountering a cold-start stall. `PARLER_ALLOW_CPU=true` allows CPU testing, but inference may be extremely slow. Set `PARLER_PRELOAD_ON_START=false` only for API contract tests where the real model is not required. Never expose an unauthenticated endpoint. For a production deployment: configure HTTPS, server-level request quotas, one GPU worker, monitoring, and a private API key.

## Connect to DUEL LAB on Vercel

Set these environment variables in **the DUEL LAB Vercel project** (not in client-side JavaScript):

- `PARLER_TTS_ENABLED=true`
- `PARLER_TTS_URL=https://your-secured-voice-host.example`
- `PARLER_TTS_API_KEY=your-unique-private-key`
- `PARLER_TTS_RATE_LIMIT_URL=https://your-redis-rest-host.example/`
- `PARLER_TTS_RATE_LIMIT_TOKEN=your-redis-rest-secret`

The rate-limit variables must point to an Upstash-compatible Redis REST endpoint. The browser talks only to its same-origin `/api/tts` proxy, which has a default per-IP budget of 20 calls/hour and global budget of 200 calls/hour. Change with `PARLER_TTS_IP_LIMIT` or `PARLER_TTS_GLOBAL_LIMIT`; the limits are approximate hour buckets, not abuse-proof identity verification. Firewall/bot mitigation is recommended before opening a paid model to public traffic.

If any necessary configuration is missing, Parler stays **disabled** and DUEL LAB's free device voice is used. No secrets should appear in GitHub or be copied to a browser.

## API contract

`POST /synthesize` JSON: `{"text":"Question one...","style":"host"}` with `X-API-Key: ...`. Returns `audio/wav`, 401 for missing authentication, 422 for bad inputs, 503 when the model can't be loaded. Clients must limit to 450 characters; the DUEL LAB proxy checks input against its published question bank.

## GPU container

The optional `Dockerfile.voice-api` uses a PyTorch/CUDA runtime and expects an NVIDIA GPU host. Example build command: `docker build -f Dockerfile.voice-api -t duel-parler-mini .` followed by running with GPU device support and `PARLER_TTS_API_KEY` supplied at runtime as a secret. Downloading model weights on first startup requires sufficient disk, memory and bandwidth; prewarm is enabled by default and startup may be slow. The adapter has not yet been run against a real GPU model in the current environment.