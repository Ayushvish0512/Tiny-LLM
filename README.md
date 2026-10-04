# Tiny-LLM — Minimal SLM on Render Free Tier

## Local Setup (venv only, no Docker)

```bash
# 1. install
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. build (fetch model if missing, no-op if present)
.venv\Scripts\python.exe model.py

# 3. run
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
```

Check it: `curl http://localhost:8000/health`

The model is fetched from Google Drive into `models/`. `model.py` verifies the file
exists and exceeds `MIN_MODEL_SIZE_MB` before use — a missing or truncated file is
re-downloaded automatically, so step 2 is safe to re-run.

## Endpoints

- `GET /health` — model load status
- `POST /chat` — `{"prompt": "Hello", "max_tokens": 64}` → streams response

## Memory Config

- Model: Qwen2.5-0.5B-Instruct Q2_K (~150MB file)
- Context: 128 tokens
- Threads: 1
- mmap: enabled
- Target: <200MB RSS

## Render Deploy

1. Push to GitHub
2. New Web Service on Render
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn --workers 1 --threads 1 --timeout 120 --bind 0.0.0.0:$PORT main:app`

Model location and Drive URL come from `.env` (`MODEL_DRIVE_URL`,
`MODEL_FILENAME`, `MIN_MODEL_SIZE_MB`). No download flag needed.