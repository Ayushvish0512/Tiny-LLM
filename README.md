# Tiny-LLM — Minimal SLM on Render Free Tier

## Quick Start

```bash
pip install -r requirements.txt
DOWNLOAD_MODEL=1 python -c "from model import load_model; load_model()"
gunicorn --workers 1 --threads 1 main:app
```

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
5. Add env var: `DOWNLOAD_MODEL=1`