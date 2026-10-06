# Tiny-LLM - Minimal SLM API on Render

Streaming chat completions from Qwen1.5-0.5B-Chat (Q2_K GGUF, 298 MB) over FastAPI,
tuned for a ~200 MB RSS footprint: 1 worker, 1 thread, mmap, 512-token context.

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/` | - | status payload (Render default health-check path) |
| GET | `/health` | - | `{"status", "model", "file"}` |
| POST | `/chat` | `{"prompt": str <=500, "max_tokens": 1..256}` | `text/plain`, streamed |

The model is fetched from Google Drive into `models/`. `model.py` refuses any file
smaller than `MIN_MODEL_SIZE_MB` and re-downloads it, so a truncated Drive
transfer heals itself instead of loading a corrupt GGUF.

## Render deploy (native Python service)

Build command:

```
pip install -r requirements.txt && python model.py
```

Start command:

```
uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120
```

Health check path: `/health`

`Procfile` already holds the start command, so leaving the dashboard field blank is
equivalent. Do not shorten the start command to `uvicorn main:app` - it binds
`127.0.0.1:8000`, which Render can never reach, and the deploy fails its health
check with no obvious error.

Running `python model.py` at build time downloads the model once into the build
directory, which native Python services reuse at runtime. Without it, every cold
start and restart re-downloads ~300 MB before accepting traffic.

### Environment variables

`llama-cpp-python` publishes no wheel on PyPI, so Render compiles llama.cpp from
the source tarball. The default build compiles a separate shared library for every
x86 ISA variant in parallel and peaks above 8 GB, where the OOM killer ends the
build with a generic `failed-wheel-build-for-install`. Setting these in the
dashboard cuts it to one variant, built serially:

```
CMAKE_BUILD_PARALLEL_LEVEL=1
CMAKE_ARGS=-DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_LLAMAFILE=OFF -DGGML_BLAS=OFF -DGGML_LTO=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
```

None of those backends are used: `model.py` runs a 0.5B Q2_K model through
`mul_mat_q`, which is plain CPU quantised matmul.

Optional - `model.py` falls back to these exact defaults, so `.env` is not needed
on Render. See `.env`.

```
MODEL_DRIVE_URL, MODEL_FILENAME, MIN_MODEL_SIZE_MB
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python model.py
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
```

Check it: `curl http://localhost:8000/health`

### Windows caveat

`llama-cpp-python` cannot be installed on Windows from PyPI alone. It ships a
source tarball for every version, and building it needs Visual Studio Build Tools
(`cl.exe` / `nmake.exe`). The failure looks like:

```
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
ERROR: Failed building wheel for llama-cpp-python
```

Install the Build Tools, or run inference through Docker or Render. `gunicorn` is
likewise Unix-only (`No module named 'fcntl'`) and will not start on Windows,
which is why the start command above invokes `uvicorn` directly.

Note that `.venv/` and `models/` are gitignored and can be cleared between
sessions; recreate the venv and re-run `model.py` if either goes missing.
## Docker deploy (no compile on Render)

The `Dockerfile` compiles llama.cpp once at build time and bakes the
298 MB model into the image, so deploys never compile and container
starts are instant with no Google Drive dependency.

```
docker build -t tiny-llm .
docker tag tiny-llm docker.io/<user>/tiny-llm:latest
docker push docker.io/<user>/tiny-llm:latest
```

Then create a **Docker Web Service** on Render from that image.

Notes:

- Docker Web Services on Render are not free tier; you pay for the
  instance.
- The image is ~1.2 GB because the model is baked in, and it is
  re-pulled on every deploy.
- The `CMD` binds `0.0.0.0:$PORT`, which is what Render requires -
  a hardcoded port makes the container unreachable.
- Local run: `docker run --rm -p 8000:8000 tiny-llm`, then
  `curl http://localhost:8000/health`.
## Troubleshooting

**Every request returns 500 with `FastAPI.__call__() missing 1 required positional argument: 'send'`**

The Start Command in the Render dashboard overrides `Procfile`. A plain
`gunicorn ... main:app` uses gunicorn's default **sync** worker, which is a WSGI
worker and calls the app as `self.wsgi(environ, resp.start_response)` - two
arguments. `FastAPI.__call__` is ASGI and needs three (`scope, receive, send`),
so every request raises `TypeError`. Either delete the dashboard Start Command so
the `Procfile` is used, or add the worker class:

```
gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 120 --bind 0.0.0.0:$PORT main:app
```

**Build stalls on `Building wheel for llama-cpp-python`, or dies with an OOM**

Two causes. Check the interpreter path in the log first: if it shows `python3.14`,
`runtime.txt` was ignored, and the C++ sdist is being compiled against a new C
API. Set `PYTHON_VERSION=3.11.9` in the dashboard and keep `.python-version`
committed. Then apply the `CMAKE_ARGS` block above to cut the compile down to a
single x86 variant.
