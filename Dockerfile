FROM python:3.11.9-slim

WORKDIR /app

# Build toolchain: PyPI ships llama-cpp-python as a source tarball only,
# so the wheel has to be compiled here rather than downloaded.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Keep the llama.cpp compile lean: a single generic x86-64 variant, built
# with low parallelism, and none of the targets this API never calls.
ENV CMAKE_BUILD_PARALLEL_LEVEL=2 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_LLAMAFILE=OFF -DGGML_BLAS=OFF -DGGML_LTO=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF"

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch the model during the build so container starts are instant and do
# not depend on Google Drive being reachable at runtime.
RUN python model.py

ENV DOWNLOAD_MODEL=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Render injects PORT and proxies only to that port, so the bind must use
# $PORT. Exec-form CMD does not expand variables, hence the shell form.
CMD ["sh", "-c", "gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:$PORT main:app"]