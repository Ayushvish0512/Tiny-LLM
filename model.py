import os
import gc
import time
import re
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = "models"
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "qwen1.5-0.5b-chat-q2_k.gguf")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

MIN_MODEL_SIZE_BYTES = int(os.getenv("MIN_MODEL_SIZE_MB", "50")) * 1024 * 1024

def extract_file_id(drive_url: str) -> str:
    """Extract file ID from various Google Drive URL formats."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'uc\?id=([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    return drive_url

FILE_ID = extract_file_id(os.getenv("MODEL_DRIVE_URL", "1QuremM9CEn1B--2k7Uz9O5n61ftsMo0u"))

def _model_is_valid() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > MIN_MODEL_SIZE_BYTES

def model_status() -> str:
    if not os.path.exists(MODEL_PATH):
        return "missing"
    if not _model_is_valid():
        return f"corrupt: {os.path.getsize(MODEL_PATH)} bytes, need > {MIN_MODEL_SIZE_BYTES}"
    return "available"

def download_model(max_retries: int = 5, retry_sleep_s: int = 10) -> None:
    status = model_status()
    if status == "available":
        print(f"Model already available at {MODEL_PATH}, skipping download.")
        return
    if status == "missing":
        print(f"No model file at {MODEL_PATH}, downloading.")
    else:
        print(f"Existing model file unusable ({status}), re-downloading.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) <= MIN_MODEL_SIZE_BYTES:
        try:
            os.remove(MODEL_PATH)
        except OSError:
            pass

    print(f"Downloading {MODEL_FILENAME} from Google Drive...")
    try:
        import gdown
    except ImportError as e:
        raise ImportError("gdown not installed") from e

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}...")
            # Use gdown with no cookies and fuzzy matching
            gdown.download(url, MODEL_PATH, quiet=False)
            if _model_is_valid():
                print("Download succeeded.")
                return
            raise RuntimeError("File too small after download")
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_sleep_s)
    raise RuntimeError(f"Download failed after {max_retries} attempts")

if __name__ == "__main__":
    download_model()
    print(f"Model status: {model_status()}")

def load_model():
    download_model()
    gc.collect()

    from llama_cpp import Llama

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_batch=8,
        n_threads=1,
        n_threads_batch=1,
        use_mlock=False,
        # Full model resident (~280 MB) stays within the 350 MB budget and
        # removes mmap page-fault stalls. CPU is already saturated on the
        # shared vCPU, so n_threads stays 1 (more threads only thrash it).
        use_mmap=False,
        verbose=False,
        logits_all=False,
        low_vram=True,
        numa=False,
        mul_mat_q=True,
        offload_kqv=True,
    )
    return llm