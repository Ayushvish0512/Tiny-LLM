import os
import gc

MODEL_DIR = "models"
MODEL_FILENAME = "qwen2.5-0.5b-instruct-q2_k.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

FILE_ID = "1iwluL_LzkdMxx7VgUw3gaCxDectPCTo8"
MIN_MODEL_SIZE_BYTES = 10 * 1024 * 1024

def _model_is_valid() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > MIN_MODEL_SIZE_BYTES

def download_model(max_retries: int = 3, retry_sleep_s: int = 5) -> None:
    if _model_is_valid():
        return

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
            gdown.download(url, MODEL_PATH, quiet=False)
            if _model_is_valid():
                print("Download succeeded.")
                return
            raise RuntimeError("File too small after download")
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                import time
                time.sleep(retry_sleep_s)
    raise RuntimeError(f"Download failed after {max_retries} attempts")

def load_model():
    download_model()
    gc.collect()

    from llama_cpp import Llama

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=128,
        n_batch=32,
        n_threads=1,
        use_mlock=False,
        use_mmap=True,
        verbose=False,
        logits_all=False,
    )
    return llm