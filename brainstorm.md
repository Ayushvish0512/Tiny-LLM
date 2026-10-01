# Tiny-LLM — Host a Small Language Model on a Free Render Instance

**Status:** Draft
**Author:** Ayush
**Date:** 30 Sep 2026

---

## 1. The Problem

I want to be able to run a language model somewhere that costs nothing and doesn't need a GPU. The obvious first instinct is to reach for Hugging Face Transformers and a small PyTorch model, but that doesn't survive contact with a free hosting tier. Render's free instances give you 512 MB of RAM, one vCPU, and an ephemeral disk. Standard `torch` alone is 3 GB+ on disk and burns 150–200 MB of RAM before you even load weights. The moment you import it, the instance is effectively dead.

So the real question isn't "which model do I want to run" — it's "what is the largest model I can run inside 200 MB of resident memory on a box that has half a gig to give." The answer turns out to be much bigger than I expected, as long as I give up PyTorch entirely.

## 2. Goals

- Run a chat-capable LLM 24/7 on Render's free tier without paying.
- Stay comfortably under 512 MB RAM so the instance doesn't get OOM-killed.
- Keep install/build time short and disk usage modest.
- Expose the model over HTTP so I can hit it from anywhere.
- Survive cold starts: a spun-down instance should come back up and be usable again on its own.

## 3. Non-Goals

- High-quality or creative output. This is a demo, not a product.
- Streaming tokens at high throughput. One vCPU is one oCPU or 0.1vCPU.
- Fine-tuning, embeddings, or a vector store. Out of scope.
- Multi-user anything. There is no auth, no accounts, no per-user state.

## 4. The Key Insight

Not using PyTorch is the whole trick. `llama-cpp-python` is a thin, C++-backed binding to `llama.cpp`, the inference engine that runs quantized GGUF models. It has no autograd, no CUDA runtime, no deep-learning framework tax. It imports in well under 30 MB of RAM and the wheel is a fraction of torch's size.

That single decision is what makes the whole project fit. Everything else is just picking a model small enough to be polite about.

## 5. How the Options Stack Up

I looked at three ways to get a model running on Render:

| Approach | Disk | Idle RAM | Pulls from HF/URL? |
|---|---|---|---|
| `llama-cpp-python` | ~50 MB | ~20 MB | Yes, via `huggingface_hub` or a direct download |
| `onnxruntime` + numpy | ~100 MB | ~35 MB | Yes, but you write the download code yourself |
| `transformers` + torch | 3.5 GB+ ❌ | ~200 MB ❌ | Yes, by default |

The third column being "yes" for all three is a bit of a trap. Transformers and torch are disqualified on the first two columns alone, so the real choice is llama.cpp vs. ONNX Runtime. I'll go with `llama-cpp-python`: smaller footprint, a much better model ecosystem in GGUF, and quantizations down to Q2 that I simply can't get anywhere else.

**Dependencies (the entire list):**

```
huggingface_hub
llama-cpp-python
```

Under 70 MB of disk combined. That's the whole dependency list, which is a nice thing to be able to say out loud.

## 6. Model Choice

**`Qwen/Qwen2.5-1.5B-Instruct-GGUF`, quantized to `q2_k`.**

- It's instruction-tuned, so it can follow a chat prompt without heavy prompt engineering.
- 1.5B parameters is small but, at Q2_K, the file is roughly **350 MB** — comfortably within Render's disk budget.
- Q2 is aggressive. It hurts coherence and it definitely hurts at code. But at n_ctx=128 nobody's grading the output quality, and the memory savings are what make the thing possible.
- Alternative if quality matters more than RAM later: swap to Q4_K_M (still under a gig, still fine) or move to a 0.5B model. Keep the model ID in one config constant so this is a one-line change.

## 7. Memory Strategy

This is where the actual engineering is. Four levers, in rough order of impact:

1. **Context window — `n_ctx=128`.** This is the biggest one. KV cache memory scales linearly with context length. Dropping from 2048 to 128 removes the majority of the model's memory overhead. 128 tokens is enough for a "hello, how are you" demo and nothing more.
2. **Quantization — Q2_K.** Rough 4-bit-ish compression of weights. The difference between Q2 and Q8 in file size is enormous.
3. **Threads — `n_threads=1`.** Matches the single vCPU Render gives us. More threads would just contend for one core and add memory for no gain.
4. **Verbose off.** Suppresses the log allocation and the startup noise. Small, but free.

Target: **under 200 MB resident**, leaving headroom against the 512 MB ceiling for the interpreter, HTTP server, and the download buffer.

## 8. The Download Problem

`hf_hub_download` fetches a single `.gguf` file from a Hub repo without cloning it. That matters — a naive `git clone` of the model repo would pull every quantization variant and blow the disk budget on files I never load.

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q2_k.gguf",
)
```

The catch: **Render's disk is ephemeral.** When the instance spins down, that 350 MB file is gone, and the next cold start re-downloads it. That's a 30–60 second tax on every first request after a spin-down, which is fine for a demo and miserable if I ever care about latency.

If that becomes annoying, the fix is to host the GGUF somewhere stable — a GitHub Release asset, or a direct Google Drive / Dropbox link — and swap `hf_hub_download` for a small `urllib` streaming download with a byte-range resume check. Same on-disk result, no cold-start re-download. **Not doing this in v1; noting it as the escape hatch.**

## 9. HTTP Layer

Flask over FastAPI: Flask pulls in less, and this is one endpoint.

```
POST /chat
{ "prompt": "Hello, how are you?" }

200 OK
{ "response": "I'm doing well, thanks for asking!" }
```

Also want:
- `GET /health` — returns 200 with model-loaded status, so Render doesn't health-check-fail during the model download window.
- Model load happens once at process start, not per request. Reloading a GGUF per request would be catastrophically slow.

## 10. Deployment

- **Build:** pip install the two requirements, `llama-cpp-python` has pre-built wheels so there's no long compile.
- **Start:** `gunicorn --workers 1 --threads 1 app:app`. One worker, one thread — more would each try to load their own copy of the model into RAM and blow the limit instantly.
- **Region:** pick the region closest to wherever the model file is hosted.
- **Cold start budget:** download + load could be 60–90 seconds on a free instance. Set generous health-check grace and keep the instance warm during demos.

## 11. Risks and Known Problems

| Risk | Impact | Mitigation |
|---|---|---|
| 350 MB model re-downloads on every spin-down | Slow cold starts | Host on a GitHub Release and stream instead (deferred) |
| `llama-cpp-python` has no prebuilt wheel for the platform | Build failures | Pin version, or build from source in the Render build step |
| Actual RAM exceeds my 200 MB estimate | OOM kill | Measure it in production; drop `n_ctx` further or switch to a 0.5B model |
| Q2 quality is noticeably bad | Poor demo | Acceptable for v1; Q4_K_M is the upgrade path |
| Free tier sleeps after inactivity | Latency | Wake it manually before demos |
| Someone finds the endpoint and runs up the bill | — | Free tier can't overcharge; no card on file means worst case is suspension |

## 12. Rough Build Order

1. Prove the model loads and generates inside the RAM budget, in a bare script. No server yet. If this fails, everything else is pointless.
2. Wrap it in Flask with `/chat` and `/health`.
3. Deploy to Render, measure real resident memory and real cold-start time.
4. Write down the actual numbers and correct the estimates in this doc.

## 13. Open Questions

- Exact `n_ctx` floor where quality collapses — worth a sweep of 64 / 128 / 256.
- Whether Render's free disk limit has room for a Q4 model instead (~1 GB).
- Whether to add a streaming (`text/event-stream`) response. Nice, but Q2 output on one core is not fast, so the stream would be trickling.
- Do I even need `huggingface_hub` as a dependency, or should I just stream from a fixed URL and drop one package?

---

## 14. Definition of Done

- [ ] Model loads from Hugging Face on a cold start and generates a coherent-enough response
- [ ] Peak RSS under 200 MB, verified in production
- [ ] `POST /chat` and `GET /health` both responding
- [ ] Deployed and publicly reachable at a Render URL
- [ ] Cold-start time measured and written down
- [ ] Total disk usage under 1 GB
