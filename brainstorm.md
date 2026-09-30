# Tiny-LLM — Product Requirements Document

## Ultra-Low Footprint Conversational API ("Micro-Chat")

**Status:** Draft
**Author:** Ayush
**Date:** 30 Sep 2026
**Supersedes:** the Python/llama-cpp-python draft. We moved to Rust — reasoning in §2.

---

## 1. Executive Summary & Objective

Build an asynchronous chat API in Rust that runs inside a hard **200 MB RAM ceiling** on a single free-tier vCPU, with support for **10 simultaneous chat sessions**. Conversation context is offloaded to an ephemeral SQLite database rather than held in RAM, and memory is released the instant a user disconnects.

The whole point is the constraint. A free Render instance has 512 MB and one core. Everything below — the language, the inference engine, the model quantization, the context cap, the SQLite offload — is downstream of that one number.

---

## 2. Why Rust Instead of Python

The Python draft assumed `llama-cpp-python`. It fits in ~20 MB idle and would probably have shipped. But for a service whose entire value proposition is *not using memory carelessly*, Python is working against me the whole way:

| Concern | Python | Rust |
|---|---|---|
| Baseline interpreter cost | ~15–25 MB, plus whatever the interpreter has already fragmented the heap | Near zero — the binary is the process |
| Release of conversation history | Refcounting + cyclic GC; freed "eventually," with fragmentation | Deterministic `Drop`, freed *exactly* at scope exit |
| Peak RSS during a burst | Transient spikes from allocator arenas | No allocator churn; flat curve |
| Concurrency model | GIL forces threads or multiprocess; a worker process means a second copy of the model | tokio async tasks on one thread; model shared by reference, never reloaded |
| Binary/model on disk | ~50 MB wheels + interpreter + 350 MB GGUF | Single stripped binary + the same GGUF |

The decisive point is the second row. Requirement 3.1 says memory must be released on disconnect — Python gives me "it'll get collected eventually, probably." Rust gives me "it is released, and the compiler enforces that the `Drop` impl runs." For a service that's going to be judged on an RSS ceiling, deterministic is the whole game.

Python also loads the model once per worker process. Two workers means two copies of 135M–500M parameters and an immediate OOM. Rust's single-threaded async queue sidesteps that: one model instance, many coroutines, no copies.

**Cost of the switch:** compile times on Render's build step (mitigated by a multi-stage Docker build that compiles on a fat image and ships a slim runtime), and a smaller talent pool if this ever needs maintaining. Both are acceptable.

---

## 3. Model Candidates & Memory Footprints

This is the decision that actually determines whether the project works. All figures below are **weights on disk** plus an **estimated resident RAM contribution** after load, at `n_ctx = 256`, `n_threads = 1`, on x86_64.

| Model | Params | Quant | File on disk | Resident contribution | Verdict |
|---|---|---|---|---|---|
| **SmolLM2-135M-Instruct** | 135M | Q4_K_M | **~100 MB** | **~110–125 MB** | ✅ **Proposed default.** Safest margin; leaves ~75 MB for everything else |
| **SmolLM2-135M-Instruct** | 135M | Q8_0 | ~145 MB | ~155–170 MB | ⚠️ Tight. Only if 135M Q4 quality is unusable |
| **Qwen2.5-0.5B-Instruct** | 494M | Q2_K | ~350 MB | ~380–410 MB | ❌ **Over budget on a 200 MB ceiling.** Good model, wrong tier |
| **Qwen2.5-1.5B-Instruct** | 1.5B | Q2_K | ~350 MB¹ | ~900 MB–1.2 GB | ❌ Far over budget. Disqualified |
| **Qwen2.5-0.5B-Instruct** | 494M | Q4_K_M | ~380 MB | ~410–440 MB | ❌ Over budget |

¹ The 1.5B Q2_K file is *smaller on disk* than the 0.5B Q4_K_M file — quantization gain beats the extra parameters — but it still lands far above the RAM ceiling once the KV cache and runtime overhead are counted. **Disk size was never the real constraint; resident size is.** That was the key correction from the Python draft.

### Decision

**Primary: `SmolLM2-135M-Instruct` at `Q4_K_M`, from `HuggingFaceTB/SmolLM2-135M-Instruct-GGUF`, file `smollm2-135m-instruct-q4_k_m.gguf`.**

Rationale:
- ~100 MB on disk, ~110–125 MB resident. This is the only candidate that leaves real headroom under 200 MB.
- It is instruction-tuned, so it holds a chat format without prompt gymnastics.
- Q4_K_M is a genuinely good quantization — noticeably better than the Q2/Q3 that make bigger models "fit." At 135M, a good 4-bit beats a bad 2-bit.
- Its quality ceiling is low. This is a demo, not a product. If 135M proves too weak, the honest next step is a 360M Q4_K_M (~250 MB) on a raised budget — **not** a jump to 0.5B, which does not fit at any quantization above Q2.

The model path is a single constant in one config module, so swapping is a one-line change.

---

## 4. Core Architectural & System Constraints

To prevent OOM kills on Render:

| Constraint | Value |
|---|---|
| Total system memory ceiling | **200 MB** |
| Target baseline (idle, model loaded) | **< 150 MB** — leaving ~50 MB for live KV cache and OS operations |
| Language / runtime | **Rust**, release binary, no heavy runtime |
| Inference engine | **`llama-cpp-2`** — native C++ `llama.cpp` behind safe Rust bindings |
| Model | **SmolLM2-135M-Instruct, Q4_K_M GGUF** |
| Context window | **256 tokens**, hard cap in LLM config |
| Concurrency strategy | Single-threaded token generation (`n_threads = 1`) with an async FIFO request queue on tokio |

### Memory Budget (working numbers)

```
llama.cpp / backend runtime        ~ 15 MB
GGUF weights (SmolLM2-135M Q4_K_M) ~115 MB
KV cache @ n_ctx=256               ~  8 MB
Tokio / axum / SQLite runtime      ~ 10 MB
10 session buffers + HTTP state    ~  5 MB
                                 --------
Total                              ~153 MB   (ceiling: 200 MB)
```

These are estimates until measured. §9 requires the real numbers and a correction to this table.

---

## 5. Functional Requirements

### 5.1 Session & Connection Management
- Accept connections via **WebSockets** (primary) and long-polling HTTP streams (fallback).
- Generate a unique `session_id` on connect (UUIDv4).
- **On disconnect, memory must be reclaimed deterministically.** The `Drop` impl for the session state releases every owned `String` immediately — no reliance on GC, no deferred cleanup.

### 5.2 Context & History Offloading (SQLite)
- Conversation history **must not** live in RAM as `Vec<String>`.
- Single-file or `:memory:` SQLite via `rusqlite` (sync, simpler, and the queries here are trivial).
- **Retain only the last 3 turns (6 messages) per session** to strictly bound token length.
- Delete all rows for a `session_id` on disconnect.

### 5.3 Inference Queue
- Never allow 10 concurrent generations — that would spike RAM immediately.
- FIFO async channel via `tokio::sync::mpsc`. All 10 clients may connect; prompts queue and are served sequentially. At <500M params on one core, a turn completes in a fraction of a second, so perceived latency stays acceptable.
- One model instance, loaded once, shared behind `Arc`. Never reloaded, never cloned per task.

### 5.4 Endpoints
```
WS   /ws/chat                      -- bidirectional chat
POST /chat                         -- single-shot HTTP fallback
GET  /health                       -- 200 with model-loaded status
```

---

## 6. Database Schema

```sql
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,          -- 'user' or 'assistant'
    content TEXT NOT NULL,       -- the text message
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_session ON chat_history(session_id);
```

---

## 7. Non-Functional Requirements & Optimizations

- **Build profile:** `--release`, `opt-level = 3`, `lto = true`, `codegen-units = 1`, `panic = "abort"`, and `strip = true` to keep the shipped binary small.
- **Context window:** `n_ctx = 256` hard cap. This bounds KV cache growth directly — the dominant controllable memory variable after weights.
- **Threads:** `n_threads = 1`. One vCPU; more threads contend and add memory for zero throughput.
- **No GC.** Deterministic drop semantics mean the RSS curve stays flat instead of sawtoothing.
- **Model download at startup:** `hf_hub_download` fetches only the single `.gguf` file — never clone the repo, which would pull every quantization variant. If startup is too slow, swap for a streaming download from a fixed URL.

---

## 8. Production Deployment Plan (Render Free Tier)

### 8.1 Containerization
Multi-stage `Dockerfile`:
- **Stage 1** — full Rust image, builds the release binary with LTO.
- **Stage 2** — `debian:stable-slim`, copies only the stripped binary and the `.gguf`. A Rust build needs no runtime, so the final image is roughly the model's size and nothing else.

### 8.2 Environment Variables
```
MODEL_PATH=./models/smollm2-135m-instruct-q4_k_m.gguf
MAX_CONCURRENT_SESSIONS=10
CONTEXT_SIZE=256
```

### 8.3 Runtime Notes
- **Ephemeral disk:** Render deletes the disk on spin-down. The ~100 MB model re-downloads on every cold start — tolerable at this size, and I can host the GGUF on a GitHub Release and stream it if it isn't.
- **Health checks:** give generous startup grace; the instance is not ready until the model finishes loading.
- **Build time:** LTO + `codegen-units = 1` makes for a slow build. Budget for it in the Render build step; it's a one-time cost per deploy.

---

## 9. Risks & Known Problems

| Risk | Impact | Mitigation |
|---|---|---|
| Resident RAM exceeds the 150 MB estimate | OOM kill under 200 MB | Measure in production; drop `n_ctx` or move to a smaller quant |
| 135M output quality is poor | Weak demo | Upgrade path is 360M Q4_K_M (~250 MB) on a raised budget — **not** 0.5B |
| Queue latency spikes at 10 sessions | Users wait | Acceptable at this model size; bound the queue and return 429 past depth N |
| SQLite file grows on disk | Disk pressure | Last-3-turns retention + purge on disconnect bounds it; consider `:memory:` |
| LTO build exceeds Render's build timeout | Failed deploy | Raise `opt-level`/`lto` tuning, or build outside and ship the binary |
| Model re-downloads on every spin-down | Slow cold starts | Host the GGUF on a GitHub Release and stream it |
| Model unavailable at boot (HF down) | Total outage | Bake the model into the Docker image as a fallback |
| Single-threaded generation | Throughput ceiling | Accepted by design; no GPU on free tier |

---

## 10. Build Order

1. **Prove the memory claim first.** Bare Rust binary, load SmolLM2-135M Q4_K_M, generate a response, print RSS. No server, no database, no WebSockets. If 135M/Q4_K_M doesn't fit under 150 MB, everything downstream is wasted work and the model decision has to change.
2. Wrap it in axum with `/chat` and `/health`.
3. Add SQLite offloading + last-3-turns trimming.
4. Add WebSockets and session lifecycle with `Drop`-based cleanup.
5. Add the FIFO inference queue and 10-session cap.
6. Multi-stage Dockerfile, deploy to Render.
7. Measure real peak RSS and cold-start time; **correct the §4 budget table with actuals.**

---

## 11. Open Questions

- What is the real RSS delta between Q4_K_M and Q8_0 for 135M? If it's small, Q8_0 might be worth taking.
- Does SmolLM2-135M hold multi-turn coherence well enough at 256 tokens, or does the last-3-turns policy need to shrink to 2?
- Should history live in `:memory:` SQLite (dies with the process, zero disk I/O) or a file (survives restart, but it's ephemeral on Render anyway)? `:memory:` looks like the right call — persistence on Render is illusory regardless.
- Is `llama-cpp-2`'s prebuilt binary good enough, or do I need to build `llama.cpp` from source for reproducible memory behavior?
- What's the queue-depth limit before returning 429 instead of accepting and hanging?
- Is WebSocket-only enough, or does the long-polling fallback earn its keep?

---

## 12. Definition of Done

- [ ] SmolLM2-135M Q4_K_M loads and generates within a **200 MB** ceiling, measured in production
- [ ] Idle baseline under **150 MB** with model loaded
- [ ] 10 concurrent WebSocket sessions served via FIFO queue, none rejected
- [ ] History lives in SQLite, never in a `Vec<String>`
- [ ] Disconnect purges DB rows and drops memory deterministically (verified via RSS drop)
- [ ] Container image contains only the stripped binary + GGUF
- [ ] Cold-start time measured and documented
- [ ] §4 memory budget table updated with real numbers
