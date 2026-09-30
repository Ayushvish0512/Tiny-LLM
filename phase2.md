# Tiny-LLM — Phase 2 PRD

## Scaling Micro-Chat on Fixed Hardware

**Status:** Draft
**Parent doc:** `brainstorm.md` (Phase 1)
**Author:** Ayush
**Date:** 30 Sep 2026

---

## 1. The Thesis of Phase 2

Phase 1 assumes **users scale RAM**: 10 sessions, one shared model, done.
Phase 2 inverts this: **RAM is fixed, users scale.**

The single most important reframe in this document:

> **RAM does not scale with users. RAM scales with *slots* (parallel decode positions). Users are served by queueing.**

Every design decision in Phase 2 follows from that. A user connecting does not get a slot. A user connecting joins a *queue* and gets a slot briefly, when one frees up. The number of slots is capped by the KV cache budget — not by the number of users we promise to serve. Users beyond that are not rejected outright; they wait, fairly, and see streaming tokens as soon as they're served.

This is how every real inference server works. It's the difference between "10 users" and "10 users who each feel like they're being served."

---

## 2. Honest Capacity Math

The Phase 1 claim that a turn takes "a fraction of a second" is optimistic and it's the assumption most likely to break Phase 2. Realistic single-core numbers for **SmolLM2-135M Q4_K_M**, `n_threads=1`, on a Render free vCPU:

| Operation | Throughput (est.) | Latency (est.) |
|---|---|---|
| Prefill | ~500–1500 tok/s | 256-token prompt ≈ **0.2–0.5 s** |
| Decode | ~20–40 tok/s | 60-token reply ≈ **1.5–3.0 s** |
| **Full turn** | — | **~2–3.5 s** |

**Global throughput ceiling: ~0.3–0.5 completed turns/second.** That is the number. One core generates one token at a time; no amount of architectural cleverness changes it.

Now the demand side. If N users each send one message every 15 seconds:

| Active users | Demand (turns/s) | Supply (turns/s) | Verdict |
|---|---|---|---|
| 3 | 0.20 | 0.40 | ✅ Comfortable |
| 6 | 0.40 | 0.40 | ⚠️ At the edge, queue starts to build |
| 10 | 0.67 | 0.40 | ❌ **Demand exceeds capacity — queue grows without bound** |

**Conclusion:** Phase 1's "10 simultaneous sessions" is not sustainable with 135M at full reply length. The honest Phase 2 target is **~5–6 sustained concurrent users**, with 10+ *connectable* sessions waiting in a fair queue.

Two levers exist to raise the ceiling without more hardware, and both cost quality:
1. **Cap `max_tokens` to ~48.** Decode dominates turn latency; halving reply length nearly doubles capacity. Free win.
2. **Cap `n_ctx` to 192 and keep 2 turns of history.** Shrinks both prefill and KV cache.

I'd take both. "Scaling on fixed hardware" means serving *more people with worse output*, not serving the same output to more people.

---

## 3. Architecture: Slot-Based Continuous Batching

Phase 1's design — one `Llama` context, prompts serialized through a FIFO — leaves the core idle during prefill and wastes a context per turn. Phase 2 replaces it with **slots inside a single shared KV cache**, the model `llama-server` itself uses.

### 3.1 The Memory Reality (this is the hard constraint)

SmolLM2-135M KV cache, GQA with 3 KV heads, 30 layers, 64-dim heads:

```
per token = 2 × layers(30) × kv_heads(3) × head_dim(64) × 2 bytes ≈ 22.5 KB
```

At `n_ctx = 192`, that's **~4.3 MB per slot.**

| Slots | KV cache | Weights | Runtime | Total | Verdict |
|---|---|---|---|---|---|
| 1 | 4 MB | 115 MB | 25 MB | ~144 MB | Phase 1 |
| 4 | 17 MB | 115 MB | 25 MB | ~157 MB | ✅ Safe |
| 6 | 26 MB | 115 MB | 25 MB | ~166 MB | ✅ Good |
| 8 | 34 MB | 115 MB | 25 MB | ~174 MB | ⚠️ Tight |
| 10 | 43 MB | 115 MB | 25 MB | ~183 MB | ⚠️ No headroom left |

**Phase 2 target: 6 slots.** The 10th concurrent *user* is served by the 7th–∞ queue, not by a 10th slot.

### 3.2 Execution Loop
- One model instance, loaded once, wrapped in `Arc`, never cloned.
- One KV cache, six slot contexts carved out of it.
- A single dedicated inference thread owns the model. It runs a **continuous batching loop**: gather one token-step's worth of work from every slot that has pending work, decode them together in a single batched forward pass, scatter results back to each stream.
- The HTTP/WebSocket layer never touches the model directly. It talks to the scheduler over channels.

Continuous batching is what makes 6 slots meaningfully better than 6 sequential turns: slots sitting idle on prefill don't block slots that are decoding, and the batch amortizes per-token overhead across all of them.

---

## 4. Scheduling & Fairness

A naive FIFO queue is the single worst thing you can ship at Phase 2 scale. One user sending long messages will starve everyone behind them.

### 4.1 Deficit Round Robin over Slots
- Each connected session accumulates **deficit credits** per wall-clock second it waits.
- When a slot frees, it goes to the session with the **highest accumulated deficit**, not the longest queue position.
- This is starvation-free: a waiting session's credit grows monotonically, so it *must* eventually win. FIFO has no such guarantee.

### 4.2 Two Queues, Two Priorities
| Queue | Contents | Behaviour |
|---|---|---|
| **Interactive** | Active WebSocket sessions waiting on a reply | Weighted 4:1 share |
| **Batch** | HTTP single-shot requests, health probes excluded | Gets the remaining 20% |

This stops a firehose of single-shot API calls from starving real interactive users.

### 4.3 Preemption (Turn Yielding)
A slot running a non-streaming or batch request is **preemptible**: it can be checkpointed and evicted mid-generation to serve an interactive user, then resumed later. Interactive slots are never preempted. Without this, one slow HTTP request can pin a slot for seconds.

### 4.4 Session Affinity
Pin each `session_id` to a specific slot index for its lifetime. Benefits:
- The system prompt and conversation prefix stays in that slot's KV cache — **prompt caching makes turn 2+ nearly free on prefill.**
- Session history stays warm in the same memory region.
- Requires that affinity is never violated, so slot assignment happens once at session creation and the slot's KV prefix is never overwritten by another session's data.

---

## 5. Admission Control & Backpressure

Connection accepted ≠ work accepted. These are separate decisions.

### 5.1 Bounded Queue, Hard Ceiling
- `mpsc` channel capacity **32**. Beyond that, refuse with `429 Too Many Requests` + `Retry-After`.
- An unbounded queue is a memory leak with extra steps — every queued prompt is RAM you no longer have.

### 5.2 Per-Session Rate Limiting
Token bucket per `session_id`: **2 prompts/min sustained, burst 4.** One client cannot occupy a slot by talking to itself. Buckets live in the session map and are dropped with the session.

### 5.3 Slow Consumer Handling
- Per-connection outbound buffer, capped at **64 KB**.
- A WebSocket client that stops reading hits the cap: send `Close(1013 Try Again Later)` and reclaim. Never buffer unboundedly on behalf of a dead client.

### 5.4 Hard Memory Guards
| Guard | Limit | Action on breach |
|---|---|---|
| Max prompt length | 4 KB | `413 Payload Too Large` |
| Max reply length | `max_tokens = 48` | hard truncation |
| History depth | 2 turns (4 messages) | trim oldest before prefill |
| Per-session resident ceiling | ~2 MB | evict from LRU, push history to SQLite |
| **Process RSS watchdog** | 185 MB | shed slots, then shed batch queue, log loudly |

The RSS watchdog is the backstop for the whole design. It runs on a timer and degrades the system rather than letting the OOM killer pick the victim.

---

## 6. Session Lifecycle

```
        connect
           │
           ▼
     ┌───────────┐   limit exceeded    ┌──────────────┐
     │CONNECTING │───────────────────▶│  REJECTED    │ 429 / close
     └─────┬─────┘                    └──────────────┘
           │ accepted
           ▼
     ┌───────────┐  slot free & DRR wins
     │  QUEUED   │───────────────────────┐
     └─────┬─────┘                       │
           │ admitted                    ▼
           │                       ┌──────────────┐
           │                       │  GENERATING  │◀──┐
           │                       └──────┬───────┘   │
           │                              │ tokens    │
           │                              ▼           │
           │                       ┌──────────────┐   │
           │        turn complete   │ STREAMING    │───┘
           │                       └──────┬───────┘
           │                              │ yield slot
           │                              ▼
           │                       ┌──────────────┐
           │                       │   COOLDOWN   │──→ next turn
           │                       └──────────────┘
           │
           │ idle > 60s / client close / error
           ▼
     ┌───────────┐   Drop: free strings, purge SQLite rows,
     │ CLOSING   │   release slot affinity, drop rate bucket
     └───────────┘
```

**Critical states:**
- **COOLDOWN** — the slot is released *here*, not after the full turn. Turn the client off-stream and immediately make the slot available to the scheduler. This is what turns "sequential turns" into "real concurrency."
- **CLOSING** — `Drop` must fire deterministically. This is the Rust guarantee Phase 1 was written around; Phase 2 leans on it harder.

### 6.1 Idle Reaping
- No message for 60s → session state dropped, DB rows purged.
- Sweeper runs every 15s over the session map.
- **This is the load-bearing feature at scale.** Without it, sessions accumulate until the OOM killer arrives. With it, memory tracks *active* users, not *total* users.

---

## 7. Streaming Is Now Mandatory

Phase 1 listed SSE streaming as an open question. **In Phase 2 it is not optional.**

With a 2–3.5s turn time and a queue ahead of you, a non-streaming client sees nothing for 5+ seconds and assumes the server is dead. Streaming tokens makes queue-wait visible and keeps people in the conversation.

- WebSocket: token-by-token frames.
- HTTP fallback: `text/event-stream`.
- **First token latency is the metric that matters**, not total latency. A slot that prefills and immediately flushes a token feels responsive even at the same total duration.

---

## 8. Degradation Ladder

Rather than fail all-or-nothing, shed load in priority order as the system saturates. Each rung is triggered by queue depth or RSS.

| Rung | Trigger | Action |
|---|---|---|
| **0 — Full** | normal | 6 slots, 4:1 interactive/batch, 2 turns history |
| **1 — Busy** | queue > 8 | Drop batch queue to 10% share; tighten `max_tokens` to 32 |
| **2 — Saturated** | queue > 16 | Refuse new batch with 429; history → 1 turn |
| **3 — Critical** | RSS > 180 MB | Evict to 4 slots; reject all new connects |
| **4 — Emergency** | RSS > 190 MB | Shed to 1 slot, serve only existing interactive sessions |

Explicit degradation is a better product than random OOM kills. It also makes load-testing interpretable.

---

## 9. Observability

Metrics that must be exported (Prometheus text format is enough):
- `queue_depth`, `queue_wait_seconds` (p50/p95/p99)
- `slots_active` / `slots_total` — the real concurrency figure
- `turns_completed_total`, `turn_latency_seconds`, `ttft_seconds` (time to first token)
- `rss_bytes`, `kv_cache_bytes`
- `rejected_total{reason}` — 429/413/1013 breakdown
- `sessions_active`, `sessions_reaped_total`

**Load test before declaring Phase 2 done:** drive 25 concurrent synthetic users at Phase 1's think-time (15s) and confirm the system finds a steady state with bounded queue depth. If the queue grows linearly, capacity math in §2 is wrong and the model or reply cap has to move.

---

## 10. What I'd Do Differently

**The strongest available option is to not build the scheduler myself.** `llama-server` — the binary that ships with `llama.cpp` — already implements slot-based continuous batching, prompt caching, slot preemption, `/slots` endpoints, and per-request `n_predict` caps. It's the same C++ engine, exposed over HTTP.

What it does *not* give me is the parts that matter for this product: the SQLite session store, the DRR fairness layer, admission control, rate limiting, the degradation ladder, and the Rust memory guarantees.

So the pragmatic Phase 2 build is: **Rust app owns sessions, DB, admission, and fairness; `llama-server` owns slots and batching as a local subprocess.** Far less code, far fewer concurrency bugs, same inference core. If writing my own batching loop becomes a multi-week rabbit hole, this is the fallback and I should start here.

**Second thing I'd change:** I'd drop the 10-user target from the public spec entirely. Advertise 6. A demo that works with 6 real users is worth more than one that OOMs in front of 10.

---

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Actual capacity < §2 estimate | Fewer usable users than planned | Cap replies at 48 tokens; measure and re-plan before promising anything |
| Continuous batching implementation is complex | Weeks of work, subtle bugs | Default to `llama-server` as the slot engine |
| KV cache prefix corruption across sessions | Users see each other's context | Strict slot affinity; test with cross-session leak assertions |
| Idle reaper bug leaks session state | Slow OOM | RSS watchdog + long-running leak test |
| Render free tier spins down under load | Apparent outages | Keep-warm pings from a single client during demos |
| Queue wait grows unboundedly under burst | Memory + timeouts | Bounded queue, 429 with `Retry-After`, degradation ladder |
| Rate limiter too aggressive | Real users get locked out | Token bucket, tune against the load test in §9 |

---

## 12. Build Order

1. **Measure.** Standalone binary: single-slot generation, print RSS and tok/s. Establish the *real* §2 numbers before building anything else.
2. **Tune the caps.** Sweep `n_ctx` (128/192/256) × `max_tokens` (32/48/64). Pick the point that maximizes turns/second while staying coherent.
3. **Decide: own batching loop or `llama-server`.** Prototype continuous batching; if it isn't working in a few days, switch to the subprocess.
4. **Scheduler core.** Slot pool + DRR + session affinity + COOLDOWN slot release. This is the heart of Phase 2.
5. **Admission control.** Bounded queue, 429 + `Retry-After`, per-session token buckets, slow-consumer close.
6. **Session lifecycle.** State machine, idle reaper, `Drop`-based cleanup, SQLite purge.
7. **Streaming.** WS token frames + SSE fallback. Measure TTFT.
8. **Degradation ladder + RSS watchdog.**
9. **Metrics endpoint.**
10. **Load test.** 25 concurrent users, 15s think time, confirm steady state. Update §2 with measured numbers.

---

## 13. Definition of Done

- [ ] **6 slots** of continuous batching running in one process
- [ ] Peak RSS **< 200 MB** under the 25-user load test
- [ ] 25 concurrent clients served with **bounded** queue depth
- [ ] No session sees another session's context (verified by assertion test)
- [ ] DRR fairness confirmed — no session starved over a 30-min run
- [ ] Idle reaper holds RSS flat across a session-churn test
- [ ] TTFT p95 under 3s at target load
- [ ] 429s carry a correct `Retry-After`; no unbounded queue growth
- [ ] Degradation ladder triggers correctly under RSS pressure
- [ ] §2 capacity table replaced with **measured** numbers
