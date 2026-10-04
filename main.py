import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import AsyncGenerator

from model import load_model, model_status
from generate import build_prompt, clean_chunk

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["file"] = model_status()
    try:
        state["llm"] = load_model()
        state["status"] = "loaded"
    except Exception as e:
        state["llm"] = None
        state["status"] = f"failed: {e}"
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str = Field(..., max_length=500)
    max_tokens: int = Field(64, ge=1, le=256)

@app.get("/health")
def health():
    return {"status": "running", "model": state.get("status", "loading"), "file": state.get("file", "unknown")}

@app.post("/chat")
async def chat(req: ChatRequest):
    llm = state.get("llm")
    if not llm:
        raise HTTPException(503, "Model not loaded")

    full_prompt = build_prompt(req.prompt)

    def stream_gen() -> AsyncGenerator[str, None]:
        try:
            for chunk in llm.create_completion(
                prompt=full_prompt,
                max_tokens=req.max_tokens,
                stream=True,
                temperature=0.7,
                stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
            ):
                token = chunk["choices"][0]["text"]
                if token:
                    yield clean_chunk(token)
        except Exception as e:
            yield f"\n[error] {e}"

    return StreamingResponse(stream_gen(), media_type="text/plain; charset=utf-8")