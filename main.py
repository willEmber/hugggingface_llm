import os
import sys
import time
import uuid
import logging
import traceback
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. Logging and Global Configuration ---

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# This will now load API_SECRET_KEY AND HF_ENDPOINT from your .env file
load_dotenv()

# We point to the model ID. The HF_ENDPOINT env var will redirect the download.
MODEL_PATH = "ByteDance-Seed/Seed-X-PPO-7B"
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "default-secret-key")
if API_SECRET_KEY == "default-secret-key":
    logger.warning("Using default API key. Set API_SECRET_KEY in .env for security.")

model = None
tokenizer = None

# --- 2. Model Loading Logic ---

def load_model():
    global model, tokenizer
    try:
        logger.info(f"Starting model loading for: {MODEL_PATH}")
        # The library automatically uses the HF_ENDPOINT environment variable.
        logger.info(f"Using Hugging Face endpoint: {os.getenv('HF_ENDPOINT', 'https://huggingface.co')}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully.")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        logger.info("High-precision model loaded successfully.")
    except Exception as e:
        logger.critical(f"Fatal error during model loading: {e}")
        logger.critical(traceback.format_exc())
        sys.exit("Application shutdown due to model loading failure. Check logs/app.log.")

# --- 3. FastAPI Application Setup (with Lifespan) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup events. Model is loaded on startup."""
    logger.info("Application lifespan: startup sequence initiated.")
    load_model()
    yield
    logger.info("Application lifespan: shutdown sequence initiated.")

app = FastAPI(lifespan=lifespan)

async def verify_key(authorization: str = Header(...)):
    if authorization != f"Bearer {API_SECRET_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# --- 4. API Data Structures (Unchanged) ---
class Message(BaseModel):
    role: str; content: str
class ChatCompletionRequest(BaseModel):
    model: str; messages: List[Message]; temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0; max_tokens: Optional[int] = 512; stream: Optional[bool] = False
class Choice(BaseModel):
    index: int; message: Message; finish_reason: str = "stop"
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}"); object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[Choice]

# --- 5. API Endpoints (Unchanged, with added logging) ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, authorized: bool = Depends(verify_key)):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported.")
    user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found.")
    
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{request_id}] Received translation request.")
    try:
        prompt = f'Translate the following English text to Chinese.\nEnglish: "{user_message}"\nChinese:'
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=request.max_tokens, temperature=request.temperature,
                top_p=request.top_p, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, num_return_sequences=1
            )
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        logger.info(f"[{request_id}] Translation successful.")
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(index=0, message=Message(role="assistant", content=response_text))]
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error during request processing: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

# --- 6. Application Runner (Unchanged) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=58000)