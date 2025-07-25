import os
import sys
import time
import uuid
import logging
import traceback
from dotenv import load_dotenv
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. Logging and Global Configuration ---

def setup_logging():
    """Configures the logging for the application."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger to write to a file and the console
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Load environment variables and global settings
load_dotenv()
MODEL_PATH = "ByteDance-Seed/Seed-X-PPO-7B"
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "default-secret-key")
if API_SECRET_KEY == "default-secret-key":
    logger.warning("You are using the default API key. Please set API_SECRET_KEY in your .env file for security.")

# Global variables for the model and tokenizer
model = None
tokenizer = None

# --- 2. Model Loading Logic ---

def load_model():
    """Loads the model and tokenizer at application startup."""
    global model, tokenizer
    try:
        logger.info(f"Starting model loading process for: {MODEL_PATH}")
        logger.info("This may take several minutes and a significant amount of VRAM...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=False,
            trust_remote_code=True
        )
        logger.info("Tokenizer loaded successfully.")

        # Load model with bfloat16 for high precision on compatible GPUs
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set the model to evaluation mode
        model.eval()
        
        logger.info("High-precision model loaded successfully and moved to device.")
        logger.info("The service is ready to accept requests.")

    except Exception as e:
        logger.critical(f"Fatal error during model loading: {e}")
        logger.critical(traceback.format_exc())
        sys.exit("Application shutdown due to model loading failure. Check logs/app.log for details.")


# --- 3. FastAPI Application Setup ---

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """FastAPI startup event handler."""
    load_model()

async def verify_key(authorization: str = Header(..., description="Bearer token for authentication")):
    """Dependency to verify the API key in the Authorization header."""
    if authorization != f"Bearer {API_SECRET_KEY}":
        logger.warning(f"Failed authentication attempt with token: {authorization}")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# --- 4. API Data Structures (OpenAI compatible) ---
# (This section remains unchanged)
class Message(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]


# --- 5. API Endpoints ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, authorized: bool = Depends(verify_key)):
    """Handles chat completion requests with error handling."""
    if request.stream:
        raise HTTPException(status_code=400, detail="This server does not support streaming responses.")

    user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    # A unique ID for tracing this specific request in logs
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{request_id}] Received translation request.")

    try:
        prompt = f"""Translate the following English text to Chinese.
English: "{user_message}"
Chinese:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        logger.info(f"[{request_id}] Successfully generated translation.")
        
        response_message = Message(role="assistant", content=response_text)
        choice = Choice(index=0, message=response_message)
        return ChatCompletionResponse(model=request.model, choices=[choice])

    except Exception as e:
        logger.error(f"[{request_id}] An error occurred during request processing: {e}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        # Return a generic 500 error to the client
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.get("/health")
async def health_check():
    """Health check endpoint to confirm service is running."""
    return {"status": "ok", "model_loaded": model is not None and tokenizer is not None}

# --- 6. Application Runner ---

if __name__ == "__main__":
    # The 'reload=True' flag is great for development but should be False in production.
    # Uvicorn will automatically run the 'startup_event' when it starts.
    uvicorn.run(app, host="0.0.0.0", port=8000)