"""
FastAPI inference API for TinyLlama fine-tuned model.
Run: uvicorn serve.inference_fastapi:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../train/checkpoints')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7

@app.post("/generate")
def generate(request: InferenceRequest):
    output = pipe(request.prompt, max_new_tokens=request.max_new_tokens, temperature=request.temperature, do_sample=True)
    return {"prompt": request.prompt, "generated": output[0]["generated_text"]}
