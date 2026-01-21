import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional
import uvicorn

app = FastAPI(title="Qalb Model API")

MODEL_PATH = "./models/Qalb-1.0-8B-Instruct"

# Global variables to store model and tokenizer
model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "آپ ایک مددگار اور بے ضرر مصنوعی ذہانت کے اسسٹنٹ ہیں۔"
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9

def load_model():
    global model, tokenizer
    print(f"Loading model from {MODEL_PATH}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/generate")
async def generate(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{request.system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{request.prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p
        )

    input_length = inputs.input_ids.shape[-1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return {"response": response.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
