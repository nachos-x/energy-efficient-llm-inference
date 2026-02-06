from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from src.model import load_quantized_bitsandbytes, create_pipeline

app = FastAPI(title="Energy-Efficient DistilBERT Inference")

# Basic API key security
API_KEY = "test123"  
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

class PredictRequest(BaseModel):
    text: str

# Load quantized robust model at startup
model, tokenizer = load_quantized_bitsandbytes("./distilbert-robust-sst2")
pipe = create_pipeline(model, tokenizer)

@app.post("/predict", dependencies=[Depends(get_api_key)])
async def predict(request: PredictRequest):
    result = pipe(request.text)[0]
    return {
        "label": result["label"],
        "score": float(result["score"]),
        "model": "distilbert-8bit-quantized-robust"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}
