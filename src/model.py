from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

def load_baseline_model(model_dir="distilbert-finetuned-sst2"):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def load_quantized_bitsandbytes(model_dir):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                   
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",           
        bnb_4bit_use_double_quant=True,      
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def create_pipeline(model, tokenizer):
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=32,
        truncation=True,
        max_length=128,
    )
