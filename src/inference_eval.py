import sys
import os

# Fix import path for Colab
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
  
import time
import torch
import mlflow
from codecarbon import OfflineEmissionsTracker
from transformers import AutoTokenizer
from src.data import load_and_preprocess_glue, preprocess_function
from src.model import load_quantized_bitsandbytes, create_pipeline, load_baseline_model



# Load test set (500 samples from GLUE validation)
_, test_ds = load_and_preprocess_glue()

model_dir = "./distilbert-robust-sst2"  # use robust model

# Baseline model
baseline_model, tokenizer = load_baseline_model(model_dir)
baseline_pipe = create_pipeline(baseline_model, tokenizer)

# Quantized model (8-bit by default – see note below for 4-bit)
quant_model, _ = load_quantized_bitsandbytes(model_dir)
quant_pipe = create_pipeline(quant_model, tokenizer)

def measure_latency(model, tokenizer, texts, runs=50, batch_size=128):
    """
    Manual batched forward pass – most accurate for throughput comparison
    """
    model.eval()
    device = next(model.parameters()).device
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            with torch.no_grad():
                _ = model(**inputs)
        times.append(time.perf_counter() - start)
    avg_time_total = sum(times) / runs
    return avg_time_total / len(texts)  # seconds per sample

# Use a large sample for stable measurement
sample_texts = list(test_ds["sentence"][:400])  # or full len(test_ds) if you want max accuracy

baseline_lat = measure_latency(baseline_model, tokenizer, sample_texts)
quant_lat    = measure_latency(quant_model, tokenizer, sample_texts)

print(f"Baseline latency: {baseline_lat:.4f} s/sample | 8-bit: {quant_lat:.4f} s/sample")
print(f"Latency reduction: {((baseline_lat - quant_lat) / baseline_lat) * 100:.1f}%")


num_loops = 100
batch_size_energy = 64

device_baseline = next(baseline_model.parameters()).device
device_quant = next(quant_model.parameters()).device

# Baseline energy
tracker = OfflineEmissionsTracker(project_name="baseline_inference")
tracker.start()
for _ in range(num_loops):
    inputs = tokenizer(
        sample_texts[:batch_size_energy],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device_baseline)
    with torch.no_grad():
        _ = baseline_model(**inputs)
baseline_emissions = tracker.stop()

# Quantized energy
tracker = OfflineEmissionsTracker(project_name="quantized_inference")
tracker.start()
for _ in range(num_loops):
    inputs = tokenizer(
        sample_texts[:batch_size_energy],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device_quant)
    with torch.no_grad():
        _ = quant_model(**inputs)
emissions = tracker.stop()

print(f"\nBaseline energy: {baseline_emissions:.6f} kg CO₂eq")
print(f"Quantized energy: {emissions:.6f} kg CO₂eq")
if baseline_emissions > 0:
    reduction = ((baseline_emissions - emissions) / baseline_emissions) * 100
    print(f"Energy reduction: {reduction:.1f}%")

print("\nEvaluation finished.")
