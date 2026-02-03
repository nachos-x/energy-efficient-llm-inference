# Energy-Efficient LLM Inference Pipeline

A production-ready inference pipeline for DistilBERT on the GLUE SST-2 sentiment classification task, optimized for low latency and energy consumption.

Key Results
- Latency reduction: 74% (from 1.3 ms → 0.3 ms per sample using 8-bit quantization)
- Energy reduction: 65% (measured via CodeCarbon on NVIDIA T4 GPU)
- Validation accuracy: ~84–86% after fine-tuning on only 1,000 training samples
- Quantized inference remains highly accurate while being dramatically more efficient

Features
- 8-bit quantization using BitsAndBytes (load_in_8bit)
- Fine-tuning on 1,000 GLUE SST-2 samples (PyTorch + Hugging Face Trainer)
- Optional adversarial robustness training with TextAttack (TextFoolerJin2019)
- Production-grade API with FastAPI + API key authentication
- Containerized with Docker
- Comprehensive tracking of latency, energy, and CO₂ emissions using MLflow & CodeCarbon

Tech Stack
- Python 3.10+
- PyTorch & Hugging Face Transformers
- BitsAndBytes (quantization)
- TextAttack (adversarial examples)
- FastAPI + Uvicorn
- Docker
- MLflow & CodeCarbon (metrics & emissions tracking)




