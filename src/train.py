import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from evaluate import load
import torch
from codecarbon import OfflineEmissionsTracker
from src.data import load_and_preprocess_glue, preprocess_function
from src.adversarial import generate_adversarial_examples, create_mixed_dataset
from src.model import load_baseline_model

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("energy-efficient-distilbert-sst2")

accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

train_ds, test_ds = load_and_preprocess_glue()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_train = train_ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)
tokenized_test  = test_ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
    report_to="none",
)

with mlflow.start_run(run_name="baseline_finetune"):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    tracker = OfflineEmissionsTracker(project_name="baseline_train", log_level="error")
    tracker.start()
    trainer.train()
    co2_kg = tracker.stop()
    
    eval_results = trainer.evaluate()
    mlflow.log_params({
        "epochs": 3,
        "batch_size": 16,
        "task": "sst2",
        "model": "distilbert-base-uncased"
    })
    mlflow.log_metric("final_accuracy", eval_results["eval_accuracy"])
    mlflow.log_metric("co2_kg", co2_kg)
    mlflow.log_metric("approx_energy_kwh", co2_kg / 0.35)
    
    trainer.save_model("./distilbert-finetuned-sst2")
    tokenizer.save_pretrained("./distilbert-finetuned-sst2")

# Adversarial training
print("Generating adversarial examples...")
adv_ds = generate_adversarial_examples(model, tokenizer, train_ds, num_examples=800)
mixed_ds = create_mixed_dataset(train_ds, adv_ds, adv_ratio=0.4)
tokenized_mixed = mixed_ds.map(lambda ex: preprocess_function(ex, tokenizer), batched=True)

with mlflow.start_run(run_name="adversarial_finetune"):
    training_args.num_train_epochs = 2
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_mixed,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    tracker = OfflineEmissionsTracker(project_name="adv_train", log_level="error")
    tracker.start()
    trainer.train()
    co2_kg_adv = tracker.stop()
    
    eval_results_adv = trainer.evaluate()
    mlflow.log_metric("final_accuracy_adv", eval_results_adv["eval_accuracy"])
    mlflow.log_metric("co2_kg_adv", co2_kg_adv)
    mlflow.log_metric("approx_energy_kwh_adv", co2_kg_adv / 0.35)
    
    trainer.save_model("./distilbert-robust-sst2")
    tokenizer.save_pretrained("./distilbert-robust-sst2")

print("Training completed. Models saved to:")
print("  - Baseline:   ./distilbert-finetuned-sst2")
print("  - Robust:     ./distilbert-robust-sst2")

