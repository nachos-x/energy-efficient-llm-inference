from datasets import load_dataset

def load_and_preprocess_glue():
    dataset = load_dataset("glue", "sst2")
    train_ds = dataset["train"].shuffle(seed=42).select(range(1000))
    test_ds = dataset["validation"].select(range(500))
    return train_ds, test_ds

def preprocess_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["sentence"], truncation=True, padding="max_length", max_length=max_length
    )
