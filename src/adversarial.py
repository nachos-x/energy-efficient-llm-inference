import textattack
from textattack import Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from datasets import Dataset, concatenate_datasets, Features, Value, ClassLabel

def generate_adversarial_examples(model, tokenizer, dataset, num_examples=800):
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    
    # Strong attack: TextFoolerJin2019
    attack = TextFoolerJin2019.build(model_wrapper)
    
    # Relax USE threshold for more successes
    for constraint in attack.constraints:
        if isinstance(constraint, UniversalSentenceEncoder):
            constraint.threshold = 0.70
            print("Relaxed USE similarity threshold to 0.70 for higher success rate")
    
    subset = dataset.shuffle(seed=42).select(range(min(num_examples, len(dataset))))
    hf_dataset = HuggingFaceDataset(subset, split="train")
    
    attacker = Attacker(attack, hf_dataset)
    results = attacker.attack_dataset()
    
    adv_texts = []
    adv_labels = []
    success_count = 0
    
    for i, result in enumerate(results):
        # Correct success check for modern TextAttack
        if isinstance(result, SuccessfulAttackResult) and result.perturbed_text() != result.original_text():
            adv_texts.append(result.perturbed_text())
            adv_labels.append(result.original_result.ground_truth_output)
            success_count += 1
            if success_count % 20 == 0:
                print(f"Success {success_count}: {result.perturbed_text()[:60]}...")
    
    print(f"\nGenerated {len(adv_texts)} successful adversarial examples")
    
    features = Features({
        "sentence": Value("string"),
        "label": ClassLabel(names=["negative", "positive"])
    })
    
    return Dataset.from_dict({"sentence": adv_texts, "label": adv_labels}, features=features)


def create_mixed_dataset(original_ds, adv_ds, adv_ratio=0.4):
    if len(adv_ds) == 0:
        print("No adversarial examples generated → using original dataset only")
        return original_ds
    
    num_adv = int(len(original_ds) * adv_ratio)
    original_subset = original_ds.select(range(len(original_ds) - num_adv))
    
    mixed = concatenate_datasets([original_subset, adv_ds])
    mixed = mixed.shuffle(seed=42)
    
    print(f"Mixed dataset size: {len(mixed)} (original: {len(original_subset)}, adv: {len(adv_ds)})")
    return mixed
