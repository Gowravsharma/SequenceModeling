from datasets import load_dataset
from adapter_loader import load_model_with_adapter
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_sentiment():
    base = "bert-base-uncased"
    adapter = "adapters/sentiment"

    model, tokenizer = load_model_with_adapter(base, adapter)
    model.to(device)

    ds = load_dataset("glue", "sst2")["validation"]

    correct = 0
    total = 0

    for item in ds:
        inputs = tokenizer(
            item["sentence"],
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = logits.argmax(dim=-1).item()

        if pred == item["label"]:
            correct += 1
        total += 1

    print(f"SST-2 Sentiment Accuracy: {correct/total:.4f}")


def evaluate_paraphrase():
    base = "bert-base-uncased"
    adapter = "adapters/paraphrase"

    model, tokenizer = load_model_with_adapter(base, adapter)
    model.to(device)

    ds = load_dataset("glue", "qqp")["validation"]

    correct = 0
    total = 0

    for item in ds:
        inputs = tokenizer(
            item["question1"],
            item["question2"],
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = logits.argmax(dim=-1).item()

        if pred == item["label"]:
            correct += 1
        total += 1

    print(f"QQP Paraphrase Accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    evaluate_sentiment()
    evaluate_paraphrase()
