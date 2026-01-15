from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model_with_adapter(base_model, adapter_path):
    config = PeftConfig.from_pretrained(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=config.num_labels if hasattr(config, "num_labels") else 2
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def route_adapter(prompt: str):
    prompt = prompt.lower()

    if "similar" in prompt or "paraphrase" in prompt:
        return "adapters/paraphrase"
    else:
        return "adapters/sentiment"
