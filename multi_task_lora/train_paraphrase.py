import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def preprocess(batch, tokenizer, max_length=128):
    return tokenizer(
        batch["question1"],
        batch["question2"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "qqp")

    encoded = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir="outputs/paraphrase",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("adapters/paraphrase")
    tokenizer.save_pretrained("adapters/paraphrase")

if __name__ == "__main__":
    main()

