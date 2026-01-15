import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def preprocess(batch, tokenizer, max_length=128):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "sst2")

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
        output_dir="outputs/sentiment",
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

    model.save_pretrained("adapters/sentiment")
    tokenizer.save_pretrained("adapters/sentiment")

if __name__ == "__main__":
    main()

#{'loss': 0.1686, 'grad_norm': 2.301307201385498, 'learning_rate': 4.750593824228029e-07, 'epoch': 2.0}
#{'eval_loss': 0.2607646584510803, 'eval_runtime': 7.1553, 'eval_samples_per_second': 121.869, 'eval_steps_per_second': 7.687, 'epoch': 2.0}
#{'train_runtime': 2334.9933, 'train_samples_per_second': 57.687, 'train_steps_per_second': 3.606, 'train_loss': 0.23441953930888776, 'epoch': 2.0}