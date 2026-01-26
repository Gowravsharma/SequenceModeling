import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_len=128):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    enc = self.tokenizer(
      self.texts[idx],
      truncation=True,
      padding="max_length",
      max_length=self.max_len,
      return_tensors="pt"
    )
    return {
      "input_ids": enc["input_ids"].squeeze(),
      "attention_mask": enc["attention_mask"].squeeze(),
      "labels": torch.tensor(self.labels[idx])
    }

def main():
  df = pd.read_csv("data/processed/data_clean.csv")

  X_train, X_val, y_train, y_val = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
  )

  tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
  model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(set(y_train))
  )
  model.to(DEVICE)

  train_ds = TextDataset(X_train, y_train, tokenizer)
  val_ds = TextDataset(X_val, y_val, tokenizer)

  train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=16)

  optimizer = AdamW(model.parameters(), lr=2e-5)

  for epoch in range(3):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
      optimizer.zero_grad()

      batch = {k: v.to(DEVICE) for k, v in batch.items()}
      outputs = model(**batch)

      loss = outputs.loss
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

  model.save_pretrained("bert_model")
  tokenizer.save_pretrained("bert_model")

if __name__ == "__main__":
  main()
