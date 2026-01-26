import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    df = pd.read_csv("data/processed/data_clean.csv").sample(10000)

    tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model")
    model = DistilBertForSequenceClassification.from_pretrained("bert_model")
    model.to(DEVICE)
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for _, row in df.iterrows():
            enc = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(DEVICE)

            output = model(**enc)
            pred = torch.argmax(output.logits, dim=1).item()

            preds.append(pred)
            labels.append(row["label"])

    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
