import pandas as pd
import re

def clean_text(text):
  text = text.lower()
  text = re.sub(r"http\S+", "", text)
  text = re.sub(r"<.*?>", "", text)
  text = re.sub(r"[^a-z\s]", "", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text

def main():
  df = pd.read_csv("data/raw/data.csv")

  df["text"] = df["text"].astype(str).apply(clean_text)
  df = df.dropna()

  df.to_csv("data/processed/data_clean.csv", index=False)
  print("Preprocessing completed.")

if __name__ == "__main__":
  main()
