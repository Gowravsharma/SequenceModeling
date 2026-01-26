from datasets import load_dataset
import pandas as pd

dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

df = pd.concat([train_df, test_df])
df = df.rename(columns={"text": "text", "label": "label"})

df.to_csv("data/raw/data.csv", index=False)
