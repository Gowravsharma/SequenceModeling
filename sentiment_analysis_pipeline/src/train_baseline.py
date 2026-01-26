import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def main():
  df = pd.read_csv("data/processed/data_clean.csv")

  X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
  )

  vectorizer = TfidfVectorizer(max_features=10000)
  X_train_vec = vectorizer.fit_transform(X_train)
  X_test_vec = vectorizer.transform(X_test)

  model = LogisticRegression(max_iter=1000)
  model.fit(X_train_vec, y_train)

  preds = model.predict(X_test_vec)
  print(classification_report(y_test, preds))

  joblib.dump(model, "baseline_model.pkl")
  joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

if __name__ == "__main__":
  main()
