import torch
from adapter_loader import load_model_with_adapter, route_adapter

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(text):
  # Pick the correct skill adapter
  adapter_path = route_adapter(text)
  print(f"Using adapter: {adapter_path}")

  # Load model + tokenizer
  model, tokenizer = load_model_with_adapter("bert-base-uncased", adapter_path)
  model.to(device)
  model.eval()

  # Encode input
  inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
  ).to(device)

  with torch.no_grad():
    logits = model(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()

  # Convert prediction to label
  label = model.config.id2label[pred_id]
  return label


if __name__ == "__main__":
  while True:
    text = input("\nEnter text (or 'exit'): ")
    if text.lower() == "exit":
      break
    print("Prediction:", predict(text))
