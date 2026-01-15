# 📘 Multi-Task LoRA Adapters for BERT

This project demonstrates how to train **multiple LoRA adapters** on different NLP tasks using a shared **BERT-base** model.  
Two separate adapters are trained:

- **Sentiment Analysis** (SST-2 – GLUE Benchmark)  
- **Paraphrase Detection** (QQP – GLUE Benchmark)

A simple **routing mechanism** selects the correct adapter at inference time, enabling modular skill composition.

---

## 🚀 Features
- Fine-tune **task-specific LoRA adapters** using the PEFT library  
- Modular loading of adapters on top of a shared backbone  
- Dynamic routing based on input query  
- Evaluation scripts for both tasks  
- Clean inference API (`predict(text)`)

---

## 📁 Project Structure
```
multi-task-lora/
│── train_sentiment.py
│── train_paraphrase.py
│── adapter_loader.py
│── inference.py
│── evaluate.py
│── adapters/
│    ├── sentiment/
│    └── paraphrase/
│── requirements.txt
│── README.md
│── __init__.py
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers datasets peft accelerate bitsandbytes
```

---

## 🏋️‍♂️ Training

### 1. Train Sentiment LoRA (SST-2)

```bash
python train_sentiment.py
```

Outputs LoRA weights in:

```
adapters/sentiment/
```

### 2. Train Paraphrase LoRA (QQP)

```bash
python train_paraphrase.py
```

Outputs:

```
adapters/paraphrase/
```

---

## 🔧 Inference

Use the unified `predict()` method:

```python
from inference import predict

print(predict("This movie was awesome!"))
print(predict("Are these two questions asking the same thing?"))
```

Example output:

```
Using adapter: adapters/sentiment
POSITIVE

Using adapter: adapters/paraphrase
duplicate
```

---

## 🔀 Dynamic Adapter Routing

```python
def route_adapter(prompt):
    if "similar" in prompt or "paraphrase" in prompt:
        return "adapters/paraphrase"
    return "adapters/sentiment"
```

---

## 📊 Evaluation

```bash
python evaluate.py
```

Output example:

```
SST-2 Accuracy: 0.92
QQP Accuracy: 0.88
```

---

## 🧠 Concepts Demonstrated
- **PEFT (LoRA, QLoRA-ready)**  
- **Multi-task adapter training**  
- **Modular skill composition**  
- **Efficient fine-tuning of Transformer models**  
- **Dynamic task routing**  

---

## 📜 License
MIT License.
