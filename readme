# 🔥 FireBERT — Fine-Tuned FinBERT for Financial News Classification

[![HuggingFace Model](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/yoganfire/FireBerth)

FireBERT is a **domain-specific financial text classifier** built by fine-tuning **FinBERT** (a BERT-based model trained on financial data) to categorize news headlines into **three focused financial domains**.

---

## 📌 Project Overview
FinBERT is originally trained for **sentiment classification** (negative, neutral, positive) using financial corpora — meaning it already understands industry-specific terms like *profit*, *revenue*, *P/E ratio*, etc.

We fine-tuned it for **topic classification** in financial news headlines:

| Label | Category | Description |
|-------|----------|-------------|
| **0** | Macro    | Government policies, inflation, GDP, RBI, trade, etc. |
| **1** | Stock    | Company-specific news (profits, expansions, deals). |
| **2** | Market   | General market trends, indices, volatility, global cues. |

---

## 🎯 Why Only Three Categories?
- **FinBERT Original Task** → 3-class sentiment classification  
- **Minimal Viable Classification** → Covers majority of impactful financial news  
- **Accuracy vs Complexity Trade-off** → More categories require more data & reduce accuracy  
- **Industry Comparison**:  
  - Bloomberg Terminal → ~200 news categories  
  - Reuters News Codes → ~1,500 categories  
  - **FireBERT** → 3 well-defined categories for better performance & simpler deployment  

---

## 🚀 Quick Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "yoganfire/FireBerth"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    label = outputs.logits.argmax(dim=1).item()
    return ["Macro", "Stock", "Market"][label]

test_texts = [
    "RBI hikes repo rate by 25 basis points to curb inflation",
    "India's GDP growth projected at 6.5% for FY2025",
    "Tata Motors reports 15% rise in quarterly net profit",
    "Infosys secures $500 million deal with US-based bank",
    "Sensex gains 450 points as banking stocks rally",
    "Oil prices surge 3% amid supply concerns"
]

for text in test_texts:
    print(text, "=>", predict(text))
#Results-----
#RBI hikes repo rate by 25 basis points to curb inflation => Macro
#India's GDP growth projected at 6.5% for FY2025 => Macro
#Tata Motors reports 15% rise in quarterly net profit => Stock
#Infosys secures $500 million deal with US-based bank => Stock
#Sensex gains 450 points as banking stocks rally => Market
#Oil prices surge 3% amid supply concerns => Macro
