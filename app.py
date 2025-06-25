data = {
    "text": [
        "India's GDP grew 6.8% in Q1, beating forecasts.",
        "RBI raises repo rate by 25 bps to combat inflation.",
        "CPI inflation eased to 4.5% in May.",
        "Government announces ₹2 trillion infrastructure push.",
        "Unemployment rate drops to 7.1% amid economic recovery.",
        "India’s export growth accelerates to 12% YoY.",
        "WPI inflation remains in negative territory for second month.",
        "Budget 2025 focuses on capex and fiscal discipline.",
        "India’s forex reserves hit a new high of $650 billion.",
        "Rural consumption picks up, driven by MNREGA payouts.",

        "TCS posts 15% YoY revenue growth in Q2.",
        "Infosys raises FY26 guidance after strong results.",
        "HUL's margins expand by 120 bps in Q4.",
        "Adani Green commissions 1.5 GW of solar capacity.",
        "Zomato turns EBITDA-positive for first time.",
        "Maruti Suzuki reports record car sales in May.",
        "ITC sees double-digit FMCG growth in last quarter.",
        "HDFC Bank net profit rises 18% YoY.",
        "Reliance Retail expands grocery delivery to 50 new cities.",
        "JSW Steel net income falls 10% due to input cost pressure.",

        "Sensex jumps 600 points on global cues.",
        "Nifty50 hits all-time high of 23,000.",
        "FII inflows cross ₹10,000 Cr in one week.",
        "India VIX drops to multi-year low at 10.2.",
        "Bank Nifty surges on dovish RBI commentary.",
        "PSU stocks rally on privatization hopes.",
        "Midcaps underperform; Nifty Midcap 100 falls 2%.",
        "Markets volatile ahead of US Fed meeting.",
        "Metal stocks drag market as China demand slows.",
        "Rupee strengthens to 82.3/$ amid weak dollar index.",

        "India’s industrial output contracts 1.2% in April.",
        "Fiscal deficit contained at 5.6% of GDP for FY2025.",
        "Core inflation fell below 4% for the first time in 2 years.",
        "Crude oil stays under $80, aiding India’s current account.",
        "Retail inflation slows as vegetable prices fall sharply.",
        "MSME sector sees pickup in loan disbursements.",

        "Biocon’s net profit rises 8% on strong biosimilar sales.",
        "Tata Motors unveils new EV lineup for 2025.",
        "L&T bags ₹10,000 Cr in defense infrastructure contracts.",
        "IRCTC reports 2x YoY growth in catering segment.",
        "Bajaj Auto exports rise 15% despite global slowdown.",
        "Nykaa expands into offline beauty stores across India.",

        "PVR-Inox reports record footfalls in Q1 FY2026.",
        "HDFC Life sees 21% jump in new business premium.",
        "ONGC gains as crude prices stabilize.",
        "Volatility spikes ahead of Fed rate decision.",
        "Rupee depreciates to 83.1/$ on global dollar strength.",
        "Nifty IT index falls 2.5% on weak global tech cues.",
        "SBI leads banking rally as credit growth surges.",
        "Auto stocks under pressure on muted monthly sales."
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  
        0, 0, 0, 0, 0, 0,             
        1, 1, 1, 1, 1, 1,             
        2, 2, 2, 2, 2, 2, 2, 2        
    ]
}

testset = {
    "text": [
        # Macro (0)
        "RBI maintains repo rate to support growth.",
        "India’s GDP projected to grow at 7.2% in FY2026.",
        "Government to boost capital expenditure by 30%.",
        "Fiscal deficit target revised to 5.4% of GDP.",
        "CPI inflation rises to 5.1% due to food prices.",
        "Exports to EU increase amid new trade agreements.",
        "GST collection hits record ₹1.9 lakh crore in June.",
        "WPI inflation turns positive after 3 negative months.",
        "India signs bilateral trade pact with UK.",
        "Foreign exchange reserves rise to $655 billion.",
        
        # Stock (1)
        "HDFC Bank reports 22% jump in Q1 profit.",
        "Infosys secures $2 billion deal from US client.",
        "Reliance Retail to launch premium fashion brand.",
        "Bajaj Auto unveils new EV scooter line-up.",
        "Tata Steel posts 10% rise in consolidated EBITDA.",
        "Zomato acquires Blinkit to strengthen quick commerce.",
        "Wipro announces 5% salary hike for employees.",
        "ITC's FMCG revenue grows 14% YoY.",
        "JSW Energy commissions 500 MW solar plant.",
        "Indigo Airlines adds 30 new domestic routes.",
        
        # Market (2)
        "Sensex gains 450 points on global optimism.",
        "Nifty50 ends above 23,200 for the first time.",
        "India VIX rises as traders hedge Fed risk.",
        "Midcap stocks outperform large caps this quarter.",
        "FII inflows cross ₹12,000 Cr amid rally.",
        "Bank Nifty hits record high on strong credit growth.",
        "Gold prices fall below ₹56,000 as dollar strengthens.",
        "Rupee slips to 83.4/$ ahead of US data.",
        "Crude oil rebounds above $80 amid supply cuts.",
        "Crypto markets dip on regulatory uncertainty.",
        
        # Macro (0)
        "RBI governor highlights inflation risks in policy meet.",
        "India’s manufacturing PMI climbs to 58.3.",
        "Government allocates ₹3 lakh crore for rural infra.",
        "Imports from China drop 8% YoY.",
        "Unemployment rate improves to 6.8% in urban areas.",
        "Cabinet approves ₹20,000 crore MSME support package.",
        
        # Stock (1)
        "TCS wins mega digital transformation contract.",
        "LIC posts 12% increase in policy premium collections.",
        "Adani Ports expands capacity at Mundra terminal.",
        "Nykaa launches new skincare line for Gen Z.",
        "Biocon signs biosimilar deal in EU markets.",
        "Maruti Suzuki sees 18% growth in SUV sales.",
        
        # Market (2)
        "Markets fall ahead of ECB rate decision.",
        "PSU stocks rally on disinvestment buzz.",
        "Smallcap index rises 2% on strong retail participation.",
        "Nifty IT index tumbles 3% on global tech rout.",
        "Volatility index hits 6-month high before Budget.",
        "Real estate stocks surge post RBI policy clarity.",
        
        # Mix more Stock (1)
        "Nestle India expands rural distribution network.",
        "L&T secures ₹7,500 Cr in new construction orders.",
        "Tata Motors sees strong EV demand in Tier-2 cities.",
        "ICICI Bank raises lending rates by 15 bps.",
        
        # Mix more Market (2)
        "Gold ETFs see record inflows amid global uncertainty.",
        "Rupee gains on soft US inflation print.",
        "Bond yields remain flat despite hawkish stance."
    ],
    "label": [
        0,0,0,0,0,0,0,0,0,0,     # 10 Macro
        1,1,1,1,1,1,1,1,1,1,     # 10 Stock
        2,2,2,2,2,2,2,2,2,2,     # 10 Market
        0,0,0,0,0,0,             # 6 Macro
        1,1,1,1,1,1,             # 6 Stock
        2,2,2,2,2,2,             # 6 Market
        1,1,1,1,                 # 4 Stock
        2,2,2                    # 3 Market
    ]
}

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import torch

#Create and split dataset
from datasets import DatasetDict
dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

#Load tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

#Preprocess (tokenize)
def preprocess(example):
    return tokenizer(example["text"], padding=True, truncation=True)

encoded_dataset = dataset.map(preprocess, batched=True)

#Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

#Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    report_to="none",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=5,
)

#Train the model using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

#Save the finetuned model
model.save_pretrained("./model_")
tokenizer.save_pretrained("./model_")

#Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=1).item()
    label_map = {0: "Macro", 1: "Stock", 2: "Market"}
    return label_map[prediction]

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
dataset = Dataset.from_dict(testset)
texts = dataset["text"]
def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.argmax(dim=1).tolist()

pred = predict(texts)
lab=dataset["label"]
without_correct=0
label_map = {0: "Macro", 1: "Stock", 2: "Market"}
print("predicted  actual  verdict")
for a in range(len(lab)): 
  if lab[a]!=pred[a]:
    print(label_map[pred[a]],"    ",label_map[lab[a]]," ","Incorrect")
  else:
    print(label_map[pred[a]],"    ",label_map[lab[a]]," ","correct")  
    without_correct+=1


model_name = "./model_"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
dataset = Dataset.from_dict(testset)
texts = dataset["text"]


def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.argmax(dim=1).tolist()

pred = predict(texts)
lab=dataset["label"]
finetunned_correct=0
label_map = {0: "Macro", 1: "Stock", 2: "Market"}
print("predicted  actual  verdict")
for a in range(len(lab)): 
  if lab[a]!=pred[a]:
    print(label_map[pred[a]],"    ",label_map[lab[a]]," ","Incorrect")
  else:
    print(label_map[pred[a]],"    ",label_map[lab[a]]," ","correct")  
    finetunned_correct+=1

print("finetunned_correct : without_finetunning_correct : total predections","",finetunned_correct,":",without_correct,":",len(lab))