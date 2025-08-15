# Import libraries
#pip install transformers datasets torch scikit-learn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
import torch

# Curated small dataset for rapid prototyping
nifty_50_data = {
    "text": [
        "Reliance Industries announces green energy investment.",
        "Tata Consultancy Services signs multi-year US banking deal.",
        "Infosys launches new GenAI platform for enterprises.",
        "HDFC Bank opens 200 new rural branches.",
        "ICICI Bank reports record profit in Q1.",
        "Kotak Mahindra Bank introduces AI-based loan approval system.",
        "Larsen & Toubro bags ₹15,000 Cr infrastructure project.",
        "State Bank of India to raise capital via bonds.",
        "Axis Bank partners with fintech for digital lending.",
        "Bajaj Finance sees 30% YoY growth in customer base.",
        "Bharti Airtel expands 5G services to 500 cities.",
        "Hindustan Unilever increases focus on health segment.",
        "ITC launches sustainable FMCG product line.",
        "Adani Enterprises begins construction of solar park.",
        "Adani Ports handles record cargo volumes in May.",
        "Wipro collaborates with NVIDIA for GenAI solutions.",
        "Tech Mahindra expands presence in Latin America.",
        "Titan Company posts strong festive season sales.",
        "Asian Paints plans ₹2,000 Cr capacity expansion.",
        "Nestle India sees double-digit growth in rural markets.",
        "Maruti Suzuki unveils new compact electric SUV.",
        "Tata Motors rolls out new electric truck lineup.",
        "Mahindra & Mahindra reports 20% rise in tractor sales.",
        "Sun Pharma gets USFDA nod for new cancer drug.",
        "Divi's Laboratories to invest in API production.",
        "Cipla receives clearance to export new inhaler globally.",
        "Dr. Reddy's Laboratories launches generic drug in US.",
        "UltraTech Cement commissions new plant in Gujarat.",
        "Grasim Industries sees strong demand in chemical segment.",
        "Shree Cement ramps up capacity to meet infra demand.",
        "JSW Steel eyes export boost amid China slowdown.",
        "Tata Steel commits ₹12,000 Cr to green steel transition.",
        "HCL Technologies reports strong digital revenue growth.",
        "Power Grid Corporation completes 765kV transmission line.",
        "NTPC to invest in solar-wind hybrid power project.",
        "Coal India increases coal output to meet power demand.",
        "BPCL modernizes fuel retail network across India.",
        "Hindalco Industries expands aluminium recycling unit.",
        "ONGC gains on stabilizing global crude oil prices.",
        "Eicher Motors announces next-gen Royal Enfield model.",
        "Hero MotoCorp launches new premium EV scooter.",
        "Bajaj Auto posts 18% growth in two-wheeler exports.",
        "Apollo Hospitals opens state-of-the-art cancer center.",
        "Britannia Industries expands dairy product offerings.",
        "Bajaj Finserv integrates insurance with mobile app.",
        "SBI Life Insurance sees 25% rise in new premium.",
        "HDFC Life launches annuity product for retirees.",
        "IndusInd Bank reports strong Q2 performance.",
        "UPL partners with agri-tech firm for digital farming.",
        "Bharat Electronics bags defence electronics order."
    ],
    "label": [1] * 50  # All Stock news
}

data = {
    "text": [
        # Macro news (0)
        "India's GDP grew 6.8% in Q1, beating forecasts.",
        "RBI raises repo rate by 25 bps to combat inflation.",
        "CPI inflation eased to 4.5% in May.",
        "Government announces ₹2 trillion infrastructure push.",
        "Unemployment rate drops to 7.1% amid economic recovery.",
        "India's export growth accelerates to 12% YoY.",
        "WPI inflation remains in negative territory for second month.",
        "Budget 2025 focuses on capex and fiscal discipline.",
        "India's forex reserves hit a new high of $650 billion.",
        "Rural consumption picks up, driven by MNREGA payouts.",
        
        # Stock news (1)
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
        
        # Market news (2)
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
        
        # Additional Macro
        "India's industrial output contracts 1.2% in April.",
        "Fiscal deficit contained at 5.6% of GDP for FY2025.",
        "Core inflation fell below 4% for the first time in 2 years.",
        "Crude oil stays under $80, aiding India's current account.",
        "Retail inflation slows as vegetable prices fall sharply.",
        "MSME sector sees pickup in loan disbursements.",
        
        # Additional Stock
        "Biocon's net profit rises 8% on strong biosimilar sales.",
        "Tata Motors unveils new EV lineup for 2025.",
        "L&T bags ₹10,000 Cr in defense infrastructure contracts.",
        "IRCTC reports 2x YoY growth in catering segment.",
        "Bajaj Auto exports rise 15% despite global slowdown.",
        "Nykaa expands into offline beauty stores across India.",
        
        # Additional Market
        "PVR-Inox reports record footfalls in Q1 FY2026.",
        "HDFC Life sees 21% jump in new business premium.",
        "ONGC gains as crude prices stabilize.",
        "Volatility spikes ahead of Fed rate decision.",
        "Rupee depreciates to 83.1/$ on global dollar strength.",
        "Nifty IT index falls 2.5% on weak global tech cues.",
        "SBI leads banking rally as credit growth surges.",
        "Auto stocks under pressure on muted monthly sales.",
        
        # Mixed categories
        "India signs free trade agreement with EU.",
        "China's GDP growth slows to 3% amid global tensions.",
        "US imposes tariffs on imported semiconductors.",
        "OPEC cuts oil production, raising global prices.",
        "Border tensions between India and China escalate.",
        "Peace talks between Russia and Ukraine show progress.",
        "Israel-Hamas ceasefire brings temporary stability.",
        "UN sanctions lifted from African nation after reforms.",
        "Government bans 25 Chinese apps, citing security.",
        "New digital privacy law could disrupt tech industry.",
        "Facebook fined $1 billion for antitrust violations.",
        "SEBI introduces stricter rules for crypto exchanges.",
        "Severe cyclone disrupts coastal supply chains.",
        "Earthquake halts operations at major manufacturing hub.",
        "Flooding damages 3 lakh hectares of farmland.",
        "Massive protests erupt over rising fuel prices.",
        "New education bill receives mixed reactions.",
        "Healthcare budget increased by 20%.",
        "Supreme Court verdict boosts corporate transparency.",
        "CEO resigns amid corruption allegations.",
        "Tesla announces major battery breakthrough.",
        "Pharma company fined for clinical trial manipulation.",
        "Indian startup becomes unicorn with $2B valuation.",
        
        # More organized data
        "Government launches new infrastructure scheme worth ₹5 lakh crore.",
        "India's fiscal deficit narrows to 4.8% of GDP.",
        "Monsoon expected to be below normal this year.",
        "PM announces new global trade policy reform.",
        "High inflation hurts rural spending patterns.",
        "IMF revises India's GDP growth forecast downward.",
        "Centre announces ₹20,000 crore for clean energy transition.",
        "India signs energy cooperation deal with UAE.",
        "RBI plans to implement new monetary policy framework.",
        "Union Budget focuses on manufacturing and exports.",
        "Tax collection exceeds budget target by 15%.",
        "Railway modernization project gets cabinet approval.",
        "India's industrial production grew 6% last quarter.",
        "New education policy expected to impact employment trends.",
        "Electricity reforms expected to cut power tariffs.",
        
        # Stock specific
        "Adani Ports signs deal to acquire Sri Lankan terminal.",
        "Infosys launches GenAI services for enterprise clients.",
        "Tata Steel to invest ₹12,000 crore in green steel production.",
        "HUL to expand into organic personal care products.",
        "Mahindra reports 20% surge in EV sales YoY.",
        "Wipro sees leadership change amid restructuring.",
        "Zomato stock spikes after delivery revenue rises 35%.",
        "ITC shareholders approve hotel business demerger.",
        "Bajaj Auto expands electric two-wheeler portfolio.",
        "Vedanta faces $200M fine over mining violations.",
        "Nykaa partners with global beauty brand for India launch.",
        "TCS bags $2 billion deal from US healthcare firm.",
        "JSW Energy signs 1.5 GW renewable power PPA.",
        "IRCTC sees record bookings during holiday season.",
        "HDFC Bank opens 500 new rural branches across India.",
        
        # Market specific
        "Sensex drops 800 points amid Fed rate hike fears.",
        "Nifty IT index gains 2.3% on strong tech earnings.",
        "FII outflows touch ₹8,000 Cr in one week.",
        "Volatility index India VIX spikes 25%.",
        "Midcap stocks outperform blue-chips this quarter.",
        "Metal sector drags market as demand slumps in China.",
        "Rupee appreciates to 81.5/$ amid weak dollar index.",
        "PSU banks rally after strong Q4 earnings.",
        "Smallcap index falls 3% on valuation concerns.",
        "Auto stocks bounce after GST cut announcement.",
        "Bank Nifty hits record high as credit demand surges.",
        "Renewables stocks gain after govt announces incentives.",
        "Nifty FMCG index up 1.8% on rural demand recovery.",
        "IPO buzz lifts startup stocks across the board.",
        "Market breadth positive: 2,000 stocks advance, 800 decline.",
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Macro
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Stock
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # Market
        0, 0, 0, 0, 0, 0,               # Additional Macro
        1, 1, 1, 1, 1, 1,               # Additional Stock
        2, 2, 2, 2, 2, 2, 2, 2,         # Additional Market
        0, 0, 0, 0,                     # Trade/Economy
        0, 0, 0, 0,                     # Geopolitics
        0, 0, 1, 2,                     # Tech/Regulation
        0, 0, 0,                        # Disasters
        0, 0, 0, 0,                     # Social/Gov
        1, 1, 1, 1,                     # Stock-specific
        0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0, # Macro
        1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1, # Stock
        2,2,2,2,2,2,2,2,2,2, 2,2,2,2,2  # Market
    ]
}

# Combine datasets
data['text'] = data['text'] + nifty_50_data['text']
data['label'] = data['label'] + nifty_50_data['label']

# Test Dataset
testset = {
    "text": [
        # Macro (0)
        "RBI maintains repo rate to support growth.",
        "India's GDP projected to grow at 7.2% in FY2026.",
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
        
        # Additional samples
        "RBI governor highlights inflation risks in policy meet.",
        "India's manufacturing PMI climbs to 58.3.",
        "Government allocates ₹3 lakh crore for rural infra.",
        "Imports from China drop 8% YoY.",
        "Unemployment rate improves to 6.8% in urban areas.",
        "Cabinet approves ₹20,000 crore MSME support package.",
        "TCS wins mega digital transformation contract.",
        "LIC posts 12% increase in policy premium collections.",
        "Adani Ports expands capacity at Mundra terminal.",
        "Nykaa launches new skincare line for Gen Z.",
        "Biocon signs biosimilar deal in EU markets.",
        "Maruti Suzuki sees 18% growth in SUV sales.",
        "Markets fall ahead of ECB rate decision.",
        "PSU stocks rally on disinvestment buzz.",
        "Smallcap index rises 2% on strong retail participation.",
        "Nifty IT index tumbles 3% on global tech rout.",
        "Volatility index hits 6-month high before Budget.",
        "Real estate stocks surge post RBI policy clarity.",
        "Nestle India expands rural distribution network.",
        "L&T secures ₹7,500 Cr in new construction orders.",
        "Tata Motors sees strong EV demand in Tier-2 cities.",
        "ICICI Bank raises lending rates by 15 bps.",
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

# Create and split dataset
dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Load tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Preprocess function
def preprocess(example):
    return tokenizer(example["text"], padding=True, truncation=True)

encoded_dataset = dataset.map(preprocess, batched=True)

# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# Training arguments
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

# Train the model
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

# Save the fine-tuned model
print("Saving model...")
model.save_pretrained("./model_")
tokenizer.save_pretrained("./model_")

# Prediction function
def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.argmax(dim=1).tolist()

# Test original model
print("\n=== TESTING ORIGINAL MODEL ===")
original_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
original_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

def predict_original(texts):
    inputs = original_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = original_model(**inputs)
    return outputs.logits.argmax(dim=1).tolist()

testset_dataset = Dataset.from_dict(testset)
texts = testset_dataset["text"]
pred = predict_original(list(texts))
lab = testset_dataset["label"]
without_correct = 0
label_map = {0: "Macro", 1: "Stock", 2: "Market"}

print("Predicted  Actual   Verdict")
for a in range(len(lab)):
    if lab[a] != pred[a]:
        print(f"{label_map[pred[a]]:<8}   {label_map[lab[a]]:<8} Incorrect")
    else:
        print(f"{label_map[pred[a]]:<8}   {label_map[lab[a]]:<8} Correct")
        without_correct += 1

print(f"Original Model Accuracy: {without_correct}/{len(lab)} = {without_correct/len(lab):.2%}")

# Test fine-tuned model
print("\n=== TESTING FINE-TUNED MODEL ===")
finetuned_tokenizer = AutoTokenizer.from_pretrained("./model_")
finetuned_model = AutoModelForSequenceClassification.from_pretrained("./model_", num_labels=3)

def predict_finetuned(texts):
    inputs = finetuned_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = finetuned_model(**inputs)
    return outputs.logits.argmax(dim=1).tolist()

pred = predict_finetuned(list(texts))
finetuned_correct = 0

print("Predicted  Actual   Verdict")
for a in range(len(lab)):
    if lab[a] != pred[a]:
        print(f"{label_map[pred[a]]:<8}   {label_map[lab[a]]:<8} Incorrect")
    else:
        print(f"{label_map[pred[a]]:<8}   {label_map[lab[a]]:<8} Correct")
        finetuned_correct += 1

print(f"Fine-tuned Model Accuracy: {finetuned_correct}/{len(lab)} = {finetuned_correct/len(lab):.2%}")
print(f"Improvement: {finetuned_correct - without_correct} more correct predictions")

