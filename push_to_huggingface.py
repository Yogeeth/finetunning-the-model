"""To Push to Hugging Face
Create a Hugging Face Account
Go to https://huggingface.co and sign up.

Generate an Access Token
Navigate to Settings → Access Tokens.
Click "New token", give it a name, and select the required role (usually write).
Copy the token and keep it safe — you’ll need it for authentication.
"""
from huggingface_hub import notebook_login

notebook_login()

# Load your trained model
model = AutoModelForSequenceClassification.from_pretrained("./model_")
tokenizer = AutoTokenizer.from_pretrained("./model_")

# Push to your account with a custom name
model.push_to_hub("model_")
tokenizer.push_to_hub("model_")

#model_id = "yoganfire/FireBerth"(Myself)
model_id = "username/modelname"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    label = outputs.logits.argmax(dim=1).item()
    return ["Macro", "Stock", "Market"][label]

print(predict("Infosys stock jumps after strong earnings."))
print(predict("RBI raises repo rate by 25 bps to combat inflation."))
print(predict("HDFC Bank reports 22% jump in Q1 profit"))
print(predict("Tata Motors unveils new EV lineup for 2025."))
