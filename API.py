import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Charger le modèle et le tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')

# Liste des intentions, chaque index correspond à une classe d'intention
classes = [
    "freeze_account", "account_blocked", "report_lost_card", "damaged_card", 
    "replacement_card_duration", "new_card", "card_declined", "credit_limit_change", 
    "credit_limit", "expiration_date", "bill_due", "pay_bill", "bill_balance", 
    "transfer", "spending_history", "transactions", "balance", "credit_score", 
    "improve_credit_score", "interest_rate", "min_payment", "redeem_rewards", 
    "rewards_balance", "report_fraud", "fraud_alert", "greeting", "thank_you", 
    "goodbye", "are_you_a_bot", "what_is_your_name", "application_status", 
    "income", "taxes", "insurance_change"
]

# Créer l'application FastAPI
app = FastAPI()

# Modèle pour la requête
class Message(BaseModel):
    text: str

# Point d'entrée pour prédire l'intention du texte
@app.post("/predict/")
async def predict(message: Message):
    # Tokenisation de l'entrée
    inputs = tokenizer(message.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Calcul des probabilités avec softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Trouver la classe prédite (indice)
    predicted_class = probs.argmax().item()
    
    # Récupérer le nom de l'intention correspondant à l'indice
    predicted_intent = classes[predicted_class]
    
    # Confiance en pourcentage
    confidence = probs.max().item() * 100  
    
    return {
        "prediction": predicted_intent,  # Renvoie le label d'intention
        "confidence": confidence
    }

# Point d'entrée pour vérifier si l'API est en ligne
@app.get("/")
async def root():
    return {"message": "API is running!"}
