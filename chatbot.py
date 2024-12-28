import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Charger le modèle et le tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')

# Liste des intentions
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

# Titre pour l'application web
st.title("💬 Banking Chatbot Assistant")

# Ajouter un style CSS personnalisé
st.markdown(
    """
    <style>
    .chat-window {
        position: fixed;
        bottom: 16px;
        right: 16px;
        background-color: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        width: 440px;
        height: 634px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        padding: 16px;
    }
    .chat-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .chat-subheader {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 16px;
    }
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        margin-bottom: 16px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .chat-message.user {
        text-align: right;
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 16px 0 16px 16px;
        max-width: 60%;
        margin-left: auto;
        margin-bottom: 10px;
        margin-top: 10px;
        color: #12412e;
    }
    .chat-message.bot {
        text-align: left;
        background-color: #0078D7;
        color: white;
        padding: 10px;
        border-radius: 0 16px 16px 16px;
        max-width: 60%;
    }
    .chat-input-container {
        display: flex;
        gap: 8px;
    }
    .chat-input {
        flex: 1;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    .chat-send-button {
        background-color: #000;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .chat-send-button:hover {
        background-color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state pour conserver l'historique
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Afficher les messages de chat
st.markdown('<div class="chat-header">Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subheader">Powered by AI Banking Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

for message in st.session_state["messages"]:
    if "user" in message:
        st.markdown(f'<div class="chat-message user">{message["user"]}</div>', unsafe_allow_html=True)
    if "bot" in message:
        st.markdown(f'<div class="chat-message bot">{message["bot"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Fin des messages

# Formulaire d'entrée utilisateur
user_input = st.text_input("Type your question...", key="user_input")

if st.button("Send") and user_input:
    # Append user message to session state
    st.session_state["messages"].append({"user": user_input, "bot": ""})

    # Send request to model
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    # Calculate probabilities with softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Find the predicted class index
    predicted_class = probs.argmax().item()

    # Retrieve the intent corresponding to the index
    predicted_intent = classes[predicted_class]

    # Confidence in percentage
    confidence = probs.max().item() * 100

    # Create bot reply
    bot_reply = (
        f"Intent: {predicted_intent}<br>"
        f"Confidence: {confidence:.2f}%"
    )

    # Update the bot's reply in session state
    st.session_state["messages"][-1]["bot"] = bot_reply