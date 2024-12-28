# Banking Chatbot Assistant

## Prérequis

Avant de commencer, vous devez installer les éléments suivants sur votre machine :

- **Python 3.7+**
- **FastAPI** (pour l'API de prédiction)
- **Streamlit** (pour l'interface utilisateur)
- **Uvicorn** (pour lancer FastAPI)
- **Requests** (pour effectuer des requêtes HTTP)

## Étapes pour démarrer l'application

1. **Installer les dépendances** :

   ```bash
   pip install -r requirements.txt

2. **Lancez le serveur FastAPI avec Uvicorn** :

    ```bash
    uvicorn API:app --reload

Le serveur sera accessible à l'URL suivante :
http://127.0.0.1:8000

3. **Lancez l'application Streamlit** :

    ```bash
    streamlit run chatbot.py


L'application sera accessible à l'URL suivante :
http://localhost:8501



