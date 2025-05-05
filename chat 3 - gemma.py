import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pinecone import Pinecone
import torch
from huggingface_hub import login

#Hugging Face
login(token="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")

# Initialisation de Pinecone
api_key = "pcsk_4A9XCs_MQy6tmPJktJKTXM1toFmGWgJXvZhZVDWgBEyHb1yeW9U2Ka9a4hF54HT8UUSYai"
pc = Pinecone(api_key=api_key)
index_name = "regulatory-documents"
index = pc.Index(index_name)

# Charger le modèle Gemma
gemma_model_name = "google/gemma-2b"  
gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
gemma_model = AutoModelForCausalLM.from_pretrained(
    gemma_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# rechercher des documents dans Pinecone
def search_documents(query_text, top_k=3):
    query_embedding = get_labse_embedding(query_text)  
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results["matches"]

# générer une réponse avec Gemma
def generate_response_with_gemma(query_text, context_texts):
    context = "\n".join(context_texts)
    prompt = f"""Vous êtes un assistant expert en réglementation médicale. Répondez à la question en utilisant les informations suivantes :

Contexte :
{context}

Question :
{query_text}

Réponse :"""

    inputs = gemma_tokenizer(prompt, return_tensors="pt").to(gemma_model.device)
    with torch.no_grad():
        outputs = gemma_model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

    response = gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()  # Nettoyer le prompt dans la réponse

# interroger et générer une réponse
def query_and_generate_response(query_text):
    relevant_docs = search_documents(query_text, top_k=3)
    context_texts = [documents[doc["id"]] for doc in relevant_docs]
    response = generate_response_with_gemma(query_text, context_texts)
    return response

# Charger les docs depuis JSON
def load_documents_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

json_path = r"C:\Users\hp\Desktop\regulatory_documents.json"
documents = load_documents_from_json(json_path)

# Exemple d'utilisation
query_text = "Quels sont les documents nécessaires pour l'enregistrement d'un dispositif médical en Europe ?"
response = query_and_generate_response(query_text)
print("Réponse générée :")
print(response)
