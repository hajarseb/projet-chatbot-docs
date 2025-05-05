import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pinecone import Pinecone

# Initialisation FastAPI
app = FastAPI()

# Initialisation de Pinecone
api_key = "pcsk_27dxcZ_CWpETpMZHG4kcGGoyY51WdfD3kR8txV1iPXVyUyAx3EUt3fpLQbqZhspgsLvYvs"
pc = Pinecone(api_key=api_key)
index = pc.Index("regulatory-documents")

# Charger LaBSE
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

# Charger le modèle Gemma
gemma_model_name = "google/gemma-2b"
gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
gemma_model = AutoModelForCausalLM.from_pretrained(
    gemma_model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Compiler le modèle pour accélérer l'inférence (si PyTorch >= 2.0)
if torch.cuda.is_available() and hasattr(torch, 'compile'):
    gemma_model = torch.compile(gemma_model)

# Charger les documents depuis un fichier JSON
def load_documents_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

json_path = r"C:\Users\USER\Desktop\DOC PFE\Data Base Regulation/regulatory_documents.json"
documents = load_documents_from_json(json_path)

# Index rapide des documents (avec vérification)
doc_lookup = {}
for doc in documents:
    if isinstance(doc, dict) and "doc_id" in doc and "text" in doc:
        doc_lookup[doc["doc_id"]] = doc["text"]


# Fonction pour obtenir un embedding avec LaBSE
def get_labse_embedding(text):
    tokens = labse_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = labse_model(**tokens)
    embeddings = output.pooler_output
    return embeddings.squeeze().tolist()

# Fonction pour générer une réponse avec Gemma
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
    return response[len(prompt):].strip()
# Fonction pour nettoyer les doublons dans la réponse générée
def clean_response(response):
    lines = response.split('\n')
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            cleaned.append(line)
            seen.add(line)
    return '\n'.join(cleaned)

# Schéma de la requête entrante
class QuestionRequest(BaseModel):
    query: str

# Endpoint principal
@app.post("/ask")
def ask_question(request: QuestionRequest):
    query = request.query

    # Embedding de la requête et recherche dans Pinecone
    query_embedding = get_labse_embedding(query)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Récupération des documents
    retrieved_docs = []
    for match in results["matches"]:
        doc_id = match["id"]
        doc_text = doc_lookup.get(doc_id, "Contenu non disponible")
        retrieved_docs.append(f"{doc_id} : {doc_text[:500]}...")

    context = "\n\n".join(retrieved_docs)
    response = generate_response_with_gemma(query, context)

    return {"response": response}
