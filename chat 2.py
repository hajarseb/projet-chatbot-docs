import json
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone, ServerlessSpec
import unicodedata

#Initialisation de Pinecone
api_key = "pcsk_27dxcZ_CWpETpMZHG4kcGGoyY51WdfD3kR8txV1iPXVyUyAx3EUt3fpLQbqZhspgsLvYvs"

pc = Pinecone(api_key=api_key)
index_name = "regulatory-documents"

# Supprimer l'index s'il existe déjà
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# Créer un nouvel index avec LaBSE 
pc.create_index(
    name=index_name,
    dimension=768,  
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index(index_name)

#Charger LaBSE
labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

#générer les embeddings avec LaBSE
def get_labse_embedding(text):
    tokens = labse_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = labse_model(**tokens)
    embeddings = output.pooler_output  # Prendre la représentation globale du texte
    return embeddings.squeeze().tolist()

#Charger le fichier .JSON contenant les documents
def load_documents_from_json(json_path):
  with open(json_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

#Charger les documents depuis le fichier .JSON
json_path = r"C:\Users\USER\Desktop\DOC PFE\Data Base Regulation/regulatory_documents.json"  # Chemin vers le fichier .JSON
documents = load_documents_from_json(json_path)

print(f"{len(documents)} documents chargés.")

#Indexation des documents dans Pinecone
for doc_id, text in documents.items():
    embedding = get_labse_embedding(text)
    index.upsert(vectors=[(doc_id, embedding)])

print("Indexation terminée.")

#Tester une requête
query_text = "Quels sont les documents nécessaires pour l'enregistrement d'un dispositif médical en Europe ?"
query_embedding = get_labse_embedding(query_text)

#Recherche dans Pinecone
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

#Affichage des résultats
for match in results["matches"]:
    print(f"Document trouvé : {match['id']} avec un score de {match['score']}")
