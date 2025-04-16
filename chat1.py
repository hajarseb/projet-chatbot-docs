import os
import fitz
import json
import unicodedata

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text") + "\n"  # Ajout d'un saut de ligne entre les pages
    return text.strip()

#Fonction pour nettoyer le nom des fichiers
def clean_file_name(file_name):
    file_name = unicodedata.normalize('NFD', file_name).encode('ascii', 'ignore').decode('ascii')
    return file_name.replace(" ", "_")

#Répertoire contenant les PDFs
pdf_directory = r"C:\Users\USER\Desktop\DOC PFE\Data Base Regulation\pdf_directory"
output_json_path = r"C:\Users\USER\Desktop\DOC PFE\Data Base Regulation/regulatory_documents.json"  # Où sauvegarder le JSON

#Stockage des documents sous format JSON
documents = {}

# **Add error handling to print a helpful message if the directory is not found.**
if not os.path.exists(pdf_directory):
    print(f"Error: The directory '{pdf_directory}' does not exist. Please check the path and try again.")
else:
    for pdf_filename in os.listdir(pdf_directory):
        if pdf_filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            if extracted_text.strip():
                doc_id = clean_file_name(pdf_filename)  # Nettoyage du nom du fichier
                documents[doc_id] = extracted_text[:5000]  # Option : Limiter à 5000 caractères

#Sauvegarde en JSON
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(documents, json_file, indent=4, ensure_ascii=False)

print(f"{len(documents)} documents extraits et enregistrés dans {output_json_path}.")

