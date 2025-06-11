import requests
import json
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime as dt
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer, util
nltk.download("stopwords")
json_schema_Offre = {
    "type": "object",
    "properties": {
        "experience_min": {
            "type": "integer",
            "description": "experience minimal requise",
        },
        "experience_max": {
            "type": "integer",
            "description": "experience maximal requise",
        },
        "poste": {
            "type": "object",
            "properties": {
                "diplome": {
                    "type": "string",
                    "enum": [
                        "licence",
                        "master",
                        "ingenieur",
                        "licence professionnelle",
                        "master professionnelle",
                        "doctorat",
                        "technicien",
                        "bachelor",
                        "maitrise"
                    ],
                    "description": "le type de diplome",
                },
                "contract": {
                    "type": "string",
                    "enum": ["stage", "cdi", "cdd", "alternance", "freelance"],
                    "description": "le type de contrat dans l'offre",
                },
                "title": {
                    "type": "string",
                    "enum": [
                        "stagiaire",
                        "technicien",
                        "junior",
                        "confirmé",
                        "senior",
                        "lead",
                        "manager",
                        "responsable",
                        "expert",
                        "directeur",
                        "jeune diplomé"
                    ],
                    "description": "le type de poste dans l'offre",
                },
            },
            "required": ["diplome","contract","title"],
            "additionalProperties": False,
        },
        "competences": {
            "type": "object",
            "properties": {
                "techniques": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "les competances techniques écrit dans l'offre' juste keywords",
                },
                "methodologies": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "les methodologies ecrit dans l'offre'",
                },
                "soft_skills": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description":"soft skills ecrit dans l'offre",
                }
            },
            "required": ["techniques", "methodologies", "soft_skills"],
            "additionalProperties": False,
        },
        "langues": {
            "type": "array",
            "items": { "type": "string" },
            "description": "les langues declarés dans l'offre",
        }
    },
    "required": [
        "experience_min",
        "experience_max",
        "poste",
        "competences",
        "langues"
    ],
    "additionalProperties": False,
}
json_schema_CV = {
    "type": "object",
    "properties": {
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "keywords de l'experience",
                    },
                    "duration_mois": { "type": "integer",
                                      "description": "duration de l'experience en mois peut egale à 0 si type d'experience et soit stage soit projet etude", },
                    "type": {
                        "type": "string",
                        "enum": ["professionnelle", "stage", "projet etude"],
                        "description": "le type de l'experience"
                    }
                },
                "required": ["keywords", "duration_mois", "type"],
                "additionalProperties": False,
            }
        },
        "education": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": [
                        "licence",
                        "master",
                        "ingenieur",
                        "licence professionnelle",
                        "master professionnelle",
                        "doctorat",
                        "technicien",
                        "bachelor",
                        "maitrise"
                    ],
                    "description": "le type de diplome",
                },
                "subjects": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "les matières etudiées dans chaque diplome si mentionnées"
                }
            },
            "required": ["level","subjects"],
            "additionalProperties": False,
        },
        "position_cherché": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "enum": [
                        "stagiaire",
                        "technicien",
                        "junior",
                        "confirmé",
                        "senior",
                        "lead",
                        "manager",
                        "responsable",
                        "expert",
                        "directeur",
                        "jeune diplomé"
                    ],
                    "description": "le type de poste recherché par le candidat",
                },
                "contract_type": {
                    "type": "string",
                    "enum": ["stage", "cdi", "cdd", "alternance", "freelance"],
                    "description": "le type de contrat recherché par le candidat si le type de contrat n'est pas mentionné dans le CV tu met la valeur par defaut qui est cdi",
                }
            },
            "required": ["title","contract_type"],
            "additionalProperties": False,
        },
        "competences": {
            "type": "object",
            "properties": {
                "techniques": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "les competances techniques écrit dans le CV juste keywords",
                },
                "methodologies": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "les methodologies ecrit dans le CV",
                },
                "soft_skills": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description":"soft skills ecrit dans le CV"
                }
            },
            "required": ["techniques", "methodologies", "soft_skills"],
            "additionalProperties": False,
        },
        "langues": {
            "type": "array",
            "items": { "type": "string" },
            "description": "les langues declarés dans le CV",
        }
    },
    "required": [
        "experiences",
        "education",
        "position_cherché",
        "competences",
        "langues"
    ],
    "additionalProperties": False,
}
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        # Lit le contenu binaire du fichier uploadé
        pdf_bytes = uploaded_file.read()
        # Ouvre PyMuPDF à partir d’un buffer mémoire (bytes)
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Erreur lors de la lecture du PDF : {e}")
    return text

def getting_prompt_offre_personalised(offre_text):
    prompt_Offre = f"""
    **Mission** :
    Analyse cette offre et extrais UNIQUEMENT les informations factuelles sous formes de keywords présentes dans le document, 
    sous forme JSON strict sans commentaires.

    **Règles STRICTES** :
    1. Ne pas inventer ou déduire des informations non explicites
    2. Utiliser uniquement les valeurs prédéfinies pour les champs normalisés
    3. experience_min et experience_max en chiffres uniquement 
    4. Dans experience_max si n'est pas specifié met 999 et pour experience_min si n'est pas specifié met 0
    5. Mots-clés techniques concrets, pas de phrases
    6. Conserver exactement les noms des sections sans modifications
    9. Si jamais tu trouve pas de valeur pour une section tu la laisses vide
    Texte du l'offre:
    {offre_text}  
    """
    return prompt_Offre
def getting_prompt_CV_personalised(cv_text):
    current_date = dt.now()
    prompt_CV = f"""
    **Mission** :
    Analyse ce CV et extrais UNIQUEMENT les informations factuelles présentes dans le document, 
    sous forme JSON strict sans commentaires.

    **Règles STRICTES** :
    1. Ne pas inventer ou déduire des informations non explicites
    2. Utiliser uniquement les valeurs prédéfinies pour les champs normalisés
    3. Durées en chiffres uniquement 
    4. Dans experience si type == stage ou internship ou projet d'etudes , met la duration_mois à 0 respecte cette regle
    5. Mots-clés techniques concrets, pas de phrases
    6. Conserver exactement les noms des sections sans modifications
    7. Pour les durées : calculer d'abord les années complètes, puis le reste en mois
    8. Date actuelle : {current_date.day}/{current_date.month}/{current_date.year}
    9. Si jamais tu trouve pas de valeur pour une section tu la laisses vide
    Texte du CV:
    {cv_text}  
    """
    return prompt_CV
def LLM_Request(jsonschema,llm_model,api_key,prompt):
  response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Authorization": f"Bearer {api_key}",
      "Content-Type": "application/json",
    },

    json={
      "model": llm_model,
      "messages": [
        {"role": "user", "content": prompt},
      ],
      "response_format": {
        "type": "json_schema",
        "json_schema": {
          "name": "json",
          "strict": True,
          "schema": jsonschema,
        },
      },
    },
  )

  data = response.json()
  info = data['choices'][0]["message"]["content"]
  return data,info

def calcule_exp(experiences):
    experience_année = 0
    for experience in experiences:
        if experience["type"] == "stage" or experience["type"] == "projet etude":
            continue
        experience_année+=experience["duration_mois"]
    return np.round(experience_année/12)
def keywords_processing(CVjson):
    # Télécharge la liste des mots vides français
    french_stopwords = set(stopwords.words('french'))

    # --- Étape 1 : Extraction des mots clés comme avant ---
    keywords_CV = []

    for experience in CVjson["experiences"]:
        keywords_CV += experience["keywords"]

    keywords_CV += [CVjson["education"]["level"]]
    keywords_CV += CVjson["education"]["subjects"]
    keywords_CV += [CVjson["position_cherché"]["title"]]
    keywords_CV += [CVjson["position_cherché"]["contract_type"]]
    keywords_CV += CVjson["competences"]["methodologies"]
    keywords_CV += CVjson["competences"]["soft_skills"]
    keywords_CV += CVjson["competences"]["techniques"]
    keywords_CV += CVjson["langues"]

    # --- Étape 2 : Tokenisation et nettoyage ---
    all_words = []
    for phrase in keywords_CV:
        words = re.findall(r'\b\w+\b', phrase.lower())  # convert to lowercase here
        all_words.extend(words)

    # --- Étape 3 : Suppression des mots vides ---
    filtered_keywords = [word for word in all_words if word not in french_stopwords]

    # --- Étape 4 : Suppression des doublons ---
    final_keywords = list(set(filtered_keywords))
    return final_keywords
def keywords_processing_offre(json_offre_info):
    # Télécharge la liste des mots vides français
    french_stopwords = set(stopwords.words('french'))

    # --- Étape 1 : Extraction des mots clés comme avant ---
    keywords_offre = []

    keywords_offre += [json_offre_info["poste"]["diplome"]]
    keywords_offre += [json_offre_info["poste"]["contract"]]
    keywords_offre += [json_offre_info["poste"]["title"]]

    keywords_offre += json_offre_info["competences"]["methodologies"]
    keywords_offre += json_offre_info["competences"]["soft_skills"]
    keywords_offre += json_offre_info["competences"]["techniques"]
    keywords_offre += json_offre_info["langues"]

    # --- Étape 2 : Tokenisation et nettoyage ---
    all_words = []
    for phrase in keywords_offre:
        words = re.findall(r'\b\w+\b', phrase.lower())  # convert to lowercase here
        all_words.extend(words)

    # --- Étape 3 : Suppression des mots vides ---
    filtered_keywords = [word for word in all_words if word not in french_stopwords]

    # --- Étape 4 : Suppression des doublons ---
    final_keywords = list(set(filtered_keywords))
    return final_keywords
def calculating_similarity(CV_inter_string, Offre_string):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    job_embedding = model.encode(CV_inter_string)
    cv_embeddings = model.encode(Offre_string)

    similarity = util.cos_sim(job_embedding, cv_embeddings)
    return np.round(similarity*100,2)
