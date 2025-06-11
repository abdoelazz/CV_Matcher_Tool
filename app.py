import streamlit as st
import time
import json
from PIL import Image

from main_script import (
    extract_text_from_pdf,
    getting_prompt_CV_personalised,
    getting_prompt_offre_personalised,
    LLM_Request,
    json_schema_CV,
    json_schema_Offre,
    calcule_exp,
    keywords_processing,
    keywords_processing_offre,
    calculating_similarity
    )
def main():
    st.set_page_config(page_title="Analyse Offre vs CV", layout="centered")
    st.logo("images/LOGO.png", size="large", link=None, icon_image=None)

    st.title("🧠 Analyse de Correspondance : Offre d'emploi ↔️ CV")

    # Choix du modèle LLM
    st.subheader("🧠 Choisissez un modèle LLM")
    llm_model_display = {
    "google/gemini-2.0-flash-001": "🔵 Gemini 2.0 Flash (payant)",
    "anthropic/claude-sonnet-4": "🟣 Claude Sonnet 4 (payant)",
    "openai/gpt-4o-mini": "🟡 GPT-4o mini (payant)",
    "deepseek/deepseek-r1-zero:free": "🟢 DeepSeek R1 Zero (gratuit)"
    }

    selected_label = st.selectbox(
        "🧠 Sélectionnez le modèle de langage à utiliser :",
        options=list(llm_model_display.values()),
        index=3
    )

    # Récupérer le modèle réel à partir du libellé sélectionné
    llm_model = [key for key, val in llm_model_display.items() if val == selected_label][0]


    
    # Saisie de la clé API
    st.subheader("🔑 Clé API du modèle LLM")
    api_key = st.text_input("Entrez votre clé API :", type="password")
    if not api_key:
        st.warning("🚨 Veuillez entrer votre clé API pour continuer.")
        st.stop()

    # Étape 1 : Coller l'offre
    st.subheader("1️⃣ Coller ici l'offre d'emploi")
    offre_text = st.text_area("📋 Contenu de l'offre d'emploi :", height=250)

    # Étape 2 : Télécharger les CVs
    st.subheader("2️⃣ Uploader un ou plusieurs CVs")
    uploaded_files = st.file_uploader("📎 Sélectionner des fichiers PDF", type=["pdf"], accept_multiple_files=True)

    # Étape 3 : Lancer l'analyse
    if offre_text and uploaded_files:
        st.subheader("3️⃣ Résultats de l’analyse")
        for file in uploaded_files:
            st.markdown(f"---\n### 📄 Résultats pour : `{file.name}`")
            with st.spinner("⏳ Analyse en cours..."):

                # Extraire le texte depuis le PDF uploadé
                cv_text = extract_text_from_pdf(file)
                if not cv_text.strip():
                    st.warning("⚠️ Le texte extrait du PDF est vide. Impossible de poursuivre l'analyse.")
                    continue  # passe au fichier suivant

                # Préparer les prompts
                prompt_CV = getting_prompt_CV_personalised(cv_text)
                prompt_offre = getting_prompt_offre_personalised(offre_text)

                # Appeler le modèle LLM
                CV_data, CV_info = LLM_Request(json_schema_CV, llm_model, api_key, prompt_CV)
                time.sleep(1)  # <-- Pause 1 seconde avant d'envoyer la requête offre (optionnel)
                offre_data, offre_info = LLM_Request(json_schema_Offre, llm_model, api_key, prompt_offre)
                time.sleep(1)  # <-- Pause 1 seconde avant d'envoyer la requête CV suivant

                # Charger le JSON retourné par le LLM
                try:
                    json_CV_info = json.loads(CV_info)
                    json_offre_info = json.loads(offre_info)
                except json.JSONDecodeError:
                    st.error("Erreur lors du décodage JSON retourné par le modèle.")
                    continue

                # Calculs
                experience = calcule_exp(json_CV_info.get("experiences", []))
                keywords_Cv = keywords_processing(json_CV_info)
                keywords_Offre = keywords_processing_offre(json_offre_info)

                CV_inter = list(set(keywords_Cv) & set(keywords_Offre))
                similarity = calculating_similarity(" ".join(CV_inter), " ".join(keywords_Offre))
                print(similarity)
            # Affichage résultats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🕒 Expérience", f"{experience} an(s)")
            with col2:
                st.metric("🔍 Score de Similarité", f"{similarity.item():.2f} %")

            st.write("✅ Mots-clés en commun :", ", ".join(CV_inter) if CV_inter else "Aucun mot-clé commun trouvé.")

            with st.expander("📘 Détails JSON du CV analysé"):
                st.json(json_CV_info)
            with st.expander("📗 Détails JSON de l'offre analysée"):
                st.json(json_offre_info)

    elif not offre_text:
        st.info("📌 Veuillez d'abord coller une offre d'emploi.")

    elif not uploaded_files:
        st.info("📌 Veuillez sélectionner au moins un CV en PDF.")
if __name__ == "__main__":
    main()