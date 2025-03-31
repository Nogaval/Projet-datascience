
import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Fraude Assurance", layout="wide")

# CSS pour personnaliser l'apparence
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }

        /* TITRE PRINCIPAL */
        .title {
            text-align: center;
            color: #6A0DAD;
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* BOÎTES D'INFORMATIONS */
        .box {
            border: 2px solid #6A0DAD;
            background: linear-gradient(to right, #f3e5f5, #e1bee7);
            color: #4A0072;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
            font-size: 18px;
            line-height: 1.6;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #6A0DAD, #9C27B0);
            color: white;
        }

        /* CONTENU DE LA SIDEBAR */
        section[data-testid="stSidebar"] .stButton>button {
            background-color: #9C27B0;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            border: 2px solid white;
        }

        section[data-testid="stSidebar"] .stButton>button:hover {
            background-color: #BA68C8;
        }

        section[data-testid="stSidebar"] .stTextInput>div>div>input {
            background-color: white;
            color: black;
            border-radius: 8px;
        }

        /* IMAGE EN HAUT À GAUCHE */
        .logo-container {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 999;
        }

    </style>
""", unsafe_allow_html=True)

# Ajouter le logo dans le coin supérieur gauche de la page
st.image("images/euria.png", width=150)  # Affichage du logo avec une largeur de 150px

# Titre principal stylisé
st.markdown("<div class='title'>🚗 Détection de Fraude en Assurance Automobile 🚗</div>", unsafe_allow_html=True)

# Espacement
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar avec image et texte
st.sidebar.image('images/car.jpeg')
st.sidebar.markdown("## 🚘 Bienvenue dans l'application de détection de fraude !")

# Ajout d'une section interactive dans la sidebar
option = st.sidebar.selectbox(
    "🔍 Sélectionnez une action :", 
    ["Accueil", "Analyse des données", "Modèle prédictif", "Résultats"]
)

st.sidebar.markdown("### 📌 Informations complémentaires")
#st.sidebar.text("Auteur : [Votre Nom]")
st.sidebar.text("Date : 2025")

# Contexte de l'étude (avec une box élégante)
st.markdown("""
    <div class="box">
        <h3>📌 CONTEXTE DE L'ÉTUDE</h3>
        La fraude à l'assurance automobile consiste à présenter des demandes d'indemnisation fausses ou exagérées concernant des dommages matériels ou corporels après un accident.
        Parmi les fraudes courantes, on trouve :
        <ul>
            <li>🚘 <b>Accidents mis en scène</b> : Les fraudeurs provoquent volontairement un accident.</li>
            <li>👥 <b>Passagers fantômes</b> : Des personnes prétendent être blessées alors qu'elles n'étaient pas présentes.</li>
            <li>⚠️ <b>Exagération des dommages</b> : Amplifier les blessures ou les réparations nécessaires.</li>
        </ul>
    </div>
    <br>
""", unsafe_allow_html=True)

# Objectifs de l'application (avec une autre box élégante)
st.markdown("""
    <div class="box">
        <h3>🎯 OBJECTIFS DE L'APPLICATION</h3>
        <ul>
            <li>📊 <b>Analyser les Données :</b> Collecter et prétraiter des données sur les assurés et les réclamations.</li>
            <li>🤖 <b>Développer un Modèle Prédictif :</b> Utiliser le Machine Learning pour détecter la fraude.</li>
            <li>🔍 <b>Interpréter les Résultats :</b> Comprendre les facteurs clés de la fraude.</li>
            <li>✅ <b>Proposer des Solutions :</b> Formuler des recommandations adaptées.</li>
        </ul>
    </div>
    <br>
""", unsafe_allow_html=True)
