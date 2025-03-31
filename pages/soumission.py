import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout="wide")
st.sidebar.image('images/car.jpeg')

st.markdown("## 📊 Prédiction sur un fichier Excel")
st.markdown("Cette section permet de prédire automatiquement la fraude pour un fichier Excel contenant des sinistres.")
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #6A0DAD, #9C27B0);
            color: white;
        }
        h2, .main-title {
            color: #6A0DAD !important;
        }
        .stFileUploader {
            background-color: #f3e5f5 !important;
            border: 2px solid #BA68C8 !important;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)
vars_ = ['Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea',
         'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'Sex',
         'MaritalStatus', 'Age', 'Fault', 'PolicyType', 'VehicleCategory',
         'VehiclePrice', 'Deductible', 'DriverRating',
         'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims',
         'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled',
         'WitnessPresent', 'AgentType', 'NumberOfSuppliments',
         'AddressChange_Claim', 'NumberOfCars', 'Year', 'BasePolicy']

z, y = st.columns([2, 1])
a = z.radio('Sélection du modèle', ['RandomForest.pkl'], horizontal=True)

model_file = y.file_uploader("📥 Charger un fichier Excel (.xlsx)", type=['xlsx'])

if model_file:
    models = joblib.load(f'models/{a}')
    base = pd.read_excel(model_file)

    # Sélection uniquement des colonnes nécessaires
    if not set(vars_).issubset(base.columns):
        st.error("⚠️ Le fichier ne contient pas toutes les colonnes requises.")
    else:
        # Assurer que toutes les colonnes ont les bons types
        base = base.copy()
        for col in vars_:
            # Appliquer les types de l'exemple (optionnel si tu as un fichier 'example' pour référence)
            if col in base.columns:
                base[col] = base[col].astype(str) if base[col].dtype == object else base[col]

        try:
            base['prediction'] = models.predict(base[vars_])
            base['proba_fraude'] = models.predict_proba(base[vars_])[:, 1]
            st.success("✅ Prédictions réalisées avec succès")
            st.dataframe(base)
        except ValueError as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.info("Cela peut venir de valeurs non reconnues par le modèle dans certaines colonnes.")