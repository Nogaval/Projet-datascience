import streamlit as st
import pandas as pd
import numpy as np
import os
from functions.mod1 import example
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.sidebar.image('images/car.jpeg')
st.markdown("<h1 style='text-align: left;color:white;'>DETECTION DE FRAUDE EN ASSURANCE AUTO</h1>", unsafe_allow_html=True)
st.sidebar.image('images/car.jpeg')

# CSS pour personnaliser l'apparence
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }

        .title {
            text-align: center;
            color: #6A0DAD;
            font-size: 40px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom, #6A0DAD, #9C27B0);
            color: white;
        }

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

        div[data-baseweb="select"] {
            width: 100% !important;
            margin-bottom: 20px !important;
        }

        .stSelectbox, .stNumberInput {
            font-size: 16px !important;
        }

        div[data-baseweb="radio"] {
            padding-top: 5px;
            padding-bottom: 5px;
            font-size: 16px;
        }

        .stButton > button {
            font-size: 18px;
            padding: 12px 35px;
            background-color: #E91E63 !important;
            border-radius: 10px;
            border: none;
            color: white;
            font-weight: 600;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        }

        .stButton > button:hover {
            background-color: #AD1457 !important;
            color: #fff;
            transition: 0.3s ease;
        }
    </style>
""", unsafe_allow_html=True)

# Créer un dataframe de prédiction
s5, s_form = st.columns([2, 3])

with s_form:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            Month = st.selectbox('Month', ['Dec', 'Jan', 'Oct', 'Jun', 'Feb', 'Nov', 'Apr', 'Mar', 'Aug','Jul', 'May', 'Sep'])
            WeekOfMonth = st.selectbox('WeekOfMonth', [1, 2, 3, 4, 5])
            DayOfWeek = st.selectbox('DayOfWeek', ['Wednesday', 'Friday', 'Saturday', 'Monday', 'Tuesday', 'Sunday','Thursday'])
            Make = st.selectbox('Make', ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac','Accura', 'Dodge', 'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab','Saturn', 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'])
            AccidentArea = st.selectbox('AccidentArea', ['Urban', 'Rural'])
            DayOfWeekClaimed = st.selectbox('DayOfWeekClaimed', ['Tuesday', 'Monday', 'Thursday', 'Friday', 'Wednesday', 'Saturday','Sunday'])
            MonthClaimed = st.selectbox('MonthClaimed', ['Jan', 'Nov', 'Jul', 'Feb', 'Mar', 'Dec', 'Apr', 'Aug', 'May','Jun', 'Sep', 'Oct', '0'])
            Age = st.number_input('Age', min_value=15)

        with col2:
            WeekOfMonthClaimed = st.selectbox('WeekOfMonthClaimed', [1, 2, 3, 4, 5])
            MaritalStatus = st.selectbox('MaritalStatus', ['Single', 'Married', 'Widow', 'Divorced'])
            Fault = st.selectbox('Fault', ['Policy Holder', 'Third Party'])
            Sex = st.radio('Sex', ['Female', 'Male'], horizontal=True)
            PolicyType = st.selectbox('PolicyType', ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability','Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision','Utility - Collision', 'Utility - Liability', 'Sport - All Perils'])
            VehicleCategory = st.selectbox('VehicleCategory', ['Sport', 'Utility', 'Sedan'])
            PoliceReportFiled = st.radio('PoliceReportFiled', ['Yes', 'No'], horizontal=True)
            WitnessPresent = st.radio('WitnessPresent', ['Yes', 'No'], horizontal=True)

        with col3:
            AgentType = st.radio('AgentType', ['Yes', 'No'], horizontal=True)
            VehiclePrice = st.selectbox('VehiclePrice', ['more than 69000', '20000 to 29000', '30000 to 39000','less than 20000', '40000 to 59000', '60000 to 69000'])
            Deductible = st.selectbox('Deductible', [300, 400, 500, 700])
            DriverRating = st.selectbox('DriverRating', [1, 4, 3, 2])
            Days_Policy_Accident = st.selectbox('Days_Policy_Accident', ['more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'])
            Days_Policy_Claim = st.selectbox('Days_Policy_Claim', ['more than 30', '15 to 30', '8 to 15', 'none'])
            PastNumberOfClaims = st.selectbox('PastNumberOfClaims', ['none', '1', '2 to 4', 'more than 4'])
            AgeOfVehicle = st.selectbox('AgeOfVehicle', ['3 years', '6 years', '7 years', 'more than 7', '5 years', 'new','4 years', '2 years'])
            AgeOfPolicyHolder = st.selectbox('AgeOfPolicyHolder', ['26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25','36 to 40', '16 to 17', 'over 65', '18 to 20'])
            NumberOfSuppliments = st.selectbox('NumberOfSuppliments', ['none', 'more than 5', '3 to 5', '1 to 2'])
            AddressChange_Claim = st.selectbox('AddressChange_Claim', ['1 year', 'no change', '4 to 8 years', '2 to 3 years','under 6 months'])
            NumberOfCars = st.selectbox('NumberOfCars', ['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'])
            BasePolicy = st.selectbox('BasePolicy', ['Liability', 'Collision', 'All Perils'])
            Year = st.selectbox('Year', [1994, 1995, 1996])

        model = st.selectbox('Sélectionner un modèle', os.listdir('models'))
        submit_button = st.form_submit_button('Prédire la fraude')

# On exécute le modèle si submit
if submit_button:
    vect = [Month, WeekOfMonth, DayOfWeek, Make, AccidentArea,
            DayOfWeekClaimed, MonthClaimed, WeekOfMonthClaimed, Sex,
            MaritalStatus, Age, Fault, PolicyType, VehicleCategory,
            VehiclePrice, Deductible, DriverRating, Days_Policy_Accident,
            Days_Policy_Claim, PastNumberOfClaims, AgeOfVehicle, AgeOfPolicyHolder,
            PoliceReportFiled, WitnessPresent, AgentType, NumberOfSuppliments,
            AddressChange_Claim, NumberOfCars, Year, BasePolicy]

    pred = pd.DataFrame([vect], columns=example.columns)
    models = joblib.load(f'models/{model}')

    prediction = models.predict(pred)[0]
    prediction_num = 1 if prediction.lower() == 'yes' else 0
    proba = models.predict_proba(pred)[0][prediction_num]

    pourcent = str(round(proba * 100, 2)) + ' %'
    message = ['Ce sinistre n\'est pas un cas de fraude ', 'Ca pourrait être une fraude'][prediction_num] + ' avec une probabilité de ' + pourcent

    s5.markdown(f"<h1 style='text-align: center; color: purple;'>{message}</h1>", unsafe_allow_html=True)
    s5.image('images/pasfraudeur.jpg' if prediction_num == 0 else 'images/fraudeur.jpeg', width=550)