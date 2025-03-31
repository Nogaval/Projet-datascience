import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards
from functions.mod1 import *
import plotly.subplots as sp
import os
st.set_page_config(layout="wide")
#st.image('images/car.jpeg')
st.sidebar.image('images/car.jpeg')
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



data = pd.read_csv( os.path.join(os.getcwd(), 'data', 'fraude_oracle.csv'),sep=';')
data.colums=data.columns.str.strip()
#data['DistanceFromHome'] = data['DistanceFromHome'].fillna(np.median(data['DistanceFromHome'].dropna()))
# Education_codes = {1: 'inférieur au collège', 2: 'collège', 3: 'licence', 4: 'master', 5: 'docteur'}
# EnvironmentSatisfaction_codes = {1: 'faible', 2: 'moyen', 3: 'élevée', 4: 'très élevée'}
# EvaluationPerformance_codes = {1: 'faible', 2: 'bon', 3: 'excellent', 4: 'exceptionnel'}
# ImplicationDansEmploi_codes = {1: 'très peu impliqué', 2: 'peu impliqué', 3: 'impliqué', 4: 'très impliqué', 5: 'exceptionnellement impliqué'}
# JobLevel_codes = {1: 'bas', 2: 'intermédiaire', 3: 'supérieur', 4: 'haut', 5: 'exceptionnel'}
# SatisfactionRelationnelle_codes = {1: 'faible', 2: 'moyen', 3: 'élevée', 4: 'très élevée'}
# SatisfactionTravail_codes = {1: 'faible', 2: 'moyen', 3: 'élevée', 4: 'très élevée'}
# StockOptionLevel_codes = {0: "pas d'option", 1: 'standard', 2: 'élevé', 3: 'exceptionnel '}
# WorkLifeBalance_codes = {1: 'mauvais', 2: 'bon', 3: 'excellent', 4: 'très élevé'}

Data_ = data.copy()
# Data_.Education = Data_.Education.replace(Education_codes)
# Data_.EnvironmentSatisfaction = Data_.EnvironmentSatisfaction.replace(EnvironmentSatisfaction_codes)
# Data_.Évaluation_performance = Data_.Évaluation_performance.replace(EvaluationPerformance_codes)
# Data_.Implication_dans_emploi = Data_.Implication_dans_emploi.replace(ImplicationDansEmploi_codes)
# Data_.JobLevel = Data_.JobLevel.replace(JobLevel_codes)
# Data_.Satisfaction_relationnelle = Data_.Satisfaction_relationnelle.replace(SatisfactionRelationnelle_codes)
# Data_.Satisfaction_travail = Data_.Satisfaction_travail.replace(SatisfactionTravail_codes)
# Data_.StockOptionLevel = Data_.StockOptionLevel.replace(StockOptionLevel_codes)
#Data_['Age_Cor'] = Data_["Age"]
#Data_=fix_age_cor(Data_)
#Data_.drop(columns={'PolicyNumber', 'RepNumber'}, inplace=True)


####partie metric
style_metric_cards('purple',border_radius_px=10,box_shadow=False)

s1,s2,s3 = st.columns(3)
s1.metric("Total Souscripteur",data.shape[0])
s2.metric("% Fraudeur",round(data['FraudFound_P'].value_counts(True)[1]*100,3))
s3.metric("Age median des Souscripteur",np.median(Data_['Age']))


st.markdown('<br>',unsafe_allow_html=True)

####partie visualisation

#### piechart

s1,s2 = st.columns([1,2])

a = s1.selectbox('',['FraudFound_P','Make','AccidentArea','Sex','MaritalStatus'])
if a == 'FraudFound_P' :
    fig = graphe_pie(Data_,'FraudFound_P')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition de fraudeurs </b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'Make' :
    fig = graphe_pie(Data_,'Make')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des fraudeurs suivant le <br> status Make d\'Affaire </b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'AccidentArea' :
    fig = graphe_pie(Data_,'AccidentArea')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des fraudeurs par <br> AccidentArea </b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'Sex' :
    fig = graphe_pie(Data_,'Sex')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des fraudeurs  par <br>Sex </b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)
if a == 'MaritalStatus' :
    fig = graphe_pie(Data_,'MaritalStatus')
    fig = fig.update_layout(
        height=500,
        title = dict(text='<b> Repartition  des employees  par <br>status MaritalStatus</b>',x=0.1,font_color="green",font_size=24)
    )
    s1.plotly_chart(fig)



#### barplot
b = s2.selectbox('',['Fault','PolicyType','VehicleCategory','VehiclePrice','Deductible','DriverRating'])

fig =  graph_bar_uni(Data_,b)
fig = fig.update_layout(
    height=500,
    title = dict(text=f'<b> Repartition  des fraudes  par status de {b} </b>',x=0.1,font_color="pink",font_size=24),
    xaxis=dict(showgrid=False,title="",color="white",showticklabels=False),yaxis = dict(title='')
)
s2.plotly_chart(fig)


###barplot bivariee
s1,s2 = st.columns([1,2])

c= s1.selectbox('',['PastNumberOfClaims',
 'AgeOfVehicle',
 'AgeOfPolicyHolder',
 'PoliceReportFiled',
 'WitnessPresent',
 'AgentType',
 'NumberOfSuppliments',
 'AddressChange_Claim',
 'NumberOfCars'])
s1.plotly_chart(bar_bi_plot(Data_,c))


quant_features = ['Age']

d = s2.selectbox('',quant_features)

s2.plotly_chart(plot_histogram_with_density(df= Data_ ,variable=d))

n = len(quant_features)
n_cols = 7
n_rows = (n + n_cols - 1) // n_cols

fig = sp.make_subplots(rows=n_rows,
                       cols=n_cols,
                       subplot_titles=['FraudFound vs<br> ' + i for i in quant_features],
                       horizontal_spacing=0.01,vertical_spacing=0.10
                       )
for idx, i in enumerate(quant_features):
    row = idx // n_cols + 1
    col = idx % n_cols + 1


    fig1 = px.box(x=Data_[i],color=Data_['FraudFound_P'] )
    fig1.update_layout(
      height=500 ,
      template='plotly_dark'
      )


    for trace in fig1.data:
        fig.add_trace(trace, row=row, col=col)
    
    fig.update_yaxes(showgrid=False,title="",color="white",showticklabels=False, row=row, col=col)
    fig.update_xaxes(showgrid=False,title="",color="white",showticklabels=False, row=row, col=col)
    
fig.update_layout(
    height=400 * n_rows,
    template='plotly_dark',showlegend=True,
    xaxis=dict(showgrid=False, color='white',
               showticklabels=False), boxmode='group'
)

st.plotly_chart(fig)