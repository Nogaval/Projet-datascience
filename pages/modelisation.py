import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, accuracy_score,
                             precision_score, recall_score, f1_score)
import os
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline

# ------ CONFIG STREAMLIT ------
st.set_page_config(layout="wide")

# ------ BARRE LATÉRALE (SIDEBAR) ------
st.sidebar.image('images/car.jpeg')

# ------ CSS PERSONNALISÉ ------
st.markdown("""
<style>
    /* Forcer le texte du bouton à rester sur une seule ligne */
    .stButton > button {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    /* Importation d'une police */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        background-color: #f8f5fc; /* Fond général blanc-violet très clair */
        color: #4A0072;           /* Texte violet sombre */
    }
    /* TITRE PRINCIPAL */
    .title {
        text-align: center;
        color: #6A0DAD;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* BOX D'INFORMATIONS */
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
    /* SIDEBAR en violet uni avec texte blanc */
    section[data-testid="stSidebar"] {
        background-color: #6A0DAD !important;
        color: white !important;
    }
    /* Boutons dans la sidebar */
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #9C27B0;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        border: 2px solid white;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background-color: #BA68C8;
        color: #fff;
    }
    /* Champ number dans la sidebar */
    section[data-testid="stSidebar"] input[type="number"] {
        background-color: white;
        border: 2px solid #6A0DAD;
        border-radius: 8px;
        padding: 4px;
        font-size: 16px;
        color: black;
    }
    /* Style pour le radio dans le contenu principal */
    div[data-baseweb="radio"] {
        background-color: #f8f5fc;
        border: 2px solid #6A0DAD;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    /* Pour enlever l'alerte label vide */
    label[for=""] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ======= CHARGEMENT ET PRÉPARATION DES DONNÉES =======
data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'fraude_oracle.csv'), sep=';')

categorial_var = ['Days_Policy_Accident','Days_Policy_Claim','PastNumberOfClaims',
                  'AgeOfVehicle','AgeOfPolicyHolder','NumberOfSuppliments','NumberOfCars']

nominal_var = ['Month','WeekOfMonth','DayOfWeek','Make','DayOfWeekClaimed','AccidentArea',
               'DayOfWeekClaimed','MonthClaimed','WeekOfMonthClaimed','Sex','MaritalStatus',
               'Fault','PolicyType','VehicleCategory','PoliceReportFiled','WitnessPresent',
               'AgentType','AddressChange_Claim','Year','BasePolicy']

quant_var = ['Age']

vars_ = ['Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed', 
         'MonthClaimed', 'WeekOfMonthClaimed', 'Sex', 'MaritalStatus', 'Age', 'Fault', 
         'PolicyType', 'VehicleCategory', 'VehiclePrice', 'Deductible', 'DriverRating', 
         'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle', 
         'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 
         'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars', 'Year', 'BasePolicy']

column_transformer = ColumnTransformer(
    transformers=[
        ('categorial_var', OrdinalEncoder(), categorial_var),
        ('nominal_var', OneHotEncoder(drop="first", handle_unknown='ignore'), nominal_var),
        ('quant_var', StandardScaler(), quant_var),
    ]
)

Data = data.copy()
column_transformer.fit_transform(Data[vars_])
list_transforme = column_transformer.get_feature_names_out().tolist()

# ======= SPLIT =======
X = Data[vars_]
Y = Data.FraudFound_P

z, w, y = st.columns([3,1,1])
rs = w.number_input('Random state', min_value=0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=rs)

# Convertir y_test en numérique
y_test = y_test.map({'No': 0, 'yes': 1})

# ======= FONCTIONS UTILES =======
def generer_model(model_name, model, column_transformer=column_transformer):
    pipeline = Pipeline([
        ('scaler', column_transformer),
        (model_name, model)
    ])
    return pipeline

def courbe_roc(model, title='RandomForest'):
    y_probs_train = model.predict_proba(x_train)[:, 1]
    y_probs_test = model.predict_proba(x_test)[:, 1]

    # Convertir labels train/test
    y_train_numeric = y_train.map({'No': 0, 'yes': 1})
    y_test_numeric = y_test

    # ROC train
    fpr_t, tpr_t, thresholds_t = roc_curve(y_train_numeric, y_probs_train, pos_label=1)
    roc_auc_t = roc_auc_score(y_train_numeric, y_probs_train)

    # ROC test
    fpr, tpr, thresholds = roc_curve(y_test_numeric, y_probs_test)
    roc_auc = roc_auc_score(y_test_numeric, y_probs_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_t, y=tpr_t,
                             mode='lines',
                             name=f'Entraînement (AUC = {roc_auc_t:.2f})',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                             mode='lines',
                             name=f'Test (AUC = {roc_auc:.2f})',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Aucune compétence',
                             line=dict(color='grey', dash='dash')))
    fig.update_layout(
        title=dict(x=0.5, font_color="#6A0DAD", text=title),
        xaxis_title='Taux de Faux Positifs',
        yaxis_title='Taux de Vrais Positifs',
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True
    )
    fig.update_xaxes(color='#4A0072')
    fig.update_yaxes(color='#4A0072')
    return fig

def confusion_matri(model, title='Logistic Regression'):
    y_pred_str = model.predict(x_test)
    # Utilisation de convert_label pour gérer chaînes et nombres
    y_pred_numeric = np.array([convert_label(label) for label in y_pred_str])

    cm = confusion_matrix(y_test, y_pred_numeric)

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['No', 'Yes'],
        y=['No', 'Yes'],
        colorscale='Purples',
        showscale=False,
        text=cm,
        texttemplate='%{text}'
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title='Valeurs Prédites', color='#4A0072'),
        yaxis=dict(title='Valeurs Réelles', color='#4A0072'),
        height=500,
        width=500,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

def convert_label(label):
    if isinstance(label, str):
        return 1 if label.lower() == 'yes' else 0
    else:
        return label

def metricss(model, name):
    y_train_numeric = np.array([convert_label(label) for label in y_train])
    y_test_numeric  = np.array([convert_label(label) for label in y_test])
    
    y_pred_train_str = model.predict(x_train)
    y_pred_test_str  = model.predict(x_test)
    y_pred_train_numeric = np.array([convert_label(label) for label in y_pred_train_str])
    y_pred_test_numeric  = np.array([convert_label(label) for label in y_pred_test_str])
    
    train_accuracy  = accuracy_score(y_train_numeric, y_pred_train_numeric)
    train_precision = precision_score(y_train_numeric, y_pred_train_numeric)
    train_recall    = recall_score(y_train_numeric, y_pred_train_numeric)
    train_f1        = f1_score(y_train_numeric, y_pred_train_numeric)
    
    test_accuracy  = accuracy_score(y_test_numeric, y_pred_test_numeric)
    test_precision = precision_score(y_test_numeric, y_pred_test_numeric)
    test_recall    = recall_score(y_test_numeric, y_pred_test_numeric)
    test_f1        = f1_score(y_test_numeric, y_pred_test_numeric)
    
    train_metrics = pd.Series({
        'concern': 'train',
        'name': name,
        'accuracy': train_accuracy,
        'precision': train_precision,
        'recall': train_recall,
        'f1_score': train_f1,
    }).to_frame().T
    
    test_metrics = pd.Series({
        'concern': 'test',
        'name': name,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
    }).to_frame().T
    
    return pd.concat([train_metrics, test_metrics], axis=0)

def roc_confusion(fig2, fig1, title):
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Confusion Matrix of ' + title + '\n',
            'ROC Curve of ' + title
        ),
        column_widths=[0.4, 0.6],
        row_heights=[600],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(
        height=500,
        template='plotly_white',
        title=dict(text=title, x=0.5, font_color='#6A0DAD'),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    fig.update_xaxes(color='#4A0072', row=1, col=1, title_text='Valeurs Prédites')
    fig.update_yaxes(color='#4A0072', row=1, col=1, title_text='Valeurs Réelles')
    fig.update_xaxes(color='#4A0072', row=1, col=2, title_text='Taux de Faux Positifs')
    fig.update_yaxes(color='#4A0072', row=1, col=2, title_text='Taux de Vrais Positifs')
    return fig

# ======= RADIO POUR LE CHOIX DE MODÈLE (deux options uniquement) =======
a = z.radio('', ['RandomForest', 'LogisticRegression'], horizontal=True)

# ======= BOUTON SAUVEGARDE (texte sur une seule ligne grâce au CSS) =======
y.button('Sauvegarder Le model', type='secondary')

if a == 'RandomForest':
    ml_RF = generer_model(
        'RandomForestClassifier',
        RandomForestClassifier(
            class_weight='balanced',
            criterion='gini',
            max_features='sqrt',
            max_depth=7,
            n_estimators=100
        )
    )
    ml_RF.fit(x_train, y_train)

    # Courbe ROC + Matrice de confusion
    fig_conf = confusion_matri(ml_RF, 'RandomForestClassifier')
    fig_roc  = courbe_roc(ml_RF, 'RandomForestClassifier')
    fig_all  = roc_confusion(fig_conf, fig_roc, 'RandomForestClassifier')
    st.plotly_chart(fig_all, use_container_width=True)
    
    # Importance des variables (top 10)
    importances = ml_RF.named_steps['RandomForestClassifier'].feature_importances_
    s = pd.DataFrame(importances, columns=['importance'])
    #s['variable'] = list_transforme
    s['variable'] = list_transforme[:len(s)]
    s = s.sort_values(by='importance', ascending=False).head(10)
    s = s.sort_values(by='importance', ascending=True)
    s['color'] = s['importance'].apply(lambda x: 'purple' if x > 0 else 'gray')
    
    # Graphique en barres pour l'importance des variables
    fig_imp = px.bar(
        x=s['importance'],
        y=s['variable'],
        color=s['color'],
        height=500
    )
    fig_imp.update_layout(
        title=dict(x=0.5, font_color="#6A0DAD", text='Importance des différentes variables'),
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(showgrid=True, color='#4A0072', title="", showticklabels=True),
        yaxis=dict(title='', color='#4A0072')
    )
    
    # Disposition en colonnes
    col1, col2 = st.columns([3, 1])
    col1.plotly_chart(fig_imp, use_container_width=True)
    
    col2.markdown('<br><br>', unsafe_allow_html=True)
    df_metrics = metricss(ml_RF, 'ml_RF_best').reset_index().drop('index', axis=1).transpose()
    df_metrics = df_metrics.astype(str)
    col2.dataframe(df_metrics, height=300, use_container_width=True)

    model = ml_RF
    name = 'RandomForest'
    


# Sauvegarde du modèle
# Sauvegarde du modèle
if y:
    if os.path.exists(f'models/{name}.pkl'):
        os.remove(f'models/{name}.pkl')
        joblib.dump(model, f'models/{name}.pkl')
    else:
        joblib.dump(model, f'models/{name}.pkl')
    st.success(f"Modèle {name} sauvegardé avec succès dans models/{name}.pkl!")
