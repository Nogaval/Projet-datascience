import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import plotly.express as px
import numpy as np


def table(Data_,var):
    inter = Data_[var].value_counts().to_frame().reset_index()
    inter[var] = inter[var].astype(str)
    inter['pourcent']=np.round(Data_[var].value_counts(normalize=True).to_frame().reset_index()['proportion']*100,2)
    inter['text']='n= ' + inter['count'].astype(str) + '  <br> percent =  ' + inter['pourcent'].astype(str) + '%'
    return inter

def graph_bar_uni(Data_,var):
    title = 'repartition de la variable ' + var
    inter = table(Data_,var).sort_values(by='count')
    return px.bar(y=inter[var],x=inter["count"],color=inter[var],text=inter['text'])


def graphe_pie(Data_,variable,size=10):
    titre="Répartion de la variable  " + variable
    fig = px.pie(Data_[variable].value_counts().to_frame().reset_index(),names=variable,
             values="count",hole=0.5,height=350,title=titre)
    fig.update_traces(textinfo='label+percent+value',showlegend=False)
    fig.update_layout(title=dict(font_color="white"))
    return fig


def bar_bi_plot(Data_,i):
    s = pd.crosstab(Data_['FraudFound_P'], Data_[i])
    s = np.round(s.divide(s.sum(axis=1) / 100, axis=0).unstack().reset_index().rename(columns={0: 'value'}), 2)


    bar_fig = px.bar(
        x=s['FraudFound_P'].astype(str),
        y=s['value'],
        color=s[i].astype(str),
        text=s[i].astype(str) + ' :  ' + s['value'].astype(str) ,
        barmode='relative',title='FraudFound_P vs  ' + str(i)
    )
    bar_fig.update_layout(showlegend=False,height=480,
                          xaxis=dict(showgrid=False,title="",color="white"),
                          yaxis=dict(showgrid=False,title="",color="white",showticklabels=False),
                          title = dict(x=0.3,font_color="pink",font_size=24),
                          margin=dict(t=34,l=0,r=0,b=5))
    return bar_fig


def plot_histogram_with_density(df, variable):
    
    
    if variable not in df.columns:
        raise ValueError(f"La variable '{variable}' n'existe pas dans la DataFrame.")
    
    histogram = go.Histogram(
        x=df[variable],
        histnorm='probability density', 
        nbinsx=30,
        name='Histogramme',
    )
    x = np.linspace(df[variable].min(), df[variable].max(), 100)
    y = stats.gaussian_kde(df[variable])(x)

    density_curve = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Courbe de Densité',
        line=dict(color='white')
    )

    
    fig = go.Figure(data=[histogram, density_curve])

    
    fig.update_layout(
        barmode='overlay',
        template='plotly_dark',
        xaxis=dict(showgrid=False,title="",color="white"),
        yaxis=dict(showgrid=False,title="",color="white",showticklabels=False),showlegend=False,height=480,
        title = dict(x=0.3,font_color="green",font_size=24,text=f'Distribution de  la variable    {variable}')
    )

    
    return fig


def parse_age_range(age_str):
    """
    Cette fonction prend en entrée la chaîne de caractères d'une classe d'âge
    (par ex. '16 to 17', 'over 65', 'under 16') et renvoie un âge "central".
    """
    if pd.isnull(age_str):
        return np.nan
    
    # Nettoyage basique
    age_str = age_str.strip().lower()

    # Cas: "16 to 17", "31 to 35", etc.
    if 'to' in age_str:
        parts = age_str.split('to')
        if len(parts) == 2:
            try:
                lower = float(parts[0])
                upper = float(parts[1])
                return (lower + upper) / 2
            except ValueError:
                return np.nan
    
    # Cas: "over 65"
    elif 'over' in age_str:
        try:
            # On enlève le mot 'over' et on convertit en float
            base_age = float(age_str.replace('over', '').strip())
            return base_age + 1  # Règle possible: "over 65" => 66
        except ValueError:
            return np.nan
    # Si rien ne correspond, on renvoie NaN
    return np.nan


def fix_age_cor(df):
    """
    Cette fonction prend un DataFrame en argument. 
    Pour toutes les lignes où Age_Cor == 0, 
    on remplace cette valeur par le "centre" de la classe dans AgeOfPolicyHolder.
    La fonction modifie le DataFrame en place et le retourne.
    """
    mask = (df['Age_Cor'] == 0) & (df['AgeOfPolicyHolder'].notnull())
    df.loc[mask, 'Age'] = df.loc[mask, 'AgeOfPolicyHolder'].astype(str).apply(parse_age_range)
    return df



example = pd.DataFrame({'Month': {0: 'Jan'}, 'WeekOfMonth': {0: '2'}, 'DayOfWeek': {0: 'Wednesday'}, 'Make': {0: 'Mazda'}, 'AccidentArea': {0: 'Urban'}, 'DayOfWeekClaimed': {0: 'Monday'}, 'MonthClaimed': {0: 'Tuesday'}, 'WeekOfMonthClaimed': {0: '1'}, 'Sex': {0: 'Male'}, 'MaritalStatus': {0: 'Single'}, 'Age': {0: 16.5}, 'Fault': {0: 'Policy Holder'}, 'PolicyType': {0: 'Sport - Liability'}, 'VehicleCategory': {0: 'Sport'}, 'VehiclePrice': {0: 'more than 69000'}, 'Deductible': {0: '400'}, 'DriverRating': {0: '3'}, 'Days_Policy_Accident': {0: 'more than 30'}, 'Days_Policy_Claim': {0: "more than 30"}, 'PastNumberOfClaims': {0: 'none'}, 'AgeOfVehicle': {0: '3 years'}, 'AgeOfPolicyHolder': {0: '26 to 30'}, 'PoliceReportFiled': {0: "No"}, 'WitnessPresent': {0: 'Yes'}, 'AgentType': {0: 'External'}, 'NumberOfSuppliments': {0: 'none'}, 'AddressChange_Claim': {0: '1 year'}, 'NumberOfCars': {0: '1 vehicle'}, 'Year': {0: '1994'}, 'BasePolicy': {0: 'Collision'}})



vars_ = ['Month',
 'WeekOfMonth',
 'DayOfWeek',
 'Make',
 'AccidentArea',
 'DayOfWeekClaimed',
 'MonthClaimed',
 'WeekOfMonthClaimed',
 'Sex',
 'MaritalStatus',
 'Age',
 'Fault',
 'PolicyType',
 'VehicleCategory',
 'VehiclePrice',
 'FraudFound_P',
 'PolicyNumber',
 'RepNumber',
 'Deductible',
 'DriverRating',
 'Days_Policy_Accident',
 'Days_Policy_Claim',
 'PastNumberOfClaims',
 'AgeOfVehicle',
 'AgeOfPolicyHolder',
 'PoliceReportFiled',
 'WitnessPresent',
 'AgentType',
 'NumberOfSuppliments',
 'AddressChange_Claim',
 'NumberOfCars',
 'Year',
 'BasePolicy']