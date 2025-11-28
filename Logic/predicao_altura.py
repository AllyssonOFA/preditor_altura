from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

from .definitions import DATASET_URL, MODEL_FILE

def avaliar_modelo(modelo, X, y, k):
    metricas = {
        'RMSE': make_scorer(root_mean_squared_error),
        'MAE': make_scorer(mean_absolute_error),
        'MAPE': make_scorer(mean_absolute_percentage_error),
        'R2': make_scorer(r2_score)
    }

    pontuacao = cross_validate(modelo, X, y, cv=k, scoring=metricas)

    avaliacao = {
        'RMSE': np.mean(pontuacao['test_RMSE']).item(),
        'MAE': np.mean(pontuacao['test_MAE']).item(),
        'MAPE': np.mean(pontuacao['test_MAPE']).item(),
        'R2': np.mean(pontuacao['test_R2']).item(),
    }

    return avaliacao

def treinar_modelo():
    dados = pd.read_csv(DATASET_URL, names=['altura_pai', 'altura_filho'], header=0)

    # de polegadas para metro
    dados = round(dados * 0.0254, 2)

    X = dados[['altura_pai']]
    y = dados['altura_filho']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LinearRegression()
    modelo.fit(X_scaled, y)

    metricas = avaliar_modelo(modelo, X, y, 5)

    return modelo, scaler, metricas

def prever_valor(modelo, scaler, valor):

    nova_altura = pd.DataFrame({
        'altura_pai': [valor]
    })

    nova_altura_scaled = scaler.transform(nova_altura)

    previsao = modelo.predict(nova_altura_scaled)

    return round(previsao[0],2)

def criar_modelo():
    modelo, scaler, metricas = treinar_modelo()

    with open(MODEL_FILE, 'wb') as file:
        pickle.dump([modelo, scaler, metricas], file)
    



if __name__ == '__main__':
    criar_modelo()
    
