# src/signal_analysis/callbacks.py

from dash import Input, Output, callback, html
import dash_bootstrap_components as dbc
import pandas as pd

from app import app  # Importar la instancia de la aplicación
from utils.calculations import calculate_parameters

# Callback para actualizar el contenido del análisis de señal
@callback(
    Output('signal-analysis-content', 'children'),
    Input('stored-data', 'data')
)
def update_signal_analysis(data_json):
    if data_json is None:
        return dbc.Alert("No hay datos cargados. Por favor, sube un archivo CSV en la sección Inicio.", color="warning")
    else:
        df = pd.read_json(data_json, orient='split')
        # Realizar los cálculos necesarios
        parameters = calculate_parameters(df)
        # Crear componentes para mostrar los resultados
        results = []
        for param_name, param_value in parameters.items():
            results.append(
                dbc.Col(html.Div([
                    html.H5(param_name),
                    html.P(f"{param_value}"),
                ]), width=4)
            )
        # Devolver el contenido
        return dbc.Row(results)
