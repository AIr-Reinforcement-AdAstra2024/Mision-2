# src/signal_analysis/layout.py

from dash import html
import dash_bootstrap_components as dbc

signal_analysis_layout = dbc.Container([
    html.H2("Análisis de Señal"),
    html.Hr(),
    # Contenido dinámico que se actualizará con el callback
    html.Div(id='signal-analysis-content'),
])
