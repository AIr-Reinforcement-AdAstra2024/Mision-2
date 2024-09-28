# src/signal_analysis/layout.py

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("Análisis de Señal"),
    html.Hr(),
    # Contenido que se actualizará dinámicamente
    html.Div(id='signal-analysis-content'),
])
