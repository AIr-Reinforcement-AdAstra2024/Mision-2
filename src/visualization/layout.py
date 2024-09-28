# src/visualization/layout.py

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("Visualización Avanzada", style={'margin-top': '20px'}),
    html.Hr(),
    # Contenedor para el contenido dinámico
    html.Div(id='visualization-content'),
])
