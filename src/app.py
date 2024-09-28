# src/app.py

import dash
import dash_bootstrap_components as dbc

# Inicializar la aplicación Dash con configuraciones
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Establecer el título de la aplicación
app.title = "Análisis de Señales de RF"

# Exponer el servidor Flask subyacente
server = app.server
