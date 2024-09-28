# src/app.py

from flask import Flask, send_from_directory
from dash import Dash
import dash_bootstrap_components as dbc

# Inicializar la aplicación Dash con configuraciones
server = Flask(__name__)
app = Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Establecer el título de la aplicación
app.title = "Análisis de Señales de RF"

# Exponer el servidor Flask subyacente
server = app.server

# Ruta para descargar el reporte
@server.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('reports', filename, as_attachment=True)
