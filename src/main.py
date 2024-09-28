# src/main.py

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

import base64
import io
import pandas as pd

# Importar los layouts de los módulos
from home.layout import home_layout
from signal_analysis.layout import signal_analysis_layout
from interferences.layout import interferences_layout
from visualization.layout import visualization_layout
from reports.layout import reports_layout
from about.layout import about_layout

# Inicializar la aplicación Dash con un tema de Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Análisis de Señales de RF"

server = app.server  # Necesario para desplegar en servicios como Heroku

# Definir la barra de navegación
navbar = dbc.Navbar(
    dbc.Container(
        [
            # Brand
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("Análisis de Señales de RF", className="ms-2")),
                ],
                align="center",
                className="g-0",
            ),
            # Navegación
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Inicio", href="/", active="exact")),
                    dbc.NavItem(dbc.NavLink("Análisis de Señal", href="/signal-analysis", active="exact")),
                    dbc.NavItem(dbc.NavLink("Interferencias", href="/interferences", active="exact")),
                    dbc.NavItem(dbc.NavLink("Visualización", href="/visualization", active="exact")),
                    dbc.NavItem(dbc.NavLink("Reportes", href="/reports", active="exact")),
                    dbc.NavItem(dbc.NavLink("Acerca de", href="/about", active="exact")),
                ],
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)

# Definir el layout de la aplicación
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dcc.Store(id='stored-data', storage_type='session'),  # Almacenar datos en la sesión
    html.Div(id='page-content')
])

# Callback para actualizar el contenido de la página según la URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return home_layout
    elif pathname == '/signal-analysis':
        return signal_analysis_layout
    elif pathname == '/interferences':
        return interferences_layout
    elif pathname == '/visualization':
        return visualization_layout
    elif pathname == '/reports':
        return reports_layout
    elif pathname == '/about':
        return about_layout
    else:
        return dbc.Container(
            [
                html.H1("404: Página no encontrada", className="text-danger"),
                html.Hr(),
                html.P(f"La ruta {pathname} no existe."),
            ],
            className="py-3",
        )
    
# Callback para procesar el archivo subido
@app.callback(
    [Output('stored-data', 'data'),
     Output('upload-message', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def process_upload(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Asumimos que es un archivo CSV
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            # Aquí puedes realizar validaciones adicionales
            # Almacenar los datos en formato JSON
            data_json = df.to_json(date_format='iso', orient='split')
            # Mensaje de éxito
            success_msg = dbc.Alert("Archivo cargado exitosamente.", color="success")
            return data_json, success_msg
        except Exception as e:
            # Mensaje de error
            error_msg = dbc.Alert(f"Hubo un error al procesar el archivo: {str(e)}", color="danger")
            return None, error_msg
    else:
        return None, ''
    
@app.callback(
    Output('signal-analysis-content', 'children'),
    Input('stored-data', 'data')
)
def update_signal_analysis(data_json):
    if data_json is None:
        return dbc.Alert("No hay datos cargados. Por favor, sube un archivo CSV en la sección Inicio.", color="warning")
    else:
        df = pd.read_json(data_json, orient='split')
        # Realizar los cálculos necesarios
        frequency = calculate_frequency(df)
        bandwidth = calculate_bandwidth(df)
        # Más cálculos...

        # Crear componentes para mostrar los resultados
        results = dbc.Row([
            dbc.Col(html.Div([
                html.H5("Frecuencia Central"),
                html.P(f"{frequency} Hz"),
            ]), width=4),
            dbc.Col(html.Div([
                html.H5("Ancho de Banda"),
                html.P(f"{bandwidth} Hz"),
            ]), width=4),
            # Más columnas con otros parámetros
        ])

        # Devolver el contenido
        return dbc.Container([results])

if __name__ == '__main__':
    app.run_server(debug=True)
