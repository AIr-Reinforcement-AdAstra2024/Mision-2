# src/interferences/callbacks.py

from dash import Input, Output, State, callback, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

from app import app  # Importar la instancia de la aplicación
from .logic import (
    detect_interferences,
    generate_interference_table,
    generate_interference_spectrum,
    apply_filters,
)

@callback(
    Output('interference-content', 'children'),
    Input('stored-data', 'data'),
    State('stored-data', 'data')
)
def update_interference_analysis(data_json, data_state):
    """
    Callback para actualizar el contenido de la página de interferencias.

    Parameters:
    - data_json: str (JSON)
        Datos almacenados en 'stored-data', en formato JSON.

    Returns:
    - content: list of Dash components
        Contenido a mostrar en la página, ya sea la alerta o los componentes de interferencia.
    """
    if data_json is None:
        # Si no hay datos, mostrar una alerta
        alert_message = dbc.Alert(
            "No hay datos cargados. Por favor, sube un archivo CSV en la sección Inicio.",
            color="warning",
            dismissable=False,
            style={'text-align': 'center'}
        )
        return alert_message
    else:
        # Convertir los datos de JSON a DataFrame
        df = pd.read_json(data_json, orient='split')

        # Detectar interferencias
        interference_list = detect_interferences(df)

        # Generar tabla de interferencias
        interference_table = generate_interference_table(interference_list)

        # Generar espectro con interferencias marcadas
        interference_spectrum_fig = generate_interference_spectrum(df, interference_list)

        # Construir el contenido completo
        content = [
            # Sección para mostrar la tabla de interferencias detectadas
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Interferencias Detectadas"),
                            interference_table,
                        ],
                        width=12,
                    ),
                ],
                className='mb-4',
            ),
            # Gráfico del Espectro con Interferencias
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Espectro con Interferencias Marcadas"),
                            dcc.Graph(figure=interference_spectrum_fig),
                        ],
                        width=12,
                    ),
                ],
                className='mb-4',
            ),
            # Opciones de Filtrado
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Opciones de Filtrado"),
                            # Aquí puedes agregar controles para seleccionar filtros
                            html.Div(id='filter-options'),
                            # Botón para aplicar filtros
                            dbc.Button("Aplicar Filtros", id='apply-filters-btn', color="primary"),
                        ],
                        width=12,
                    ),
                ],
                className='mb-4',
            ),
            # Gráfico del Espectro Filtrado
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Espectro Después del Filtrado"),
                            dcc.Graph(id='filtered-spectrum'),
                        ],
                        width=12,
                    ),
                ],
                className='mb-4',
            ),
        ]

        return content

# Callback para aplicar los filtros y actualizar el espectro filtrado
@callback(
    Output('filtered-spectrum', 'figure'),
    Input('apply-filters-btn', 'n_clicks'),
    State('stored-data', 'data'),
    prevent_initial_call=True
)
def update_filtered_spectrum(n_clicks, data_json):
    if n_clicks is None or data_json is None:
        return {}
    else:
        df = pd.read_json(data_json, orient='split')
        # Aplicar filtros
        filtered_df = apply_filters(df)
        # Generar espectro filtrado
        filtered_spectrum_fig = generate_interference_spectrum(filtered_df, [])
        return filtered_spectrum_fig
