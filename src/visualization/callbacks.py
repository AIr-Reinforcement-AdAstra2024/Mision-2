# src/visualization/callbacks.py

from dash import Input, Output, State, callback, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

from app import app  # Importar la instancia de la aplicación
from .logic import (
    generate_spectrogram,
    generate_temporal_spectrum_analysis,
    generate_comparison_graphs,
)

@callback(
    Output('visualization-content', 'children'),
    Input('stored-data', 'data')
)
def update_visualization(data_json):
    """
    Callback para actualizar el contenido de la página de visualización.

    Parameters:
    - data_json: str (JSON)
        Datos almacenados en 'stored-data', en formato JSON.

    Returns:
    - content: list of Dash components
        Contenido a mostrar en la página, ya sea la alerta o las visualizaciones.
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

        # Generar espectrograma
        spectrogram_fig = generate_spectrogram(df)

        # Generar análisis de espectro temporal
        temporal_spectrum_fig = generate_temporal_spectrum_analysis(df)

        # Generar gráficos de comparación antes y después del filtrado
        comparison_figs = generate_comparison_graphs(df)

        # Construir el contenido completo
        content = [
            # Espectrograma
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Espectrograma (Waterfall Display)"),
                            dcc.Graph(figure=spectrogram_fig),
                        ],
                        width=12,
                    ),
                ],
                className='mb-4',
            ),
            # Análisis de espectro temporal
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Análisis de Espectro Temporal"),
                            dcc.Graph(figure=temporal_spectrum_fig),
                        ],
                        width=12,
                    ),
                ],
                className='mb-4',
            ),
            # Comparación antes y después del filtrado
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Comparación Antes y Después del Filtrado"),
                            dcc.Graph(figure=comparison_figs['before']),
                            dcc.Graph(figure=comparison_figs['after']),
                        ],
                        width=12,
                    ),
                ],
                className='mb-4',
            ),
        ]

        return content
