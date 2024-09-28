# src/interferences/logic.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def detect_interferences(df):
    """
    Detecta interferencias en el espectro de la señal.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.

    Returns:
    - interference_list: list of dict
        Lista de diccionarios con información de las interferencias detectadas.
    """
    # Placeholder para la detección real de interferencias
    # Por ahora, simulamos algunas interferencias

    # Simulamos que las interferencias son los puntos donde la amplitud supera un umbral
    threshold = df['Amplitude'].mean() + 2 * df['Amplitude'].std()
    interference_df = df[df['Amplitude'] > threshold]

    interference_list = []
    for index, row in interference_df.iterrows():
        interference = {
            'Frequency': row['Frequency'],
            'Amplitude': row['Amplitude'],
            'Type': 'Interferencia detectada'  # Placeholder
        }
        interference_list.append(interference)

    return interference_list

def generate_interference_table(interference_list):
    """
    Genera una tabla Dash con las interferencias detectadas.

    Parameters:
    - interference_list: list of dict
        Lista de diccionarios con información de las interferencias detectadas.

    Returns:
    - table: dash_table.DataTable
        Tabla con las interferencias.
    """
    from dash import dash_table

    if not interference_list:
        return html.P("No se detectaron interferencias.")

    df_interferences = pd.DataFrame(interference_list)

    table = dash_table.DataTable(
        data=df_interferences.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df_interferences.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    )

    return table

def generate_interference_spectrum(df, interference_list):
    """
    Genera una figura de Plotly con el espectro de la señal y las interferencias marcadas.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.
    - interference_list: list of dict
        Lista de interferencias detectadas.

    Returns:
    - fig: plotly.graph_objs._figure.Figure
        Figura del espectro con interferencias marcadas.
    """
    fig = go.Figure()
    # Agregar la señal original
    fig.add_trace(go.Scatter(
        x=df['Frequency'],
        y=df['Amplitude'],
        mode='lines',
        name='Señal Original'
    ))

    # Marcar las interferencias
    if interference_list:
        frequencies = [interf['Frequency'] for interf in interference_list]
        amplitudes = [interf['Amplitude'] for interf in interference_list]
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=amplitudes,
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Interferencias'
        ))

    fig.update_layout(
        title='Espectro de la Señal con Interferencias Marcadas',
        xaxis_title='Frecuencia (Hz)',
        yaxis_title='Amplitud (dBm)'
    )

    return fig

def apply_filters(df):
    """
    Aplica filtros para eliminar las interferencias detectadas.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.

    Returns:
    - filtered_df: pandas.DataFrame
        DataFrame con las interferencias eliminadas.
    """
    # Placeholder para la aplicación real de filtros
    # Por ahora, eliminamos los puntos donde la amplitud supera un umbral
    threshold = df['Amplitude'].mean() + 2 * df['Amplitude'].std()
    filtered_df = df[df['Amplitude'] <= threshold]
    return filtered_df
