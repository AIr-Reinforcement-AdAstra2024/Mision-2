# src/visualization/logic.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def generate_spectrogram(df):
    """
    Genera un espectrograma (Waterfall display) de la señal.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.

    Returns:
    - fig: plotly.graph_objs._figure.Figure
        Figura del espectrograma.
    """
    # Placeholder para el espectrograma real
    # Supongamos que df tiene columnas 'Time', 'frecuency', 'magnitude'

    # Crear una matriz para el espectrograma
    if 'Time' not in df.columns:
        # Si no hay columna 'Time', simulamos tiempos
        df['Time'] = np.linspace(0, 1, len(df))

    pivot_df = df.pivot(index='frecuency', columns='Time', values='magnitude')

    fig = go.Figure(data=go.Heatmap(
        x=pivot_df.columns,
        y=pivot_df.index,
        z=pivot_df.values,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title='Espectrograma de la Señal',
        xaxis_title='Tiempo (s)',
        yaxis_title='Frecuencia (Hz)'
    )

    return fig

def generate_temporal_spectrum_analysis(df):
    """
    Genera una figura que muestra cómo varía el espectro a lo largo del tiempo.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.

    Returns:
    - fig: plotly.graph_objs._figure.Figure
        Figura del análisis de espectro temporal.
    """
    # Placeholder para el análisis real
    # Podrías mostrar la potencia total vs. tiempo, etc.

    if 'Time' not in df.columns:
        df['Time'] = np.linspace(0, 1, len(df))

    # Calcular la potencia total en cada instante de tiempo
    power_by_time = df.groupby('Time')['magnitude'].sum().reset_index()

    fig = px.line(power_by_time, x='Time', y='magnitude', title='Potencia Total vs. Tiempo')
    fig.update_xaxes(title='Tiempo (s)')
    fig.update_yaxes(title='Potencia (dBm)')

    return fig

def generate_comparison_graphs(df):
    """
    Genera gráficos de comparación antes y después del filtrado.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.

    Returns:
    - figs: dict
        Diccionario con las figuras 'before' y 'after'.
    """
    # Placeholder para los datos filtrados
    # Supongamos que aplicamos un filtrado simple

    # Gráfico antes del filtrado
    fig_before = px.line(df, x='frecuency', y='magnitude', title='Espectro Antes del Filtrado')
    fig_before.update_xaxes(title='Frecuencia (Hz)')
    fig_before.update_yaxes(title='Amplitud (dBm)')

    # Aplicar un filtro simple (por ejemplo, eliminar frecuencias por encima de un umbral)
    threshold = df['magnitude'].mean() + 2 * df['magnitude'].std()
    df_filtered = df[df['magnitude'] <= threshold]

    # Gráfico después del filtrado
    fig_after = px.line(df_filtered, x='frecuency', y='magnitude', title='Espectro Después del Filtrado')
    fig_after.update_xaxes(title='Frecuencia (Hz)')
    fig_after.update_yaxes(title='Amplitud (dBm)')

    figs = {
        'before': fig_before,
        'after': fig_after
    }

    return figs
