# src/reports/logic.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from fpdf import FPDF
import uuid

def get_available_content_options():
    """
    Proporciona las opciones de contenido disponibles para el reporte.

    Returns:
    - options: list of dict
        Lista de opciones para el componente dbc.Checklist.
    """
    options = [
        {'label': 'Parámetros de la Señal', 'value': 'signal_parameters'},
        {'label': 'Parámetros del Enlace', 'value': 'link_parameters'},
        {'label': 'Gráfico del Espectro', 'value': 'spectrum_graph'},
        {'label': 'Espectrograma', 'value': 'spectrogram'},
        {'label': 'Interferencias Detectadas', 'value': 'interferences'},
        {'label': 'Visualizaciones Avanzadas', 'value': 'advanced_visualizations'},
        # Agregar más opciones según sea necesario
    ]
    return options

def generate_report(df, selected_content, comments):
    """
    Genera un reporte en PDF con el contenido seleccionado.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.
    - selected_content: list of str
        Lista de contenidos seleccionados por el usuario.
    - comments: str
        Comentarios adicionales proporcionados por el usuario.

    Returns:
    - report_file_path: str
        Ruta al archivo PDF generado.
    """
    # Crear una instancia de FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título del reporte
    pdf.cell(200, 10, txt="Reporte de Análisis de Señal", ln=True, align='C')
    pdf.ln(10)

    # Agregar comentarios adicionales
    if comments:
        pdf.multi_cell(0, 10, txt=f"Comentarios adicionales:\n{comments}")
        pdf.ln(10)

    # Incluir el contenido seleccionado
    if 'signal_parameters' in selected_content:
        pdf.cell(0, 10, txt="Parámetros de la Señal:", ln=True)
        # Aquí deberías obtener los parámetros de la señal
        pdf.cell(0, 10, txt="(Parámetros de la señal...)", ln=True)
        pdf.ln(5)

    if 'link_parameters' in selected_content:
        pdf.cell(0, 10, txt="Parámetros del Enlace:", ln=True)
        # Aquí deberías obtener los parámetros del enlace
        pdf.cell(0, 10, txt="(Parámetros del enlace...)", ln=True)
        pdf.ln(5)

    if 'spectrum_graph' in selected_content:
        # Generar y guardar el gráfico del espectro
        spectrum_fig = generate_spectrum_figure(df)
        spectrum_fig_path = save_figure(spectrum_fig, 'spectrum_graph.png')
        # Insertar el gráfico en el PDF
        pdf.image(spectrum_fig_path, w=180)
        pdf.ln(5)

    if 'spectrogram' in selected_content:
        # Generar y guardar el espectrograma
        spectrogram_fig = generate_spectrogram_figure(df)
        spectrogram_fig_path = save_figure(spectrogram_fig, 'spectrogram.png')
        # Insertar el gráfico en el PDF
        pdf.image(spectrogram_fig_path, w=180)
        pdf.ln(5)

    # Agregar más contenido según sea necesario

    # Generar un nombre único para el archivo
    report_file_name = f"reporte_{uuid.uuid4()}.pdf"
    report_file_path = os.path.join('reports', report_file_name)

    # Crear el directorio si no existe
    os.makedirs('reports', exist_ok=True)

    # Guardar el PDF
    pdf.output(report_file_path)

    return report_file_path

def generate_spectrum_figure(df):
    """
    Genera la figura del espectro de la señal.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.

    Returns:
    - fig: plotly.graph_objs._figure.Figure
        Figura del espectro.
    """
    fig = px.line(df, x='Frequency', y='Amplitude', title='Espectro de la Señal')
    fig.update_xaxes(title='Frecuencia (Hz)')
    fig.update_yaxes(title='Amplitud (dBm)')
    return fig

def generate_spectrogram_figure(df):
    """
    Genera la figura del espectrograma de la señal.

    Parameters:
    - df: pandas.DataFrame
        DataFrame con los datos de la señal.

    Returns:
    - fig: plotly.graph_objs._figure.Figure
        Figura del espectrograma.
    """
    # Placeholder para el espectrograma real
    if 'Time' not in df.columns:
        df['Time'] = np.linspace(0, 1, len(df))

    pivot_df = df.pivot(index='Frequency', columns='Time', values='Amplitude')

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

def save_figure(fig, filename):
    """
    Guarda una figura de Plotly en un archivo de imagen.

    Parameters:
    - fig: plotly.graph_objs._figure.Figure
        Figura a guardar.
    - filename: str
        Nombre del archivo.

    Returns:
    - filepath: str
        Ruta al archivo guardado.
    """
    filepath = os.path.join('reports', filename)
    os.makedirs('reports', exist_ok=True)
    fig.write_image(filepath)
    return filepath
