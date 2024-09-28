# src/home/logic.py

import pandas as pd

def read_csv_content(contents, filename):
    """
    Función para leer el contenido del archivo CSV cargado.

    Parameters:
    - contents: str
        Contenido del archivo en formato base64.
    - filename: str
        Nombre del archivo cargado.

    Returns:
    - df: pandas.DataFrame
        DataFrame con los datos del archivo CSV.
    """
    # Por el momento, devolver un DataFrame de prueba
    data = {
        'Frequency': [100, 200, 300, 400, 500],
        'Amplitude': [-50, -45, -55, -60, -58]
    }
    df = pd.DataFrame(data)
    return df

    # Código original para leer el CSV (comentado por ahora)
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Asumimos que es un archivo CSV
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    except Exception as e:
        raise ValueError(f"Error al leer el archivo CSV: {str(e)}")
    """
