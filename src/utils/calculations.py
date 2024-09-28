# src/utils/calculations.py

def calculate_parameters(df):
    # Implementa los cálculos de los parámetros según la Tabla 1
    # Por ejemplo:
    frequency_central = df['Frequency'].mean()  # Ejemplo simplificado
    bandwidth = df['Frequency'].max() - df['Frequency'].min()
    # Añade más cálculos aquí

    parameters = {
        "Frecuencia Central": f"{frequency_central:.2f} Hz",
        "Ancho de Banda": f"{bandwidth:.2f} Hz",
        # Agrega más parámetros
    }
    return parameters
