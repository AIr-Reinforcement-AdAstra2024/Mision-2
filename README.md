# Proyecto: Caracterización de Señales de Radiofrecuencia - Reto Codefest Ad Astra 2024

## Tabla de Contenidos

- [Proyecto: Caracterización de Señales de Radiofrecuencia - Reto Codefest Ad Astra 2024](#proyecto-caracterización-de-señales-de-radiofrecuencia---reto-codefest-ad-astra-2024)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [Descripción del Proyecto](#descripción-del-proyecto)
  - [Características](#características)
  - [Requerimientos del Sistema](#requerimientos-del-sistema)
    - [Software](#software)
    - [Hardware](#hardware)
  - [Instrucciones de Instalación](#instrucciones-de-instalación)
    - [1. Clonar el Repositorio](#1-clonar-el-repositorio)
    - [2. Navegar al Directorio del Proyecto](#2-navegar-al-directorio-del-proyecto)
    - [3. Instalar Dependencias](#3-instalar-dependencias)
    - [4. Ejecutar el Software](#4-ejecutar-el-software)
  - [Uso del Software](#uso-del-software)
    - [Video de Demostración](#video-de-demostración)
  - [Arquitectura del Proyecto](#arquitectura-del-proyecto)
    - [Estructura del Proyecto](#estructura-del-proyecto)
    - [Componentes Clave](#componentes-clave)
  - [Pruebas y Ejemplos](#pruebas-y-ejemplos)
    - [Archivos de Prueba](#archivos-de-prueba)
    - [Ejemplo de Ejecución](#ejemplo-de-ejecución)
  - [Colaboradores](#colaboradores)

## Descripción del Proyecto

Este proyecto es parte del **Reto Codefest Ad Astra 2024**, y tiene como objetivo desarrollar un software que permita la **caracterización de señales de radiofrecuencia** (RF) en enlaces de comunicación satelital. El sistema debe ser capaz de calcular parámetros como la frecuencia central, el ancho de banda, la potencia de la señal y la relación señal-ruido, entre otros.

El software está diseñado para recibir como entrada archivos CSV generados por analizadores de espectro, analizar los datos y proporcionar una interfaz gráfica intuitiva para visualizar y exportar los resultados.

## Características

- **Cálculo automático de parámetros de señal**: Frecuencia central, amplitud, ruido, ancho de banda, entre otros.
- **Detección de interferencias**: Identificación de señales no deseadas y eliminación de interferencias.
- **Visualización de espectrogramas**: Gráficos en tiempo real y visualización de la señal RF.
- **Exportación de resultados**: Generación de informes en formato PDF y CSV.
- **Interfaz gráfica amigable**: Fácil uso e interacción con el usuario.

## Requerimientos del Sistema

### Software

- **Python 3.x** o superior
- **Bibliotecas necesarias**:
  - `pandas`
  - `matplotlib`
  - `plotly`
  - `dash`
  - `numpy`

Para instalar las dependencias, utiliza el siguiente comando:

```bash
pip install -r requirements.txt
```

### Hardware

- Ordenador con soporte para Python 3.x.
- Analizador de espectro (opcional para pruebas en tiempo real).

## Instrucciones de Instalación

### 1. Clonar el Repositorio

Clona el repositorio en tu máquina local con el siguiente comando:

```bash
git clone https://github.com/AIr-Reinforcement-AdAstra2024/Mision-2
```

### 2. Navegar al Directorio del Proyecto

Dirígete al directorio del proyecto:

```bash
cd Mision-2
```

### 3. Instalar Dependencias

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el Software

Ejecuta la aplicación con el siguiente comando:

```bash
python src/main.py
```

## Uso del Software

1. **Cargar Archivo CSV**: Abre el software e importa los archivos CSV generados por el analizador de espectro.
2. **Visualizar los Resultados**: Observa la representación gráfica de la señal RF y ajusta los filtros según las necesidades.
3. **Aplicar Filtros**: Usa las herramientas del software para aplicar filtros de frecuencia y eliminar interferencias no deseadas.
4. **Exportar Resultados**: Puedes exportar los resultados obtenidos en formato CSV o PDF.

### Video de Demostración

Encuentre el video en la ruta `resources/Demo.mp4`

Este video muestra cómo se utiliza el software desarrollado para caracterizar señales RF, desde la carga de un archivo CSV hasta la visualización y exportación de los resultados. **Haz clic en la imagen para ver el video.**

## Arquitectura del Proyecto

El proyecto sigue una estructura modular, organizada para facilitar el mantenimiento y la escalabilidad.

### Estructura del Proyecto

```
/
├── LICENSE
├── README.md
├── requirements.txt
├── resources
│   └── recurso1.csv
└── src
    ├── __pycache__
    ├── about
    ├── app.py
    ├── assets
    ├── home
    ├── interferences
    ├── main.py
    ├── reports
    ├── signal_analysis
    └── utils
```

### Componentes Clave

- **main.py**: El archivo principal que contiene la lógica de la aplicación y arranca la interfaz gráfica.
- **resources/**: Archivos CSV para probar la aplicación.
- **src/**: Directorio con las distintas pantallas de la aplicación.

## Pruebas y Ejemplos

### Archivos de Prueba

El directorio `resources/` incluye varios archivos CSV de prueba que puedes usar para verificar el funcionamiento del software.

### Ejemplo de Ejecución

1. Abre el software y carga uno de los archivos CSV de prueba.
2. Observa cómo el software detecta automáticamente los parámetros clave de la señal.
3. Aplica filtros para eliminar interferencias no deseadas.
4. Exporta los resultados en formato PDF para su análisis posterior.

## Overview de Funcionalidades

### Carga de archivos

<img width="1728" alt="image" src="https://github.com/user-attachments/assets/0d487d63-8b53-4d8e-8f3a-f0a5c04da3a4">

### Análisis de señales

<img width="1728" alt="image" src="https://github.com/user-attachments/assets/d2317863-a3fc-4b22-84c6-a7afdcb51b4a">

### Espectro de la señal y Análisis de espectro temporal

<img width="1728" alt="image" src="https://github.com/user-attachments/assets/96c2fcfb-a80b-4f3a-b8c8-bc8954413ad4">

### Transformada inversa de Fourier

<img width="1728" alt="image" src="https://github.com/user-attachments/assets/7c4de8f6-8e0e-472a-a563-548c3d52473f">

### Detección de interferencias

<img width="1728" alt="image" src="https://github.com/user-attachments/assets/15203359-9fba-4a5e-b0ae-01e26c0e02e0">

### Generación de reportes

<img width="1728" alt="image" src="https://github.com/user-attachments/assets/4a7bd81f-646f-4df3-858a-482ccbe0f7ef">

### Acerca de

<img width="1728" alt="image" src="https://github.com/user-attachments/assets/2cc35698-5366-46b1-8911-a8acc72a1c67">


## Colaboradores

Este proyecto fue desarrollado por:

- **Sergio Oliveros**
- **Daniel Álvarez**
- **Sebastian Urrea**
- **Haider Fonseca**
- **Daniel Perea**
