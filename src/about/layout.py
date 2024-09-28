# src/about/layout.py

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container(
    [
        html.H2("Acerca de Air Reinforcement"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Motivación del Proyecto"),
                        html.P(
                            """
                            El proyecto surge de la necesidad de contar con herramientas accesibles y eficientes para el análisis de señales de radiofrecuencia. 
                            Con el creciente uso del espectro electromagnético, es crucial disponer de soluciones que permitan identificar interferencias y optimizar 
                            las comunicaciones inalámbricas.
                            """
                        ),
                        html.H4("Tecnologías Utilizadas"),
                        html.Ul(
                            [
                                html.Li("Python 3"),
                                html.Li("Dash y Plotly para la interfaz gráfica y visualizaciones"),
                                html.Li("Dash Bootstrap Components para el diseño responsivo"),
                                html.Li("Pandas y NumPy para el procesamiento de datos"),
                                # Agrega más tecnologías según corresponda
                            ]
                        ),
                        html.H4("Agradecimientos"),
                        html.P(
                            """
                            Agradecemos a todos los colaboradores y a la comunidad de código abierto por proporcionar herramientas y recursos que han hecho posible este proyecto.
                            """
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.H4("Miembros del Equipo"),
                                html.Ul(
                                    [
                                        html.Li("Daniel Perea"),
                                        html.Li("Daniel Vanegas"),
                                        html.Li("Sergio Oliveros"),
                                        html.Li("Haider Fonseca"),
                                        html.Li("Sebastian Urrea"),
                                        # Agrega más miembros según corresponda
                                    ]
                                ),
                            ], className='col-6'),
                            # Imagen del equipo
                            html.Div(
                                [
                                    html.Img(src="assets/foto_equipo.jpeg", style={'width': '100%', 'height': 'auto', 'max-width': '400px', 'border-radius': '10px'}),
                                ],
                                style={'text-align': 'center'},
                                className='col-6'
                            ),
                        ]),
                        html.H4("Contacto"),
                        html.P(
                            """
                            Si tienes preguntas o comentarios, no dudes en ponerte en contacto con nosotros a través de:
                            """
                        ),
                        html.Ul(
                            [
                                html.Li("GitHub: https://github.com/AIr-Reinforcement-AdAstra2024"),
                                # Agrega más métodos de contacto si es necesario
                            ]
                        ),
                        html.H4("Licencia"),
                        html.P("Este proyecto está licenciado bajo la Licencia MIT."),
                        html.A(
                            "Ver Licencia MIT",
                            href="assets/LICENSE.txt",
                            target="_blank",
                            className="btn btn-link",
                        ),
                    ],
                    width=12,
                ),
            ]
        ),
    ],
    className="mt-4",
)
