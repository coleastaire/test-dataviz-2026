"""Colores, formato de números y estilo de los gráficos (los mismos del libro)."""
import plotly.graph_objects as go
import plotly.io as pio

# Paleta fija de playa: un color significa una sola cosa en todo el dashboard
# (la misma de jb-book/py/utils.py y de assets/estilo.css)
MAR = "#0B3B5C"        # ganador y marca
CORAL = "#E4572E"      # perdedor y sorpresas
TURQUESA = "#1A8A9A"   # FIVB
ARENA = "#D9A441"      # AVP

PAL = {"FIVB": TURQUESA, "AVP": ARENA,
       "Ganador": MAR, "Perdedor": CORAL,
       "M": "#2F6690", "W": "#C97B63"}
ETAPAS = ["Clasificatoria", "Fase de grupos", "Cuadro principal", "Semifinales", "Finales"]
# Etapas: más oscuro = más avanzado en el torneo
PAL_ETAPA = dict(zip(ETAPAS, ["#BFE3E6", "#86C7CE", "#4FA3AE", "#1F6F86", MAR]))
# Poder discriminante: mismo tono, más intenso = más poder
PAL_PODER = {"Alto": MAR, "Moderado": "#5B8DB0", "Bajo": "#C9D6E3"}
# Modelos: más oscuro = más complejo; la regla de referencia en gris
PAL_MODELO = {"Regresión logística": "#86C7CE", "Random Forest": "#1F6F86",
              "Gradient Boosting": MAR, "Regla del ranking": "#64748B"}
# Perfiles de juego, ordenados por % de victorias: de mar (más gana) a coral (más pierde)
PAL_PERFIL = [MAR, "#5B8DB0", "#EFA58C", CORAL]
NEUTRO = MAR
GRIS = "#64748B"
# Correlación: coral (negativa) – blanco – mar (positiva)
ESCALA_CORR = [[0, CORAL], [0.5, "#F7F7F7"], [1, MAR]]
# Escala de una sola dirección (por ejemplo, matriz de confusión)
ESCALA_SEC = [[0, "#F4EBDD"], [0.5, "#4FA3AE"], [1, MAR]]
GENERO = {"M": "Hombres", "W": "Mujeres"}

TINTA = "#1F2937"
TINTA_SUAVE = "#6B7280"
REJILLA = "#ECEEF1"
FUENTE = "Fuente: vb_matches.csv (BigTimeStats vía TidyTuesday). Elaboración propia."
LETRA = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"


def fmt(x, d=2):
    """Número con coma decimal y punto de miles: fmt(76749, 0) -> '76.749'."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x, d=1):
    return f"{fmt(x, d)} %"


def fmt_p(p):
    return "< 0,001" if p < 0.001 else fmt(p, 3)


# Plantilla de plotly para todos los gráficos
pio.templates["vb"] = go.layout.Template(layout=dict(
    font=dict(family=LETRA, size=13, color=TINTA),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", separators=",.",
    xaxis=dict(gridcolor=REJILLA, zeroline=False, linecolor="#D1D5DB",
               ticks="outside", tickcolor="#D1D5DB", automargin=True),
    yaxis=dict(gridcolor=REJILLA, zeroline=False, automargin=True),
    legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                title=None),
    margin=dict(l=10, r=20, t=40, b=10),
    hoverlabel=dict(bgcolor="white", bordercolor="#D1D5DB",
                    font=dict(family=LETRA, color=TINTA)),
    colorway=[MAR, TURQUESA, ARENA, CORAL, GRIS]))
pio.templates.default = "vb"


def vacia(mensaje="No hay datos para estos filtros"):
    """Figura en blanco con un mensaje, para cuando un filtro deja 0 filas."""
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=TINTA_SUAVE))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320)
    return fig
