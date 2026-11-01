"""Colores, formato de números y estilo de los gráficos (los mismos del libro)."""
import plotly.graph_objects as go
import plotly.io as pio

# Paleta fija: un color significa una sola cosa en todo el dashboard
PAL = {"FIVB": "#D95F02", "AVP": "#1B9E77",
       "Ganador": "#2166AC", "Perdedor": "#B2182B",
       "M": "#7570B3", "W": "#E7298A"}
ETAPAS = ["Clasificatoria", "Fase de grupos", "Cuadro principal", "Semifinales", "Finales"]
PAL_ETAPA = dict(zip(ETAPAS, ["#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"]))
PAL_PODER = {"Alto": "#1A1A1A", "Moderado": "#737373", "Bajo": "#BDBDBD"}
NEUTRO = "#4682B4"
GRIS = "#4D4D4D"
ESCALA_CORR = [[0, "#8E0152"], [0.25, "#DE77AE"], [0.5, "#F7F7F7"],
               [0.75, "#7FBC41"], [1, "#276419"]]
GENERO = {"M": "Hombres", "W": "Mujeres"}

TINTA = "#1F2937"
TINTA_SUAVE = "#6B7280"
REJILLA = "#ECEEF1"
FUENTE = "Fuente: vb_matches.csv (BigTimeStats vía TidyTuesday). Elaboración propia."


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
    font=dict(family="Inter, Arial, sans-serif", size=13, color=TINTA),
    paper_bgcolor="white", plot_bgcolor="white", separators=",.",
    xaxis=dict(gridcolor=REJILLA, zeroline=False, linecolor="#D1D5DB",
               ticks="outside", tickcolor="#D1D5DB", automargin=True),
    yaxis=dict(gridcolor=REJILLA, zeroline=False, automargin=True),
    legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                title=None),
    margin=dict(l=10, r=20, t=40, b=10),
    hoverlabel=dict(bgcolor="white", bordercolor="#D1D5DB",
                    font=dict(family="Inter, Arial, sans-serif", color=TINTA)),
    colorway=[NEUTRO, GRIS]))
pio.templates.default = "vb"


def vacia(mensaje="No hay datos para estos filtros"):
    """Figura en blanco con un mensaje, para cuando un filtro deja 0 filas."""
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=TINTA_SUAVE))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320)
    return fig