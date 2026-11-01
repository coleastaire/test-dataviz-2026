"""Utilidades comunes del libro en Python (equivale a r-book/R/utils.R).
Cada capítulo empieza con:  from py.utils import *
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Para que los gráficos plotly aparezcan en el HTML de Quarto
pio.renderers.default = "plotly_mimetype+notebook_connected"

# ---------------------------------------------------------------
# 1. Semilla (guía 1.1)
# ---------------------------------------------------------------
RANDOM_STATE = 2026
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------
# 2. Paleta fija (guía 1.2): un color = una sola cosa en todo el libro
# ---------------------------------------------------------------
PAL = {"FIVB": "#D95F02", "AVP": "#1B9E77",
       "Ganador": "#2166AC", "Perdedor": "#B2182B",
       "M": "#7570B3", "W": "#E7298A"}

ETAPAS = ["Clasificatoria", "Fase de grupos", "Cuadro principal",
          "Semifinales", "Finales"]
PAL_ETAPA = dict(zip(ETAPAS, ["#C6DBEF", "#9ECAE1", "#6BAED6",
                              "#3182BD", "#08519C"]))

PAL_PODER = {"Alto": "#1A1A1A", "Moderado": "#737373", "Bajo": "#BDBDBD"}
PAL_MODELO = {"Regresión logística": "#E6AB02", "Random Forest": "#A6761D",
              "XGBoost": "#6A3D9A", "Regla del ranking": "#999999"}
NEUTRO = "#4682B4"
GRIS_TOTAL = "#4D4D4D"

ESCALA_CORR = [[0, "#8E0152"], [0.25, "#DE77AE"], [0.5, "#F7F7F7"],
               [0.75, "#7FBC41"], [1, "#276419"]]

FUENTE = ("Fuente: vb_matches.csv (BigTimeStats vía TidyTuesday). "
          "Elaboración propia.")

# ---------------------------------------------------------------
# 3. Rutas (los capítulos se ejecutan desde py-book/)
# ---------------------------------------------------------------
RAW = "../data/raw/vb_matches.csv"
CLEAN = "../data/clean/"
CIFRAS_PATH = "../compartido/cifras_control_py.csv"


# ---------------------------------------------------------------
# 4. Formato español: coma decimal y punto de miles
# ---------------------------------------------------------------
def fmt(x, d=2):
    """fmt(76756, 0) -> '76.756' ; fmt(19.04, 1) -> '19,0'"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_p(p):
    """p-valor con 3 decimales o '< 0,001'."""
    return "< 0,001" if p < 0.001 else fmt(p, 3)


def fmt_pct(x, d=1):
    """fmt_pct(19.04) -> '19,0 %'"""
    return f"{fmt(x, d)} %"


# ---------------------------------------------------------------
# 5. Estilo común de todos los gráficos (guía 1.4)
# ---------------------------------------------------------------
def estilo_plotly(fig, titulo, x, y):
    fig.update_layout(
        template="plotly_white", title=dict(text=titulo, x=0, y=0.97),
        xaxis=dict(title=x, gridcolor="#E5E5E5", showticklabels=True, automargin=True),
        yaxis=dict(title=y, gridcolor="#E5E5E5", showticklabels=True, automargin=True),
        separators=",.", font=dict(family="Arial", size=13, color="#333333"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom"),
        margin=dict(t=90, b=100, l=70, r=20))
    fig.add_annotation(text=FUENTE, xref="paper", yref="paper", x=0, y=0,
                       yshift=-70, xanchor="left", yanchor="top", showarrow=False,
                       font=dict(size=10, color="#737373"))
    return fig

# ---------------------------------------------------------------
# 6. Registro de cifras de control para comparar con R (guía 7.3)
# ---------------------------------------------------------------
CIFRAS = []


def registrar(id_, valor):
    """Guarda una cifra en el CSV común sin borrar las de otros capítulos."""
    import os
    previas = (pd.read_csv(CIFRAS_PATH) if os.path.exists(CIFRAS_PATH)
               else pd.DataFrame(columns=["id", "valor"]))
    previas = previas[previas["id"] != id_]                 # reemplaza si ya existe
    nueva = pd.DataFrame([{"id": id_, "valor": float(valor)}])
    pd.concat([previas, nueva], ignore_index=True).to_csv(CIFRAS_PATH, index=False)
    CIFRAS[:] = [c for c in CIFRAS if c["id"] != id_] + [{"id": id_, "valor": float(valor)}]
    return valor


def cifra(id_):
    """Lee una cifra ya registrada (por ejemplo, del pipeline)."""
    return pd.read_csv(CIFRAS_PATH).set_index("id")["valor"][id_]

# ---------------------------------------------------------------
# 7. Estadísticos descriptivos (guía 4.2: misma fórmula que en R)
# ---------------------------------------------------------------
def describir(s):
    s = s.dropna()
    return pd.Series({"n": len(s), "media": s.mean(), "mediana": s.median(),
                      "de": s.std(),            # muestral (ddof = 1)
                      "p25": s.quantile(.25), "p75": s.quantile(.75),
                      "min": s.min(), "max": s.max(),
                      "asimetria": s.skew()})   # Fisher-Pearson ajustada
    
    

# ---------------------------------------------------------------
# 8. Cifras dentro del texto (Jupyter Book: glue de myst-nb)
# ---------------------------------------------------------------
class _Texto(str):
    """Texto que se muestra sin comillas al pegarlo en el libro."""
    def __repr__(self):
        return str(self)


def pegar(nombre, valor):
    """Guarda un texto para usarlo en un párrafo con {glue}`nombre`."""
    from myst_nb import glue
    glue(nombre, _Texto(valor), display=False)