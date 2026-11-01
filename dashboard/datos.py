"""Carga de datos y cálculos estadísticos del dashboard.

Usa las mismas fórmulas que el libro: Mann-Whitney U con correlación
rango-biserial, solapamiento IQR, chi-cuadrado con V de Cramér, Spearman y VIF.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.linear_model import LinearRegression

from estilo import ETAPAS

CARPETA = Path(__file__).resolve().parent / "datos"
P = pd.read_parquet(CARPETA / "partidos.parquet")    # una fila por partido
E = pd.read_parquet(CARPETA / "equipos.parquet")     # una fila por pareja en cada partido
J = pd.read_parquet(CARPETA / "jugadores.parquet")   # una fila por jugador en cada partido
RES = json.loads((CARPETA / "resumen.json").read_text(encoding="utf-8"))

ANIOS = (int(P.year.min()), int(P.year.max()))


def filtrar(d, circuito="Todos", genero="Todos", anios=ANIOS, etapa="Todas"):
    """Aplica los filtros generales a cualquiera de las tres tablas."""
    if circuito != "Todos":
        d = d[d.circuit == circuito]
    if genero != "Todos":
        d = d[d.gender == genero]
    if etapa != "Todas":
        d = d[d.etapa == etapa]
    return d[d.year.between(anios[0], anios[1])]


# --------------------------------------------------------------------------
# Variables que se pueden explorar
# (tabla, columna, etiqueta, unidad, solo filas con estadísticas de juego)
# --------------------------------------------------------------------------
VARS_UNI = {
    "edad": ("J", "edad", "Edad del jugador", "años", False),
    "estatura": ("J", "estatura_cm", "Estatura del jugador", "cm", False),
    "ranking": ("E", "rank_main", "Ranking del equipo", "posición", False),
    "duracion": ("P", "duracion_min", "Duración del partido", "min", False),
    "sets": ("P", "n_sets", "Número de sets", "sets", False),
    "puntos": ("P", "puntos_totales", "Puntos totales del partido", "puntos", False),
    "kills": ("E", "kills", "Kills del equipo", "ataques ganadores", True),
    "aces": ("E", "aces", "Aces del equipo", "saques directos", True),
    "blocks": ("E", "blocks", "Bloqueos del equipo", "bloqueos", True),
    "digs": ("E", "digs", "Defensas del equipo (digs)", "defensas", True),
    "hitpct": ("E", "hitpct", "% de ataque del equipo", "proporción", True),
}

VARS_BI = {
    "rank_main": ("Ranking principal", "posición", False),
    "edad_media": ("Edad media de la pareja", "años", False),
    "estatura_media_cm": ("Estatura media de la pareja", "cm", False),
    "dif_edad_pareja": ("Diferencia de edad entre compañeros", "años", False),
    "dif_estatura_pareja_cm": ("Diferencia de estatura entre compañeros", "cm", False),
    "hitpct": ("% de ataque", "proporción", True),
    "errors": ("Errores de ataque", "errores", True),
    "blocks": ("Bloqueos", "bloqueos", True),
    "kills": ("Kills", "ataques ganadores", True),
    "aces": ("Aces", "saques directos", True),
    "digs": ("Defensas (digs)", "defensas", True),
    "attacks": ("Ataques", "ataques", True),
    "serve_errors": ("Errores de saque", "errores", True),
}

TABLAS = {"P": P, "E": E, "J": J}


def serie_uni(clave, circuito, genero, anios):
    """Devuelve la tabla filtrada y la columna de una variable univariada."""
    tabla, col, _, _, solo_stats = VARS_UNI[clave]
    d = filtrar(TABLAS[tabla], circuito, genero, anios)
    if tabla == "P":
        d = d[d.es_forfeit == 0]
    if solo_stats:
        d = d[d.tiene_stats_completas == 1]
    return d, col


def describir(s):
    s = s.dropna()
    return {"n": len(s), "Media": s.mean(), "Mediana": s.median(), "DE": s.std(),
            "P25": s.quantile(.25), "P75": s.quantile(.75),
            "Mín": s.min(), "Máx": s.max(), "Asimetría": s.skew()}


def poder(rbc):
    a = abs(rbc)
    return "Alto" if a >= 0.5 else "Moderado" if a >= 0.3 else "Bajo"


def comparar(d, var):
    """Mann-Whitney U entre ganadores y perdedores, con RBC y solapamiento IQR."""
    g = d.loc[d.resultado == "Ganador", var].dropna()
    p = d.loc[d.resultado == "Perdedor", var].dropna()
    if len(g) < 10 or len(p) < 10:
        return None
    U, pv = mannwhitneyu(g, p, alternative="two-sided", method="asymptotic")
    qg, qp = g.quantile([.25, .75]).values, p.quantile([.25, .75]).values
    inter = max(0, min(qg[1], qp[1]) - max(qg[0], qp[0]))
    union = max(qg[1], qp[1]) - min(qg[0], qp[0])
    rbc = 2 * U / (len(g) * len(p)) - 1
    return {"n": len(g) + len(p), "med_g": g.median(), "med_p": p.median(),
            "p": pv, "rbc": rbc, "solap": inter / union if union > 0 else 1.0,
            "poder": poder(rbc)}


def solo_avp(d):
    """Estadísticas de juego: solo AVP con estadísticas completas (igual que el libro)."""
    return d[(d.circuit == "AVP") & (d.tiene_stats_completas == 1)]


def tabla_poder(circuito, genero, anios):
    """RBC de todas las variables numéricas con los filtros actuales."""
    d = filtrar(E, circuito, genero, anios)
    filas = []
    for var, (etq, _, solo_stats) in VARS_BI.items():
        dd = solo_avp(d) if solo_stats else d
        r = comparar(dd, var)
        if r:
            filas.append({"var": var, "Variable": etq + (" (AVP)" if solo_stats else ""), **r})
    return pd.DataFrame(filas)


def chi_v(x, y):
    t = pd.crosstab(x, y)
    if min(t.shape) < 2:
        return None
    chi2, pv, gl, _ = chi2_contingency(t, correction=False)
    return {"v": np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))), "p": pv}


def curva_victoria(d, col, ancho):
    """% de victorias según la ventaja sobre el rival, con ventana móvil de +/- ancho."""
    s = d[[col, "resultado"]].dropna()
    s = s.assign(v=s[col].round().astype(int), gana=(s.resultado == "Ganador").astype(int))
    tab = s.groupby("v").gana.agg(["sum", "count"])
    tab = tab.reindex(range(tab.index.min(), tab.index.max() + 1), fill_value=0)
    roll = tab.rolling(2 * ancho + 1, center=True, min_periods=1).sum()
    roll["pct"] = 100 * roll["sum"] / roll["count"]
    return roll.rename(columns={"count": "n"})


def vif():
    """VIF de las variables conocidas antes del partido (todos los partidos)."""
    X = P.dropna(subset=["dif_rank", "dif_edad", "dif_estatura_cm"]).copy()
    niveles = ["Cuadro principal"] + [e for e in ETAPAS if e != "Cuadro principal"]
    X["etapa"] = pd.Categorical(X.etapa.astype(str), categories=niveles)
    X["circuit"] = X.circuit.astype(str)
    X["gender"] = X.gender.astype(str)
    X = pd.concat([X[["dif_rank", "dif_edad", "dif_estatura_cm", "dif_clasif"]],
                   pd.get_dummies(X[["circuit", "gender", "etapa"]], drop_first=True,
                                  dtype=float)], axis=1)
    nombres = {"dif_rank": "Diferencia de ranking", "dif_edad": "Diferencia de edad media",
               "dif_estatura_cm": "Diferencia de estatura media",
               "dif_clasif": "Diferencia en origen de clasificatoria",
               "circuit_FIVB": "Circuito: FIVB (ref. AVP)",
               "gender_W": "Género: mujeres (ref. hombres)"}
    filas = []
    for v in X.columns:
        otras = X.drop(columns=v)
        r2 = LinearRegression().fit(otras, X[v]).score(otras, X[v])
        filas.append({"Variable": nombres.get(v, v.replace("etapa_", "Etapa: ") + " (ref. Cuadro principal)"),
                      "VIF": 1 / (1 - r2)})
    t = pd.DataFrame(filas)
    t["Diagnóstico"] = np.select([t.VIF <= 5, t.VIF <= 10], ["Aceptable", "Revisar"], "Grave")
    return t, len(X)


# Cifras generales que se muestran en varias páginas
r_todos = P[P.rank_diff.notna()]
CIFRAS = {
    "n_partidos": len(P),
    "n_jugadores": J.player_id.nunique(),
    "n_paises": P.country.nunique(),
    "n_rank": len(r_todos),
    "pct_fav": 100 * (1 - r_todos.es_upset.mean()),
    "pct_stats": 100 * P.tiene_stats_completas.mean(),
    "dur_med": P[P.es_forfeit == 0].duracion_min.median(),
    "dur_3": P[(P.es_forfeit == 0) & (P.n_sets == 3)].duracion_min.median(),
}