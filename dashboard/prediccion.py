"""Lee los resultados que deja entrenar_modelos.py en dashboard/modelos/.

El dashboard nunca entrena: si faltan los archivos, las pestañas de modelos lo
avisan y piden correr  python dashboard/entrenar_modelos.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent / "modelos"
NOMBRES = ["Regresión logística", "Random Forest", "Gradient Boosting"]


def _json(nombre):
    ruta = CARPETA / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


MET = _json("metricas.json")       # métricas, política de división, importancias
CUR = _json("curvas.json")         # curvas ROC / PR, umbrales y distribución de probabilidades
PERF = _json("perfiles.json")      # perfiles de juego (K-means, AVP)

MODELO = None
if (CARPETA / "modelo_recomendado.joblib").exists():
    try:
        import joblib
        MODELO = joblib.load(CARPETA / "modelo_recomendado.joblib")
    except Exception as err:       # p. ej. otra versión de scikit-learn
        print(f"Aviso: no se pudo cargar el modelo guardado ({err}). "
              "Corra python dashboard/entrenar_modelos.py")


def fila(rank_propio, rank_rival, edad_propia, edad_rival, est_propia, est_rival,
         clasif_propia, clasif_rival, circuito, genero, etapa):
    """Una fila con los predictores del modelo, vista desde la pareja 'propia'."""
    ventaja = (rank_rival - rank_propio) if rank_propio and rank_rival else np.nan
    return {"ventaja_rank": ventaja, "edad_media": edad_propia, "ventaja_edad": edad_propia - edad_rival,
            "estatura_media_cm": est_propia, "ventaja_estatura_cm": est_propia - est_rival,
            "es_clasif": int(clasif_propia), "es_clasif_rival": int(clasif_rival),
            "circuit": circuito, "gender": genero, "etapa": etapa}


def probabilidad(**p):
    """Probabilidad de que gane la pareja propia según el modelo recomendado.

    Se evalúa desde los dos lados (propia y rival) y se promedia, para que las dos
    probabilidades sumen exactamente 1 sin importar qué pareja se ponga primero.
    """
    if MODELO is None:
        return None
    a = fila(**p)
    b = fila(p["rank_rival"], p["rank_propio"], p["edad_rival"], p["edad_propia"], p["est_rival"],
             p["est_propia"], p["clasif_rival"], p["clasif_propia"], p["circuito"], p["genero"], p["etapa"])
    X = pd.DataFrame([a, b])
    pa, pb = MODELO.predict_proba(X[MET["politica"]["predictores"]])[:, 1]
    return float((pa + 1 - pb) / 2)
