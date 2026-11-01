"""Entrena, fuera del dashboard, los modelos de la página Predicción y los perfiles de juego.

El dashboard NO entrena nada al arrancar: solo lee lo que este script deja en
dashboard/modelos/ (métricas, curvas, importancias, perfiles y el modelo elegido).
Hay que correrlo una vez, y de nuevo solo si cambian los datos de dashboard/datos/.

Uso, desde la carpeta raíz del repositorio:
    python dashboard/entrenar_modelos.py

Qué hace:
1. Modelo previo al partido. Usa SOLO información que se conoce antes de jugar:
   ventaja de ranking, edad, estatura, origen en clasificatoria, circuito, género
   y etapa. Nada de kills, aces, bloqueos, defensas ni errores (se producen durante
   el partido: usarlos sería una fuga de información).
   - Cada partido aparece dos veces en equipos.parquet (ganador y perdedor). Se
     conserva UNA fila por partido, elegida al azar con semilla fija, para que la
     clase esté balanceada y un mismo partido no quede en entrenamiento y prueba.
   - División entrenamiento/prueba por partido (GroupShuffleSplit, 75/25).
   - Todo el preprocesamiento (imputación, estandarización, variables indicadoras)
     va dentro de un Pipeline: se ajusta solo con el conjunto de entrenamiento.
   - Se comparan regresión logística, Random Forest y Gradient Boosting con
     validación cruzada en entrenamiento; el modelo recomendado se elige con la
     validación cruzada, nunca con la prueba (la logística se prefiere si queda a
     menos de una desviación estándar del mejor).
2. Perfiles de juego (solo AVP): K-means sobre estadísticas por set estandarizadas.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_samples, silhouette_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMILLA = 2026          # la misma del libro (jb-book/py/utils.py)
CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "datos"
SALIDA = CARPETA / "modelos"
ETAPAS = ["Clasificatoria", "Fase de grupos", "Cuadro principal", "Semifinales", "Finales"]

# Predictores permitidos: solo información previa al partido
NUM = ["ventaja_rank", "edad_media", "ventaja_edad", "estatura_media_cm",
       "ventaja_estatura_cm", "es_clasif", "es_clasif_rival"]
CAT = ["circuit", "gender", "etapa"]
PREDICTORES = NUM + CAT
ETQ_PRED = {"ventaja_rank": "Ventaja de ranking", "edad_media": "Edad media",
            "ventaja_edad": "Ventaja de edad", "estatura_media_cm": "Estatura media",
            "ventaja_estatura_cm": "Ventaja de estatura", "es_clasif": "Viene de clasificatoria",
            "es_clasif_rival": "Rival viene de clasificatoria", "circuit": "Circuito",
            "gender": "Género", "etapa": "Etapa del torneo"}
# Variables que se producen durante o después del partido: prohibidas como predictores
PROHIBIDAS = {"kills", "aces", "blocks", "digs", "attacks", "errors", "serve_errors", "hitpct",
              "duracion_min", "n_sets", "puntos_totales", "resultado", "es_upset", "rank_diff",
              "w_rank", "l_rank", "tiene_stats_completas", "es_forfeit", "es_retirado"}
assert not PROHIBIDAS & set(PREDICTORES)

NOMBRES = ["Regresión logística", "Random Forest", "Gradient Boosting"]

# Perfiles de juego (AVP): estadísticas por set y % de ataque
STATS = ["kills", "aces", "blocks", "digs", "attacks", "errors", "serve_errors"]
VARS_PERFIL = [f"{s}_set" for s in STATS] + ["hitpct"]
ETQ_PERFIL = {"kills_set": "kills", "aces_set": "aces", "blocks_set": "bloqueos",
              "digs_set": "defensas", "attacks_set": "ataques", "errors_set": "errores de ataque",
              "serve_errors_set": "errores de saque", "hitpct": "% de ataque"}
K_PERFILES = 4


# --------------------------------------------------------------------------
# Datos del modelo previo al partido
# --------------------------------------------------------------------------
def tabla_partidos(E, P, semilla=SEMILLA):
    """Una fila por partido (ganador o perdedor, al azar con semilla fija).

    Devuelve la tabla y un dict con cuántas filas o partidos se descartaron y por qué.
    """
    e = E.copy()
    for c in ["circuit", "gender", "etapa", "resultado"]:
        e[c] = e[c].astype(str)
    n_inicial = e.match_id.nunique()
    # Defensa ante duplicados: cada partido debe tener exactamente un ganador y un perdedor
    g = e.groupby("match_id").resultado.agg(n="size", n_g=lambda s: int((s == "Ganador").sum()))
    validos = g[(g.n == 2) & (g.n_g == 1)].index
    e = e[e.match_id.isin(validos)]
    # Origen en clasificatoria del rival (la otra fila del mismo partido)
    rival = e[["match_id", "resultado", "es_clasif"]].copy()
    rival["resultado"] = rival.resultado.map({"Ganador": "Perdedor", "Perdedor": "Ganador"})
    e = e.merge(rival.rename(columns={"es_clasif": "es_clasif_rival"}),
                on=["match_id", "resultado"], how="left", validate="one_to_one")
    # Sin forfeits: en ellos no se jugó, así que el resultado no depende de las parejas
    e = e.merge(P[["match_id", "es_forfeit"]], on="match_id", how="left", validate="many_to_one")
    n_forfeit = int(e.loc[e.es_forfeit == 1, "match_id"].nunique())
    e = e[e.es_forfeit != 1]
    # Una fila por partido: orden fijo y número aleatorio con semilla
    e = e.sort_values(["match_id", "resultado"]).reset_index(drop=True)
    e["_azar"] = np.random.default_rng(semilla).random(len(e))
    e = e.loc[e.groupby("match_id")._azar.idxmax()].drop(columns="_azar").reset_index(drop=True)
    e["gana"] = (e.resultado == "Ganador").astype(int)
    assert e.match_id.is_unique
    descartes = {"partidos_iniciales": int(n_inicial),
                 "partidos_sin_un_ganador_y_un_perdedor": int(n_inicial - len(validos)),
                 "partidos_forfeit": n_forfeit, "partidos_modelo": int(len(e))}
    return e, descartes


def preproceso(escalar):
    num = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        num.append(("esc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), NUM),
        ("cat", OneHotEncoder(categories=[["AVP", "FIVB"], ["M", "W"], ETAPAS],
                              handle_unknown="ignore"), CAT)])


def modelos():
    return {
        "Regresión logística": Pipeline([("prep", preproceso(True)),
                                         ("mod", LogisticRegression(max_iter=2000))]),
        "Random Forest": Pipeline([("prep", preproceso(False)),
                                   ("mod", RandomForestClassifier(
                                       n_estimators=200, min_samples_leaf=50, max_features="sqrt",
                                       n_jobs=-1, random_state=SEMILLA))]),
        "Gradient Boosting": Pipeline([("prep", preproceso(False)),
                                       ("mod", GradientBoostingClassifier(
                                           n_estimators=250, learning_rate=0.05, max_depth=3,
                                           subsample=0.8, random_state=SEMILLA))]),
    }


HIPERPARAMETROS = {
    "Regresión logística": "L2, C = 1, max_iter = 2000; predictores numéricos estandarizados",
    "Random Forest": "200 árboles, min_samples_leaf = 50, max_features = sqrt",
    "Gradient Boosting": "250 árboles, learning_rate = 0,05, max_depth = 3, subsample = 0,8",
}


def metricas(y, prob, umbral=0.5):
    pred = (prob >= umbral).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob),
            "brier": brier_score_loss(y, prob), "n": int(len(y))}


def submuestra(*arrs, maximo=300):
    """Reduce una curva a ~maximo puntos conservando los extremos."""
    n = len(arrs[0])
    idx = np.unique(np.linspace(0, n - 1, min(n, maximo)).round().astype(int))
    return [np.asarray(a)[idx].round(4).tolist() for a in arrs]


def curvas(y, prob):
    fpr, tpr, _ = roc_curve(y, prob)
    prec, rec, _ = precision_recall_curve(y, prob)
    fpr, tpr = submuestra(fpr, tpr)
    rec, prec = submuestra(rec[::-1], prec[::-1])
    return {"roc": {"fpr": fpr, "tpr": tpr}, "pr": {"recall": rec, "precision": prec}}


def umbrales(y, prob):
    t = np.round(np.arange(0.05, 0.951, 0.01), 2)
    filas = {"umbral": t.tolist(), "accuracy": [], "precision": [], "recall": [], "f1": [],
             "pct_pred_gana": []}
    for u in t:
        pred = (prob >= u).astype(int)
        filas["accuracy"].append(round(accuracy_score(y, pred), 4))
        # sin ninguna predicción "gana", la precisión no está definida: se deja vacía (no 0)
        filas["precision"].append(round(precision_score(y, pred), 4) if pred.any() else None)
        filas["recall"].append(round(recall_score(y, pred), 4))
        filas["f1"].append(round(f1_score(y, pred, zero_division=0), 4))
        filas["pct_pred_gana"].append(round(100 * pred.mean(), 2))
    return filas


def distribucion(y, prob):
    bordes = np.linspace(0, 1, 41)
    return {"bordes": bordes.round(3).tolist(),
            "gana": np.histogram(prob[y == 1], bordes)[0].tolist(),
            "pierde": np.histogram(prob[y == 0], bordes)[0].tolist()}


def c3(x):
    """Tres decimales con coma, como en el resto del dashboard."""
    return f"{x:.3f}".replace(".", ",")


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def redondear(x):
    """Convierte números de numpy a tipos de JSON (y redondea a 6 decimales)."""
    if isinstance(x, dict):
        return {k: redondear(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [redondear(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 6)
    return x


def entrenar_prediccion(E, P):
    t0 = time.time()
    d, descartes = tabla_partidos(E, P)
    X, y, grupos = d[PREDICTORES], d.gana.values, d.match_id.values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEMILLA)
    i_tr, i_te = next(gss.split(X, y, groups=grupos))
    assert not set(grupos[i_tr]) & set(grupos[i_te]), "un partido quedó en entrenamiento y prueba"
    Xtr, Xte, ytr, yte = X.iloc[i_tr], X.iloc[i_te], y[i_tr], y[i_te]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

    res, curv, imp, ajustados, probs = {}, {}, {}, {}, {}
    con_rank = (Xte.ventaja_rank.notna() & (Xte.ventaja_rank != 0)).values
    for nombre, pipe in modelos().items():
        print(f"  {nombre}: validación cruzada…", flush=True)
        auc_cv = cross_val_score(pipe, Xtr, ytr, groups=grupos[i_tr], cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        m = metricas(yte, prob)
        cm = confusion_matrix(yte, (prob >= 0.5).astype(int), labels=[1, 0])
        m_rank = metricas(yte[con_rank], prob[con_rank])
        res[nombre] = {"auc_cv_media": auc_cv.mean(), "auc_cv_de": auc_cv.std(ddof=1),
                       "auc_cv_pliegues": auc_cv.tolist(), "prueba": m,
                       "prueba_con_ranking": m_rank,
                       "matriz_confusion": {"etiquetas": ["Gana", "Pierde"], "valores": cm.tolist()},
                       "hiperparametros": HIPERPARAMETROS[nombre]}
        curv[nombre] = {"todos": curvas(yte, prob), "con_ranking": curvas(yte[con_rank], prob[con_rank]),
                        "umbrales": umbrales(yte, prob), "distribucion": distribucion(yte, prob)}
        print(f"  {nombre}: importancia por permutación…", flush=True)
        pi = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=5,
                                    random_state=SEMILLA, n_jobs=1)
        imp[nombre] = sorted([{"variable": v, "etiqueta": ETQ_PRED[v], "media": pi.importances_mean[i],
                               "de": pi.importances_std[i]} for i, v in enumerate(PREDICTORES)],
                             key=lambda r: -r["media"])
        ajustados[nombre], probs[nombre] = pipe, prob

    # Regla "gana el mejor rankeado": solo en partidos con ranking de las dos parejas y sin empate
    v = Xte.ventaja_rank.values[con_rank]
    yr = yte[con_rank]
    pred_regla = (v > 0).astype(int)
    regla = {"accuracy": accuracy_score(yr, pred_regla), "precision": precision_score(yr, pred_regla),
             "recall": recall_score(yr, pred_regla), "f1": f1_score(yr, pred_regla),
             "roc_auc": roc_auc_score(yr, v), "n": int(con_rank.sum()),
             "cobertura_prueba_pct": 100 * con_rank.mean(),
             "cobertura_total_pct": 100 * (X.ventaja_rank.notna() & (X.ventaja_rank != 0)).mean(),
             "partidos_sin_ranking_de_alguna_pareja_pct": 100 * Xte.ventaja_rank.isna().mean(),
             "partidos_con_empate_de_ranking_pct": 100 * (Xte.ventaja_rank == 0).mean()}
    curv["Regla del ranking"] = {"con_ranking": curvas(yr, v.astype(float))}

    # Modelo recomendado: el de mayor AUC media en validación cruzada; si la regresión logística queda a
    # menos de una desviación estándar de él, se prefiere la logística (más fácil de interpretar)
    mejor = max(NOMBRES, key=lambda n: res[n]["auc_cv_media"])
    corte = res[mejor]["auc_cv_media"] - res[mejor]["auc_cv_de"]
    lr = "Regresión logística"
    recomendado = lr if res[lr]["auc_cv_media"] >= corte else mejor
    if recomendado == mejor:
        otros = ", ".join(f"{n} {c3(res[n]['auc_cv_media'])}" for n in NOMBRES if n != mejor)
        razon = (f"{mejor} tiene la mayor AUC media en la validación cruzada de entrenamiento "
                 f"({c3(res[mejor]['auc_cv_media'])} ± {c3(res[mejor]['auc_cv_de'])}; {otros}). La regresión "
                 "logística, más fácil de interpretar, queda a más de una desviación estándar, así que no se prefiere. "
                 "La elección no usa el conjunto de prueba.")
    else:
        razon = (f"{mejor} tiene la mayor AUC media en validación cruzada ({c3(res[mejor]['auc_cv_media'])}), "
                 f"pero la regresión logística queda a menos de una desviación estándar "
                 f"({c3(res[lr]['auc_cv_media'])} frente al corte de {c3(corte)}) y es más fácil de interpretar: "
                 "se prefiere el modelo más simple que rinde igual. La elección no usa el conjunto de prueba.")

    # Robustez temporal (solo informativa): el modelo recomendado entrenado con 2000–2016 y probado con 2017–2019
    anio = E.drop_duplicates("match_id").set_index("match_id").year
    yrs = d.match_id.map(anio).values
    tr_t, te_t = yrs <= 2016, yrs >= 2017
    pipe_t = modelos()[recomendado].fit(X[tr_t], y[tr_t])
    prob_t = pipe_t.predict_proba(X[te_t])[:, 1]
    rk_t = (X[te_t].ventaja_rank.notna() & (X[te_t].ventaja_rank != 0)).values
    temporal = {"modelo": recomendado, "n_entrenamiento": int(tr_t.sum()), "n_prueba": int(te_t.sum()),
                "prueba": metricas(y[te_t], prob_t),
                "regla_accuracy": accuracy_score(y[te_t][rk_t], (X[te_t].ventaja_rank.values[rk_t] > 0).astype(int)),
                "regla_n": int(rk_t.sum())}

    SALIDA.mkdir(exist_ok=True)
    joblib.dump(ajustados[recomendado], SALIDA / "modelo_recomendado.joblib", compress=3)
    ids_te = np.sort(grupos[i_te])
    politica = {
        "semilla": SEMILLA,
        "unidad": "Un partido = una fila. De las dos filas de cada partido (ganador y perdedor) se "
                  "conserva una al azar con numpy.random.default_rng(semilla) sobre la tabla ordenada "
                  "por match_id y resultado. La variable respuesta es 'gana' (1 si esa pareja ganó).",
        "division": "GroupShuffleSplit por match_id, 75 % entrenamiento y 25 % prueba, semilla fija. "
                    "Se verifica que ningún match_id esté en los dos conjuntos.",
        "validacion_cruzada": "StratifiedGroupKFold de 5 pliegues (agrupado por match_id) dentro del "
                              "entrenamiento. El modelo recomendado se elige con esta AUC, no con la prueba.",
        "preprocesamiento": "Pipeline de scikit-learn ajustado solo con entrenamiento: imputación por "
                            "mediana con indicador de faltante (ranking ausente = pareja sin ranking "
                            "numérico), estandarización (solo regresión logística) y variables indicadoras "
                            "de circuito, género y etapa.",
        "predictores": PREDICTORES, "etiquetas_predictores": ETQ_PRED,
        "excluidas_por_fuga": sorted(PROHIBIDAS),
        "descartes": descartes,
        "n_entrenamiento": int(len(i_tr)), "n_prueba": int(len(i_te)),
        "pct_gana_entrenamiento": 100 * ytr.mean(), "pct_gana_prueba": 100 * yte.mean(),
        "huella_ids_prueba": hashlib.sha256(",".join(map(str, ids_te)).encode()).hexdigest(),
        "alcance": "Partidos FIVB y AVP de 2000 a 2019. Asociaciones observacionales, no causales.",
    }
    salida = {"generado": time.strftime("%Y-%m-%d %H:%M:%S"), "segundos": round(time.time() - t0, 1),
              "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                            "pandas": pd.__version__, "numpy": np.__version__},
              "datos_sha256": {f: sha256(DATOS / f) for f in ["equipos.parquet", "partidos.parquet"]},
              "politica": politica, "modelos": res, "regla": regla, "recomendado": recomendado,
              "razon": razon, "importancias": imp, "temporal": temporal}
    (SALIDA / "metricas.json").write_text(json.dumps(redondear(salida), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (SALIDA / "curvas.json").write_text(json.dumps(redondear(curv), ensure_ascii=False), encoding="utf-8")
    return salida


# --------------------------------------------------------------------------
# Perfiles de juego (solo AVP)
# --------------------------------------------------------------------------
def tabla_perfiles(E, P):
    a = E[(E.circuit.astype(str) == "AVP") & (E.tiene_stats_completas == 1)]
    a = a.merge(P[["match_id", "n_sets", "es_forfeit"]], on="match_id", how="left", validate="many_to_one")
    a = a[(a.es_forfeit != 1) & (a.n_sets > 0)].copy()
    for s in STATS:
        a[f"{s}_set"] = a[s] / a.n_sets
    return a.replace([np.inf, -np.inf], np.nan).dropna(subset=VARS_PERFIL).reset_index(drop=True)


def nombre_perfil(z):
    """Nombre descriptivo a partir de los dos rasgos más marcados del centroide (en DE)."""
    orden = z.abs().sort_values(ascending=False)
    rasgos = [f"{'más' if z[v] > 0 else 'menos'} {ETQ_PERFIL[v]}" for v in orden.index[:2] if abs(z[v]) >= 0.25]
    if not rasgos:
        return "Perfil promedio"
    texto = " y ".join(rasgos)
    return texto[0].upper() + texto[1:]


def entrenar_perfiles(E, P):
    a = tabla_perfiles(E, P)
    esc = StandardScaler().fit(a[VARS_PERFIL])
    Z = esc.transform(a[VARS_PERFIL])
    n_sil = min(10000, len(a))
    sil_k = {}
    for k in range(2, 7):
        lab = KMeans(k, n_init=10, random_state=SEMILLA).fit_predict(Z)
        sil_k[k] = silhouette_score(Z, lab, sample_size=n_sil, random_state=SEMILLA)
    km = KMeans(K_PERFILES, n_init=20, random_state=SEMILLA).fit(Z)
    a["perfil"] = km.labels_
    rng = np.random.default_rng(SEMILLA)
    idx = rng.choice(len(a), n_sil, replace=False)
    s_muestra = silhouette_samples(Z[idx], km.labels_[idx])
    pca = PCA(2, random_state=SEMILLA).fit(Z)
    idx_plot = idx[:2000]
    xy = pca.transform(Z[idx_plot])

    zc = pd.DataFrame(km.cluster_centers_, columns=VARS_PERFIL)
    perfiles = []
    for c in range(K_PERFILES):
        g = a[a.perfil == c]
        p = (g.resultado.astype(str) == "Ganador").mean()
        perfiles.append({"cluster": c, "nombre": nombre_perfil(zc.loc[c]), "n": len(g),
                         "pct_filas": 100 * len(g) / len(a), "pct_victorias": 100 * p,
                         "ic95": 196 * np.sqrt(p * (1 - p) / len(g)),
                         "z": zc.loc[c].to_dict(), "medias": g[VARS_PERFIL].mean().to_dict(),
                         "silueta_media": s_muestra[km.labels_[idx] == c].mean()})
    perfiles.sort(key=lambda r: -r["pct_victorias"])
    orden = {r["cluster"]: i for i, r in enumerate(perfiles)}
    salida = {
        "generado": time.strftime("%Y-%m-%d %H:%M:%S"), "semilla": SEMILLA, "k": K_PERFILES,
        "variables": VARS_PERFIL, "etiquetas": ETQ_PERFIL,
        "filas": len(a), "partidos": int(a.match_id.nunique()),
        "anios": [int(a.year.min()), int(a.year.max())],
        "filtro": "AVP, estadísticas completas, sin forfeits; una fila por pareja en cada partido; "
                  "estadísticas divididas por el número de sets (el % de ataque no se divide).",
        "silueta": silhouette_score(Z, km.labels_, sample_size=n_sil, random_state=SEMILLA),
        "silueta_por_k": sil_k, "pct_silueta_negativa": 100 * (s_muestra < 0).mean(),
        "pca_varianza": pca.explained_variance_ratio_.tolist(),
        "perfiles": perfiles, "media_global": dict(zip(VARS_PERFIL, esc.mean_)),
        "de_global": dict(zip(VARS_PERFIL, esc.scale_)),
        "muestra": {"x": xy[:, 0].round(3).tolist(), "y": xy[:, 1].round(3).tolist(),
                    "orden": [orden[int(c)] for c in km.labels_[idx_plot]]},
    }
    (SALIDA / "perfiles.json").write_text(json.dumps(redondear(salida), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return salida


def main():
    E = pd.read_parquet(DATOS / "equipos.parquet")
    P = pd.read_parquet(DATOS / "partidos.parquet")
    SALIDA.mkdir(exist_ok=True)
    print("Modelo previo al partido")
    m = entrenar_prediccion(E, P)
    for n in NOMBRES:
        r = m["modelos"][n]
        print(f"  {n:20s} AUC CV {r['auc_cv_media']:.3f} ± {r['auc_cv_de']:.3f} · "
              f"AUC prueba {r['prueba']['roc_auc']:.3f} · accuracy {r['prueba']['accuracy']:.3f}")
    print(f"  Regla del ranking    AUC {m['regla']['roc_auc']:.3f} · accuracy {m['regla']['accuracy']:.3f} "
          f"(cobertura {m['regla']['cobertura_prueba_pct']:.1f} % de la prueba)")
    print(f"  Recomendado: {m['recomendado']}")
    print("Perfiles de juego (AVP)")
    p = entrenar_perfiles(E, P)
    for r in p["perfiles"]:
        print(f"  {r['nombre']:45s} n={r['n']:6d}  victorias {r['pct_victorias']:.1f} %")
    print(f"  Silueta: {p['silueta']:.3f}")
    print(f"Listo: archivos en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
