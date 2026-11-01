"""Los resultados guardados en dashboard/modelos/ son coherentes y se pueden recalcular desde los datos."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import entrenar_modelos as T

MET = json.loads((T.SALIDA / "metricas.json").read_text(encoding="utf-8"))
CUR = json.loads((T.SALIDA / "curvas.json").read_text(encoding="utf-8"))
PERF = json.loads((T.SALIDA / "perfiles.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def division():
    E = pd.read_parquet(T.DATOS / "equipos.parquet")
    P = pd.read_parquet(T.DATOS / "partidos.parquet")
    d, _ = T.tabla_partidos(E, P)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=MET["politica"]["semilla"])
    i_tr, i_te = next(gss.split(d, d.gana, groups=d.match_id))
    return E, d, i_tr, i_te


def test_datos_son_los_mismos_del_entrenamiento():
    for f, h in MET["datos_sha256"].items():
        assert T.sha256(T.DATOS / f) == h, f"{f} cambió: vuelva a correr entrenar_modelos.py"


def test_signos_de_ventaja(division):
    E = division[0]
    w = E[E.resultado == "Ganador"].set_index("match_id")
    l = E[E.resultado == "Perdedor"].set_index("match_id")
    for c in ["ventaja_rank", "ventaja_edad", "ventaja_estatura_cm"]:
        s = (w[c] + l[c].reindex(w.index)).dropna()
        assert np.allclose(s, 0), c


def test_division_por_partido_reproducible(division):
    _, d, i_tr, i_te = division
    pol = MET["politica"]
    assert not set(d.match_id.iloc[i_tr]) & set(d.match_id.iloc[i_te])
    assert (len(i_tr), len(i_te)) == (pol["n_entrenamiento"], pol["n_prueba"])
    ids = np.sort(d.match_id.values[i_te])
    assert hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() == pol["huella_ids_prueba"]
    assert set(pol["predictores"]) == set(T.PREDICTORES)
    assert not set(pol["predictores"]) & set(pol["excluidas_por_fuga"])


def test_metricas_guardadas_se_recalculan(division):
    _, d, _, i_te = division
    te = d.iloc[i_te]
    modelo = joblib.load(T.SALIDA / "modelo_recomendado.joblib")
    prob = modelo.predict_proba(te[T.PREDICTORES])[:, 1]
    r = MET["modelos"][MET["recomendado"]]["prueba"]
    assert roc_auc_score(te.gana, prob) == pytest.approx(r["roc_auc"], abs=1e-6)
    assert accuracy_score(te.gana, prob >= 0.5) == pytest.approx(r["accuracy"], abs=1e-6)
    rk = te.ventaja_rank.notna() & (te.ventaja_rank != 0)
    assert accuracy_score(te.gana[rk], te.ventaja_rank[rk] > 0) == pytest.approx(MET["regla"]["accuracy"], abs=1e-6)
    assert rk.sum() == MET["regla"]["n"]


def test_estructura_de_metricas():
    assert MET["recomendado"] in T.NOMBRES and MET["razon"]
    for n in T.NOMBRES:
        r = MET["modelos"][n]
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert 0 <= r["prueba"][k] <= 1
        assert len(r["auc_cv_pliegues"]) == 5
        assert np.array(r["matriz_confusion"]["valores"]).sum() == MET["politica"]["n_prueba"]
        assert {"todos", "con_ranking", "umbrales", "distribucion"} <= set(CUR[n])
        assert sum(CUR[n]["distribucion"]["gana"]) + sum(CUR[n]["distribucion"]["pierde"]) == MET["politica"]["n_prueba"]
        assert {i["variable"] for i in MET["importancias"][n]} == set(T.PREDICTORES)
    assert 0 < MET["regla"]["cobertura_prueba_pct"] <= 100


def test_perfiles():
    assert PERF["k"] == 4 and len(PERF["perfiles"]) == 4
    assert sum(p["n"] for p in PERF["perfiles"]) == PERF["filas"]
    assert -1 <= PERF["silueta"] <= 1
    assert set(PERF["variables"]) == {f"{s}_set" for s in T.STATS} | {"hitpct"}
    v = [p["pct_victorias"] for p in PERF["perfiles"]]
    assert v == sorted(v, reverse=True)
    assert len({p["nombre"] for p in PERF["perfiles"]}) == 4
