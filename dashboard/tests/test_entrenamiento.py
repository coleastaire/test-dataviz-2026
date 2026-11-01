"""Política de datos del modelo previo al partido (con datos sintéticos, sin entrenar nada)."""
import numpy as np
import pandas as pd
import pytest

import entrenar_modelos as T


def sintetico(n=200, semilla=0):
    rng = np.random.default_rng(semilla)
    filas = []
    for m in range(n):
        base = {"match_id": m, "year": 2000 + m % 20, "circuit": "AVP" if m % 2 else "FIVB",
                "gender": "M" if m % 3 else "W", "etapa": T.ETAPAS[m % 5]}
        v = float(rng.integers(-20, 20))
        for res, signo in [("Ganador", 1), ("Perdedor", -1)]:
            filas.append({**base, "resultado": res, "ventaja_rank": signo * v, "edad_media": 28.0,
                          "ventaja_edad": 0.0, "estatura_media_cm": 190.0, "ventaja_estatura_cm": 0.0,
                          "es_clasif": int(res == "Perdedor" and m % 4 == 0)})
    E = pd.DataFrame(filas)
    P = pd.DataFrame({"match_id": range(n), "es_forfeit": [int(m % 50 == 0) for m in range(n)]})
    return E, P


def test_una_fila_por_partido_y_reproducible():
    E, P = sintetico()
    a, desc = T.tabla_partidos(E, P)
    b, _ = T.tabla_partidos(E, P)
    assert a.match_id.is_unique
    pd.testing.assert_frame_equal(a, b)                       # misma semilla, misma muestra
    assert desc["partidos_forfeit"] == 4 and len(a) == 196    # forfeits fuera
    # la elección es al azar: aparecen ganadores y perdedores
    assert 0.3 < a.gana.mean() < 0.7
    c, _ = T.tabla_partidos(E, P, semilla=1)
    assert not a.gana.equals(c.gana)


def test_defensa_ante_duplicados_y_rival():
    E, P = sintetico(20)
    dup = E[E.match_id == 3]                                  # partido con filas repetidas
    raro = E[E.match_id == 5].assign(resultado="Ganador")     # dos ganadores
    E2 = pd.concat([E[~E.match_id.isin([3, 5])], dup, dup, raro])
    d, desc = T.tabla_partidos(E2, P)
    assert desc["partidos_sin_un_ganador_y_un_perdedor"] == 2
    assert not {3, 5} & set(d.match_id)
    # origen en clasificatoria del rival = el de la otra fila del partido
    m0 = d[d.match_id == 4].iloc[0]
    assert m0.es_clasif_rival == (1 if m0.resultado == "Ganador" else 0)


def test_predictores_sin_fuga():
    prohibidas = {"kills", "aces", "blocks", "digs", "errors", "attacks", "serve_errors", "hitpct", "duracion_min"}
    assert not prohibidas & set(T.PREDICTORES)
    assert set(T.PREDICTORES) == {"ventaja_rank", "edad_media", "ventaja_edad", "estatura_media_cm",
                                  "ventaja_estatura_cm", "es_clasif", "es_clasif_rival", "circuit", "gender", "etapa"}


def test_preprocesamiento_se_ajusta_solo_con_entrenamiento():
    """Imputación y estandarización salen solo del entrenamiento, aunque la prueba sea muy distinta."""
    E, P = sintetico(400)
    d, _ = T.tabla_partidos(E, P)
    tr, te = d.iloc[:300].copy(), d.iloc[300:].copy()
    te["edad_media"] = 99.0
    pipe = T.modelos()["Regresión logística"].fit(tr[T.PREDICTORES], tr.gana)
    pipe.predict_proba(te[T.PREDICTORES])
    num = pipe.named_steps["prep"].named_transformers_["num"]
    i = T.NUM.index("edad_media")
    assert num.named_steps["imp"].statistics_[i] == pytest.approx(28.0)
    assert num.named_steps["esc"].mean_[i] == pytest.approx(tr.edad_media.mean())
