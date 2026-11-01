"""Prepara los datos del dashboard a partir de los archivos limpios del libro.

Solo hay que correrlo si cambian los datos limpios. Los archivos que genera
(carpeta dashboard/datos) ya vienen incluidos en el repositorio, así que quien
solo quiera ver el dashboard NO necesita correr este script.

Uso, desde la carpeta raíz del repositorio:
    python dashboard/preparar_datos.py
"""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
LIMPIOS = RAIZ / "data" / "clean"
CIFRAS = RAIZ / "compartido" / "cifras_control_py.csv"
SALIDA = Path(__file__).resolve().parent / "datos"
SALIDA.mkdir(exist_ok=True)

# Solo las columnas que usa el dashboard (así los archivos pesan poco)
COLS_PARTIDOS = ["match_id", "year", "circuit", "gender", "etapa", "country",
                 "duracion_min", "n_sets", "puntos_totales", "es_forfeit", "es_retirado",
                 "w_rank", "l_rank", "rank_diff", "es_upset", "tiene_stats_completas",
                 "dif_rank", "dif_edad", "dif_estatura_cm", "dif_clasif"]
COLS_EQUIPOS = ["match_id", "year", "circuit", "gender", "etapa", "resultado",
                "rank_main", "rank_rival", "es_clasif", "edad_media", "estatura_media_cm",
                "dif_edad_pareja", "dif_estatura_pareja_cm", "ventaja_rank", "ventaja_edad",
                "ventaja_estatura_cm", "duracion_min", "tiene_stats_completas",
                "kills", "aces", "blocks", "digs", "attacks", "errors", "serve_errors", "hitpct"]
COLS_JUGADORES = ["year", "circuit", "gender", "etapa", "resultado", "player_id",
                  "edad", "estatura_cm", "pais"]

archivos = {"partidos": ("vb_matches_limpio.csv", COLS_PARTIDOS),
            "equipos": ("vb_long.csv", COLS_EQUIPOS),
            "jugadores": ("vb_jugadores.csv", COLS_JUGADORES)}

for nombre, (archivo, cols) in archivos.items():
    d = pd.read_csv(LIMPIOS / archivo, usecols=cols, low_memory=False)
    for c in ["circuit", "gender", "etapa", "resultado", "country", "pais"]:
        if c in d.columns:
            d[c] = d[c].astype("category")
    d.to_parquet(SALIDA / f"{nombre}.parquet", index=False)
    print(f"{nombre}: {len(d):,} filas x {d.shape[1]} columnas")

# Cifras del proceso de limpieza (pasos 0 a 11), para la página "Datos y limpieza"
c = pd.read_csv(CIFRAS).set_index("id").valor
resumen = {k: float(v) for k, v in c.items() if k.split("_")[0][0] == "p"}
(SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
print(f"resumen: {len(resumen)} cifras de limpieza")