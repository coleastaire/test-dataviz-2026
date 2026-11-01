"""Pipeline de limpieza canónico (guía, sección 2).

Lee el CSV crudo, aplica los pasos 0 a 12 y escribe los CSV limpios en
data/clean/. Registra las cifras de control en compartido/cifras_control_py.csv.

Cómo ejecutarlo (desde la carpeta py-book, con el entorno activado):
    python -m py.limpieza
"""
from py.utils import *

# Los 4 jugadores de cada partido y la columna con su nombre
JUG = ["w_p1", "w_p2", "l_p1", "l_p2"]
NOMBRE = {"w_p1": "w_player1", "w_p2": "w_player2",
          "l_p1": "l_player1", "l_p2": "l_player2"}


# ---------------------------------------------------------------
# Paso 0. Cargar el crudo y numerar filas
# ---------------------------------------------------------------
def paso0_cargar():
    raw = pd.read_csv(RAW, dtype={"w_rank": str, "l_rank": str,
                                  "duration": str, "score": str})
    df = raw.copy()
    df["row_id"] = np.arange(1, len(df) + 1)      # posición original (1..76.756)
    df["date"] = pd.to_datetime(df["date"])
    registrar("p0_filas_crudas", len(df))
    return df


# ---------------------------------------------------------------
# Paso 1. Fechas de nacimiento en el futuro (problema 1)
# ---------------------------------------------------------------
def paso1_fechas_nacimiento(df):
    for j in JUG:
        bd = pd.to_datetime(df[f"{j}_birthdate"])
        malo = bd.dt.year > 2005                   # año imposible para un jugador
        df[f"{j}_bd_corregida"] = malo.astype(int)
        df[f"{j}_birthdate"] = bd.mask(malo, bd - pd.DateOffset(years=100))
    cols = [f"{j}_bd_corregida" for j in JUG]
    df["bd_corregida"] = (df[cols].sum(axis=1) > 0).astype(int)
    registrar("p1_partidos_bd_corregida", df["bd_corregida"].sum())
    return df


# ---------------------------------------------------------------
# Paso 2. Duración HH:MM:SS -> minutos (problema 12)
# ---------------------------------------------------------------
def paso2_duracion(df):
    df["duracion_min"] = (pd.to_timedelta(df["duration"], errors="coerce")
                          .dt.total_seconds() / 60)
    registrar("p2_duracion_na", df["duracion_min"].isna().sum())
    return df


# ---------------------------------------------------------------
# Paso 3. Países (problema 7)
# ---------------------------------------------------------------
MAPA_PAIS = {"Slovak Republic": "Slovakia",
             "CÃ´te d'Ivoire": "Côte d'Ivoire",
             "Principality of Monaco": "Monaco"}


def paso3_paises(df):
    df["country"] = df["country"].replace(MAPA_PAIS)
    for j in JUG:
        df[f"{j}_country"] = (df[f"{j}_country"].replace(MAPA_PAIS)
                              .fillna("Desconocido"))
    cols = [f"{j}_country" for j in JUG]
    registrar("p3_pais_desconocido", (df[cols] == "Desconocido").sum().sum())
    return df


# ---------------------------------------------------------------
# Paso 4. Rankings: "17, Q2" -> rank_main 17, rank_qual 2, es_clasif 1
# ---------------------------------------------------------------
def paso4_rankings(df):
    for lado in ["w", "l"]:
        r = df[f"{lado}_rank"]
        df[f"{lado}_rank_main"] = pd.to_numeric(
            r.str.split(",").str[0].str.strip(), errors="coerce")
        df[f"{lado}_rank_qual"] = r.str.extract(r"Q(\d+)")[0].astype(float)
        df[f"{lado}_es_clasif"] = r.str.contains("Q", na=False).astype(int)
    registrar("p4_w_rank_main_validos", df["w_rank_main"].notna().sum())
    registrar("p4_l_rank_main_validos", df["l_rank_main"].notna().sum())
    return df


# ---------------------------------------------------------------
# Paso 5. Marcador: sets, puntos, forfeits y retiros (problema 5)
# ---------------------------------------------------------------
def paso5_marcador(df):
    # Cada set viene como "21-17"; se toman hasta 3 sets
    sets = df["score"].fillna("").str.findall(r"(\d+)-(\d+)").str[:3]
    pg = sets.apply(lambda L: [int(a) for a, b in L])   # puntos del ganador
    pp = sets.apply(lambda L: [int(b) for a, b in L])   # puntos del perdedor

    df["es_forfeit"] = (df["score"].isna() |
                        df["score"].str.contains("Forfeit", na=False)).astype(int)
    df["es_retirado"] = df["score"].str.contains("retired", case=False,
                                                 na=False).astype(int)
    df["n_sets"] = sets.str.len()
    df["sets_ganador"] = [sum(a > b for a, b in zip(g, p)) for g, p in zip(pg, pp)]
    df["sets_perdedor"] = [sum(a < b for a, b in zip(g, p)) for g, p in zip(pg, pp)]
    df["puntos_ganador"] = [sum(g) if g else np.nan for g in pg]
    df["puntos_perdedor"] = [sum(p) if p else np.nan for p in pp]
    df["puntos_totales"] = df["puntos_ganador"] + df["puntos_perdedor"]
    registrar("p5_forfeit", df["es_forfeit"].sum())
    registrar("p5_retirado", df["es_retirado"].sum())
    return df


# ---------------------------------------------------------------
# Paso 6. Etapa del torneo (36 brackets -> 5 etapas) y round (problema 6)
# ---------------------------------------------------------------
def paso6_etapa(df):
    b = df["bracket"]
    df["etapa"] = np.select(
        [b.str.startswith("Pool"),
         b.isin(["Winner's Bracket", "Contender's Bracket"]),
         b.isin(["Qualifier Bracket", "Qualifier Playoff",
                 "Country Quota Matches", "Lucky Losers"]),
         b.eq("Semifinals"),
         b.isin(["Finals", "Gold Medal", "Bronze Medal", "3rd Place"])],
        ["Fase de grupos", "Cuadro principal", "Clasificatoria",
         "Semifinales", "Finales"],
        default="Otra")
    df["etapa"] = pd.Categorical(df["etapa"], categories=ETAPAS + ["Otra"],
                                 ordered=True)
    df["round"] = df["round"].fillna("No aplica")   # vacío = no aplica, no es NA
    registrar("p6_etapa_otra", (df["etapa"] == "Otra").sum())
    return df


# ---------------------------------------------------------------
# Paso 7. Llaves del partido y filas inválidas (problemas 4 y 13)
# ---------------------------------------------------------------
def paso7_llaves(df):
    claves = ["circuit", "tournament", "country", "year"]
    df = df.sort_values(claves + ["date", "row_id"]).reset_index(drop=True)

    # (a) un torneo se parte en eventos si hay > 14 días entre partidos seguidos
    salto = (df.groupby(claves)["date"].diff().dt.days > 14).astype(int)
    df["evento"] = salto.groupby([df[c] for c in claves]).cumsum() + 1

    # (b) llaves únicas
    df["tournament_id"] = (df[claves].astype(str).agg("|".join, axis=1)
                           + "|" + df["evento"].astype(str))
    df["match_id"] = (df["tournament_id"] + "|" + df["gender"] + "|"
                      + df["bracket"] + "|" + df["match_num"].astype(str)
                      + "|" + df["date"].dt.strftime("%Y-%m-%d"))

    # (c) se eliminan las filas con el mismo jugador en ambos equipos
    mismo = ((df.w_player1 == df.l_player1) | (df.w_player1 == df.l_player2) |
             (df.w_player2 == df.l_player1) | (df.w_player2 == df.l_player2))
    registrar("p7_filas_invalidas", mismo.sum())
    df = df[~mismo].reset_index(drop=True)
    registrar("p7_match_id_duplicados", df["match_id"].duplicated().sum())

    # (d) identificador de jugador: nombre | fecha de nacimiento (o género)
    for j in JUG:
        df[f"{j}_player_id"] = (
            df[NOMBRE[j]] + "|" +
            df[f"{j}_birthdate"].dt.strftime("%Y-%m-%d").fillna(df["gender"]))
    registrar("p7_filas_finales", len(df))
    return df


# ---------------------------------------------------------------
# Paso 8. Estadísticas de juego (problemas 3 y 8)
# ---------------------------------------------------------------
def paso8_estadisticas(df):
    # (a) banderas de cobertura (antes de anular celdas)
    df["tiene_stats_completas"] = (df[[f"{j}_tot_kills" for j in JUG]]
                                   .notna().all(axis=1).astype(int))
    df["tiene_stats_basicas"] = (df[[f"{j}_tot_aces" for j in JUG]]
                                 .notna().all(axis=1).astype(int))
    df["stat_invalida"] = 0
    celdas = 0
    for j in JUG:
        k, a, e, h = (f"{j}_tot_{s}" for s in ["kills", "attacks", "errors", "hitpct"])
        # (b) valores imposibles -> NA
        malo = df[a].notna() & df[k].notna() & ((df[a] < 0) | (df[k] > df[a]))
        celdas += malo.sum()
        df.loc[malo, "stat_invalida"] = 1
        df.loc[malo, [k, a]] = np.nan
        # (c) hitpct recalculado; fuera de [-1, 1] -> NA
        df[h] = np.where(df[a] > 0, (df[k] - df[e]) / df[a], np.nan)
        df.loc[df[h].abs() > 1, h] = np.nan
    registrar("p8_stats_completas", df["tiene_stats_completas"].sum())
    registrar("p8_stats_basicas", df["tiene_stats_basicas"].sum())
    registrar("p8_celdas_invalidas", celdas)
    return df

# ---------------------------------------------------------------
# Paso 9. Estatura en cm y edad inconsistente (problemas 9 y 10)
# ---------------------------------------------------------------
def paso9_estatura_edad(df):
    df["edad_inconsistente"] = 0
    for j in JUG:
        df[f"{j}_hgt_cm"] = df[f"{j}_hgt"] * 2.54          # pulgadas -> cm
        calc = (df["date"] - df[f"{j}_birthdate"]).dt.days / 365.25
        dif = df[f"{j}_age"] - calc
        df.loc[dif.abs() > 0.5, "edad_inconsistente"] = 1
    registrar("p9_edad_inconsistente", df["edad_inconsistente"].sum())
    return df


# ---------------------------------------------------------------
# Paso 10. Variables de pareja, upset y orientación A/B del modelo
# ---------------------------------------------------------------
def paso10_pareja(df):
    # (a) media de la pareja: NA si falta uno de los dos (nunca mean(axis=1))
    df["w_edad"] = (df.w_p1_age + df.w_p2_age) / 2
    df["l_edad"] = (df.l_p1_age + df.l_p2_age) / 2
    df["w_est"] = (df.w_p1_hgt_cm + df.w_p2_hgt_cm) / 2
    df["l_est"] = (df.l_p1_hgt_cm + df.l_p2_hgt_cm) / 2

    # (b) upset: gana el peor rankeado (rank mayor = peor)
    df["rank_diff"] = df.l_rank_main - df.w_rank_main
    df["es_upset"] = np.where(df.rank_diff.isna(), np.nan,
                              (df.rank_diff < 0).astype(float))

    # (c) equipo A = ganador si row_id es impar (regla fija, sin azar)
    df["a_gana"] = (df.row_id % 2 == 1).astype(int)
    signo = np.where(df.a_gana == 1, 1, -1)

    # (d) diferencias A - B
    df["dif_rank"] = signo * (df.w_rank_main - df.l_rank_main)
    df["dif_edad"] = signo * (df.w_edad - df.l_edad)
    df["dif_estatura_cm"] = signo * (df.w_est - df.l_est)
    df["dif_clasif"] = signo * (df.w_es_clasif - df.l_es_clasif)

    registrar("p10_partidos_con_ambos_rank", df.rank_diff.notna().sum())
    registrar("p10_pct_upset", 100 * df.es_upset.mean())
    return df


# ---------------------------------------------------------------
# Paso 11. Formato largo: equipo-partido y jugador-partido
# ---------------------------------------------------------------
STATS = ["kills", "aces", "blocks", "digs", "attacks", "errors", "serve_errors"]


def equipo(d, yo, rival, etiqueta):
    """Una fila por partido para el equipo 'yo' ('w' o 'l')."""
    out = pd.DataFrame({
        "match_id": d.match_id, "tournament_id": d.tournament_id,
        "year": d.year, "circuit": d.circuit, "gender": d.gender,
        "etapa": d.etapa, "resultado": etiqueta,
        "rank_main": d[f"{yo}_rank_main"], "rank_rival": d[f"{rival}_rank_main"],
        "es_clasif": d[f"{yo}_es_clasif"],
        "edad_media": d[f"{yo}_edad"], "estatura_media_cm": d[f"{yo}_est"],
        "dif_edad_pareja": (d[f"{yo}_p1_age"] - d[f"{yo}_p2_age"]).abs(),
        "dif_estatura_pareja_cm": (d[f"{yo}_p1_hgt_cm"] - d[f"{yo}_p2_hgt_cm"]).abs(),
        "ventaja_edad": d[f"{yo}_edad"] - d[f"{rival}_edad"],
        "ventaja_estatura_cm": d[f"{yo}_est"] - d[f"{rival}_est"],
        "duracion_min": d.duracion_min, "n_sets": d.n_sets,
        "es_forfeit": d.es_forfeit,
        "tiene_stats_completas": d.tiene_stats_completas,
        "tiene_stats_basicas": d.tiene_stats_basicas})
    out["ventaja_rank"] = out.rank_rival - out.rank_main   # + = mejor rankeado
    for s in STATS:                                          # suma de la pareja
        out[s] = d[f"{yo}_p1_tot_{s}"] + d[f"{yo}_p2_tot_{s}"]
    out["hitpct"] = np.where(out.attacks > 0,
                             (out.kills - out.errors) / out.attacks, np.nan)
    return out


def paso11_formato_largo(df):
    vb_long = (pd.concat([equipo(df, "w", "l", "Ganador"),
                          equipo(df, "l", "w", "Perdedor")])
               .sort_values(["match_id", "resultado"]).reset_index(drop=True))

    vb_jugadores = (pd.concat([pd.DataFrame({
        "match_id": df.match_id, "year": df.year, "circuit": df.circuit,
        "gender": df.gender, "etapa": df.etapa, "rol": j,
        "resultado": "Ganador" if j[0] == "w" else "Perdedor",
        "player_id": df[f"{j}_player_id"], "edad": df[f"{j}_age"],
        "estatura_cm": df[f"{j}_hgt_cm"], "pais": df[f"{j}_country"]})
        for j in JUG])
        .sort_values(["match_id", "rol"]).reset_index(drop=True))

    registrar("p11_filas_long", len(vb_long))
    registrar("p11_filas_jugadores", len(vb_jugadores))
    return vb_long, vb_jugadores


# ---------------------------------------------------------------
# Paso 12. Exportar los CSV limpios
# ---------------------------------------------------------------
def paso12_exportar(df, vb_long, vb_jugadores):
    for nombre, d in [("vb_matches_limpio", df), ("vb_long", vb_long),
                      ("vb_jugadores", vb_jugadores)]:
        d = d.copy()
        decimales = d.select_dtypes("float").columns       # solo columnas decimales
        d[decimales] = d[decimales].round(4)
        d.to_csv(f"{CLEAN}{nombre}.csv", index=False, na_rep="",
                 date_format="%Y-%m-%d", encoding="utf-8")
        print(f"  {nombre}.csv: {len(d):,} filas x {d.shape[1]} columnas")

# ---------------------------------------------------------------
# Ejecución completa
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = paso0_cargar()
    df = paso1_fechas_nacimiento(df)
    df = paso2_duracion(df)
    df = paso3_paises(df)
    df = paso4_rankings(df)
    df = paso5_marcador(df)
    df = paso6_etapa(df)
    df = paso7_llaves(df)
    df = paso8_estadisticas(df)
    df = paso9_estatura_edad(df)
    df = paso10_pareja(df)
    vb_long, vb_jugadores = paso11_formato_largo(df)
    
    print("Archivos escritos en data/clean/:")
    paso12_exportar(df, vb_long, vb_jugadores)

    print("Cifras de control:")
    print(pd.DataFrame(CIFRAS).to_string(index=False))