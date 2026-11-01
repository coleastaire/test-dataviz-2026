---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Limpieza y preparación

```{code-cell} ipython3
:tags: [remove-cell]
from py.utils import *
from great_tables import GT, style, loc

raw = pd.read_csv(RAW, low_memory=False)
m = pd.read_csv(CLEAN + "vb_matches_limpio.csv", low_memory=False, parse_dates=["date"])
lg = pd.read_csv(CLEAN + "vb_long.csv", low_memory=False)
jg = pd.read_csv(CLEAN + "vb_jugadores.csv", low_memory=False)
JUG = ["w_p1", "w_p2", "l_p1", "l_p2"]
```

La limpieza se hace una sola vez, con el script `py/limpieza.py`, que aplica doce pasos sobre el archivo original y escribe tres archivos limpios. La versión en R repite los mismos pasos y debe obtener las mismas cifras de control. Este capítulo resume qué se encontró y qué se decidió; ninguna fila se elimina por calidad salvo las que son lógicamente imposibles.

## Duplicados y llave del partido

```{code-cell} ipython3
:tags: [remove-cell]
n_dup_exactos = raw.duplicated().sum()
n_llave_falsa = raw.duplicated(["tournament", "year", "gender", "match_num"]).sum()
n_eventos = (m.groupby(["circuit", "tournament", "country", "year"]).evento.max() > 1).sum()
n_match_rep = m.match_id.duplicated().sum()
n_invalidas = int(cifra("p7_filas_invalidas"))

tabla_51 = pd.DataFrame([
    ("Filas duplicadas exactas", n_dup_exactos, "Ninguna acción"),
    ("Torneo + año + género + número de partido repetidos", n_llave_falsa,
     "No es una llave válida: el número se reinicia en cada bracket"),
    ("Torneos con más de un evento en el mismo año", n_eventos,
     "Se separan en eventos si hay más de 14 días entre partidos"),
    ("match_id repetidos", n_match_rep, "Ninguna: la llave nueva es única"),
    ("Partidos con el mismo jugador en ambos equipos", n_invalidas, "Se eliminan")],
    columns=["Chequeo", "Resultado", "Acción"])

registrar("t5_1_n_dup_exactos", n_dup_exactos)
registrar("t5_1_n_llave_falsa", n_llave_falsa)
registrar("t5_1_n_torneos_multi_evento", n_eventos)

pegar("t51_llave_falsa", fmt(n_llave_falsa, 0))
pegar("t51_eventos", fmt(n_eventos, 0))
pegar("t51_invalidas", fmt(n_invalidas, 0))
```

**Tabla 5.1.** Verificación de duplicados y de la llave del partido.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_51)
 .fmt_number(columns=["Resultado"], decimals=0, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

No hay filas duplicadas exactas, pero eso engaña: la combinación torneo-año-género-número de partido se repite {glue}`t51_llave_falsa` veces, porque el número de partido vuelve a empezar en cada bracket (Tabla 5.1). Por eso se construyó una llave propia, `match_id`, que combina el torneo, el evento, el género, el bracket, el número y la fecha; con ella no hay repetidos. Además, {glue}`t51_eventos` torneos se jugaron dos veces el mismo año en el mismo país y se separaron como eventos distintos. Solo se eliminaron {glue}`t51_invalidas` partidos en los que el mismo jugador aparece en los dos equipos, algo imposible.

## Valores faltantes

```{code-cell} ipython3
def cols(sufijos):
    return [f"{j}_{s}" for j in JUG for s in sufijos]

familias = {
    "Aces y bloqueos": (raw, cols(["tot_aces", "tot_blocks"])),
    "Ataques, kills, % de ataque y defensas": (raw, cols(["tot_attacks", "tot_kills", "tot_hitpct", "tot_digs"])),
    "Errores de ataque y de saque": (raw, cols(["tot_errors", "tot_serve_errors"])),
    "Estatura": (raw, cols(["hgt"])),
    "Edad": (raw, cols(["age"])),
    "Duración": (m, ["duracion_min"]),
    "Ranking del ganador": (m, ["w_rank_main"]),
    "Ranking del perdedor": (m, ["l_rank_main"])}

filas = []
for fam, (df_, cs) in familias.items():
    for grupo, sub in [("FIVB", df_[df_.circuit == "FIVB"]),
                       ("AVP", df_[df_.circuit == "AVP"]), ("Total", df_)]:
        filas.append({"familia": fam, "grupo": grupo,
                      "pct": 100 * sub[cs].isna().values.mean()})
na = pd.DataFrame(filas)
orden = (na[na.grupo == "Total"].sort_values("pct").familia.tolist())

fig = go.Figure()
for grupo, color in [("FIVB", PAL["FIVB"]), ("AVP", PAL["AVP"]), ("Total", GRIS_TOTAL)]:
    s = na[na.grupo == grupo].set_index("familia").loc[orden]
    fig.add_bar(x=s.pct, y=s.index, orientation="h", name=grupo, marker_color=color,
                hovertemplate="%{y}<br>" + grupo + ": %{x:.1f} % vacías<extra></extra>")
fig.update_layout(barmode="group", height=560)
fig.update_xaxes(range=[0, 100], ticksuffix=" %")
estilo_plotly(fig, "Porcentaje de celdas vacías por familia de variables y circuito",
              "% de celdas vacías", "")
```

**Figura 5.1.** Porcentaje de celdas vacías por familia de variables y circuito.

```{code-cell} ipython3
:tags: [remove-cell]
tot = na[na.grupo == "Total"].set_index("familia").pct
pct_cob = m.groupby("circuit").tiene_stats_completas.mean() * 100
for fam, v in tot.items():
    registrar("f5_1_pct_" + fam.split()[0].lower(), v)
registrar("f5_1_pct_cobertura_fivb", pct_cob["FIVB"])
registrar("f5_1_pct_cobertura_avp", pct_cob["AVP"])

pegar("f51_aces", fmt_pct(tot["Aces y bloqueos"]))
pegar("f51_errores", fmt_pct(tot["Errores de ataque y de saque"]))
pegar("f51_ataques", fmt_pct(tot["Ataques, kills, % de ataque y defensas"]))
pegar("f51_cob_fivb", fmt_pct(pct_cob["FIVB"]))
pegar("f51_cob_avp", fmt_pct(pct_cob["AVP"]))
pegar("f51_rank_w", fmt_pct(tot["Ranking del ganador"], 2))
pegar("f51_rank_l", fmt_pct(tot["Ranking del perdedor"], 2))
pegar("f51_estatura", fmt_pct(tot["Estatura"]))
pegar("f51_edad", fmt_pct(tot["Edad"]))
pegar("f51_duracion", fmt_pct(tot["Duración"]))
```

Las estadísticas de juego faltan en el {glue}`f51_aces`, {glue}`f51_errores` y {glue}`f51_ataques` de las celdas, según la familia (Figura 5.1). El faltante no es aleatorio: depende del circuito. Solo el {glue}`f51_cob_fivb` de los partidos de la FIVB tiene estadísticas completas, frente al {glue}`f51_cob_avp` de la AVP. También falta el ranking numérico en el {glue}`f51_rank_w` de los ganadores y el {glue}`f51_rank_l` de los perdedores, porque muchas parejas de clasificatoria solo tienen una posición "Q". La estatura falta en el {glue}`f51_estatura` de las celdas; la edad, en el {glue}`f51_edad`, y la duración, en el {glue}`f51_duracion` de los partidos. Ninguno de estos valores se imputa: rellenarlos inventaría información y, en el caso de las estadísticas, haría parecer que la FIVB tiene datos que no tiene.

```{code-cell} ipython3
cob = (m.groupby(["year", "circuit"])
         .agg(pct=("tiene_stats_completas", "mean"), n=("match_id", "size"))
         .reset_index())
cob["pct"] = 100 * cob.pct
anios = pd.DataFrame({"year": range(2000, 2020)})

fig = go.Figure()
for c in ["FIVB", "AVP"]:
    s = anios.merge(cob[cob.circuit == c], on="year", how="left")
    fig.add_scatter(x=s.year, y=s.pct, name=c, mode="lines+markers",
                    line=dict(color=PAL[c], width=2), marker=dict(size=7),
                    customdata=s.n, connectgaps=False,
                    hovertemplate="%{x} · " + c +
                    "<br>%{y:.1f} % con estadísticas<br>%{customdata:,} partidos<extra></extra>")
fig.update_xaxes(dtick=1, tickangle=-45)
fig.update_yaxes(range=[0, 100], ticksuffix=" %")
estilo_plotly(fig, "Cobertura de estadísticas de juego por año y circuito (% de partidos)",
              "Año", "% de partidos con estadísticas completas")
```

**Figura 5.2.** Cobertura de estadísticas de juego por año y circuito (% de partidos).

```{code-cell} ipython3
:tags: [remove-cell]
n_stats = int(m.tiene_stats_completas.sum())
pct_stats = 100 * m.tiene_stats_completas.mean()
pct_stats_avp = 100 * m[m.tiene_stats_completas == 1].circuit.eq("AVP").mean()
avp_cob = cob[cob.circuit == "AVP"].set_index("year").pct
anios_avp_bajos = avp_cob[avp_cob < 10].index.tolist()
registrar("f5_2_n_stats", n_stats)
registrar("f5_2_pct_stats_avp", pct_stats_avp)

pegar("f52_n_stats", fmt(n_stats, 0))
pegar("f52_pct_stats", fmt_pct(pct_stats))
pegar("f52_pct_avp", fmt_pct(pct_stats_avp))
pegar("f52_anios_bajos", ", ".join(str(a) for a in anios_avp_bajos))
```

Solo {glue}`f52_n_stats` partidos ({glue}`f52_pct_stats`) tienen estadísticas completas, y el {glue}`f52_pct_avp` de ellos son de la AVP (Figura 5.2). Incluso dentro de la AVP la cobertura no es pareja: es casi nula en {glue}`f52_anios_bajos`. En la FIVB es mínima todos los años. Por eso, en este libro, las estadísticas de juego describen a la AVP de 2003 a 2019 y no al voleibol de playa en general; cualquier gráfico que las use lo indica en su título.

## Correcciones de calidad

```{code-cell} ipython3
:tags: [remove-cell]
c = lambda i: fmt(cifra(i), 0)
tabla_52 = pd.DataFrame([
    ("Fechas de nacimiento en el futuro (siglo mal leído)", f"{c('p1_partidos_bd_corregida')} partidos", "Restar 100 años y marcar; la edad se toma de la columna age", 1, "Alta"),
    ("Ranking guardado como texto (\"17, Q2\", \"Q3\")", "Todos los partidos", "Separar ranking principal, posición de clasificatoria y bandera es_clasif", 4, "Alta"),
    ("Estadísticas de juego vacías y concentradas en la AVP", f"{fmt_pct(100 - pct_stats)} de los partidos", "Bandera de cobertura; no se imputan", 8, "Alta"),
    ("Llave del partido repetida", f"{fmt(n_llave_falsa, 0)} repeticiones", "Crear match_id y tournament_id", 7, "Alta"),
    ("Mismo jugador en ambos equipos", f"{c('p7_filas_invalidas')} partidos", "Eliminar las filas", 7, "Alta"),
    ("Estadísticas imposibles (kills > ataques)", f"{c('p8_celdas_invalidas')} celdas", "Anular las celdas y recalcular el % de ataque", 8, "Media"),
    ("Marcador con forfeits, retiros o vacío", f"{c('p5_forfeit')} forfeits y {c('p5_retirado')} retiros", "Banderas; forfeits fuera de duración y puntos", 5, "Media"),
    ("36 nombres de bracket sin agrupar", "Todos los partidos", "Agrupar en 5 etapas ordenadas", 6, "Media"),
    ("Round vacío", f"{fmt_pct(100 * raw['round'].isna().mean())} de los partidos", "Etiquetar como \"No aplica\" (no es dato perdido)", 6, "Baja"),
    ("Duración como texto HH:MM:SS", f"{c('p2_duracion_na')} vacías", "Convertir a minutos; no se imputa", 2, "Media"),
    ("Nombres de país duplicados y países vacíos", f"{c('p3_pais_desconocido')} celdas vacías", "Unificar 3 nombres y usar \"Desconocido\"", 3, "Baja"),
    ("Estatura en pulgadas y con faltantes", f"{fmt_pct(tot['Estatura'])} de celdas vacías", "Convertir a cm; no se imputa", 9, "Media"),
    ("Edad que no cuadra con las fechas", f"{c('p9_edad_inconsistente')} partidos", "Marcar con bandera; se conservan", 9, "Baja"),
    ("Cobertura temporal desigual (2000, AVP 2011–2012, 2019)", "3 periodos", "Documentar; los conteos por año se leen también como cobertura del conjunto de datos", "—", "Media")],
    columns=["Problema", "Alcance", "Decisión", "Paso", "Prioridad"])
tabla_52.insert(0, "#", range(1, len(tabla_52) + 1))
tabla_52["Paso"] = tabla_52["Paso"].astype(str)

pegar("t52_n", str(len(tabla_52)))
pegar("t52_bd", c("p1_partidos_bd_corregida"))
```

**Tabla 5.2.** Registro de problemas de calidad y decisiones tomadas.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_52)
 .tab_style(style=style.text(weight="bold"),
            locations=loc.body(columns="Prioridad", rows=tabla_52.index[tabla_52.Prioridad == "Alta"].tolist()))
 .cols_align(align="center", columns=["#", "Paso", "Prioridad"])
 .opt_stylize(style=1, color="gray"))
```

Se registraron {glue}`t52_n` problemas de calidad (Tabla 5.2). La mayoría son silenciosos: no producen ningún error al abrir el archivo, pero distorsionan gráficos y conclusiones si no se corrigen. Los más graves son las fechas de nacimiento en el futuro ({glue}`t52_bd` partidos), el ranking guardado como texto y el faltante estructural de las estadísticas. La columna "Paso" indica en qué paso del script de limpieza se corrige cada uno.

## Formato largo y archivos limpios

```{code-cell} ipython3
:tags: [remove-cell]
tabla_53 = pd.DataFrame([
    ("vb_matches_limpio.csv", "Partido", len(m), m.shape[1], "1, 4, 5 y 6"),
    ("vb_long.csv", "Equipo en un partido (Ganador / Perdedor)", len(lg), lg.shape[1], "6"),
    ("vb_jugadores.csv", "Jugador en un partido", len(jg), jg.shape[1], "1, 4 y 6")],
    columns=["Archivo", "Unidad de fila", "Filas", "Columnas", "Capítulos"])
for _, f in tabla_53.iterrows():
    registrar("t5_3_n_filas_" + f.Archivo.split(".")[0], f.Filas)
    registrar("t5_3_n_columnas_" + f.Archivo.split(".")[0], f.Columnas)

pegar("t53_m", fmt(len(m), 0))
pegar("t53_lg", fmt(len(lg), 0))
pegar("t53_jg", fmt(len(jg), 0))
```

**Tabla 5.3.** Archivos limpios generados.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_53)
 .fmt_number(columns=["Filas", "Columnas"], decimals=0, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

El análisis parte de tres archivos con distinta unidad de observación (Tabla 5.3): {glue}`t53_m` partidos, {glue}`t53_lg` filas equipo-partido y {glue}`t53_jg` filas jugador-partido. El formato equipo-partido es el que permite comparar ganadores y perdedores en el capítulo siguiente: cada partido aparece dos veces, una por pareja, con una columna `resultado` que vale "Ganador" o "Perdedor".

