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

# Descripción del conjunto de datos

```{code-cell} ipython3
:tags: [remove-cell]
from py.utils import *
from great_tables import GT

raw = pd.read_csv(RAW, low_memory=False)
m = pd.read_csv(CLEAN + "vb_matches_limpio.csv", low_memory=False, parse_dates=["date"])
lg = pd.read_csv(CLEAN + "vb_long.csv", low_memory=False)
jg = pd.read_csv(CLEAN + "vb_jugadores.csv", low_memory=False)
JUG = ["w_p1", "w_p2", "l_p1", "l_p2"]
```

Este capítulo describe el conjunto de datos antes de cualquier análisis: qué variables contiene, cómo se reparten los partidos entre circuitos, géneros y etapas, de dónde vienen los jugadores y qué tan frecuentes son los valores atípicos.

## Estructura y variables

```{code-cell} ipython3
:tags: [remove-cell]
grupos = {
    "Identificación del partido": ["circuit", "tournament", "country", "year",
                                   "date", "gender", "match_num"],
    "Jugadores (× 4)": ["w_player1", "w_player2", "l_player1", "l_player2"]
        + [f"{j}_{c}" for j in JUG for c in ["birthdate", "age", "hgt", "country"]],
    "Ranking": ["w_rank", "l_rank"],
    "Resultado del partido": ["score", "duration", "bracket", "round"],
    "Estadísticas de juego (× 4)": [c for c in raw.columns if "_tot_" in c]}
tipos = {"Identificación del partido": "Texto, entero y fecha",
         "Jugadores (× 4)": "Texto, fecha y decimal",
         "Ranking": "Texto (\"17\", \"20, Q6\", \"Q3\")",
         "Resultado del partido": "Texto",
         "Estadísticas de juego (× 4)": "Decimal"}
pct_na = raw.isna().mean() * 100
tabla_41 = pd.DataFrame([{
    "Grupo": g, "Columnas": len(cols),
    "Ejemplos": ", ".join(cols[:3]), "Tipo": tipos[g],
    "% vacío (mín.)": pct_na[cols].min(), "% vacío (máx.)": pct_na[cols].max()}
    for g, cols in grupos.items()])
n_filas, n_cols = raw.shape
st = grupos["Estadísticas de juego (× 4)"]
registrar("t4_1_n_filas", n_filas)
registrar("t4_1_n_columnas", n_cols)

pegar("t41_filas", fmt(n_filas, 0))
pegar("t41_cols", fmt(n_cols, 0))
pegar("t41_n_st", fmt(len(st), 0))
pegar("t41_na_min", fmt_pct(pct_na[st].min()))
pegar("t41_na_max", fmt_pct(pct_na[st].max()))
```

**Tabla 4.1.** Estructura del conjunto de datos original.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_41)
 .fmt_number(columns=["% vacío (mín.)", "% vacío (máx.)"], decimals=1,
             sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

El archivo original tiene {glue}`t41_filas` partidos y {glue}`t41_cols` columnas, organizadas en cinco grupos (Tabla 4.1). Las {glue}`t41_n_st` columnas de estadísticas de juego concentran los faltantes: entre el {glue}`t41_na_min` y el {glue}`t41_na_max` de sus celdas están vacías. El ranking, la duración y el marcador vienen como texto, por lo que requieren conversión antes de analizarse (capítulo de limpieza).

## Cobertura por circuito, género y etapa

```{code-cell} ipython3
d = m.groupby(["circuit", "gender"]).size().reset_index(name="n")
d["pct"] = 100 * d.n / d.n.sum()

fig = go.Figure()
for g in ["M", "W"]:
    dd = d[d.gender == g].set_index("circuit").loc[["FIVB", "AVP"]].reset_index()
    fig.add_bar(x=dd.circuit, y=dd.n, name=g, marker_color=PAL[g],
                text=[fmt(v, 0) for v in dd.n], textposition="outside",
                customdata=dd.pct,
                hovertemplate="Circuito: %{x}<br>Género: " + g +
                "<br>Partidos: %{y:,}<br>%{customdata:.1f} % del total<extra></extra>")
fig.update_layout(barmode="group")
fig.update_yaxes(range=[0, d.n.max() * 1.15], tickformat=",d")
estilo_plotly(fig, "Partidos por circuito y género, 2000–2019",
              "Circuito", "Número de partidos")
```

**Figura 4.1.** Partidos por circuito y género, 2000–2019.

```{code-cell} ipython3
:tags: [remove-cell]
n_fivb = (m.circuit == "FIVB").sum()
n_avp = (m.circuit == "AVP").sum()
pct_m = 100 * (m.gender == "M").mean()
anio_avp = m[m.circuit == "AVP"].year
for _, f in d.iterrows():
    registrar(f"f4_1_n_{f.circuit}_{f.gender}", f.n)

pegar("f41_pct_fivb", fmt_pct(100 * n_fivb / len(m), 2))
pegar("f41_n_fivb", fmt(n_fivb, 0))
pegar("f41_pct_avp", fmt_pct(100 * n_avp / len(m), 2))
pegar("f41_n_avp", fmt(n_avp, 0))
pegar("f41_pct_m", fmt_pct(pct_m, 2))
pegar("f41_pct_w", fmt_pct(100 - pct_m, 2))
pegar("f41_avp_desde", str(anio_avp.min()))
pegar("f41_avp_hasta", str(anio_avp.max()))
```

La FIVB concentra el {glue}`f41_pct_fivb` de los partidos ({glue}`f41_n_fivb`) y la AVP el {glue}`f41_pct_avp` ({glue}`f41_n_avp`). El género está equilibrado: {glue}`f41_pct_m` de partidos masculinos y {glue}`f41_pct_w` femeninos (Figura 4.1). Todo análisis que mezcle circuitos está dominado por la FIVB en proporción de dos a uno. Además, los partidos de la AVP solo cubren {glue}`f41_avp_desde`–{glue}`f41_avp_hasta`: el periodo 2000–2019 describe el conjunto, no a cada circuito.

```{code-cell} ipython3
e = m.etapa.value_counts().reindex(ETAPAS).reset_index()
e.columns = ["etapa", "n"]
e["pct"] = 100 * e.n / len(m)
e["texto"] = [f"{fmt(n, 0)} ({fmt_pct(p)})" for n, p in zip(e.n, e.pct)]

fig = go.Figure(go.Bar(
    x=e.n, y=e.etapa, orientation="h",
    marker_color=[PAL_ETAPA[x] for x in e.etapa],
    marker_line=dict(color="#737373", width=0.5),
    text=e.texto, textposition="outside", customdata=e.pct,
    hovertemplate="%{y}<br>Partidos: %{x:,}<br>%{customdata:.1f} % del total<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(range=[0, e.n.max() * 1.25])
estilo_plotly(fig, "Partidos por etapa del torneo", "Número de partidos", "")
```

**Figura 4.2.** Partidos por etapa del torneo.

```{code-cell} ipython3
:tags: [remove-cell]
et = e.set_index("etapa")
for k, v in et.n.items():
    registrar(f"f4_2_n_{k.replace(' ', '_')}", v)

pegar("f42_n_cp", fmt(et.n["Cuadro principal"], 0))
pegar("f42_pct_cp", fmt_pct(et.pct["Cuadro principal"]))
pegar("f42_n_cl", fmt(et.n["Clasificatoria"], 0))
pegar("f42_pct_cl", fmt_pct(et.pct["Clasificatoria"]))
pegar("f42_n_sf", fmt(et.n["Semifinales"] + et.n["Finales"], 0))
```

El cuadro principal reúne {glue}`f42_n_cp` partidos ({glue}`f42_pct_cp`) y la clasificatoria {glue}`f42_n_cl` ({glue}`f42_pct_cl`), que es la etapa donde aparecen los rankings tipo "Q" (Figura 4.2). Semifinales y finales suman {glue}`f42_n_sf` partidos: son pocos, pero son los que definen los torneos.

## Países sede y de procedencia

```{code-cell} ipython3
:tags: [remove-cell]
sede = m.country.value_counts().head(10)
pj = (jg.groupby("pais")
        .agg(registros=("player_id", "size"), unicos=("player_id", "nunique"))
        .sort_values("registros", ascending=False).head(10))
tabla_42 = pd.DataFrame({
    "Puesto": range(1, 11),
    "País sede": sede.index, "Partidos": sede.values,
    "% de partidos": 100 * sede.values / len(m),
    "País del jugador": pj.index, "Registros": pj.registros.values,
    "Jugadores únicos": pj.unicos.values})
registrar("t4_2_n_sede_usa", sede["United States"])
registrar("t4_2_n_registros_usa", pj.registros["United States"])
registrar("t4_2_n_unicos_usa", pj.unicos["United States"])

pegar("t42_sede_usa", fmt(sede["United States"], 0))
pegar("t42_sede2", sede.index[1])
pegar("t42_sede2_n", fmt(sede.iloc[1], 0))
pegar("t42_reg_usa", fmt(pj.registros.iloc[0], 0))
pegar("t42_uni_usa", fmt(pj.unicos.iloc[0], 0))
pegar("t42_pais2", pj.index[1])
pegar("t42_reg2", fmt(pj.registros.iloc[1], 0))
pegar("t42_uni2", fmt(pj.unicos.iloc[1], 0))
pegar("t42_pct_usa", fmt_pct(100 * pj.registros.iloc[0] / len(jg), 0))
```

**Tabla 4.2.** Top 10 de países sede y de países de procedencia de los jugadores.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_42)
 .tab_spanner(label="Dónde se juega", columns=["País sede", "Partidos", "% de partidos"])
 .tab_spanner(label="De dónde vienen los jugadores",
              columns=["País del jugador", "Registros", "Jugadores únicos"])
 .fmt_number(columns=["Partidos", "Registros", "Jugadores únicos"], decimals=0,
             sep_mark=".", dec_mark=",")
 .fmt_number(columns=["% de partidos"], decimals=1, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

Estados Unidos es sede de {glue}`t42_sede_usa` partidos, muy por encima de {glue}`t42_sede2` ({glue}`t42_sede2_n`), porque ahí se juega todo el circuito AVP (Tabla 4.2). También aporta la mayor cantidad de registros de jugador ({glue}`t42_reg_usa`) y de jugadores distintos ({glue}`t42_uni_usa`), seguido por {glue}`t42_pais2` con {glue}`t42_reg2` registros pero solo {glue}`t42_uni2` jugadores: pocos jugadores brasileños, pero que juegan muchos partidos, lo que sugiere un grupo reducido de parejas muy competitivas.

## Datos atípicos

```{code-cell} ipython3
dd = m[(m.es_forfeit == 0) & m.duracion_min.notna()]
fig = go.Figure()
for k in [1, 2, 3]:
    s = dd[dd.n_sets == k].duracion_min
    fig.add_trace(go.Box(y=s, name=f"{k} set" + ("" if k == 1 else "s"),
                         marker_color=NEUTRO, boxpoints="outliers",
                         marker=dict(size=3, opacity=0.5),
                         hovertemplate="%{y:.0f} min<extra></extra>"))
fig.update_layout(showlegend=False)
estilo_plotly(fig, "Duración del partido según el número de sets (min)",
              "Número de sets", "Duración (min)")
```

**Figura 4.3.** Duración del partido según el número de sets (min).

```{code-cell} ipython3
:tags: [remove-cell]
def limites(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)

def atipicos(s):
    lo, hi = limites(s)
    return ((s < lo) | (s > hi)).sum()

med_sets = dd.groupby("n_sets").duracion_min.median()
lo_g, hi_g = limites(dd.duracion_min)
n_glob = atipicos(dd.duracion_min)
n_por_sets = sum(atipicos(g.duracion_min) for _, g in dd.groupby("n_sets"))
cortos = dd[dd.duracion_min < 10]
for k, v in med_sets.items():
    registrar(f"f4_3_mediana_{k}_sets", v)
registrar("f4_3_n_atipicos_global", n_glob)
registrar("f4_3_n_atipicos_por_sets", n_por_sets)

pegar("f43_med1", fmt(med_sets[1], 0))
pegar("f43_med2", fmt(med_sets[2], 0))
pegar("f43_med3", fmt(med_sets[3], 0))
pegar("f43_lo", fmt(lo_g, 1))
pegar("f43_hi", fmt(hi_g, 1))
pegar("f43_n_glob", fmt(n_glob, 0))
pegar("f43_n_sets", fmt(n_por_sets, 0))
pegar("f43_n_cortos", fmt(len(cortos), 0))
pegar("f43_n_ret", fmt(cortos.es_retirado.sum(), 0))
```

La mediana de duración es de {glue}`f43_med1`, {glue}`f43_med2` y {glue}`f43_med3` minutos para partidos de 1, 2 y 3 sets (Figura 4.3). Un criterio de rango intercuartílico (IQR) aplicado a todos los partidos juntos fija los límites en [{glue}`f43_lo`; {glue}`f43_hi`] minutos y marca {glue}`f43_n_glob` partidos como atípicos, casi todos partidos a 3 sets que simplemente son más largos. Por eso los atípicos se buscan dentro de cada número de sets: con ese criterio son {glue}`f43_n_sets`, porque cada grupo es más homogéneo y sus límites más estrechos. Los {glue}`f43_n_cortos` partidos de menos de 10 minutos son, en {glue}`f43_n_ret` casos, partidos con retiro de una pareja: son legítimos, no errores de registro.

```{code-cell} ipython3
:tags: [remove-cell]
filas = []
def fila(clave, variable, regla, s, unidad, decision, d=1):
    s = s.dropna()
    lo, hi = limites(s)
    n = ((s < lo) | (s > hi)).sum()
    filas.append({"Variable": variable, "Regla": regla,
                  "Límites": f"[{fmt(lo, d)}; {fmt(hi, d)}] {unidad}",
                  "N atípicos": n, "% atípicos": 100 * n / len(s),
                  "Decisión": decision})
    registrar(f"t4_3_n_{clave}", n)

fila("edad", "Edad del jugador", "IQR", jg.edad, "años", "Se conservan: juniors y veteranos reales")
fila("estatura_h", "Estatura (hombres)", "IQR por género", jg[jg.gender == "M"].estatura_cm, "cm", "Se conservan: jugadores muy altos o bajos reales")
fila("estatura_m", "Estatura (mujeres)", "IQR por género", jg[jg.gender == "W"].estatura_cm, "cm", "Se conservan")
fila("ranking", "Ranking principal", "IQR", lg.rank_main, "posición", "Se conservan: equipos de ranking bajo", d=0)
filas.append({"Variable": "Duración", "Regla": "IQR por número de sets",
              "Límites": "Según sets (ver texto)", "N atípicos": n_por_sets,
              "% atípicos": 100 * n_por_sets / len(dd),
              "Decisión": "Se conservan; < 10 min son retiros"})
celdas = int(cifra("p8_celdas_invalidas"))      # viene del pipeline
filas.append({"Variable": "Estadísticas de juego", "Regla": "Lógica: kills > ataques o ataques < 0",
              "Límites": "—", "N atípicos": celdas, "% atípicos": np.nan,
              "Decisión": "Se anulan (celdas imposibles)"})
tabla_43 = pd.DataFrame(filas)

pegar("t43_pct_max", fmt_pct(tabla_43["% atípicos"].max(), 1))
pegar("t43_var_max", tabla_43.loc[tabla_43["% atípicos"].idxmax(), "Variable"].lower())
pegar("t43_celdas", fmt(celdas, 0))
```

**Tabla 4.3.** Resumen de datos atípicos por variable.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_43)
 .fmt_number(columns=["N atípicos"], decimals=0, sep_mark=".", dec_mark=",")
 .fmt_number(columns=["% atípicos"], decimals=2, sep_mark=".", dec_mark=",")
 .sub_missing(columns=["% atípicos"], missing_text="—")
 .opt_stylize(style=1, color="gray"))
```

Ninguna variable supera el {glue}`t43_pct_max` de valores atípicos; el máximo corresponde a la {glue}`t43_var_max` (Tabla 4.3). En todos los casos los atípicos son valores reales: jugadores muy jóvenes o veteranos, jugadores excepcionalmente altos, equipos de ranking bajo y partidos especialmente largos o con retiro. Por eso no se eliminan. Solo se anulan las {glue}`t43_celdas` celdas de estadísticas lógicamente imposibles, como más ataques convertidos que ataques realizados.