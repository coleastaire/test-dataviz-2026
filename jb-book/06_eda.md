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

# Análisis exploratorio (EDA)

```{code-cell} ipython3
:tags: [remove-cell]
from py.utils import *
from great_tables import GT
from plotly.subplots import make_subplots

m = pd.read_csv(CLEAN + "vb_matches_limpio.csv", low_memory=False)
lg = pd.read_csv(CLEAN + "vb_long.csv", low_memory=False)
jg = pd.read_csv(CLEAN + "vb_jugadores.csv", low_memory=False)
mf = m[m.es_forfeit == 0]                       # partidos jugados (sin forfeits)
st = lg[lg.tiene_stats_completas == 1]          # equipos con estadísticas completas
```

El análisis exploratorio avanza en tres niveles: cada variable por separado (univariado), cada variable frente al resultado del partido (bivariado) y varias variables a la vez (multivariado).

## Análisis univariado

Esta sección describe cómo se distribuye cada variable numérica, sin compararla todavía con el resultado. Sirve para conocer el perfil típico de un partido y de una pareja profesional, y para decidir si alguna variable necesita una transformación antes de las pruebas.

```{code-cell} ipython3
:tags: [remove-cell]
pct_avp_st = 100 * st.circuit.eq("AVP").mean()
etq_st = f"Equipo-partido con estadísticas ({fmt(pct_avp_st, 0)} % AVP)"
filas = [
    ("Edad", "años", "Jugador-partido", jg.edad),
    ("Estatura (hombres)", "cm", "Jugador-partido", jg[jg.gender == "M"].estatura_cm),
    ("Estatura (mujeres)", "cm", "Jugador-partido", jg[jg.gender == "W"].estatura_cm),
    ("Ranking principal", "posición", "Equipo-partido", lg.rank_main),
    ("Duración", "min", "Partido sin forfeits", mf.duracion_min),
    ("Número de sets", "sets", "Partido sin forfeits", mf.n_sets),
    ("Puntos totales", "puntos", "Partido sin forfeits", mf.puntos_totales),
    ("Kills del equipo", "ataques ganadores", etq_st, st.kills),
    ("Aces del equipo", "saques directos", etq_st, st.aces),
    ("Bloqueos del equipo", "bloqueos", etq_st, st.blocks),
    ("Defensas del equipo (digs)", "defensas", etq_st, st.digs),
    ("% de ataque del equipo", "proporción", etq_st, st.hitpct)]
tabla_61 = pd.DataFrame([{"Variable": v, "Unidad": u, "Muestra": mu, **describir(s)}
                         for v, u, mu, s in filas])
tabla_61.columns = ["Variable", "Unidad", "Muestra", "n", "Media", "Mediana", "DE",
                    "P25", "P75", "Mín", "Máx", "Asimetría"]
d61 = tabla_61.set_index("Variable")
for v, f in d61.iterrows():
    clave = v.split()[0].lower() + ("_" + v.split()[-1].strip("()") if "(" in v else "")
    registrar(f"t6_1_n_{clave}", f["n"])
    registrar(f"t6_1_media_{clave}", f["Media"])
    registrar(f"t6_1_asim_{clave}", f["Asimetría"])
asim_altas = d61[d61["Asimetría"].abs() > 1]

pegar("t61_edad_media", fmt(d61.Media["Edad"]))
pegar("t61_edad_mediana", fmt(d61.Mediana["Edad"]))
pegar("t61_edad_asim", fmt(d61["Asimetría"]["Edad"]))
pegar("t61_est_h", fmt(d61.Media["Estatura (hombres)"], 1))
pegar("t61_est_m", fmt(d61.Media["Estatura (mujeres)"], 1))
pegar("t61_dur_media", fmt(d61.Media["Duración"]))
pegar("t61_dur_mediana", fmt(d61.Mediana["Duración"], 0))
pegar("t61_dur_asim", fmt(d61["Asimetría"]["Duración"]))
pegar("t61_asim_altas", " y ".join(v.lower() for v in asim_altas.index))
pegar("t61_asim_max", fmt(asim_altas["Asimetría"].max()))
```

**Tabla 6.1.** Estadísticos descriptivos de las variables numéricas.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_61)
 .fmt_number(columns=["n"], decimals=0, sep_mark=".", dec_mark=",")
 .fmt_number(columns=["Media", "Mediana", "DE", "P25", "P75", "Mín", "Máx", "Asimetría"],
             decimals=2, sep_mark=".", dec_mark=",")
 .tab_spanner(label="Posición", columns=["Media", "Mediana"])
 .tab_spanner(label="Dispersión", columns=["DE", "P25", "P75", "Mín", "Máx"])
 .opt_stylize(style=1, color="gray"))
```

La edad tiene una media de {glue}`t61_edad_media` años y una mediana de {glue}`t61_edad_mediana`, con una cola leve hacia jugadores mayores (asimetría de {glue}`t61_edad_asim`). La estatura es casi simétrica en ambos géneros: la media es de {glue}`t61_est_h` cm en hombres y {glue}`t61_est_m` cm en mujeres. La duración tiene una media de {glue}`t61_dur_media` minutos y una mediana de {glue}`t61_dur_mediana` (asimetría de {glue}`t61_dur_asim`). Solo {glue}`t61_asim_altas` supera una asimetría de 1 ({glue}`t61_asim_max`), algo esperable en un conteo con muchos partidos de 0, 1 o 2 aces (Tabla 6.1). No se transforma ninguna variable: las pruebas del análisis bivariado comparan rangos y no se ven afectadas por la asimetría.

```{code-cell} ipython3
med = jg.edad.median()
fig = go.Figure(go.Histogram(
    x=jg.edad, xbins=dict(start=12, end=70, size=1),
    marker=dict(color=NEUTRO, line=dict(color="white", width=0.5)),
    hovertemplate="Edad: %{x} años<br>Registros: %{y:,}<extra></extra>"))
fig.add_vline(x=med, line_dash="dash", line_color=GRIS_TOTAL,
              annotation_text=f"Mediana: {fmt(med, 1)} años",
              annotation_position="top right")
estilo_plotly(fig, "Distribución de la edad de los jugadores (años)",
              "Edad (años)", "Registros jugador-partido")
```

**Figura 6.1.** Distribución de la edad de los jugadores (años).

```{code-cell} ipython3
:tags: [remove-cell]
q25, q75 = jg.edad.quantile([.25, .75])
n_menor16, n_mayor45 = (jg.edad < 16).sum(), (jg.edad > 45).sum()
registrar("f6_1_mediana_edad", med)
registrar("f6_1_n_menor_16", n_menor16)
registrar("f6_1_n_mayor_45", n_mayor45)

pegar("f61_q25", fmt(q25, 1))
pegar("f61_q75", fmt(q75, 1))
pegar("f61_menor16", fmt(n_menor16, 0))
pegar("f61_mayor45", fmt(n_mayor45, 0))
```

El 50 % central de los registros está entre {glue}`f61_q25` y {glue}`f61_q75` años (Figura 6.1). Hay {glue}`f61_menor16` registros de jugadores menores de 16 años y {glue}`f61_mayor45` de mayores de 45: son juniors y veteranos reales, no errores. El voleibol de playa profesional se juega sobre todo entre los 25 y los 32 años.

```{code-cell} ipython3
fig = go.Figure()
for g, nombre in [("M", "Hombres"), ("W", "Mujeres")]:
    fig.add_histogram(x=jg[jg.gender == g].estatura_cm, name=nombre,
                      xbins=dict(start=153.67, end=218.44, size=2.54),
                      marker_color=PAL[g], opacity=0.6,
                      hovertemplate=nombre + "<br>Estatura: %{x} cm<br>Registros: %{y:,}<extra></extra>")
fig.update_layout(barmode="overlay")
estilo_plotly(fig, "Distribución de la estatura por género (cm)",
              "Estatura (cm)", "Registros jugador-partido")
```

**Figura 6.2.** Distribución de la estatura por género (cm).

```{code-cell} ipython3
:tags: [remove-cell]
med_est = jg.groupby("gender").estatura_cm.median()
hcols = ["w_p1_hgt", "w_p2_hgt", "l_p1_hgt", "l_p2_hgt"]
falta = m[hcols].isna().any(axis=1)
pct_falta = 100 * falta.mean()
pct_falta_circ = 100 * falta.groupby(m.circuit).mean()
pct_falta_g = 100 * m[["w_p1_hgt", "w_p2_hgt"]].isna().values.mean()
pct_falta_p = 100 * m[["l_p1_hgt", "l_p2_hgt"]].isna().values.mean()
registrar("f6_2_mediana_M", med_est["M"])
registrar("f6_2_mediana_W", med_est["W"])
registrar("f6_2_pct_partidos_sin_estatura", pct_falta)

pegar("f62_med_h", fmt(med_est["M"], 1))
pegar("f62_med_m", fmt(med_est["W"], 1))
pegar("f62_dif", fmt(med_est["M"] - med_est["W"], 1))
pegar("f62_falta", fmt_pct(pct_falta))
pegar("f62_falta_avp", fmt_pct(pct_falta_circ["AVP"]))
pegar("f62_falta_fivb", fmt_pct(pct_falta_circ["FIVB"]))
pegar("f62_falta_p", fmt_pct(pct_falta_p))
pegar("f62_falta_g", fmt_pct(pct_falta_g))
```

Los hombres miden en mediana {glue}`f62_med_h` cm y las mujeres {glue}`f62_med_m` cm, una diferencia de {glue}`f62_dif` cm (Figura 6.2). Por eso la estatura se analiza siempre separada por género: mezclarla crearía una distribución con dos picos que no describe a nadie. Falta al menos una estatura en el {glue}`f62_falta` de los partidos, más en la AVP ({glue}`f62_falta_avp`) que en la FIVB ({glue}`f62_falta_fivb`), y más en perdedores ({glue}`f62_falta_p` de las celdas) que en ganadores ({glue}`f62_falta_g`). Esa diferencia hay que tenerla en cuenta al comparar ganadores y perdedores: los jugadores sin estatura registrada tienden a ser los menos conocidos.

```{code-cell} ipython3
fig = go.Figure(go.Histogram(
    x=lg.rank_main, xbins=dict(start=0.5, end=64.5, size=1),
    marker=dict(color=NEUTRO, line=dict(color="white", width=0.5)),
    hovertemplate="Ranking: %{x}<br>Equipos-partido: %{y:,}<extra></extra>"))
estilo_plotly(fig, "Distribución del ranking principal de los equipos",
              "Ranking principal (1 = mejor)", "Equipos-partido")
```

**Figura 6.3.** Distribución del ranking principal de los equipos.

```{code-cell} ipython3
:tags: [remove-cell]
n_con_rank = lg.rank_main.notna().sum()
pct_sin_rank = 100 * lg.rank_main.isna().mean()
pct_top10 = 100 * (lg.rank_main <= 10).sum() / n_con_rank
registrar("f6_3_n_con_rank", n_con_rank)
registrar("f6_3_pct_sin_rank", pct_sin_rank)

pegar("f63_min", fmt(lg.rank_main.min(), 0))
pegar("f63_max", fmt(lg.rank_main.max(), 0))
pegar("f63_sin_rank", fmt_pct(pct_sin_rank))
pegar("f63_con_rank", fmt(n_con_rank, 0))
pegar("f63_top10", fmt_pct(pct_top10))
```

El ranking va de {glue}`f63_min` a {glue}`f63_max` (Figura 6.3). El {glue}`f63_sin_rank` de los equipos-partido no tiene ranking numérico porque solo trae su posición de clasificatoria ("Q"). Por eso todo análisis de ranking cubre {glue}`f63_con_rank` equipos-partido y no el total. Entre los que sí tienen ranking, el {glue}`f63_top10` está entre los 10 primeros: los equipos mejor rankeados avanzan más en los torneos y, por lo tanto, juegan más partidos.

```{code-cell} ipython3
dur = mf.duracion_min.dropna()
med_sets = mf.groupby("n_sets").duracion_min.median()
fig = go.Figure(go.Histogram(
    x=dur, xbins=dict(start=0, end=136, size=2),
    marker=dict(color=NEUTRO, line=dict(color="white", width=0.5)),
    hovertemplate="Duración: %{x} min<br>Partidos: %{y:,}<extra></extra>"))
for k, pos in [(2, "top left"), (3, "top right")]:
    fig.add_vline(x=med_sets[k], line_dash="dash", line_color=GRIS_TOTAL,
                  annotation_text=f"Mediana {k} sets: {fmt(med_sets[k], 0)} min",
                  annotation_position=pos)
estilo_plotly(fig, "Distribución de la duración de los partidos (min)",
              "Duración (min)", "Partidos")
```

**Figura 6.4.** Distribución de la duración de los partidos (min).

```{code-cell} ipython3
:tags: [remove-cell]
p25, p50, p75 = dur.quantile([.25, .5, .75])
registrar("f6_4_p25", p25)
registrar("f6_4_mediana", p50)
registrar("f6_4_p75", p75)

pegar("f64_p25", fmt(p25, 0))
pegar("f64_p75", fmt(p75, 0))
pegar("f64_p50", fmt(p50, 0))
pegar("f64_med2", fmt(med_sets[2], 0))
pegar("f64_med3", fmt(med_sets[3], 0))
pegar("f64_min", fmt(dur.min(), 0))
pegar("f64_max", fmt(dur.max(), 0))
```

La mitad de los partidos dura entre {glue}`f64_p25` y {glue}`f64_p75` minutos, con una mediana de {glue}`f64_p50` (Figura 6.4). La distribución tiene dos zonas: los partidos a 2 sets, alrededor de {glue}`f64_med2` minutos, y los partidos a 3 sets, alrededor de {glue}`f64_med3`, que forman la cola derecha. El mínimo es de {glue}`f64_min` minutos y el máximo de {glue}`f64_max`.

```{code-cell} ipython3
sets = mf.n_sets.value_counts().sort_index()
pct_sets = 100 * sets / sets.sum()
fig = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65], horizontal_spacing=0.12)
fig.add_bar(x=[f"{k} set" + ("" if k == 1 else "s") for k in sets.index], y=sets.values,
            marker_color=NEUTRO, text=[fmt_pct(p) for p in pct_sets], textposition="outside",
            hovertemplate="%{x}<br>Partidos: %{y:,}<extra></extra>", showlegend=False,
            row=1, col=1)
fig.add_histogram(x=mf.puntos_totales, xbins=dict(start=0, end=170, size=2),
                  marker=dict(color=NEUTRO, line=dict(color="white", width=0.5)),
                  hovertemplate="Puntos: %{x}<br>Partidos: %{y:,}<extra></extra>",
                  showlegend=False, row=1, col=2)
estilo_plotly(fig, "Número de sets y puntos totales por partido", "", "")
fig.update_yaxes(range=[0, sets.max() * 1.15], row=1, col=1)
fig.update_xaxes(title_text="Número de sets", row=1, col=1)
fig.update_yaxes(title_text="Partidos", row=1, col=1)
fig.update_xaxes(title_text="Puntos totales del partido", row=1, col=2)
fig
```

**Figura 6.5.** Número de sets y puntos totales por partido.

```{code-cell} ipython3
:tags: [remove-cell]
uno = mf[mf.n_sets == 1]
n_uno_ret = uno.es_retirado.sum()
uno_jugado = uno[uno.es_retirado == 0]
anios_uno = uno_jugado.groupby(["circuit", "year"]).size().sort_values(ascending=False).head(2)
med_pts = mf.puntos_totales.median()
med_pts_sets = mf.groupby("n_sets").puntos_totales.median()
for k, v in sets.items():
    registrar(f"f6_5_n_{k}_sets", v)
registrar("f6_5_n_1_set_sin_retiro", len(uno) - n_uno_ret)
registrar("f6_5_mediana_puntos", med_pts)

pegar("f65_pct2", fmt_pct(pct_sets[2]))
pegar("f65_pct3", fmt_pct(pct_sets[3]))
pegar("f65_n1", fmt(sets[1], 0))
pegar("f65_n1_ret", fmt(n_uno_ret, 0))
pegar("f65_n1_jug", fmt(len(uno_jugado), 0))
pegar("f65_uno_a", f"{anios_uno.index[0][0]} de {anios_uno.index[0][1]}")
pegar("f65_uno_b", f"{anios_uno.index[1][0]} de {anios_uno.index[1][1]}")
pegar("f65_med_pts", fmt(med_pts, 0))
pegar("f65_pts2", fmt(med_pts_sets[2], 0))
pegar("f65_pts3", fmt(med_pts_sets[3], 0))
pegar("f65_med_h", fmt(med_est["M"], 0))
pegar("f65_med_m", fmt(med_est["W"], 0))
```

El {glue}`f65_pct2` de los partidos se decide en 2 sets y el {glue}`f65_pct3` en 3 (Figura 6.5). Hay {glue}`f65_n1` partidos de un solo set: {glue}`f65_n1_ret` terminaron por retiro y los otros {glue}`f65_n1_jug` se jugaron a un único set de 15 o 25 puntos, un formato que usaron algunos torneos y que se concentra en la {glue}`f65_uno_a` y la {glue}`f65_uno_b`. No se mezclan con los demás al comparar duraciones. La mediana es de {glue}`f65_med_pts` puntos por partido: {glue}`f65_pts2` en los partidos a 2 sets y {glue}`f65_pts3` en los de 3, lo que explica la forma de dos picos del panel derecho.

```{admonition} Para el entrenador
:class: tip
Un partido típico dura unos {glue}`f64_p50` minutos, pero uno de cada tres se va a un tercer set y dura cerca de {glue}`f64_med3`: la preparación física debe pensarse para el partido largo, no para el promedio. En estatura, la referencia del circuito es de unos {glue}`f65_med_h` cm en hombres y {glue}`f65_med_m` cm en mujeres; la sección siguiente muestra si estar por encima de esa referencia se asocia con ganar más.
```

## Análisis bivariado frente al resultado

Esta sección compara a las parejas ganadoras con las perdedoras. Cada partido aporta dos filas, una por pareja, así que solo se comparan variables que distinguen a una pareja de la otra: ranking, edad, estatura, diferencias entre compañeros, origen en clasificatoria y estadísticas de juego. Las variables del partido (duración, sets, circuito, etapa) valen lo mismo para las dos parejas; para ellas se estudia otra pregunta: si cambian la probabilidad de una sorpresa (apartado "Upsets y diferencia de ranking").

```{code-cell} ipython3
:tags: [remove-cell]
from scipy.stats import mannwhitneyu, wilcoxon, chi2_contingency
from great_tables import style, loc

def prueba_mw(d, var, muestra, etiqueta):
    g = d.loc[d.resultado == "Ganador", var].dropna()
    p = d.loc[d.resultado == "Perdedor", var].dropna()
    U, pv = mannwhitneyu(g, p, alternative="two-sided", use_continuity=True,
                         method="asymptotic")
    qg, qp = g.quantile([.25, .75]).values, p.quantile([.25, .75]).values
    inter = max(0, min(qg[1], qp[1]) - max(qg[0], qp[0]))
    union = max(qg[1], qp[1]) - min(qg[0], qp[0])
    rbc = 2 * U / (len(g) * len(p)) - 1
    poder = "Alto" if abs(rbc) >= 0.5 else "Moderado" if abs(rbc) >= 0.3 else "Bajo"
    return {"var": var, "Variable": etiqueta, "Muestra": muestra,
            "n Ganador": len(g), "n Perdedor": len(p),
            "Mediana G": g.median(), "Mediana P": p.median(), "U": U, "p-valor": pv,
            "RBC": rbc, "Solapamiento IQR": inter / union if union > 0 else 1,
            "Poder": poder}

avp = lg[(lg.circuit == "AVP") & (lg.tiene_stats_completas == 1)]
pruebas = [
    (lg, "rank_main", "Todos", "Ranking principal (posición)"),
    (lg, "edad_media", "Todos", "Edad media de la pareja (años)"),
    (lg[lg.gender == "M"], "estatura_media_cm", "Todos (hombres)", "Estatura media de la pareja (cm)"),
    (lg[lg.gender == "W"], "estatura_media_cm", "Todos (mujeres)", "Estatura media de la pareja (cm)"),
    (lg, "dif_edad_pareja", "Todos", "Diferencia de edad entre compañeros (años)"),
    (lg, "dif_estatura_pareja_cm", "Todos", "Diferencia de estatura entre compañeros (cm)"),
    (avp, "kills", "AVP con estadísticas", "Kills"),
    (avp, "aces", "AVP con estadísticas", "Aces"),
    (avp, "blocks", "AVP con estadísticas", "Bloqueos"),
    (avp, "digs", "AVP con estadísticas", "Defensas (digs)"),
    (avp, "attacks", "AVP con estadísticas", "Ataques"),
    (avp, "errors", "AVP con estadísticas", "Errores de ataque"),
    (avp, "serve_errors", "AVP con estadísticas", "Errores de saque"),
    (avp, "hitpct", "AVP con estadísticas", "% de ataque")]
tabla_62 = pd.DataFrame([prueba_mw(*x) for x in pruebas])
tabla_62["orden"] = tabla_62.Muestra.str.startswith("AVP").astype(int)
tabla_62 = (tabla_62.assign(abs_rbc=tabla_62.RBC.abs())
            .sort_values(["orden", "abs_rbc"], ascending=[True, False])
            .drop(columns=["orden", "abs_rbc"]).reset_index(drop=True))
for _, f in tabla_62.iterrows():
    clave = f["var"] + ("_" + f.Muestra.split("(")[-1][0] if "(" in f.Muestra else "")
    registrar(f"t6_2_rbc_{clave}", f.RBC)
    registrar(f"t6_2_solap_{clave}", f["Solapamiento IQR"])
    registrar(f"t6_2_ng_{clave}", f["n Ganador"])
t62 = tabla_62.set_index("var")
max_p = tabla_62["p-valor"].max()
n_eq = len(lg)
```

### Variables numéricas: Mann-Whitney U y correlación rango-biserial

**Tabla 6.2.** Comparación Ganador vs Perdedor: Mann-Whitney U, correlación rango-biserial (RBC) y solapamiento IQR.

```{code-cell} ipython3
:tags: [remove-input]
t = tabla_62.drop(columns=["var"]).copy()
t["p-valor"] = t["p-valor"].map(fmt_p)
gt = (GT(t)
      .fmt_number(columns=["n Ganador", "n Perdedor", "U"], decimals=0, sep_mark=".", dec_mark=",")
      .fmt_number(columns=["Mediana G", "Mediana P", "RBC", "Solapamiento IQR"], decimals=2,
                  sep_mark=".", dec_mark=",")
      .cols_align(align="center", columns=["Poder"]))
for nivel, color in PAL_PODER.items():
    filas = t.index[t.Poder == nivel].tolist()
    if filas:
        gt = gt.tab_style(style=[style.fill(color=color),
                                 style.text(color="white" if nivel != "Bajo" else "#1A1A1A", weight="bold")],
                          locations=loc.body(columns="Poder", rows=filas))
gt.opt_stylize(style=1, color="gray")
```

```{code-cell} ipython3
:tags: [hide-input, remove-output]
gana = lg[lg.resultado == "Ganador"]
pareado = {}
for v in ["ventaja_rank", "ventaja_estatura_cm", "ventaja_edad"]:
    s = gana[v].dropna()
    s = s[s != 0]
    w = wilcoxon(s)
    pareado[v] = {"n": len(s), "pct": 100 * (s > 0).mean(), "p": w.pvalue, "w": w.statistic}
    registrar(f"t6_2_wilcoxon_pct_{v}", pareado[v]["pct"])
avp_fuertes = tabla_62[tabla_62.Muestra.str.startswith("AVP") & (tabla_62.Poder != "Bajo")]

pegar("t62_n_eq", fmt(n_eq, 0))
pegar("t62_max_p", fmt_p(max_p))
pegar("t62_rbc_rank", fmt(t62.RBC["rank_main"]))
pegar("t62_u_rank", fmt(t62.U["rank_main"], 0))
pegar("t62_p_rank", fmt_p(t62["p-valor"]["rank_main"]))
pegar("t62_w_stat", fmt(pareado["ventaja_rank"]["w"], 0))
pegar("t62_w_n", fmt(pareado["ventaja_rank"]["n"], 0))
pegar("t62_w_p_rank", fmt_p(pareado["ventaja_rank"]["p"]))
pegar("t62_u_hit", fmt(t62.U["hitpct"], 0))
pegar("t62_p_hit", fmt_p(t62["p-valor"]["hitpct"]))
pegar("t62_med_g_rank", fmt(t62["Mediana G"]["rank_main"], 0))
pegar("t62_med_p_rank", fmt(t62["Mediana P"]["rank_main"], 0))
pegar("t62_max_fis", fmt(t62.loc[["edad_media", "estatura_media_cm"], "RBC"].abs().max()))
pegar("t62_w_rank", fmt_pct(pareado["ventaja_rank"]["pct"]))
pegar("t62_w_est", fmt_pct(pareado["ventaja_estatura_cm"]["pct"]))
pegar("t62_w_edad", fmt_pct(pareado["ventaja_edad"]["pct"]))
pegar("t62_w_p", fmt_p(max(x["p"] for x in pareado.values())))
pegar("t62_rbc_hit", fmt(t62.RBC["hitpct"]))
pegar("t62_solap_hit", fmt(t62["Solapamiento IQR"]["hitpct"]))
pegar("t62_avp_fuertes", ", ".join(avp_fuertes.Variable.iloc[1:].str.lower()))
```

Para cada variable se aplicó la prueba U de Mann-Whitney con un nivel de significancia α = 0,05. La hipótesis nula (H₀) es que la variable se distribuye igual en ganadores y en perdedores, y la alternativa (H₁) es que las distribuciones son distintas. En todas las variables el p-valor fue menor que α (el más alto fue {glue}`t62_max_p`), así que se rechaza H₀ en todos los casos (Tabla 6.2). Como la muestra tiene {glue}`t62_n_eq` equipos-partido, incluso diferencias muy pequeñas resultan significativas; por eso también se revisa el tamaño del efecto (RBC).

Para el ranking se obtuvo U = {glue}`t62_u_rank`, p {glue}`t62_p_rank` y RBC = {glue}`t62_rbc_rank`, un efecto bajo. La mediana de los ganadores es {glue}`t62_med_g_rank` y la de los perdedores {glue}`t62_med_p_rank`, es decir, los ganadores suelen estar mejor rankeados. La edad y la estatura tienen efectos todavía menores (|RBC| ≤ {glue}`t62_max_fis`), y las diferencias de edad y estatura entre compañeros casi no separan el resultado. En la muestra completa ninguna de estas variables pasa de poder bajo.

Mann-Whitney compara a todos los ganadores con todos los perdedores, sin tener en cuenta contra quién jugó cada uno. Como cada partido aporta un ganador y un perdedor, también se aplicó la prueba de Wilcoxon de rangos con signo sobre la ventaja de cada pareja ganadora frente a su rival (H₀: la mediana de la ventaja es 0, α = 0,05). Para el ranking se obtuvo W = {glue}`t62_w_stat` con {glue}`t62_w_n` partidos y p {glue}`t62_w_p_rank`, por lo que se rechaza H₀: el ganador tiene mejor ranking que su rival en el {glue}`t62_w_rank` de los partidos. Con la estatura y la edad pasa lo mismo, pero con porcentajes más cercanos al 50 %: el ganador es más alto en el {glue}`t62_w_est` de los partidos y mayor en el {glue}`t62_w_edad` (p {glue}`t62_w_p` en los tres casos). Esto indica que el ranking sirve más como diferencia con el rival que como valor por sí solo.

En la AVP, las estadísticas de juego separan más que las variables físicas. Para el % de ataque se obtuvo U = {glue}`t62_u_hit`, p {glue}`t62_p_hit` y RBC = {glue}`t62_rbc_hit`, que corresponde a un poder alto; además, los rangos centrales de ganadores y perdedores no se tocan (solapamiento de {glue}`t62_solap_hit`). Le siguen {glue}`t62_avp_fuertes`, todas con poder moderado. Hay que tener en cuenta que estas estadísticas se registran durante el partido, así que sirven para describir el juego de los ganadores, pero no para anticipar el resultado antes del partido.

```{code-cell} ipython3
paneles = [(lg, "rank_main", "Ranking (posición)"),
           (lg, "edad_media", "Edad media (años)"),
           (lg[lg.gender == "M"], "estatura_media_cm", "Estatura media, hombres (cm)"),
           (lg[lg.gender == "W"], "estatura_media_cm", "Estatura media, mujeres (cm)")]
fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.12)
for i, (d, var, tit) in enumerate(paneles):
    fila, col = i // 2 + 1, i % 2 + 1
    for res in ["Ganador", "Perdedor"]:
        y = d.loc[d.resultado == res, var].dropna()
        fig.add_trace(go.Box(x=[res] * len(y), y=y, name=res, marker_color=PAL[res],
                             boxpoints=False, legendgroup=res, showlegend=(i == 0)),
                      row=fila, col=col)
    fig.update_yaxes(title_text=tit, row=fila, col=col)
fig.update_layout(height=650)
estilo_plotly(fig, "Ranking, edad y estatura del equipo según resultado", "", "")
fig.update_yaxes(title_text="Ranking (posición)", row=1, col=1)
fig
```

**Figura 6.6.** Ranking, edad y estatura del equipo según resultado.

```{code-cell} ipython3
:tags: [remove-cell]
pegar("f66_solap_rank", fmt(t62["Solapamiento IQR"]["rank_main"]))
```

Las cajas del ranking están desplazadas: la mediana es {glue}`t62_med_g_rank` en ganadores y {glue}`t62_med_p_rank` en perdedores, con un solapamiento IQR de {glue}`f66_solap_rank` (Figura 6.6). En edad y estatura las cajas casi se superponen, coherente con su poder bajo: conocer la edad o la estatura de una pareja, sin mirar al rival, dice muy poco sobre si va a ganar.

### Poder discriminante

```{code-cell} ipython3
p = tabla_62.copy()
p["etq"] = p.Variable.str.replace(r" \(.*\)", "", regex=True) + p.Muestra.map(
    lambda s: " (AVP)" if s.startswith("AVP") else " (H)" if "hombres" in s
    else " (M)" if "mujeres" in s else "")
p["abs"] = p.RBC.abs()
p = p.sort_values("abs")
fig = go.Figure(go.Bar(
    x=p["abs"], y=p.etq, orientation="h", marker_color=[PAL_PODER[x] for x in p.Poder],
    customdata=np.stack([p.RBC, p["p-valor"].map(fmt_p), p["n Ganador"] + p["n Perdedor"], p.Poder], axis=1),
    hovertemplate="%{y}<br>RBC: %{customdata[0]:.3f}<br>p-valor: %{customdata[1]}"
                  "<br>n: %{customdata[2]:,}<br>Poder: %{customdata[3]}<extra></extra>"))
for x in [0.3, 0.5]:
    fig.add_vline(x=x, line_dash="dot", line_color=GRIS_TOTAL)
fig.update_xaxes(range=[0, 1])
fig.update_layout(height=560)
estilo_plotly(fig, "Poder discriminante de las variables (|RBC|)",
              "|Correlación rango-biserial| (0–1)", "")
```

**Figura 6.7.** Poder discriminante de las variables (|RBC|).

```{code-cell} ipython3
:tags: [remove-cell]
cuenta = tabla_62.Poder.value_counts().reindex(["Alto", "Moderado", "Bajo"], fill_value=0)
altas = p[p.Poder == "Alto"].etq.tolist()
for k, v in cuenta.items():
    registrar(f"f6_7_n_{k.lower()}", v)

pegar("f67_alto", str(cuenta["Alto"]))
pegar("f67_moderado", str(cuenta["Moderado"]))
pegar("f67_bajo", str(cuenta["Bajo"]))
pegar("f67_altas", " y ".join(altas))
```

{glue}`f67_alto` variable tiene poder alto, {glue}`f67_moderado` tienen poder moderado y {glue}`f67_bajo` poder bajo (Figura 6.7). La de poder alto es {glue}`f67_altas`, y todas las de poder moderado son también estadísticas de juego de la AVP. Las variables físicas y el ranking absoluto quedan en poder bajo: por sí solas no anticipan quién gana. La H y la M entre paréntesis indican hombres y mujeres.

### Variables categóricas: chi-cuadrado y V de Cramér

```{code-cell} ipython3
:tags: [remove-cell]
def chi_v(x, y, cruce):
    t = pd.crosstab(x, y)
    chi2, pv, gl, _ = chi2_contingency(t, correction=False)
    n = t.values.sum()
    v = np.sqrt(chi2 / (n * (min(t.shape) - 1)))
    mag = ("Despreciable" if v < 0.1 else "Pequeña" if v < 0.3
           else "Media" if v < 0.5 else "Grande")
    return {"Cruce": cruce, "Tabla (r × c)": f"{t.shape[0]} × {t.shape[1]}", "n": n,
            "χ²": chi2, "gl": gl, "p-valor": pv, "V de Cramér": v, "Magnitud": mag}

r = m[m.rank_diff.notna()]
tabla_63 = pd.DataFrame([
    chi_v(r.es_upset, r.circuit, "Upset × circuito"),
    chi_v(r.es_upset, r.gender, "Upset × género"),
    chi_v(r.es_upset, r.etapa, "Upset × etapa"),
    chi_v(lg.resultado, lg.es_clasif, "Resultado × viene de clasificatoria")])
for _, f in tabla_63.iterrows():
    registrar("t6_3_v_" + f.Cruce.split("×")[1].strip().split()[0], f["V de Cramér"])
t63 = tabla_63.set_index("Cruce")
perd_clasif = 100 * lg.groupby("es_clasif").resultado.apply(lambda s: (s == "Perdedor").mean())
ups_circ = 100 * r.groupby("circuit").es_upset.mean()
ups_gen = 100 * r.groupby("gender").es_upset.mean()

pegar("t63_ups_fivb", fmt_pct(ups_circ["FIVB"]))
pegar("t63_ups_avp", fmt_pct(ups_circ["AVP"]))
pegar("t63_v_circ", fmt(t63["V de Cramér"]["Upset × circuito"], 3))
pegar("t63_ups_m", fmt_pct(ups_gen["M"]))
pegar("t63_ups_w", fmt_pct(ups_gen["W"]))
pegar("t63_v_gen", fmt(t63["V de Cramér"]["Upset × género"], 3))
pegar("t63_v_etapa", fmt(t63["V de Cramér"]["Upset × etapa"], 3))
pegar("t63_perd_clasif", fmt_pct(perd_clasif[1]))
pegar("t63_perd_directo", fmt_pct(perd_clasif[0]))
pegar("t63_v_clasif", fmt(t63["V de Cramér"]["Resultado × viene de clasificatoria"], 3))
pegar("t63_chi_circ", fmt(t63["χ²"]["Upset × circuito"]))
pegar("t63_gl_circ", fmt(t63["gl"]["Upset × circuito"], 0))
pegar("t63_p_max", fmt_p(t63["p-valor"].max()))
```

**Tabla 6.3.** Pruebas chi-cuadrado y V de Cramér para variables categóricas.

```{code-cell} ipython3
:tags: [remove-input]
t = tabla_63.copy()
t["p-valor"] = t["p-valor"].map(fmt_p)
(GT(t)
 .fmt_number(columns=["n", "gl"], decimals=0, sep_mark=".", dec_mark=",")
 .fmt_number(columns=["χ²"], decimals=2, sep_mark=".", dec_mark=",")
 .fmt_number(columns=["V de Cramér"], decimals=3, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

Para las variables categóricas se aplicó la prueba chi-cuadrado de independencia con α = 0,05, donde H₀ es que las dos variables son independientes. En los cuatro cruces el p-valor fue {glue}`t63_p_max`, así que se rechaza H₀, pero la V de Cramér es menor que 0,1 en todos: la asociación existe, aunque es despreciable (Tabla 6.3). Por ejemplo, para sorpresa × circuito se obtuvo χ²({glue}`t63_gl_circ`) = {glue}`t63_chi_circ`, p {glue}`t63_p_max` y V = {glue}`t63_v_circ`. La tasa de sorpresas es algo mayor en la FIVB ({glue}`t63_ups_fivb`) que en la AVP ({glue}`t63_ups_avp`; V = {glue}`t63_v_circ`) y casi igual entre hombres ({glue}`t63_ups_m`) y mujeres ({glue}`t63_ups_w`; V = {glue}`t63_v_gen`). Según la etapa, la asociación también es despreciable (V = {glue}`t63_v_etapa`). Los equipos que vienen de clasificatoria pierden más: el {glue}`t63_perd_clasif` de sus partidos, frente al {glue}`t63_perd_directo` de los que entraron directo (V = {glue}`t63_v_clasif`).

### Upsets y diferencia de ranking

Un **upset** (sorpresa) es un partido que gana la pareja peor rankeada. Solo puede calcularse cuando las dos parejas tienen ranking numérico.

```{code-cell} ipython3
fig = go.Figure()
for nombre, cond, color in [("Upset: gana el peor rankeado", r.rank_diff < 0, PAL["Perdedor"]),
                            ("Gana el mejor rankeado", r.rank_diff > 0, PAL["Ganador"])]:
    fig.add_histogram(x=r.rank_diff[cond], name=nombre, marker_color=color,
                      xbins=dict(start=-64.5, end=64.5, size=1),
                      hovertemplate="Diferencia: %{x}<br>Partidos: %{y:,}<extra></extra>")
fig.add_vline(x=0, line_color=GRIS_TOTAL)
fig.update_layout(barmode="overlay")
estilo_plotly(fig, "Distribución de la diferencia de ranking (perdedor − ganador)",
              "Diferencia de ranking (posiciones)", "Partidos")
```

**Figura 6.8.** Distribución de la diferencia de ranking (perdedor − ganador).

```{code-cell} ipython3
:tags: [remove-cell]
pct_upset = 100 * r.es_upset.mean()
puro = r[~r.w_rank.str.contains(",", na=True) & ~r.l_rank.str.contains(",", na=True)]
registrar("f6_8_media_rank_diff", r.rank_diff.mean())
registrar("f6_8_mediana_rank_diff", r.rank_diff.median())
registrar("f6_8_n", len(r))
registrar("f6_8_pct_upset", pct_upset)
registrar("f6_8_n_puro", len(puro))
registrar("f6_8_pct_upset_puro", 100 * puro.es_upset.mean())

pegar("f68_media", fmt(r.rank_diff.mean()))
pegar("f68_mediana", fmt(r.rank_diff.median(), 0))
pegar("f68_pct_upset", fmt_pct(pct_upset, 2))
pegar("f68_n", fmt(len(r), 0))
pegar("f68_n_puro", fmt(len(puro), 0))
pegar("f68_pct_puro", fmt_pct(100 * puro.es_upset.mean()))
```

La diferencia media es de {glue}`f68_media` posiciones (mediana de {glue}`f68_mediana`): en general gana el mejor rankeado (Figura 6.8). Aun así, el {glue}`f68_pct_upset` de los {glue}`f68_n` partidos con ambos rankings son upsets, casi uno de cada tres. Como prueba de sensibilidad, si se usan solo los rankings simples (sin el formato compuesto "17, Q2"), quedan {glue}`f68_n_puro` partidos y la tasa es del {glue}`f68_pct_puro`: la conclusión no cambia.

```{code-cell} ipython3
etiquetas = ["1–2", "3–5", "6–10", "11–20", "21 o más"]
grupo = pd.cut(r.rank_diff.abs(), [0, 2, 5, 10, 20, np.inf], labels=etiquetas)
fav = (r.assign(grupo=grupo, gana_fav=(r.rank_diff > 0))
         .groupby("grupo", observed=True)
         .agg(p=("gana_fav", "mean"), n=("gana_fav", "size")).reset_index())
fav["ic"] = 1.96 * np.sqrt(fav.p * (1 - fav.p) / fav.n) * 100
fav["pct"] = 100 * fav.p
fig = go.Figure(go.Scatter(
    x=fav.grupo, y=fav.pct, mode="lines+markers+text",
    text=[fmt_pct(v) for v in fav.pct], textposition="top center",
    line=dict(color=PAL["Ganador"], width=2), marker=dict(size=9),
    error_y=dict(type="data", array=fav.ic, color=PAL["Ganador"]),
    customdata=fav.n,
    hovertemplate="Diferencia: %{x}<br>Gana el favorito: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>"))
fig.add_hline(y=50, line_dash="dash", line_color=GRIS_TOTAL,
              annotation_text="50 %: moneda al aire", annotation_position="bottom right")
fig.update_yaxes(range=[40, 100], ticksuffix=" %")
estilo_plotly(fig, "Probabilidad de que gane el mejor rankeado según la diferencia de ranking",
              "Diferencia de ranking (posiciones)", "% de victorias del mejor rankeado")
```

**Figura 6.9.** Probabilidad de que gane el mejor rankeado según la diferencia de ranking.

```{code-cell} ipython3
:tags: [remove-cell]
fv = fav.set_index("grupo")
for g_, f in fv.iterrows():
    registrar(f"f6_9_pct_{g_}", f.pct)
    registrar(f"f6_9_n_{g_}", f.n)

pegar("f69_cerca", fmt_pct(fv.pct["1–2"]))
pegar("f69_lejos", fmt_pct(fv.pct["21 o más"]))
```

Con 1 o 2 posiciones de diferencia, el favorito gana el {glue}`f69_cerca` de las veces: casi una moneda al aire (Figura 6.9). La probabilidad sube de forma constante con la diferencia y, con más de 20 posiciones, llega al {glue}`f69_lejos`. Los intervalos de confianza son muy estrechos porque cada grupo tiene miles de partidos. El ranking es informativo, pero su valor depende de la distancia con el rival.

```{code-cell} ipython3
ue = (r.groupby(["etapa", "circuit"], observed=True)
        .agg(pct=("es_upset", "mean"), n=("es_upset", "size")).reset_index())
ue["pct"] = 100 * ue.pct
excluidos = ue[ue.n < 30]
ue = ue[ue.n >= 30]
fig = go.Figure()
for c in ["FIVB", "AVP"]:
    s = ue[ue.circuit == c].set_index("etapa").reindex(ETAPAS).reset_index()
    fig.add_bar(x=s.etapa, y=s.pct, name=c, marker_color=PAL[c], customdata=s.n,
                text=[fmt_pct(v) if pd.notna(v) else "" for v in s.pct], textposition="outside",
                hovertemplate="%{x} · " + c + "<br>Upsets: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>")
fig.update_layout(barmode="group")
fig.update_yaxes(range=[0, ue.pct.max() * 1.2], ticksuffix=" %")
estilo_plotly(fig, "Tasa de upsets por etapa del torneo y circuito",
              "Etapa del torneo", "% de upsets")
```

**Figura 6.10.** Tasa de upsets por etapa del torneo y circuito.

```{code-cell} ipython3
:tags: [remove-cell]
for _, f in ue.iterrows():
    registrar(f"f6_10_pct_{f.etapa.replace(' ', '_')}_{f.circuit}", f.pct)
ue_f = ue[ue.circuit == "FIVB"].set_index("etapa").pct
ue_a = ue[ue.circuit == "AVP"].set_index("etapa").pct
comunes = ue_f.index.intersection(ue_a.index)
fivb_mayor = (ue_f[comunes] > ue_a[comunes]).all()
exc_txt = "; ".join(f"{e.etapa} {e.circuit} (n = {e.n})" for e in excluidos.itertuples())
dist_etapa = r.assign(d=r.rank_diff.abs()).groupby("etapa").d.median()

pegar("f610_max_etapa", ue_f.idxmax().lower())
pegar("f610_max", fmt_pct(ue_f.max()))
pegar("f610_min_etapa", ue_f.idxmin().lower())
pegar("f610_min", fmt_pct(ue_f.min()))
pegar("f610_d_grupos", fmt(dist_etapa["Fase de grupos"], 0))
pegar("f610_d_semis", fmt(dist_etapa["Semifinales"], 0))
pegar("f610_d_clasif", fmt(dist_etapa["Clasificatoria"], 0))
pegar("f610_frase", "En todas las etapas comparables, la FIVB tiene más sorpresas que la AVP."
      if fivb_mayor else "La FIVB y la AVP no siguen el mismo patrón en todas las etapas.")
pegar("f610_exc", exc_txt)
```

En la FIVB, la tasa de upsets es mayor en la {glue}`f610_max_etapa` ({glue}`f610_max`) y menor en la {glue}`f610_min_etapa` ({glue}`f610_min`) (Figura 6.10). La explicación está en la distancia entre rivales: en la fase de grupos los rivales están separados, en mediana, por {glue}`f610_d_grupos` posiciones, mientras que en semifinales y finales solo por {glue}`f610_d_semis`; a menor distancia, más sorpresas. La clasificatoria tiene la tasa más alta, pero también la muestra más pequeña con ranking: ahí se enfrentan equipos de ranking bajo y parecido ({glue}`f610_d_clasif` posiciones de diferencia en mediana). {glue}`f610_frase` No se dibuja {glue}`f610_exc`, porque tiene menos de 30 partidos.

### Diferencias de pareja

```{code-cell} ipython3
cortes_est = [-np.inf, -10, -5, 0, 5, 10, np.inf]
etq_est = ["< −10", "−10 a −5", "−5 a 0", "0 a 5", "5 a 10", "≥ 10"]
cortes_edad = [-np.inf, -5, -2, 0, 2, 5, np.inf]
etq_edad = ["< −5", "−5 a −2", "−2 a 0", "0 a 2", "2 a 5", "≥ 5"]

def tasa(var, cortes, etq):
    g = pd.cut(lg[var], cortes, labels=etq, right=False)
    return (lg.assign(g=g, gana=lg.resultado.eq("Ganador"))
              .groupby("g", observed=True).agg(pct=("gana", "mean"), n=("gana", "size"))
              .assign(pct=lambda d: 100 * d.pct).reset_index())

v_est, v_edad = tasa("ventaja_estatura_cm", cortes_est, etq_est), tasa("ventaja_edad", cortes_edad, etq_edad)
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08)
for col, d in [(1, v_est), (2, v_edad)]:
    fig.add_scatter(x=d.g, y=d.pct, mode="lines+markers", showlegend=False,
                    line=dict(color=PAL["Ganador"], width=2), marker=dict(size=8),
                    customdata=d.n,
                    hovertemplate="Ventaja: %{x}<br>Victorias: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>",
                    row=1, col=col)
    fig.add_hline(y=50, line_dash="dash", line_color=GRIS_TOTAL, row=1, col=col)
estilo_plotly(fig, "Tasa de victoria según la ventaja de estatura y de edad sobre el rival", "", "")
fig.update_xaxes(title_text="Ventaja de estatura (cm)", row=1, col=1)
fig.update_xaxes(title_text="Ventaja de edad (años)", row=1, col=2)
fig.update_yaxes(title_text="% de victorias", ticksuffix=" %", range=[35, 65], row=1, col=1)
fig
```

**Figura 6.11.** Tasa de victoria según la ventaja de estatura y de edad sobre el rival.

```{code-cell} ipython3
:tags: [remove-cell]
ve, va = v_est.set_index("g").pct, v_edad.set_index("g").pct
for k, v in ve.items():
    registrar(f"f6_11_pct_est_{k}", v)
for k, v in va.items():
    registrar(f"f6_11_pct_edad_{k}", v)

pegar("f611_est_mas10", fmt_pct(ve["≥ 10"]))
pegar("f611_est_menos10", fmt_pct(ve["< −10"]))
pegar("f611_edad_mas5", fmt_pct(va["≥ 5"]))
```

Una pareja 10 cm o más alta que su rival gana el {glue}`f611_est_mas10` de las veces, y una 10 cm o más baja, solo el {glue}`f611_est_menos10` (Figura 6.11). La ventaja de edad pesa menos: con 5 años o más de ventaja, la tasa de victoria es del {glue}`f611_edad_mas5`. La curva es simétrica por construcción, porque cada partido aporta una ventaja +x para una pareja y −x para la otra. Las diferencias entre compañeros de una misma pareja están en la Tabla 6.2 y no se asocian con el resultado.

### Estadísticas de juego (solo AVP)

```{code-cell} ipython3
stats = [("kills", "Kills"), ("aces", "Aces"), ("blocks", "Bloqueos"),
         ("digs", "Defensas (digs)"), ("errors", "Errores de ataque"), ("hitpct", "% de ataque")]
fig = make_subplots(rows=2, cols=3, subplot_titles=[t for _, t in stats],
                    vertical_spacing=0.16, horizontal_spacing=0.08)
for i, (var, tit) in enumerate(stats):
    fila, col = i // 3 + 1, i % 3 + 1
    for res in ["Ganador", "Perdedor"]:
        y = avp.loc[avp.resultado == res, var].dropna()
        fig.add_trace(go.Box(x=[res] * len(y), y=y, name=res, marker_color=PAL[res],
                             boxpoints=False, legendgroup=res, showlegend=(i == 0)),
                      row=fila, col=col)
fig.update_layout(height=650)
estilo_plotly(fig, "Estadísticas de juego del equipo según resultado (solo AVP)", "", "")
```

**Figura 6.12.** Estadísticas de juego del equipo según resultado (solo AVP con estadísticas completas).

```{code-cell} ipython3
:tags: [remove-cell]
n_avp_part = avp.match_id.nunique()
mayores = [t for v, t in stats if t62["Mediana G"][v] > t62["Mediana P"][v]]
top = tabla_62[tabla_62.Muestra.str.startswith("AVP")].iloc[0]
registrar("f6_12_n_partidos_avp", n_avp_part)

pegar("f612_n", fmt(n_avp_part, 0))
pegar("f612_mayores", ", ".join(x.lower() for x in mayores))
pegar("f612_err_g", fmt(t62["Mediana G"]["errors"], 0))
pegar("f612_err_p", fmt(t62["Mediana P"]["errors"], 0))
pegar("f612_top", top.Variable.lower())
pegar("f612_top_rbc", fmt(top.RBC))
pegar("b_cerca", fmt_pct(fv.pct["1–2"], 0))
pegar("b_lejos", fmt_pct(fv.pct["21 o más"], 0))
pegar("b_est10", fmt_pct(ve["≥ 10"], 0))
pegar("b_clasif", fmt_pct(perd_clasif[1], 0))
pegar("b_directo", fmt_pct(perd_clasif[0], 0))
```

En los {glue}`f612_n` partidos de la AVP con estadísticas, el ganador tiene una mediana más alta en {glue}`f612_mayores` (Figura 6.12), y menos errores de ataque: {glue}`f612_err_g` frente a {glue}`f612_err_p`. La mayor separación está en el {glue}`f612_top` (RBC = {glue}`f612_top_rbc`): el ganador convierte más de sus ataques y falla menos. Este resultado describe a la AVP de 2003 a 2019 y no se generaliza a la FIVB.

```{admonition} Para el entrenador
:class: tip
Antes del partido, lo que más informa es la **distancia de ranking con el rival**, no el ranking propio: con 1 o 2 posiciones de diferencia el partido está abierto ({glue}`b_cerca` para el favorito), y solo con más de 20 posiciones el favorito es claro ({glue}`b_lejos`). Ser más alto que el rival suma, pero poco: 10 cm de ventaja llevan la tasa de victoria a {glue}`b_est10`. Durante el partido, lo que separa al ganador en la AVP es la eficacia en ataque: convertir más y fallar menos, por encima de atacar más veces. Si su pareja viene de clasificatoria, espere una desventaja moderada ({glue}`b_clasif` de derrotas frente a {glue}`b_directo`).
```

## Análisis multivariado

Esta sección mira varias variables a la vez: qué variables se mueven juntas y si las variables que se conocen antes del partido aportan información distinta o son redundantes entre sí. Solo se usan variables disponibles en ambos circuitos; las estadísticas de juego quedan fuera para no mezclar la AVP con la FIVB.

### Correlación de Spearman

```{code-cell} ipython3
vars_ = ["rank_main", "rank_rival", "edad_media", "estatura_media_cm",
         "dif_edad_pareja", "dif_estatura_pareja_cm", "duracion_min"]
etq = ["Ranking propio", "Ranking rival", "Edad media", "Estatura media",
       "Dif. edad pareja", "Dif. estatura pareja", "Duración"]

def triangulo(d):
    rho = d[vars_].corr(method="spearman").values.copy()
    rho[np.triu_indices_from(rho, k=1)] = np.nan
    return rho

rho = triangulo(lg)
fig = go.Figure(go.Heatmap(
    x=etq, y=etq, z=rho, colorscale=ESCALA_CORR, zmin=-1, zmax=1,
    texttemplate="%{z:.2f}", colorbar=dict(title="rho"),
    hovertemplate="%{y} – %{x}<br>rho = %{z:.3f}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
fig.update_layout(height=560)
estilo_plotly(fig, "Matriz de correlación de Spearman entre variables del equipo", "", "")
```

**Figura 6.13.** Matriz de correlación de Spearman entre variables del equipo.

```{code-cell} ipython3
:tags: [remove-cell]
pares = (pd.DataFrame(rho, index=etq, columns=etq).stack()
           .loc[lambda s: [a != b for a, b in s.index]])
orden = pares.abs().sort_values(ascending=False)
(a1, b1), (a2, b2) = orden.index[0], orden.index[1]
resto = orden.iloc[2:].max()
for (a, b), v in pares.items():
    registrar(f"f6_13_rho_{a}_{b}".replace(" ", "_").replace(".", ""), v)
rho_est_rank = {g: lg[lg.gender == g][["estatura_media_cm", "rank_main"]]
                    .corr(method="spearman").iloc[0, 1] for g in ["M", "W"]}

pegar("f613_par1", f"{b1.lower()} y {a1.lower()}")
pegar("f613_rho1", fmt(pares[(a1, b1)]))
pegar("f613_par2", f"{b2.lower()} y {a2.lower()}")
pegar("f613_rho2", fmt(pares[(a2, b2)]))
pegar("f613_resto", fmt(resto))
pegar("f613_rho_rival", fmt(pares[("Ranking rival", "Ranking propio")]))
pegar("f613_est_h", fmt(rho_est_rank["M"]))
pegar("f613_est_m", fmt(rho_est_rank["W"]))
```

Ninguna correlación es fuerte (Figura 6.13). Las más altas en valor absoluto son entre {glue}`f613_par1` (rho = {glue}`f613_rho1`) y entre {glue}`f613_par2` (rho = {glue}`f613_rho2`); el resto no supera {glue}`f613_resto` en valor absoluto. Las parejas de más edad tienden a tener un mejor ranking (un número más bajo) y a combinar compañeros de edades más distintas, como un veterano con un jugador joven. El ranking propio y el del rival casi no se relacionan (rho = {glue}`f613_rho_rival`), lo esperable en torneos donde se cruzan equipos de todos los niveles. La estatura mezcla hombres y mujeres; separada por género, su correlación con el ranking es de {glue}`f613_est_h` en hombres y {glue}`f613_est_m` en mujeres: los hombres más altos tienden a estar algo mejor rankeados.

```{code-cell} ipython3
rho_g = triangulo(lg[lg.resultado == "Ganador"])
rho_p = triangulo(lg[lg.resultado == "Perdedor"])
fig = make_subplots(rows=1, cols=2, subplot_titles=["Ganador", "Perdedor"],
                    horizontal_spacing=0.18)
for col, z in [(1, rho_g), (2, rho_p)]:
    fig.add_trace(go.Heatmap(
        x=etq, y=etq, z=z, colorscale=ESCALA_CORR, zmin=-1, zmax=1,
        texttemplate="%{z:.2f}", textfont=dict(size=10), showscale=(col == 2),
        colorbar=dict(title="rho"),
        hovertemplate="%{y} – %{x}<br>rho = %{z:.3f}<extra></extra>"), row=1, col=col)
    fig.update_yaxes(autorange="reversed", row=1, col=col)
fig.update_layout(height=560)
estilo_plotly(fig, "Correlación de Spearman según resultado (Ganador vs Perdedor)", "", "")
fig.update_yaxes(autorange="reversed")
fig
```

**Figura 6.14.** Correlación de Spearman según resultado (Ganador vs Perdedor).

```{code-cell} ipython3
:tags: [remove-cell]
dif = pd.DataFrame(np.abs(rho_g - rho_p), index=etq, columns=etq).stack()
dif = dif[[a != b for a, b in dif.index]].sort_values(ascending=False)
(da, db), dmax = dif.index[0], dif.iloc[0]
dur = {res: lg[lg.resultado == res][["duracion_min", "rank_main", "rank_rival"]]
               .corr(method="spearman").loc["duracion_min"] for res in ["Ganador", "Perdedor"]}
sig = dif.iloc[2] if len(dif) > 2 else 0
registrar("f6_14_max_dif", dmax)

pegar("f614_dmax", fmt(dmax))
pegar("f614_par", f"{da.lower()}–{db.lower()}")
pegar("f614_sig", fmt(sig))
pegar("f614_dur_propio", fmt(dur["Ganador"]["rank_main"]))
pegar("f614_dur_rival", fmt(dur["Ganador"]["rank_rival"]))
```

Las matrices de ganadores y perdedores se parecen en casi todo (Figura 6.14). La excepción está en la duración: su mayor diferencia es de {glue}`f614_dmax` en el par {glue}`f614_par`, y la siguiente diferencia relevante cae a {glue}`f614_sig`. Visto desde los ganadores, la duración casi no depende de su propio ranking (rho = {glue}`f614_dur_propio`), pero sí del ranking de su rival (rho = {glue}`f614_dur_rival`): **los partidos se alargan cuando la pareja que pierde es buena**. Las dos matrices son espejo una de la otra, porque el ranking propio del ganador es el ranking rival del perdedor. Fuera de ese par, las variables no se relacionan de forma distinta según el resultado.

### Redundancia entre variables previas al partido (VIF)

```{code-cell} ipython3
:tags: [remove-cell]
from sklearn.linear_model import LinearRegression

def calcular_vif(ref_etapa):
    X = m.dropna(subset=["dif_rank", "dif_edad", "dif_estatura_cm"]).copy()
    niveles = [ref_etapa] + [e for e in ETAPAS if e != ref_etapa]
    X["etapa"] = pd.Categorical(X.etapa, categories=niveles)
    X = pd.concat([X[["dif_rank", "dif_edad", "dif_estatura_cm", "dif_clasif"]],
                   pd.get_dummies(X[["circuit", "gender", "etapa"]],
                                  drop_first=True, dtype=float)], axis=1)
    vif = {}
    for v in X.columns:
        otras = X.drop(columns=v)
        r2 = LinearRegression().fit(otras, X[v]).score(otras, X[v])
        vif[v] = 1 / (1 - r2)
    return pd.Series(vif), len(X)

vif, n_vif = calcular_vif("Cuadro principal")
vif_clasif, _ = calcular_vif("Clasificatoria")
n_clasif_completos = (m.dropna(subset=["dif_rank", "dif_edad", "dif_estatura_cm"])
                        .etapa.eq("Clasificatoria").sum())
nombres = {"dif_rank": "Diferencia de ranking (A − B)",
           "dif_edad": "Diferencia de edad media (años)",
           "dif_estatura_cm": "Diferencia de estatura media (cm)",
           "dif_clasif": "Diferencia en origen de clasificatoria",
           "circuit_FIVB": "Circuito: FIVB (ref. AVP)",
           "gender_W": "Género: mujeres (ref. hombres)"}
tabla_64 = pd.DataFrame({
    "Variable": [nombres.get(k, k.replace("etapa_", "Etapa: ") + " (ref. Cuadro principal)")
                 for k in vif.index],
    "VIF": vif.values})
tabla_64["Diagnóstico"] = np.select([tabla_64.VIF <= 5, tabla_64.VIF <= 10],
                                    ["Aceptable", "Revisar"], "Grave")
registrar("t6_4_n_completos", n_vif)
for k, v in vif.items():
    registrar(f"t6_4_vif_{k}".replace(" ", "_"), v)

pegar("t64_n", fmt(n_vif, 0))
pegar("t64_min", fmt(vif.min()))
pegar("t64_max", fmt(vif.max()))
pegar("t64_max_clasif", fmt(vif_clasif.max(), 1))
pegar("t64_n_clasif", fmt(n_clasif_completos, 0))
```

**Tabla 6.4.** Factor de inflación de la varianza (VIF) de las variables conocidas antes del partido.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_64)
 .fmt_number(columns=["VIF"], decimals=2, sep_mark=".", dec_mark=",")
 .cols_align(align="center", columns=["Diagnóstico"])
 .opt_stylize(style=1, color="gray"))
```

Con los {glue}`t64_n` partidos que tienen ranking, edad y estatura de las dos parejas, todos los VIF están entre {glue}`t64_min` y {glue}`t64_max` (Tabla 6.4): ninguna variable conocida antes del partido es una combinación de las demás, así que pueden analizarse juntas sin contar dos veces la misma información. La etapa se codifica tomando como referencia el **cuadro principal**, la etapa más frecuente. Si se usara la clasificatoria como referencia, los VIF de las etapas subirían hasta {glue}`t64_max_clasif`, pero no por redundancia real: solo {glue}`t64_n_clasif` partidos de clasificatoria tienen todos los datos, y una categoría de referencia tan pequeña vuelve inestables las demás.

```{admonition} Para el entrenador
:class: tip
Las variables que se conocen antes del partido aportan información distinta y no se repiten entre sí: el ranking, la edad y la estatura pueden mirarse juntas sin contar dos veces lo mismo. Un dato útil para planificar: la duración de un partido depende sobre todo de **qué tan buena es la pareja que pierde**. Si su pareja enfrenta a un rival fuerte, prepárese para un partido largo aunque gane; si su pareja es la débil, alargar el partido es una señal de que está compitiendo bien.
```