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

# Análisis temporal

```{code-cell} ipython3
:tags: [remove-cell]
from py.utils import *
from great_tables import GT
from plotly.subplots import make_subplots
from scipy.stats import kendalltau

m = pd.read_csv(CLEAN + "vb_matches_limpio.csv", low_memory=False, parse_dates=["date"])
jg = pd.read_csv(CLEAN + "vb_jugadores.csv", low_memory=False)
ANIOS = pd.Index(range(2000, 2020), name="year")


def bandas(fig, **kw):
    """Bandas grises en los años con cobertura incompleta."""
    for x0, x1 in [(1999.5, 2000.5), (2010.5, 2012.5), (2018.5, 2019.5)]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="#C9D6E3", opacity=0.35, line_width=0, **kw)
    return fig


def por_anio(s):
    """Completa los 20 años (vacío donde no hay datos)."""
    return s.reindex(ANIOS)


pegar("c7_nota", "Bandas grises: años con cobertura incompleta (2000: un torneo; AVP 2011–2012: "
      "casi sin partidos; 2019: hasta el 29 de agosto).")
```

Este capítulo describe cómo cambian los partidos y los jugadores entre 2000 y 2019. No se ajusta un modelo de series de tiempo (ARIMA, suavizado exponencial o similares) por tres razones: hay como máximo 20 puntos anuales por serie, cuando un modelo así necesita del orden de 50; varias series tienen años con cobertura incompleta, que el modelo confundiría con cambios reales; y el objetivo es describir, no pronosticar. En su lugar, cada gráfico muestra la serie anual, algunas con una media móvil de 3 años, y la Tabla 7.1 evalúa si la tendencia es significativa con la prueba de Mann-Kendall (sección 3.5).

Tres periodos se marcan con una banda gris en todos los gráficos y se excluyen de las pruebas de tendencia: 2000, con un solo torneo de la FIVB desde el 16 de septiembre; 2011–2012 en la AVP, cuando el circuito suspendió operaciones y se reorganizó tras su bancarrota {cite:p}`wiki_avp`; y 2019, con datos solo hasta el 29 de agosto (este último solo afecta a los conteos de partidos).

## Cobertura del conjunto de datos por año

**Figura 7.1.** Partidos por año y circuito, 2000–2019.

```{code-cell} ipython3
conteo = m.groupby(["year", "circuit"]).size().unstack()
fig = go.Figure()
for c in ["FIVB", "AVP"]:
    s = por_anio(conteo[c])
    fig.add_scatter(x=s.index, y=s.values, name=c, mode="lines+markers",
                    line=dict(color=PAL[c], width=2), marker=dict(size=7),
                    hovertemplate="%{x} · " + c + "<br>Partidos: %{y:,}<extra></extra>")
bandas(fig)
fig.update_xaxes(dtick=1, tickangle=-45)
estilo_plotly(fig, "Partidos por año y circuito, 2000–2019", "Año", "Número de partidos")
```

```{code-cell} ipython3
:tags: [remove-cell]
cf, ca = conteo["FIVB"], conteo["AVP"].dropna()
for (anio, c), v in m.groupby(["year", "circuit"]).size().items():
    registrar(f"f7_1_n_{c}_{anio}", v)
antes = ca[ca.index < 2011]
pegar("f71_fivb_2000", fmt(cf[2000], 0))
pegar("f71_fivb_max", fmt(cf.max(), 0))
pegar("f71_fivb_max_anio", str(cf.idxmax()))
pegar("f71_avp_max", fmt(antes.max(), 0))
pegar("f71_avp_max_anio", str(antes.idxmax()))
pegar("f71_avp_2011", fmt(ca[2011], 0))
pegar("f71_avp_2012", fmt(ca[2012], 0))
```

La FIVB pasa de {glue}`f71_fivb_2000` partidos en 2000 a un máximo de {glue}`f71_fivb_max` en {glue}`f71_fivb_max_anio` (Figura 7.1). La AVP crece hasta {glue}`f71_avp_max` partidos en {glue}`f71_avp_max_anio`, cae a {glue}`f71_avp_2011` en 2011 y {glue}`f71_avp_2012` en 2012, y se recupera desde 2013. Por esos cortes, la curva mide la cobertura del conjunto de datos tanto como la actividad del deporte: un año con pocos partidos no significa necesariamente un año con poco voleibol. {glue}`c7_nota`

**Figura 7.2.** Partidos por torneo, por año y circuito.

```{code-cell} ipython3
torneos = m.groupby(["year", "circuit"]).tournament_id.nunique().unstack()
ppt = conteo / torneos
fig = go.Figure()
for c in ["FIVB", "AVP"]:
    s, t = por_anio(ppt[c]), por_anio(torneos[c])
    fig.add_scatter(x=s.index, y=s.values, name=c, mode="lines+markers",
                    line=dict(color=PAL[c], width=2), marker=dict(size=7), customdata=t.values,
                    hovertemplate="%{x} · " + c + "<br>Partidos por torneo: %{y:.1f}"
                                  "<br>Torneos: %{customdata}<extra></extra>")
bandas(fig)
fig.update_xaxes(dtick=1, tickangle=-45)
estilo_plotly(fig, "Partidos por torneo, por año y circuito", "Año", "Partidos por torneo")
```

```{code-cell} ipython3
:tags: [remove-cell]
tf = torneos["FIVB"]
ppt_f = ppt["FIVB"]
ta = torneos["AVP"].dropna().loc[2013:]
for (anio, c), v in m.groupby(["year", "circuit"]).tournament_id.nunique().items():
    registrar(f"f7_2_torneos_{c}_{anio}", v)
pegar("f72_tf_min", fmt(tf.loc[2001:2016].min(), 0))
pegar("f72_tf_max", fmt(tf.loc[2001:2016].max(), 0))
pegar("f72_tf_2017", fmt(tf[2017], 0))
pegar("f72_tf_2018", fmt(tf[2018], 0))
pegar("f72_tf_2019", fmt(tf[2019], 0))
pegar("f72_ppt_antes", fmt(ppt_f.loc[2001:2016].median(), 0))
pegar("f72_ppt_despues", fmt(ppt_f.loc[2017:2019].mean(), 0))
pegar("f72_ta_min", fmt(ta.min(), 0))
pegar("f72_ta_max", fmt(ta.max(), 0))
```

La FIVB registra entre {glue}`f72_tf_min` y {glue}`f72_tf_max` torneos por año hasta 2016, y salta a {glue}`f72_tf_2017` en 2017, {glue}`f72_tf_2018` en 2018 y {glue}`f72_tf_2019` en 2019 (Figura 7.2). Al mismo tiempo, los partidos por torneo bajan de unos {glue}`f72_ppt_antes` a unos {glue}`f72_ppt_despues`. El aumento de partidos de la FIVB en 2018 se explica, entonces, por más torneos y más pequeños, no por torneos más grandes. Coincide con la estructura de torneos por categorías de 1 a 5 estrellas que la FIVB introdujo en 2017 {cite:p}`wiki_protour`, que sumó al circuito mundial muchos torneos de menor categoría. En la AVP el número de torneos es estable (entre {glue}`f72_ta_min` y {glue}`f72_ta_max` por año desde 2013).

## Duración, edad y estatura por año

**Figura 7.3.** Duración media de los partidos por año y circuito (min).

```{code-cell} ipython3
dur = m[m.es_forfeit == 0].groupby(["year", "circuit"]).duracion_min.mean().unstack()
fig = go.Figure()
for c in ["FIVB", "AVP"]:
    s = por_anio(dur[c])
    fig.add_scatter(x=s.index, y=s.values, name=c, mode="lines+markers",
                    line=dict(color=PAL[c], width=2), marker=dict(size=7),
                    hovertemplate="%{x} · " + c + "<br>Duración media: %{y:.1f} min<extra></extra>")
    fig.add_scatter(x=s.index, y=s.rolling(3, center=True).mean(), name=f"{c} (media móvil 3 años)",
                    mode="lines", line=dict(color=PAL[c], width=1.5, dash="dash"),
                    hovertemplate="%{x} · " + c + "<br>Media móvil: %{y:.1f} min<extra></extra>")
bandas(fig)
fig.update_xaxes(dtick=1, tickangle=-45)
fig.update_yaxes(range=[30, 55])
estilo_plotly(fig, "Duración media de los partidos por año y circuito (min)", "Año", "Duración media (min)")
```

```{code-cell} ipython3
:tags: [remove-cell]
df_, da_ = dur["FIVB"], dur["AVP"].dropna()
da_ok = da_.loc[da_.index.difference([2011, 2012])]
for (anio, c), v in m[m.es_forfeit == 0].groupby(["year", "circuit"]).duracion_min.mean().items():
    registrar(f"f7_3_dur_{c}_{anio}", v)
pegar("f73_f_2001", fmt(df_[2001], 1))
pegar("f73_f_max", fmt(df_.max(), 1))
pegar("f73_f_max_anio", str(df_.idxmax()))
pegar("f73_f_rec", fmt(df_.loc[2016:2019].mean(), 0))
pegar("f73_a_min", fmt(da_ok.min(), 0))
pegar("f73_a_max", fmt(da_ok.max(), 0))
```

Los partidos de la AVP duran más que los de la FIVB casi todos los años (Figura 7.3). En la FIVB, la duración media sube de {glue}`f73_f_2001` minutos en 2001 a un máximo de {glue}`f73_f_max` en {glue}`f73_f_max_anio`, y desde 2016 se mantiene alrededor de {glue}`f73_f_rec` minutos. En la AVP oscila entre {glue}`f73_a_min` y {glue}`f73_a_max` minutos, sin un patrón claro. La baja reciente de la FIVB coincide con la entrada de torneos más pequeños (Figura 7.2).

**Figura 7.4.** Edad media y estatura media de los jugadores por año y género.

```{code-cell} ipython3
edad = jg.groupby(["year", "gender"]).edad.mean().unstack()
est = jg.groupby(["year", "gender"]).estatura_cm.mean().unstack()
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                    subplot_titles=["Edad media (años)", "Estatura media (cm)"])
for col, d, u in [(1, edad, "años"), (2, est, "cm")]:
    for g, nombre in [("M", "Hombres"), ("W", "Mujeres")]:
        s = por_anio(d[g])
        fig.add_scatter(x=s.index, y=s.values, name=nombre, mode="lines+markers",
                        line=dict(color=PAL[g], width=2), marker=dict(size=6),
                        legendgroup=g, showlegend=(col == 1),
                        hovertemplate="%{x} · " + nombre + "<br>%{y:.1f} " + u + "<extra></extra>",
                        row=1, col=col)
    bandas(fig, row=1, col=col)
fig.update_xaxes(dtick=2, tickangle=-45)
estilo_plotly(fig, "Edad media y estatura media de los jugadores por año y género", "", "")
fig.update_xaxes(title_text="Año")
fig.update_layout(height=520, margin=dict(t=140), title=dict(y=0.98),
                  legend=dict(y=1.16, yanchor="bottom"))
fig
```

```{code-cell} ipython3
:tags: [remove-cell]
e_ = edad.loc[2001:]
h_ = est.loc[2001:]
rango_h = h_.max() - h_.min()
for (anio, g), v in jg.groupby(["year", "gender"]).edad.mean().items():
    registrar(f"f7_4_edad_{g}_{anio}", v)
for (anio, g), v in jg.groupby(["year", "gender"]).estatura_cm.mean().items():
    registrar(f"f7_4_est_{g}_{anio}", v)
pegar("f74_e_min", fmt(e_.min().min(), 1))
pegar("f74_e_max", fmt(e_.max().max(), 1))
pegar("f74_w_2018", fmt(e_["W"].loc[2018], 1))
pegar("f74_rango_h", fmt(rango_h["M"], 1))
pegar("f74_rango_w", fmt(rango_h["W"], 1))
pegar("f74_w_inicio", fmt(h_["W"].loc[2001:2004].mean(), 0))
pegar("f74_w_final", fmt(h_["W"].loc[2015:2019].mean(), 0))
```

Entre 2001 y 2019, la edad media oscila entre {glue}`f74_e_min` y {glue}`f74_e_max` años, siempre algo mayor en hombres que en mujeres (Figura 7.4). En los últimos años las mujeres son más jóvenes: su edad media baja a {glue}`f74_w_2018` años en 2018. La estatura media cambia poco: varía {glue}`f74_rango_h` cm en hombres y {glue}`f74_rango_w` cm en mujeres en todo el periodo, aunque en las mujeres hay un aumento gradual, de unos {glue}`f74_w_inicio` cm a principios de siglo a unos {glue}`f74_w_final` cm al final.

## Tasa de sorpresas por año

**Figura 7.5.** Tasa de sorpresas por año y circuito.

```{code-cell} ipython3
r = m[m.rank_diff.notna()]
ua = r.groupby(["year", "circuit"]).es_upset.agg(["mean", "size"])
ua = ua[ua["size"] >= 30]["mean"].unstack() * 100
ut = r.groupby("year").es_upset.mean() * 100
tasa_global = 100 * r.es_upset.mean()
fig = go.Figure()
for c in ["FIVB", "AVP"]:
    s = por_anio(ua[c])
    fig.add_scatter(x=s.index, y=s.values, name=c, mode="lines+markers",
                    line=dict(color=PAL[c], width=2), marker=dict(size=7),
                    hovertemplate="%{x} · " + c + "<br>Sorpresas: %{y:.1f} %<extra></extra>")
s = por_anio(ut)
fig.add_scatter(x=s.index, y=s.values, name="Total", mode="lines",
                line=dict(color=PAL["Perdedor"], width=2.5),
                hovertemplate="%{x} · Total<br>Sorpresas: %{y:.1f} %<extra></extra>")
fig.add_hline(y=tasa_global, line_dash="dot", line_color=GRIS_TOTAL,
              annotation_text=f"Tasa global: {fmt_pct(tasa_global)}", annotation_position="top left")
bandas(fig)
fig.update_xaxes(dtick=1, tickangle=-45)
fig.update_yaxes(range=[15, 45], ticksuffix=" %")
estilo_plotly(fig, "Tasa de sorpresas por año y circuito", "Año", "% de sorpresas")
```

```{code-cell} ipython3
:tags: [remove-cell]
ut_ok = ut.loc[2001:]
excl_ups = r.groupby(["year", "circuit"]).size().loc[lambda s: s < 30]
txt_excl = ("No se dibuja " + ", ".join(f"{c} {a}" for a, c in excl_ups.index)
            + ", con menos de 30 partidos con ranking." if len(excl_ups)
            else "La AVP no tiene partidos con ranking de ambas parejas en 2011, por eso su línea se corta.")
periodos = {"2001–2010": ut.loc[2001:2010].mean(), "2013–2016": ut.loc[2013:2016].mean(),
            "2017–2019": ut.loc[2017:2019].mean()}
for (anio, c), v in (r.groupby(["year", "circuit"]).es_upset.mean() * 100).items():
    registrar(f"f7_5_pct_{c}_{anio}", v)
registrar("f7_5_ups_2001_2010", periodos["2001–2010"])
registrar("f7_5_ups_2017_2019", periodos["2017–2019"])
pegar("f75_min", fmt_pct(ut_ok.min()))
pegar("f75_max", fmt_pct(ut_ok.max()))
pegar("f75_global", fmt_pct(tasa_global))
pegar("f75_p1", fmt_pct(periodos["2001–2010"]))
pegar("f75_p2", fmt_pct(periodos["2013–2016"]))
pegar("f75_p3", fmt_pct(periodos["2017–2019"]))
pegar("f75_p1_0", fmt_pct(periodos["2001–2010"], 0))
pegar("f75_p3_0", fmt_pct(periodos["2017–2019"], 0))
pegar("f75_excl", txt_excl)
```

La tasa total de sorpresas se mueve entre {glue}`f75_min` y {glue}`f75_max` desde 2001, alrededor de la tasa global de {glue}`f75_global` (Figura 7.5), pero no de forma estable: pasa de un promedio de {glue}`f75_p1` en 2001–2010 a {glue}`f75_p2` en 2013–2016 y {glue}`f75_p3` en 2017–2019. La FIVB tiene más sorpresas que la AVP en casi todos los años, coherente con la Figura 6.11. {glue}`f75_excl` La prueba de tendencia está en la Tabla 7.1.

## Prueba de tendencia

```{code-cell} ipython3
:tags: [remove-cell]
def tendencia(serie, nombre, excluir, unidad):
    s = serie.drop(index=[a for a in excluir if a in serie.index]).dropna()
    t = kendalltau(s.index.values, s.values)
    pend = np.polyfit(s.index.values, s.values, 1)[0]
    tend = ("Sin tendencia" if t.pvalue >= 0.05
            else "Creciente" if t.statistic > 0 else "Decreciente")
    quitados = [str(a) for a in excluir if a in serie.index and a > s.index.min()]
    return {"Serie": nombre,
            "Años usados": f"{s.index.min()}–{s.index.max()}" + (f" (sin {', '.join(quitados)})" if quitados else ""),
            "n": len(s), "tau de Kendall": t.statistic, "p-valor": t.pvalue,
            "Pendiente": pend, "Unidad": f"{unidad}/año", "Tendencia": tend}


series = [
    (ppt["FIVB"], "Partidos por torneo (FIVB)", [2000], "partidos"),
    (ppt["AVP"], "Partidos por torneo (AVP)", [2011, 2012], "partidos"),
    (dur["FIVB"], "Duración media (FIVB)", [2000], "min"),
    (dur["AVP"], "Duración media (AVP)", [2011, 2012], "min"),
    (edad["M"], "Edad media (hombres)", [2000], "años"),
    (edad["W"], "Edad media (mujeres)", [2000], "años"),
    (est["M"], "Estatura media (hombres)", [2000], "cm"),
    (est["W"], "Estatura media (mujeres)", [2000], "cm"),
    (ut, "Tasa de sorpresas (total)", [2000], "pp"),
    (ua["FIVB"], "Tasa de sorpresas (FIVB)", [2000], "pp"),
    (ua["AVP"], "Tasa de sorpresas (AVP)", [2011, 2012], "pp")]
tabla_71 = pd.DataFrame([tendencia(*x) for x in series])
for _, f in tabla_71.iterrows():
    clave = f.Serie.lower().replace(" ", "_").replace("(", "").replace(")", "")
    registrar(f"t7_1_tau_{clave}", f["tau de Kendall"])
    registrar(f"t7_1_p_{clave}", f["p-valor"])
sig = tabla_71[tabla_71.Tendencia != "Sin tendencia"]
t71 = tabla_71.set_index("Serie")
pegar("t71_n", str(len(tabla_71)))
pegar("t71_n_sig", str(len(sig)))
pegar("t71_lista", "; ".join(f"{f.Serie.lower()}, {f.Tendencia.lower()} ({fmt(f.Pendiente, 2)} {f.Unidad})"
                             for f in sig.itertuples()))
pegar("t71_tau_tot", fmt(t71["tau de Kendall"]["Tasa de sorpresas (total)"]))
pegar("t71_p_tot", fmt_p(t71["p-valor"]["Tasa de sorpresas (total)"]))
pegar("t71_tau_fivb", fmt(t71["tau de Kendall"]["Tasa de sorpresas (FIVB)"]))
pegar("t71_tau_avp", fmt(t71["tau de Kendall"]["Tasa de sorpresas (AVP)"]))
pegar("t71_nmin", str(tabla_71.n.min()))
pegar("t71_nmax", str(tabla_71.n.max()))
```

**Tabla 7.1.** Prueba de tendencia de Mann-Kendall y pendiente lineal por serie.

```{code-cell} ipython3
:tags: [remove-input]
t = tabla_71.copy()
t["p-valor"] = t["p-valor"].map(fmt_p)
(GT(t)
 .fmt_number(columns=["tau de Kendall", "Pendiente"], decimals=3, sep_mark=".", dec_mark=",")
 .cols_align(align="center", columns=["n", "Tendencia"])
 .opt_stylize(style=1, color="gray"))
```

De las {glue}`t71_n` series, {glue}`t71_n_sig` muestran una tendencia significativa con α = 0,05 (Tabla 7.1): {glue}`t71_lista`. El resultado más relevante es la tasa de sorpresas: crece en el total (tau = {glue}`t71_tau_tot`, p {glue}`t71_p_tot`), en la FIVB (tau = {glue}`t71_tau_fivb`) y en la AVP (tau = {glue}`t71_tau_avp`). Como aparece en los dos circuitos y ya antes de 2017, no se explica solo por los torneos pequeños de la FIVB: el circuito profesional se ha vuelto más parejo y el ranking anticipa el resultado algo peor que hace 20 años. Con tan pocos años, la potencia de la prueba es baja: una serie "sin tendencia" no prueba que el deporte sea estable, solo que estos datos no alcanzan para afirmar un cambio. Esto confirma la decisión de no ajustar un modelo de series de tiempo: con {glue}`t71_nmin` a {glue}`t71_nmax` puntos por serie no hay información suficiente para pronosticar.

```{admonition} Para el entrenador
:class: tip
El circuito es hoy más parejo que hace 20 años: las sorpresas pasaron de cerca del {glue}`f75_p1_0` de los partidos a cerca del {glue}`f75_p3_0`. Para preparar un partido, eso significa que una ventaja de ranking vale algo menos que antes y que conviene usar las probabilidades de los años recientes. La edad y la estatura cambian poco, con dos matices: las jugadoras son hoy algo más altas y más jóvenes, y desde 2017 la FIVB tiene muchos más torneos pequeños, con partidos más cortos.
```
