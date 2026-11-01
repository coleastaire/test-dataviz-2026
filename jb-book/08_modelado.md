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

# Modelado

```{code-cell} ipython3
:tags: [remove-cell]
from py.utils import *
from great_tables import GT, style, loc
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import statsmodels.formula.api as smf

m = pd.read_csv(CLEAN + "vb_matches_limpio.csv", low_memory=False)
ETAPAS_MOD = ["Cuadro principal", "Clasificatoria", "Fase de grupos",
              "Semifinales", "Finales"]           # referencia: Cuadro principal (sección 6.3.2)
CLAVE = {"Regla \"gana el mejor rankeado\"": "regla", "Regresión logística": "logistica",
         "Random Forest": "rf", "XGBoost": "xgb"}  # nombres sin tilde para las cifras de control
```

Este capítulo responde la cuarta pregunta del entrenador (sección 1.2): con la información disponible antes del partido, ¿se puede anticipar quién gana mejor que con la regla simple "gana el mejor rankeado"?

## Planteamiento del problema

```{code-cell} ipython3
:tags: [remove-cell]
num = ["dif_rank", "dif_edad", "dif_estatura_cm", "dif_clasif"]
cat = ["circuit", "gender", "etapa"]
d = m[m.es_forfeit == 0].dropna(subset=["dif_rank", "dif_edad", "dif_estatura_cm"])
pct_modelable = 100 * len(d) / len(m)
registrar("m9_n_modelable", len(d))
pegar("m8_n", fmt(len(d), 0))
pegar("m8_pct", fmt_pct(pct_modelable))
```

Cada fila es un partido y la variable respuesta es `a_gana`: vale 1 si gana la pareja A y 0 si gana la B. La pareja A se asignó con una regla fija, ajena al resultado (sección 5.4), así que A gana aproximadamente la mitad de las veces y ambas clases están balanceadas (sección 6.1.1). Los predictores son las diferencias A − B en ranking, edad media, estatura media y origen en clasificatoria, más el contexto del partido: circuito, género y etapa. El contexto es el mismo para las dos parejas, así que solo puede aportar combinado con las diferencias (por ejemplo, si una ventaja de estatura pesa más en hombres), algo que pueden captar Random Forest y XGBoost, pero no una regresión logística sin interacciones.

Las estadísticas de juego (kills, aces, bloqueos, defensas) no se usan como predictores: se producen durante el partido, así que un entrenador no las conoce antes de jugar. Usarlas sería una fuga de información. Se analizan aparte, en un modelo explicativo (sección 8.11).

Solo se usan partidos jugados (sin forfeits) con ranking, edad y estatura de las dos parejas: {glue}`m8_n` partidos, el {glue}`m8_pct` del total. Todo partido en el que una pareja solo tiene posición de clasificatoria ("Q") queda fuera, porque la diferencia de ranking no se puede calcular.

## Variable respuesta y división de los datos

```{code-cell} ipython3
:tags: [remove-cell]
train, test = d[d.year <= 2016], d[d.year >= 2017]
Xtr, ytr = train[num + cat], train.a_gana
Xte, yte = test[num + cat], test.a_gana
tabla_81 = pd.DataFrame([
    ("Entrenamiento", "2000–2016", len(train), 100 * len(train) / len(d), 100 * ytr.mean()),
    ("Prueba", "2017–2019", len(test), 100 * len(test) / len(d), 100 * yte.mean())],
    columns=["Conjunto", "Años", "Partidos", "% del total modelable", "% en que gana A"])
registrar("t9_1_n_train", len(train))
registrar("t9_1_n_test", len(test))
pegar("t81_ntr", fmt(len(train), 0))
pegar("t81_nte", fmt(len(test), 0))
pegar("t81_pct_te", fmt_pct(tabla_81.iloc[1, 3]))
pegar("t81_ytr", fmt_pct(100 * ytr.mean()))
pegar("t81_yte", fmt_pct(100 * yte.mean()))
```

**Figura 8.1.** Variable respuesta en los conjuntos de entrenamiento y prueba.

```{code-cell} ipython3
fig = go.Figure()
for valor, nombre, color in [(1, "Gana A", PAL["Ganador"]), (0, "Gana B", PAL["Perdedor"])]:
    n = [(train.a_gana == valor).sum(), (test.a_gana == valor).sum()]
    pct = [100 * n[0] / len(train), 100 * n[1] / len(test)]
    fig.add_bar(x=["Entrenamiento (2000–2016)", "Prueba (2017–2019)"], y=n, name=nombre,
                marker_color=color, text=[f"{fmt(a, 0)} ({fmt_pct(b)})" for a, b in zip(n, pct)],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{x}<br>" + nombre + ": %{y:,} partidos<extra></extra>")
fig.update_layout(barmode="group")
fig.update_yaxes(range=[0, (train.a_gana == 1).sum() * 1.2], tickformat=",d")
estilo_plotly(fig, "Variable respuesta en los conjuntos de entrenamiento y prueba", "", "Partidos")
```

**Tabla 8.1.** División temporal de los datos.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_81)
 .fmt_number(columns=["Partidos"], decimals=0, sep_mark=".", dec_mark=",")
 .fmt_number(columns=["% del total modelable", "% en que gana A"], decimals=1, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

La división es temporal: el modelo se entrena con {glue}`t81_ntr` partidos de 2000–2016 y se prueba con {glue}`t81_nte` de 2017–2019 ({glue}`t81_pct_te` del total), que nunca ve durante el entrenamiento (Tabla 8.1). Así se imita el uso real: predecir partidos futuros con lo aprendido del pasado. La variable respuesta está balanceada en los dos conjuntos: gana A en el {glue}`t81_ytr` de los partidos de entrenamiento y en el {glue}`t81_yte` de los de prueba (Figura 8.1). Por eso la accuracy es interpretable y no hace falta corregir desbalance de clases.

## Validación cruzada estratificada

Para elegir los hiperparámetros se usa validación cruzada de 5 pliegues estratificada dentro del conjunto de entrenamiento: los partidos de 2000–2016 se dividen en 5 partes con la misma proporción de victorias de A; cada combinación se entrena con 4 partes y se evalúa con la quinta, rotando, y se elige la de mayor AUC media. El conjunto de prueba solo se usa una vez, al final, para evaluar el modelo elegido.

## Preprocesamiento

```{code-cell} ipython3
prep = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first",
                           categories=[["AVP", "FIVB"], ["M", "W"], ETAPAS_MOD]), cat)],
    remainder="passthrough")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

Las tres variables de contexto se convierten en variables indicadoras, con referencia en la AVP, los hombres y el cuadro principal; el resultado son 10 columnas. Después, todas se estandarizan (media 0 y desviación 1) dentro de cada modelo. La estandarización no cambia los árboles, pero mantiene el mismo flujo en los tres modelos. Todo el preprocesamiento va dentro de un `Pipeline`, de modo que en la validación cruzada se ajusta solo con los pliegues de entrenamiento y no se filtra información del pliegue de validación.

## Modelos de referencia

```{code-cell} ipython3
:tags: [remove-cell]
prob_regla = -Xte.dif_rank                         # mejor ranking de A = mayor puntaje
pred_regla = (Xte.dif_rank < 0).astype(int)
regla = {"Modelo": "Regla \"gana el mejor rankeado\"",
         "AUC CV": roc_auc_score(ytr, -Xtr.dif_rank), "AUC CV DE": np.nan,
         "AUC prueba": roc_auc_score(yte, prob_regla),
         "Accuracy prueba": accuracy_score(yte, pred_regla),
         "F1 prueba": f1_score(yte, pred_regla)}
acc_regla_train = accuracy_score(ytr, (Xtr.dif_rank < 0).astype(int))
pegar("m8_regla_acc", fmt_pct(100 * regla["Accuracy prueba"]))
pegar("m8_regla_auc", fmt(regla["AUC prueba"], 3))
```

El punto de comparación es lo que haría un entrenador sin modelo: apostar por la pareja mejor rankeada. En el conjunto de prueba, esa regla acierta el {glue}`m8_regla_acc` de los partidos, con una AUC de {glue}`m8_regla_auc` (para la AUC se usa la diferencia de ranking como puntaje). La regresión logística es el segundo punto de referencia: es el modelo más simple que combina todos los predictores. Un modelo más complejo solo vale la pena si supera a estos dos.

## Optimización de hiperparámetros

```{code-cell} ipython3
modelos = {
    "Regresión logística": (LogisticRegression(penalty=None, max_iter=1000), {}),
    "Random Forest": (RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
                      {"mod__max_features": [2, 3, 5], "mod__min_samples_split": [5, 20, 50]}),
    "XGBoost": (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
                {"mod__n_estimators": [200, 500], "mod__max_depth": [3, 5],
                 "mod__learning_rate": [0.05, 0.1]})}

res, ajustados, busquedas, probs = [], {}, {}, {}
for nombre, (mod, grilla) in modelos.items():
    pipe = Pipeline([("prep", prep), ("esc", StandardScaler()), ("mod", mod)])
    gs = GridSearchCV(pipe, grilla, cv=cv, scoring="roc_auc").fit(Xtr, ytr)
    prob = gs.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    ajustados[nombre], busquedas[nombre], probs[nombre] = gs.best_estimator_, gs, prob
    res.append({"Modelo": nombre, "AUC CV": gs.best_score_,
                "AUC CV DE": gs.cv_results_["std_test_score"][gs.best_index_],
                "AUC prueba": roc_auc_score(yte, prob),
                "Accuracy prueba": accuracy_score(yte, pred), "F1 prueba": f1_score(yte, pred)})
```

```{code-cell} ipython3
:tags: [remove-cell]
filas = [("Regresión logística", "—", "Sin ajuste", "—", busquedas["Regresión logística"].best_score_)]
nombres_hp = {"mod__max_features": "max_features", "mod__min_samples_split": "min_samples_split",
              "mod__n_estimators": "n_estimators", "mod__max_depth": "max_depth",
              "mod__learning_rate": "learning_rate"}


def valor(v):
    return fmt(v, 2) if isinstance(v, float) else str(v)


rango_cv = {}
for nombre in ["Random Forest", "XGBoost"]:
    gs = busquedas[nombre]
    rango_cv[nombre] = gs.cv_results_["mean_test_score"].max() - gs.cv_results_["mean_test_score"].min()
    for hp, valores in modelos[nombre][1].items():
        filas.append((nombre, nombres_hp[hp], ", ".join(valor(v) for v in valores),
                      valor(gs.best_params_[hp]), gs.best_score_))
tabla_82 = pd.DataFrame(filas, columns=["Modelo", "Hiperparámetro", "Valores probados",
                                        "Valor elegido", "AUC media en CV"])
for nombre in ["Random Forest", "XGBoost"]:
    registrar(f"t9_2_auc_cv_{CLAVE[nombre]}", busquedas[nombre].best_score_)


def txt_params(p):
    return ", ".join(f"{nombres_hp[k]} = {valor(v)}" for k, v in p.items())


pegar("t82_rf", txt_params(busquedas["Random Forest"].best_params_))
pegar("t82_xgb", txt_params(busquedas["XGBoost"].best_params_))
pegar("t82_rango_xgb", fmt(rango_cv["XGBoost"], 3))
pegar("t82_rango_rf", fmt(rango_cv["Random Forest"], 3))
```

**Tabla 8.2.** Hiperparámetros evaluados y seleccionados.

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_82, groupname_col="Modelo")
 .fmt_number(columns=["AUC media en CV"], decimals=3, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

La mejor combinación de Random Forest fue {glue}`t82_rf` y la de XGBoost, {glue}`t82_xgb` (Tabla 8.2). En XGBoost, la AUC en validación cruzada cambia menos de {glue}`t82_rango_xgb` entre todas las combinaciones: el resultado casi no depende de los hiperparámetros. En Random Forest el rango es mayor ({glue}`t82_rango_rf`) porque los árboles que se dividen con muy pocos partidos memorizan ruido; exigir más partidos por división lo corrige.

## Comparación final

```{code-cell} ipython3
:tags: [remove-cell]
tabla_83 = pd.DataFrame([regla] + res)
tabla_83["AUC CV (media ± DE)"] = [
    fmt(a, 3) + ("" if pd.isna(s) else f" ± {fmt(s, 3)}")
    for a, s in zip(tabla_83["AUC CV"], tabla_83["AUC CV DE"])]
t83 = tabla_83.set_index("Modelo")
mejor = tabla_83.iloc[1:].sort_values("AUC prueba", ascending=False).iloc[0]
ganancia = mejor["AUC prueba"] - regla["AUC prueba"]
for _, f in tabla_83.iterrows():
    registrar(f"t9_3_auc_{CLAVE[f.Modelo]}", f["AUC prueba"])
    registrar(f"t9_3_acc_{CLAVE[f.Modelo]}", f["Accuracy prueba"])
pegar("t83_auc_log", fmt(t83["AUC prueba"]["Regresión logística"], 3))
pegar("t83_mejor", mejor.Modelo)
pegar("t83_mejor_auc", fmt(mejor["AUC prueba"], 3))
pegar("t83_mejor_acc", fmt_pct(100 * mejor["Accuracy prueba"]))
pegar("t83_ganancia", fmt(ganancia, 3))
pegar("t83_cv_media", fmt(t83["AUC CV"].iloc[1:].mean(), 2))
pegar("t83_te_media", fmt(t83["AUC prueba"].iloc[1:].mean(), 2))
pegar("t83_acc_regla_tr", fmt_pct(100 * acc_regla_train))
```

**Tabla 8.3.** Comparación final de modelos (AUC CV en 2000–2016; métricas de prueba en 2017–2019).

```{code-cell} ipython3
:tags: [remove-input]
t = tabla_83[["Modelo", "AUC CV (media ± DE)", "AUC prueba", "Accuracy prueba", "F1 prueba"]]
gt = (GT(t)
      .fmt_number(columns=["AUC prueba", "Accuracy prueba", "F1 prueba"], decimals=3,
                  sep_mark=".", dec_mark=","))
for col in ["AUC prueba", "Accuracy prueba", "F1 prueba"]:
    gt = gt.tab_style(style=style.text(weight="bold"),
                      locations=loc.body(columns=col, rows=[int(t[col].idxmax())]))
(gt.tab_source_note("La regla no se entrena: su AUC en la columna CV se calcula sobre todo el periodo 2000–2016.")
   .opt_stylize(style=1, color="gray"))
```

La regla del ranking ya acierta el {glue}`m8_regla_acc` de los partidos de prueba (Tabla 8.3). La regresión logística llega a una AUC de {glue}`t83_auc_log` y el mejor modelo, {glue}`t83_mejor`, a {glue}`t83_mejor_auc`. La ganancia sobre la regla es de apenas {glue}`t83_ganancia` puntos de AUC, y la accuracy de los tres modelos queda prácticamente igual a la de la regla. La diferencia de ranking concentra casi toda la información disponible antes del partido: edad, estatura, clasificatoria y contexto agregan muy poco.

Todos los modelos, incluida la regla, rinden menos en prueba que en la validación cruzada (AUC cercana a {glue}`t83_cv_media` en 2000–2016 frente a {glue}`t83_te_media` en 2017–2019). No es sobreajuste, porque también le pasa a la regla, que no aprende nada: su accuracy cae de {glue}`t83_acc_regla_tr` a {glue}`m8_regla_acc`. Es el mismo cambio que mostró el análisis temporal (sección 7.3): en 2017–2019 hay más sorpresas y el ranking anticipa peor.

**Figura 8.2.** Curvas ROC en el conjunto de prueba (2017–2019).

```{code-cell} ipython3
fig = go.Figure()
for nombre, p in [("Regla del ranking", prob_regla), *probs.items()]:
    fpr, tpr, _ = roc_curve(yte, p)
    auc = roc_auc_score(yte, p)
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{nombre} (AUC = {fmt(auc, 3)})",
                    line=dict(color=PAL_MODELO[nombre], width=2,
                              dash="dash" if nombre == "Regla del ranking" else "solid"),
                    hovertemplate=nombre + "<br>FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>")
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                line=dict(color="#C9D6E3", dash="dot"))
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])
fig.update_layout(height=560)
estilo_plotly(fig, "Curvas ROC en el conjunto de prueba (2017–2019)",
              "Tasa de falsos positivos (1 − especificidad)", "Tasa de verdaderos positivos (sensibilidad)")
fig.update_layout(legend=dict(orientation="v", x=0.55, y=0.05, yanchor="bottom", xanchor="left"))
```

Las curvas de los tres modelos casi se superponen con la de la regla del ranking (Figura 8.2). La regla se escalona porque la diferencia de ranking toma valores enteros y tiene muchos empates; los modelos solo la suavizan, sin separarse de forma visible en ninguna zona de la curva.

## Matriz de confusión y distribución de probabilidades

```{code-cell} ipython3
:tags: [remove-cell]
pred_mejor = (probs[mejor.Modelo] >= 0.5).astype(int)
mc = {"Regla del ranking": confusion_matrix(yte, pred_regla, labels=[1, 0]),
      mejor.Modelo: confusion_matrix(yte, pred_mejor, labels=[1, 0])}
cm = mc[mejor.Modelo]
sens = cm[0, 0] / cm[0].sum()
espec = cm[1, 1] / cm[1].sum()
prec = cm[0, 0] / cm[:, 0].sum()
cambios = int((pred_mejor != pred_regla.values).sum())
for k, v in [("tp", cm[0, 0]), ("fn", cm[0, 1]), ("fp", cm[1, 0]), ("tn", cm[1, 1])]:
    registrar(f"f8_3_{k}_{CLAVE[mejor.Modelo]}", v)
pegar("f83_tp", fmt(cm[0, 0], 0))
pegar("f83_tn", fmt(cm[1, 1], 0))
pegar("f83_fp", fmt(cm[1, 0], 0))
pegar("f83_fn", fmt(cm[0, 1], 0))
pegar("f83_sens", fmt_pct(100 * sens))
pegar("f83_espec", fmt_pct(100 * espec))
pegar("f83_prec", fmt_pct(100 * prec))
pegar("f83_cambios", fmt(cambios, 0))
pegar("f83_pct_cambios", fmt_pct(100 * cambios / len(yte)))
```

**Figura 8.3.** Matriz de confusión en el conjunto de prueba: regla del ranking y mejor modelo.

```{code-cell} ipython3
fig = make_subplots(rows=1, cols=2, subplot_titles=list(mc.keys()), horizontal_spacing=0.16)
etq_real, etq_pred = ["Real: gana A", "Real: gana B"], ["Predice: gana A", "Predice: gana B"]
for col, (nombre, c) in enumerate(mc.items(), start=1):
    pct = 100 * c / c.sum()
    fig.add_trace(go.Heatmap(
        z=c, x=etq_pred, y=etq_real, colorscale=ESCALA_SEC, showscale=False,
        text=[[f"{fmt(c[i, j], 0)}<br>({fmt_pct(pct[i, j])})" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", textfont=dict(size=14),
        hovertemplate="%{y}<br>%{x}<br>Partidos: %{z:,}<extra></extra>"), row=1, col=col)
    fig.update_yaxes(autorange="reversed", row=1, col=col)
fig.update_layout(height=430)
estilo_plotly(fig, "Matriz de confusión en el conjunto de prueba (umbral 0,5)", "", "")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig
```

Con un umbral de 0,5, el mejor modelo acierta {glue}`f83_tp` partidos en los que gana A y {glue}`f83_tn` en los que gana B, y se equivoca en {glue}`f83_fp` (predice que gana A y gana B) y {glue}`f83_fn` (al revés) (Figura 8.3). Su sensibilidad es del {glue}`f83_sens`, su especificidad del {glue}`f83_espec` y su precisión del {glue}`f83_prec`: los dos tipos de error son de tamaño parecido, lo esperable con clases balanceadas. Las dos matrices casi coinciden: el modelo cambia la predicción de la regla en solo {glue}`f83_cambios` partidos ({glue}`f83_pct_cambios` de la prueba).

**Figura 8.4.** Distribución de la probabilidad predicha de que gane A, según el resultado real.

```{code-cell} ipython3
p_mejor = probs[mejor.Modelo]
fig = go.Figure()
for valor, nombre, color in [(1, "Real: gana A", PAL["Ganador"]), (0, "Real: gana B", PAL["Perdedor"])]:
    fig.add_histogram(x=p_mejor[yte.values == valor], name=nombre, marker_color=color, opacity=0.6,
                      xbins=dict(start=0, end=1, size=0.025),
                      hovertemplate=nombre + "<br>Probabilidad: %{x}<br>Partidos: %{y:,}<extra></extra>")
fig.add_vline(x=0.5, line_dash="dash", line_color=GRIS_TOTAL,
              annotation_text="Umbral 0,5", annotation_position="top")
fig.update_layout(barmode="overlay")
estilo_plotly(fig, f"Probabilidad predicha de que gane A ({mejor.Modelo}, prueba 2017–2019)",
              "Probabilidad predicha de que gane A", "Partidos")
```

```{code-cell} ipython3
:tags: [remove-cell]
claros = (np.abs(p_mejor - 0.5) >= 0.25)
pegar("f84_pct_claros", fmt_pct(100 * claros.mean()))
pegar("f84_acc_claros", fmt_pct(100 * ((p_mejor[claros] >= 0.5) == yte.values[claros]).mean()))
pegar("f84_acc_dudosos", fmt_pct(100 * ((p_mejor[~claros] >= 0.5) == yte.values[~claros]).mean()))
```

Las dos distribuciones se separan, pero se superponen mucho alrededor de 0,5 (Figura 8.4): en buena parte de los partidos el modelo no tiene una preferencia clara. Solo el {glue}`f84_pct_claros` de los partidos de prueba recibe una probabilidad por debajo de 0,25 o por encima de 0,75; en ellos el modelo acierta el {glue}`f84_acc_claros`, frente al {glue}`f84_acc_dudosos` en los demás. Para el entrenador, la probabilidad es más útil que la predicción binaria: indica cuándo el partido está realmente abierto.

## Importancia de variables

**Figura 8.5.** Importancia de variables por permutación (caída de AUC en prueba).

```{code-cell} ipython3
etq_var = {"dif_rank": "Diferencia de ranking", "dif_edad": "Diferencia de edad",
           "dif_estatura_cm": "Diferencia de estatura", "dif_clasif": "Diferencia en clasificatoria",
           "circuit": "Circuito", "gender": "Género", "etapa": "Etapa"}
imp = []
for nombre, modelo in ajustados.items():
    pi = permutation_importance(modelo, Xte, yte, scoring="roc_auc",
                                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    imp.append(pd.DataFrame({"modelo": nombre, "variable": Xte.columns,
                             "media": pi.importances_mean, "de": pi.importances_std}))
imp = pd.concat(imp)
orden_var = (imp.groupby("variable").media.mean().sort_values().index.tolist())
fig = go.Figure()
for nombre in ajustados:
    s = imp[imp.modelo == nombre].set_index("variable").loc[orden_var]
    fig.add_bar(x=s.media, y=[etq_var[v] for v in s.index], orientation="h", name=nombre,
                marker_color=PAL_MODELO[nombre], error_x=dict(type="data", array=s.de),
                text=[fmt(v, 3) for v in s.media], textposition="outside", cliponaxis=False,
                textfont=dict(size=10),
                hovertemplate=nombre + "<br>%{y}<br>Caída de AUC: %{x:.4f}<extra></extra>")
fig.update_layout(barmode="group", height=620)
estilo_plotly(fig, "Importancia de variables por permutación (caída de AUC en prueba)",
              "Caída de AUC", "")
```

```{code-cell} ipython3
:tags: [remove-cell]
im = imp.pivot(index="variable", columns="modelo", values="media")
top_rank = im.loc["dif_rank"]
resto_max = im.drop(index="dif_rank").max().max()
contexto_max = im.loc[["circuit", "gender", "etapa"]].max().max()
segunda = im.drop(index="dif_rank").mean(axis=1).idxmax()
for (nombre, var), v in imp.set_index(["modelo", "variable"]).media.items():
    registrar(f"f9_2_imp_{CLAVE[nombre]}_{var}", v)
pegar("f85_rank_min", fmt(top_rank.min(), 3))
pegar("f85_rank_max", fmt(top_rank.max(), 3))
pegar("f85_resto", fmt(resto_max, 3))
pegar("f85_segunda", etq_var[segunda].lower())
pegar("f85_contexto", fmt(contexto_max, 3))
```

En los tres modelos, la variable más importante es la diferencia de ranking: al desordenarla al azar, la AUC cae entre {glue}`f85_rank_min` y {glue}`f85_rank_max` (Figura 8.5). Ninguna otra variable baja la AUC más de {glue}`f85_resto`; la segunda en importancia es la {glue}`f85_segunda`. El contexto (circuito, género y etapa) aporta como máximo {glue}`f85_contexto`: prácticamente nada. Se usa la importancia por permutación, y no la importancia interna de cada algoritmo, porque mide lo mismo en los tres modelos.

## Interpretación con la regresión logística

```{code-cell} ipython3
:tags: [remove-cell]
tr = train.copy()
tr["etapa"] = pd.Categorical(tr.etapa, categories=ETAPAS_MOD)
logit = smf.logit("a_gana ~ dif_rank + dif_edad + dif_estatura_cm + dif_clasif"
                  " + C(circuit, Treatment('AVP')) + C(gender, Treatment('M')) + C(etapa)",
                  data=tr).fit(disp=0)
ic = logit.conf_int()
nombres_or = {"Intercept": "Intercepto", "dif_rank": "Diferencia de ranking (por posición)",
              "dif_edad": "Diferencia de edad media (por año)",
              "dif_estatura_cm": "Diferencia de estatura media (por cm)",
              "dif_clasif": "Diferencia en origen de clasificatoria",
              "C(circuit, Treatment('AVP'))[T.FIVB]": "Circuito: FIVB (ref. AVP)",
              "C(gender, Treatment('M'))[T.W]": "Género: mujeres (ref. hombres)"}
tabla_84 = pd.DataFrame({
    "Predictor": [nombres_or.get(k, k.replace("C(etapa)[T.", "Etapa: ").replace("]", "") + " (ref. Cuadro principal)")
                  for k in logit.params.index],
    "Coeficiente": logit.params.values, "OR": np.exp(logit.params.values),
    "IC 95 %": [f"{fmt(np.exp(a), 3)} – {fmt(np.exp(b), 3)}" for a, b in ic.values],
    "p-valor": [fmt_p(p) for p in logit.pvalues.values]})
orr = pd.Series(np.exp(logit.params.values), index=logit.params.index)
ic_rank = np.exp(ic.loc["dif_rank"])
for k in ["dif_rank", "dif_edad", "dif_estatura_cm", "dif_clasif"]:
    registrar(f"t9_4_coef_{k}", logit.params[k])
pegar("t84_or_rank", fmt(1 / orr["dif_rank"], 3))
pegar("t84_ic_lo", fmt(1 / ic_rank[1], 3))
pegar("t84_ic_hi", fmt(1 / ic_rank[0], 3))
pegar("t84_or_rank10", fmt((1 / orr["dif_rank"]) ** 10, 2))
pegar("t84_or_est", fmt(orr["dif_estatura_cm"], 3))
pegar("t84_or_est10", fmt(orr["dif_estatura_cm"] ** 10, 2))
pegar("t84_or_edad", fmt(orr["dif_edad"], 3))
pegar("t84_or_clasif", fmt(orr["dif_clasif"], 2))
pegar("t84_pct_rank", fmt_pct(100 * (1 / orr["dif_rank"] - 1), 0))
```

**Tabla 8.4.** Regresión logística: *odds ratios* de los predictores (entrenamiento, variables sin estandarizar).

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_84)
 .fmt_number(columns=["Coeficiente", "OR"], decimals=3, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

La regresión logística permite leer el efecto de cada variable en unidades reales (Tabla 8.4). Como las diferencias son A − B y un número de ranking más bajo es mejor, cada posición de ventaja en el ranking multiplica las *odds* de ganar por {glue}`t84_or_rank` (IC 95 % {glue}`t84_ic_lo`–{glue}`t84_ic_hi`); 10 posiciones de ventaja las multiplican por {glue}`t84_or_rank10`. Cada centímetro de ventaja en estatura media las multiplica por {glue}`t84_or_est` (10 cm, por {glue}`t84_or_est10`), y cada año de ventaja en edad media, por {glue}`t84_or_edad`. A igual ranking, la pareja que viene de clasificatoria tiene *odds* {glue}`t84_or_clasif` veces mayores. Esto no contradice la Tabla 6.3, donde esas parejas perdían más: pierden más porque suelen tener peor ranking, no por venir de clasificatoria. Una posible explicación es que llegan con partidos encima y que su ranking se queda corto frente a su nivel real; con estos datos no se puede comprobar. Las variables de contexto no son significativas, como se esperaba: valen lo mismo para las dos parejas.

## Modelo secundario: AVP con estadísticas de juego

```{admonition} Alcance de este modelo
:class: important
Este modelo es explicativo, no predictivo. Las estadísticas se conocen después del partido, así que una AUC alta solo dice qué acciones acompañan a la victoria. Describe a la AVP de 2003 a 2019 y no se aplica a la FIVB.
```

```{code-cell} ipython3
:tags: [remove-cell]
a = m[(m.circuit == "AVP") & (m.tiene_stats_completas == 1) & (m.es_forfeit == 0)].copy()
for s in ["kills", "aces", "blocks", "digs", "errors", "attacks"]:
    a[f"w_{s}"] = a[f"w_p1_tot_{s}"] + a[f"w_p2_tot_{s}"]
    a[f"l_{s}"] = a[f"l_p1_tot_{s}"] + a[f"l_p2_tot_{s}"]
    a[f"dif_{s}"] = np.where(a.a_gana == 1, a[f"w_{s}"] - a[f"l_{s}"], a[f"l_{s}"] - a[f"w_{s}"])
hit_w = (a.w_kills - a.w_errors) / a.w_attacks
hit_l = (a.l_kills - a.l_errors) / a.l_attacks
a["dif_hitpct"] = np.where(a.a_gana == 1, hit_w - hit_l, hit_l - hit_w)
pa = ["dif_kills", "dif_aces", "dif_blocks", "dif_digs", "dif_errors", "dif_hitpct"]
etq_pa = {"dif_kills": "kills", "dif_aces": "aces", "dif_blocks": "bloqueos",
          "dif_digs": "defensas", "dif_errors": "errores de ataque", "dif_hitpct": "% de ataque"}
a = a.replace([np.inf, -np.inf], np.nan).dropna(subset=pa)
a_tr, a_te = a[a.year <= 2016], a[a.year >= 2017]

secundarios = {
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=500, max_features=3, min_samples_split=50,
        random_state=RANDOM_STATE, n_jobs=-1))}
filas = []
for nombre, modelo in secundarios.items():
    modelo.fit(a_tr[pa], a_tr.a_gana)
    p = modelo.predict_proba(a_te[pa])[:, 1]
    pi = permutation_importance(modelo, a_te[pa], a_te.a_gana, scoring="roc_auc",
                                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    top3 = pd.Series(pi.importances_mean, index=pa).sort_values(ascending=False).head(3)
    filas.append({"Modelo": nombre, "n entrenamiento": len(a_tr), "n prueba": len(a_te),
                  "AUC": roc_auc_score(a_te.a_gana, p),
                  "Accuracy": accuracy_score(a_te.a_gana, p >= 0.5),
                  "F1": f1_score(a_te.a_gana, p >= 0.5),
                  "3 predictores más importantes": ", ".join(etq_pa[v] for v in top3.index)})
tabla_85 = pd.DataFrame(filas)
t85 = tabla_85.set_index("Modelo")
registrar("t9_5_n_train", len(a_tr))
registrar("t9_5_n_test", len(a_te))
registrar("t9_5_auc_logistica", t85.AUC["Regresión logística"])
registrar("t9_5_auc_rf", t85.AUC["Random Forest"])
pegar("t85_auc", fmt(t85.AUC["Regresión logística"], 3))
pegar("t85_nte", fmt(len(a_te), 0))
pegar("t85_acc", fmt_pct(100 * t85.Accuracy["Regresión logística"]))
pegar("t85_top", t85["3 predictores más importantes"]["Regresión logística"])
```

**Tabla 8.5.** Modelo secundario: AVP con estadísticas de juego (explicativo).

```{code-cell} ipython3
:tags: [remove-input]
(GT(tabla_85)
 .fmt_number(columns=["n entrenamiento", "n prueba"], decimals=0, sep_mark=".", dec_mark=",")
 .fmt_number(columns=["AUC", "Accuracy", "F1"], decimals=3, sep_mark=".", dec_mark=",")
 .opt_stylize(style=1, color="gray"))
```

Con las diferencias de kills, aces, bloqueos, defensas, errores y % de ataque entre las dos parejas, la regresión logística alcanza una AUC de {glue}`t85_auc` en los {glue}`t85_nte` partidos de prueba y acierta el {glue}`t85_acc` (Tabla 8.5). Que sea tan alta no significa que se pueda anticipar el resultado: estas cifras se producen durante el partido y casi lo describen. Su utilidad es otra: mostrar qué acciones pesan más en la victoria. Las más importantes son {glue}`t85_top`, lo que coincide con el análisis bivariado (sección 6.2.6): en la AVP, la eficacia en ataque es lo que más separa a quien gana.

```{admonition} Para el entrenador
:class: tip
Antes del partido, la diferencia de ranking es casi todo lo que se puede saber: ningún modelo con edad, estatura o contexto la mejora de forma apreciable. Como regla práctica, cada posición de ventaja aumenta las *odds* de ganar alrededor de un {glue}`t84_pct_rank`. La probabilidad del modelo sirve más que su predicción binaria: cerca de 0,5, el partido está abierto. Lo que el ranking no anticipa se decide en la cancha, y ahí, al menos en la AVP, pesa sobre todo la eficacia en ataque: convertir los ataques y no regalar errores. El dashboard puede usar el modelo como predicción base y las estadísticas de juego como guía de entrenamiento, sin mezclar las dos cosas.
```
