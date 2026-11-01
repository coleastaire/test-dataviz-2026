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

# Conclusiones

```{code-cell} ipython3
:tags: [remove-cell]
from py.utils import *

# Cifras ya calculadas en los capítulos 4 a 8 (compartido/cifras_control_py.csv)
c = pd.read_csv(CIFRAS_PATH).set_index("id").valor

# Pocas cifras que se calculan aquí porque no quedaron registradas
m = pd.read_csv(CLEAN + "vb_matches_limpio.csv", low_memory=False)
jg = pd.read_csv(CLEAN + "vb_jugadores.csv", low_memory=False)
lg = pd.read_csv(CLEAN + "vb_long.csv", usecols=["resultado", "es_clasif"])
n_total = len(m)
avp_anio = m[m.circuit == "AVP"].groupby("year").size()
perd_clasif = 100 * lg.groupby("es_clasif").resultado.apply(lambda s: (s == "Perdedor").mean())
mf = m[m.es_forfeit == 0]

def pct_atipicos(s):
    s = s.dropna()
    q1, q3 = s.quantile([.25, .75])
    return 100 * ((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).mean()
max_atip = max(pct_atipicos(jg[jg.gender == g].estatura_cm) for g in ["M", "W"])
pct_fivb = 100 * (c["f4_1_n_FIVB_M"] + c["f4_1_n_FIVB_W"]) / n_total

pegar("c7_pct_fivb", fmt_pct(pct_fivb, 2))
pegar("c7_pct_avp", fmt_pct(100 - pct_fivb, 2))
pegar("c7_n_stats", fmt(c["f5_2_n_stats"], 0))
pegar("c7_pct_stats", fmt_pct(100 * c["f5_2_n_stats"] / n_total))
pegar("c7_pct_stats_avp", fmt_pct(c["f5_2_pct_stats_avp"]))
pegar("c7_max_atip", fmt_pct(max_atip))
pegar("c7_celdas", fmt(c["p8_celdas_invalidas"], 0))
pegar("c7_avp_2010", fmt(avp_anio[2010], 0))
pegar("c7_avp_2011", fmt(avp_anio[2011], 0))
pegar("c7_avp_2012", fmt(avp_anio[2012], 0))
pegar("c7_pct_upset", fmt_pct(c["f6_8_pct_upset"], 2))
pegar("c7_n_rank", fmt(c["f6_8_n"], 0))
pegar("c7_pct_upset_puro", fmt_pct(c["f6_8_pct_upset_puro"]))
pegar("c7_rbc_rank", fmt(c["t6_2_rbc_rank_main"]))
pegar("c7_w_rank", fmt_pct(c["t6_2_wilcoxon_pct_ventaja_rank"]))
pegar("c7_fav_cerca", fmt_pct(c["f6_9_pct_1–2"]))
pegar("c7_fav_lejos", fmt_pct(c["f6_9_pct_21 o más"]))
pegar("c7_rbc_fis", fmt(max(abs(c["t6_2_rbc_edad_media"]), abs(c["t6_2_rbc_estatura_media_cm_h"]),
                            abs(c["t6_2_rbc_estatura_media_cm_m"]))))
pegar("c7_est10", fmt_pct(c["f6_11_pct_est_≥ 10"]))
pegar("c7_perd_clasif", fmt_pct(perd_clasif[1]))
pegar("c7_perd_directo", fmt_pct(perd_clasif[0]))
pegar("c7_v_clasif", fmt(c["t6_3_v_viene"], 3))
pegar("c7_rbc_hit", fmt(c["t6_2_rbc_hitpct"]))
pegar("c7_rbc_err", fmt(c["t6_2_rbc_errors"]))
pegar("c7_rbc_blk", fmt(c["t6_2_rbc_blocks"]))
pegar("c7_rbc_kil", fmt(c["t6_2_rbc_kills"]))
pegar("c7_rbc_ace", fmt(c["t6_2_rbc_aces"]))
pegar("c7_rho_max", fmt(c[c.index.str.startswith("f6_13_rho")].abs().max()))
pegar("c7_vif_max", fmt(c[c.index.str.startswith("t6_4_vif")].max()))
pegar("c7_dur_med", fmt(mf.duracion_min.median(), 0))
pegar("c7_dur_3", fmt(mf[mf.n_sets == 3].duracion_min.median(), 0))
pegar("c7_sin_estatura", fmt_pct(c["f6_2_pct_partidos_sin_estatura"]))
pegar("c7_fivb_2018", fmt(c["f7_1_n_FIVB_2018"], 0))
pegar("c7_ups_antes", fmt_pct(c["f7_5_ups_2001_2010"]))
pegar("c7_ups_despues", fmt_pct(c["f7_5_ups_2017_2019"]))
pegar("c7_tau_ups", fmt(c["t7_1_tau_tasa_de_sorpresas_total"]))
pegar("c7_dur_2008", fmt(c["f7_3_dur_FIVB_2008"], 1))
pegar("c7_dur_rec", fmt(np.mean([c[f"f7_3_dur_FIVB_{a}"] for a in range(2016, 2020)]), 0))
modelos_auc = {"Regresión logística": c["t9_3_auc_logistica"], "Random Forest": c["t9_3_auc_rf"],
               "XGBoost": c["t9_3_auc_xgb"]}
mejor_mod = max(modelos_auc, key=modelos_auc.get)
imp_rank = c[c.index.str.startswith("f9_2_imp_") & c.index.str.endswith("dif_rank")]
pegar("c7_mejor_mod", mejor_mod)
pegar("c7_mejor_auc", fmt(modelos_auc[mejor_mod], 3))
pegar("c7_regla_auc", fmt(c["t9_3_auc_regla"], 3))
pegar("c7_regla_acc", fmt_pct(100 * c["t9_3_acc_regla"]))
pegar("c7_imp_min", fmt(imp_rank.min(), 2))
pegar("c7_imp_max", fmt(imp_rank.max(), 2))
pegar("c7_pct_modelable", fmt_pct(100 * c["m9_n_modelable"] / n_total))
pegar("c7_auc_avp", fmt(c["t9_5_auc_logistica"], 3))
```

Este capítulo reúne lo que el libro encontró y lo traduce en decisiones para un entrenador. Cada hallazgo lleva su cifra y la figura o tabla que lo respalda.

## Hallazgos

**Sobre los datos**

1. **El circuito define la estructura del conjunto de datos.** La FIVB tiene el {glue}`c7_pct_fivb` de los partidos y la AVP el {glue}`c7_pct_avp`, pero casi solo la AVP registra estadísticas de juego (Figuras 4.1 y 5.2).
2. **El faltante de estadísticas no es aleatorio.** Solo {glue}`c7_n_stats` partidos ({glue}`c7_pct_stats`) tienen estadísticas completas, y el {glue}`c7_pct_stats_avp` son de la AVP. Todo resultado con kills, aces o % de ataque describe a la AVP de 2003 a 2019 (Figura 5.2).
3. **Los atípicos son legítimos.** Ninguna variable supera el {glue}`c7_max_atip` de valores atípicos; solo se anularon {glue}`c7_celdas` celdas lógicamente imposibles (Tabla 4.3).
4. **La serie temporal refleja la cobertura del conjunto de datos.** La AVP pasa de {glue}`c7_avp_2010` partidos en 2010 a {glue}`c7_avp_2011` en 2011 y {glue}`c7_avp_2012` en 2012, durante su crisis; 2000 y 2019 son años parciales, y la FIVB llega a {glue}`c7_fivb_2018` partidos en 2018 con su nueva estructura de torneos. Los conteos por año describen la cobertura del conjunto de datos tanto como la actividad del deporte (Figuras 7.1 y 7.2).

**Sobre qué hace ganar a una pareja**

5. **Casi uno de cada tres partidos es una sorpresa.** Gana la pareja peor rankeada en el {glue}`c7_pct_upset` de los {glue}`c7_n_rank` partidos con ambos rankings; con rankings simples, en el {glue}`c7_pct_upset_puro` (Figura 6.9).
6. **Lo que informa es la distancia con el rival, no el ranking propio.** Comparando a todos los ganadores con todos los perdedores, el ranking tiene poder bajo (RBC = {glue}`c7_rbc_rank`). Comparando cada pareja con su rival, el mejor rankeado gana el {glue}`c7_w_rank` de los partidos: el {glue}`c7_fav_cerca` con 1 o 2 posiciones de diferencia y el {glue}`c7_fav_lejos` con más de 20 (Tabla 6.2 y Figura 6.10).
7. **Edad y estatura pesan poco.** Tienen poder bajo (|RBC| ≤ {glue}`c7_rbc_fis`). Una pareja 10 cm más alta que su rival gana el {glue}`c7_est10` de las veces (Figuras 6.8 y 6.12).
8. **Venir de clasificatoria pesa poco.** Esas parejas pierden el {glue}`c7_perd_clasif` de sus partidos frente al {glue}`c7_perd_directo` de las que entran directo, pero la asociación es despreciable (V = {glue}`c7_v_clasif`; Tabla 6.3).
9. **En la AVP, gana la eficacia en ataque.** El % de ataque es la única variable con poder alto (RBC = {glue}`c7_rbc_hit`), seguido de los errores de ataque ({glue}`c7_rbc_err`), los bloqueos ({glue}`c7_rbc_blk`), los kills ({glue}`c7_rbc_kil`) y los aces ({glue}`c7_rbc_ace`) (Figura 6.13).
10. **Las variables no repiten información.** Todas las correlaciones de Spearman son ≤ {glue}`c7_rho_max` en valor absoluto y todos los VIF ≤ {glue}`c7_vif_max` (Figura 6.14 y Tabla 6.4).

**Sobre el tiempo y la predicción**

11. **El circuito es cada vez más parejo.** Las sorpresas pasan del {glue}`c7_ups_antes` en 2001–2010 al {glue}`c7_ups_despues` en 2017–2019, con tendencia significativa (tau = {glue}`c7_tau_ups`). En la FIVB, la duración media baja de {glue}`c7_dur_2008` minutos en 2008 a unos {glue}`c7_dur_rec` en 2016–2019 (Figura 7.5 y Tabla 7.1).
12. **Ningún modelo supera de forma apreciable a la regla del ranking.** El mejor, {glue}`c7_mejor_mod`, alcanza una AUC de {glue}`c7_mejor_auc` frente a {glue}`c7_regla_auc` de la regla "gana el mejor rankeado", que acierta el {glue}`c7_regla_acc` de los partidos de prueba. La diferencia de ranking es la variable más importante en los tres modelos: desordenarla baja la AUC entre {glue}`c7_imp_min` y {glue}`c7_imp_max` (Tabla 8.3 y Figura 8.5).
13. **Las estadísticas de juego explican, pero no predicen.** En la AVP, un modelo con las diferencias de kills, errores y % de ataque alcanza una AUC de {glue}`c7_auc_avp`, pero esas cifras se conocen al terminar el partido; sirven para orientar el entrenamiento, no para anticipar el resultado (Tabla 8.5).

## Recomendaciones para el entrenador

La respuesta a la pregunta de este libro (sección 1.2) es: **antes del partido, el factor medible que más se asocia con ganar es la ventaja de ranking sobre el rival; la estatura, la edad y el contexto aportan muy poco en comparación, y ningún modelo los convierte en una mejor predicción. Durante el partido, al menos en la AVP, lo que separa al ganador es la eficacia en ataque.**

**Qué puede usar**

- **Para preparar un partido:** la probabilidad de ganar según la diferencia de ranking (Figura 6.10), usando como referencia los años recientes, porque hoy hay más sorpresas que antes (Figura 7.5). Con 1 o 2 posiciones de diferencia, el partido está abierto; solo con más de 20 el favorito es claro.
- **Para planificar la exigencia física:** un partido típico dura unos {glue}`c7_dur_med` minutos, pero uno de cada tres se va a un tercer set y se acerca a los {glue}`c7_dur_3`. Los partidos se alargan cuando el rival que pierde es bueno (Figura 6.15).
- **Para armar parejas:** la estatura y la edad pueden inclinar la balanza, pero poco; no deberían pesar más que el nivel competitivo reflejado en el ranking. Las diferencias de edad o estatura entre compañeros no se asocian con el resultado.
- **Para priorizar el entrenamiento (referencia AVP):** la eficacia en ataque (convertir los ataques y reducir los errores) separa más al ganador que atacar más veces; le siguen el bloqueo y el saque.

**Qué no se puede concluir**

- Que las estadísticas de juego describan a la FIVB: solo describen a la AVP de 2003 a 2019.
- Que un factor **cause** la victoria: todos los resultados son asociaciones.
- Que un modelo complejo anticipe mejor que el ranking: con estos datos, no lo hace.

**Qué datos harían falta**

- Estadísticas de juego de la FIVB, para comparar circuitos y extender las recomendaciones de entrenamiento.
- Ranking numérico de las parejas de clasificatoria, que hoy quedan fuera de casi todo análisis de ranking.
- Información de contexto que hoy no existe en el conjunto de datos: viento, lesiones, tiempo de la pareja junta o resultados previos entre los mismos rivales.

**Base para el dashboard.** Los hallazgos sugieren tres módulos: (1) una calculadora de probabilidad de victoria según la diferencia de ranking, filtrable por circuito, género y etapa; (2) un perfil de parejas con su ranking, edad y estatura frente al rival; y (3) un panel de estadísticas de juego de la AVP, claramente rotulado como tal, para orientar el entrenamiento.

## Limitaciones

1. **Faltante no aleatorio.** Las estadísticas de juego dependen del circuito; describen a la AVP, no se usan para comparar circuitos y no se imputan.
2. **Cobertura temporal desigual.** 2000 es parcial, la AVP casi no tiene partidos en 2011–2012 y 2019 llega hasta el 29 de agosto. Los conteos por año miden cobertura, no solo actividad.
3. **Ranking incompleto.** Falta el ranking numérico en una parte importante de los perdedores, sobre todo los que vienen de clasificatoria; los análisis de ranking y el modelo se inclinan hacia el cuadro principal (el modelo usa el {glue}`c7_pct_modelable` de los partidos).
4. **Supuestos sin confirmar con la fuente.** El significado del ranking compuesto ("17, Q2") y la corrección de 100 años en las fechas de nacimiento.
5. **Filas no independientes.** El ganador y el perdedor de un mismo partido forman un par; por eso Mann-Whitney se complementó con la prueba de Wilcoxon pareada.
6. **Muestra muy grande.** Casi todos los p-valores son < 0,001; las conclusiones se basan en el tamaño del efecto.
7. **Estatura faltante.** Falta al menos una estatura en el {glue}`c7_sin_estatura` de los partidos, más en perdedores y en la AVP, lo que puede inflar levemente la ventaja de estatura de los ganadores.
8. **Cambio en el tiempo.** El ranking anticipa peor en 2017–2019 que antes; las probabilidades calculadas con todo el periodo pueden sobrestimar la ventaja del favorito hoy.
9. **Modelo secundario explicativo.** El modelo de la AVP usa estadísticas que se conocen después del partido; no sirve para predecir.
10. **Fuente secundaria.** Los datos fueron recopilados por terceros a partir de los sitios de los circuitos; no son un registro oficial.