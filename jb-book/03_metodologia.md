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

# Metodología

Este capítulo describe las técnicas usadas en el libro y por qué se eligieron. Todos los procesos aleatorios usan la misma semilla (2026), de modo que cualquier lector obtiene exactamente las mismas cifras al ejecutar el código.

## Marco teórico

### Enfoque del estudio

El estudio combina dos enfoques complementarios. Primero, un análisis descriptivo y comparativo: describir los datos, compararlos y buscar patrones antes de plantear cualquier explicación, con técnicas que un entrenador pueda interpretar directamente. Segundo, un enfoque predictivo: evaluar si la información conocida antes del partido permite anticipar al ganador mejor que el ranking (capítulo 8).

La parte descriptiva avanza en tres niveles:

- Univariado: cada variable por separado, con estadísticos descriptivos (media, mediana, desviación estándar, percentiles y asimetría) y gráficos de distribución.
- Bivariado: cada variable frente al resultado del partido (ganador o perdedor), con pruebas no paramétricas y tamaños del efecto.
- Multivariado: varias variables a la vez, para ver cuáles se mueven juntas y si aportan información distinta.

### Unidades de análisis

Un partido puede mirarse desde tres niveles, y cada pregunta usa el que le corresponde:

- **Partido:** una fila por partido. Sirve para la duración, los sets, los puntos y la diferencia de ranking entre las dos parejas.
- **Equipo-partido:** dos filas por partido, una para la pareja ganadora y otra para la perdedora. Es la base de las comparaciones entre ganadores y perdedores.
- **Jugador-partido:** cuatro filas por partido, una por jugador. Sirve para la edad, la estatura y el país.

### Favorito y sorpresa

Cuando las dos parejas tienen ranking numérico, el **favorito** es la pareja con mejor ranking (el número más bajo). Una **sorpresa** (*upset*) es un partido que gana la pareja peor rankeada. La diferencia de ranking entre las dos parejas mide qué tan clara era la ventaja del favorito antes del partido.

### Clasificación supervisada

El capítulo de modelado plantea un problema de **clasificación binaria supervisada**: a partir de información conocida antes del partido, predecir cuál de las dos parejas gana. Formalmente, se busca una función $f: X \rightarrow \{0, 1\}$ que, para cada partido, asigne una probabilidad de que la pareja A gane, $P(Y = 1 \mid X)$. El modelo aprende $f$ con partidos de 2000–2016 y se evalúa con partidos de 2017–2019 que nunca vio, para imitar el uso real: predecir partidos futuros.

Los predictores son diferencias entre las dos parejas (ranking, edad media, estatura media y si vienen de clasificatoria) y el contexto del partido (circuito, género y etapa del torneo). Las estadísticas de juego (kills, aces, bloqueos) no se usan para predecir, porque solo se conocen cuando el partido ya terminó. Como la pareja A se asigna con una regla fija y ajena al resultado, la clase está balanceada: A gana aproximadamente la mitad de las veces.

Se comparan tres modelos de complejidad creciente con una **regla de referencia** sin aprendizaje (Tabla 3.1). Si ningún modelo supera esa regla, ningún modelo aporta información nueva.

**Tabla 3.1.** Modelos evaluados en el capítulo de modelado.

| Modelo | Cómo decide | Ventaja | Limitación |
|---|---|---|---|
| Regla "gana el mejor rankeado" | Predice que gana la pareja con mejor ranking | Es lo que hace un entrenador sin modelo; fija el piso a superar | Ignora edad, estatura, etapa y contexto |
| Regresión logística | Combina linealmente los predictores y los transforma en probabilidad con la función logística | Interpretable: cada coeficiente se lee como un *odds ratio* | Solo capta relaciones lineales en el logit, sin interacciones |
| Random Forest {cite:p}`breiman2001` | Promedia cientos de árboles de decisión entrenados sobre muestras distintas de los datos | Capta interacciones y relaciones no lineales; robusto ante atípicos | Menos interpretable; requiere ajustar hiperparámetros |
| XGBoost {cite:p}`chen2016` | Construye árboles en secuencia, cada uno corrigiendo los errores del anterior (*gradient boosting*) | Suele ser el más preciso en datos tabulares | El más complejo; riesgo de sobreajuste si no se regula |

La regresión logística modela el logaritmo de las *odds* de victoria como una combinación lineal de los predictores:

$$
\log \frac{P(Y=1)}{1 - P(Y=1)} = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k
$$

donde $e^{\beta_j}$ es el *odds ratio*: cuánto se multiplican las *odds* de ganar por cada unidad adicional de $x_j$.

### Métricas de evaluación

- **Matriz de confusión:** tabla que cruza el resultado real con el predicho y muestra los aciertos y los dos tipos de error (predecir que gana A cuando gana B, y al revés).
- **AUC (área bajo la curva ROC):** probabilidad de que el modelo asigne mayor probabilidad de victoria a una pareja ganadora que a una perdedora tomada al azar. Vale 0,5 si el modelo no discrimina y 1 si discrimina perfectamente. Es la métrica principal porque no depende de un umbral.
- **Accuracy:** proporción de partidos con el ganador bien predicho, usando un umbral de 0,5. Es interpretable porque la clase está balanceada.
- **F1:** media armónica de la precisión y la sensibilidad para la clase "gana A".

Los hiperparámetros se eligen con **validación cruzada estratificada de 5 pliegues** dentro del periodo de entrenamiento: el conjunto se divide en 5 partes con la misma proporción de victorias, se entrena con 4 y se evalúa con la restante, rotando. Así la elección no usa los datos de prueba.

## Tratamiento estadístico y pruebas no paramétricas

Las variables del análisis no siguen una distribución normal: el ranking es discreto y acotado, la duración es asimétrica y las estadísticas de juego son conteos. Por eso se usan **pruebas no paramétricas**, que comparan órdenes (rangos) en lugar de medias y no suponen normalidad.

En todas las pruebas se usa un nivel de significancia **α = 0,05**. La hipótesis nula (H₀) es que no hay diferencia entre ganadores y perdedores o, en el caso del chi-cuadrado, que las dos variables son independientes. Si el p-valor es menor que α, se rechaza H₀. Después se revisa el tamaño del efecto para saber si la diferencia encontrada es grande o pequeña.

**Mann-Whitney U** {cite:p}`mann1947` compara una variable numérica entre ganadores y perdedores. Contrasta si, al tomar una pareja ganadora y una perdedora al azar, una tiende a tener valores mayores que la otra. Su **tamaño del efecto** es la **correlación rango-biserial** (RBC) {cite:p}`kerby2014`:

$$
RBC = \frac{2U}{n_1 n_2} - 1
$$

donde $U$ es el estadístico del grupo ganador y $n_1$, $n_2$ son los tamaños de cada grupo. Va de −1 a 1: un valor positivo indica que los ganadores tienden a tener valores más altos. El **poder discriminante** se clasifica con los umbrales de Cohen {cite:p}`cohen1988`: alto si $|RBC| \geq 0{,}5$, moderado si $0{,}3 \leq |RBC| < 0{,}5$ y bajo si $|RBC| < 0{,}3$.

**Chi-cuadrado de independencia** evalúa si dos variables categóricas están asociadas, por ejemplo, si la tasa de sorpresas depende de la etapa del torneo. Su tamaño del efecto es la **V de Cramér** {cite:p}`cramer1946`:

$$
V = \sqrt{\frac{\chi^2}{n \, (\min(r, c) - 1)}}
$$

donde $r$ y $c$ son el número de filas y columnas de la tabla. Va de 0 (sin asociación) a 1 (asociación perfecta). Se interpreta como despreciable si $V < 0{,}1$, pequeña entre 0,1 y 0,3, media entre 0,3 y 0,5 y grande si $V \geq 0{,}5$. No se aplica la corrección de Yates.

**Por qué se interpreta el tamaño del efecto y no el p-valor.** Con más de 150.000 filas, casi cualquier diferencia, por mínima que sea, resulta estadísticamente significativa ($p < 0{,}001$). El p-valor dice si una diferencia existe; el tamaño del efecto dice **si importa**. Para un entrenador, solo lo segundo es útil.

**Filas no independientes.** Cada partido aporta dos filas, una del ganador y otra del perdedor, que no son independientes entre sí. Por eso Mann-Whitney se complementa con la **prueba de rangos con signo de Wilcoxon** {cite:p}`wilcoxon1945` sobre la ventaja de cada pareja frente a su rival, calculada a nivel de partido.

## Análisis de solapamiento

Además del tamaño del efecto, se mide cuánto se superponen las distribuciones de ganadores y perdedores. El **solapamiento IQR** compara el 50 % central de cada grupo, es decir, el rango entre sus percentiles 25 y 75:

$$
\text{Solapamiento IQR} = \frac{\text{longitud de la intersección de } [P_{25}, P_{75}]_G \text{ y } [P_{25}, P_{75}]_P}{\text{longitud de su unión}}
$$

Vale 0 si los rangos centrales no se tocan y 1 si son idénticos. Un solapamiento alto significa que, aunque la diferencia sea significativa, una pareja típica ganadora y una perdedora se parecen mucho en esa variable; por lo tanto, la variable sirve poco para distinguirlas en la práctica.

## Análisis multivariado y multicolinealidad

Para ver qué variables se mueven juntas se usa la **correlación de Spearman** ($\rho$) {cite:p}`spearman1904`, que mide la asociación monótona entre dos variables usando sus rangos. Va de −1 a 1 y no exige relación lineal ni normalidad.

Antes del modelado se verifica la **multicolinealidad**, es decir, que ningún predictor sea casi una combinación de los demás, lo que volvería inestables los coeficientes de la regresión logística. Se usa el **factor de inflación de la varianza** (VIF):

$$
VIF_j = \frac{1}{1 - R^2_j}
$$

donde $R^2_j$ es el coeficiente de determinación de una regresión del predictor $j$ sobre todos los demás. Un VIF menor o igual a 5 se considera aceptable, entre 5 y 10 requiere revisión y mayor que 10 indica multicolinealidad grave.

## Análisis de tendencias en el tiempo

Para el análisis temporal (capítulo 7) no se ajusta un modelo de series de tiempo: hay como máximo 20 puntos anuales, varios con cobertura incompleta. En su lugar, cada serie se muestra con una media móvil de 3 años, la tendencia se evalúa con la **prueba de Mann-Kendall** {cite:p}`mann1945`, equivalente a la $\tau$ de Kendall {cite:p}`kendall1938` entre el año y el valor de la serie, y su magnitud con la pendiente de una recta ajustada por mínimos cuadrados.

## Fases del procesamiento

El trabajo sigue cinco fases. Cada una produce un resultado que usa la siguiente (Tabla 3.2).

**Tabla 3.2.** Fases del procesamiento.

| Fase | Qué se hace | Resultado |
|---|---|---|
| 1. Datos crudos | Se lee el archivo original, que nunca se modifica | `vb_matches.csv`, 76.756 partidos |
| 2. Limpieza | Un único proceso aplica doce pasos de corrección (fechas, rankings, marcador, etapas, llaves y estadísticas) y registra cifras de control | Datos verificables (capítulo 5) |
| 3. Archivos limpios | Se generan tres tablas con distinta unidad de análisis | Partido · equipo-partido · jugador-partido |
| 4. Análisis | Descripción de los datos, análisis exploratorio, análisis temporal y modelado | Capítulos 4, 6, 7 y 8 |
| 5. Conclusiones | Los hallazgos se traducen en recomendaciones para el entrenador | Capítulo 9 y dashboard |

Todos los capítulos de análisis parten de los archivos limpios.

```{admonition} Limitaciones del análisis
:class: note
- **Faltante no aleatorio:** las estadísticas de juego dependen del circuito y describen casi solo a la AVP; no se imputan ni se usan para comparar circuitos.
- **Cobertura desigual en el tiempo:** 2000 es parcial, la AVP casi no tiene datos en 2011–2012 y 2019 llega hasta agosto, así que los conteos por año miden también la cobertura del conjunto de datos.
- **Ranking incompleto:** muchas parejas que vienen de clasificatoria solo tienen posición "Q", sin ranking numérico; los análisis de ranking y el modelo se inclinan hacia el cuadro principal.
- **Supuestos sin confirmar con la fuente:** el significado del ranking compuesto ("17, Q2") y la corrección de 100 años en fechas de nacimiento.
- **Asociación no es causalidad:** los resultados muestran qué factores acompañan a la victoria, no qué la provoca.
- **Fuente secundaria:** los datos fueron recopilados por terceros a partir de los sitios de los circuitos; no son un registro oficial.
```
