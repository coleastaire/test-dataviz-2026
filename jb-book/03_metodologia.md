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

Este capítulo describe las técnicas usadas en el libro y por qué se eligieron. Todas se aplican igual en la versión en Python y en la versión en R, con las mismas reglas y la misma semilla (2026), de modo que ambos libros reportan las mismas cifras.

## Marco teórico

### Análisis exploratorio de datos

El libro sigue un enfoque de **análisis exploratorio de datos** (EDA): describir los datos, compararlos y buscar patrones antes de plantear cualquier explicación. No se ajustan modelos predictivos. El objetivo es medir qué factores conocidos antes del partido acompañan a la victoria y con qué fuerza, con técnicas que un entrenador pueda interpretar directamente.

El análisis avanza en tres niveles:

- **Univariado:** cada variable por separado, con estadísticos descriptivos (media, mediana, desviación estándar, percentiles y asimetría) y gráficos de distribución.
- **Bivariado:** cada variable frente al resultado del partido (ganador o perdedor), con pruebas no paramétricas y tamaños del efecto.
- **Multivariado:** varias variables a la vez, para ver cuáles se mueven juntas y si aportan información distinta.

### Unidades de análisis

Un partido puede mirarse desde tres niveles, y cada pregunta usa el que le corresponde:

- **Partido:** una fila por partido. Sirve para la duración, los sets, los puntos y la diferencia de ranking entre las dos parejas.
- **Equipo-partido:** dos filas por partido, una para la pareja ganadora y otra para la perdedora. Es la base de las comparaciones entre ganadores y perdedores.
- **Jugador-partido:** cuatro filas por partido, una por jugador. Sirve para la edad, la estatura y el país.

### Favorito y sorpresa

Cuando las dos parejas tienen ranking numérico, el **favorito** es la pareja con mejor ranking (el número más bajo). Una **sorpresa** (*upset*) es un partido que gana la pareja peor rankeada. La diferencia de ranking entre las dos parejas mide qué tan clara era la ventaja del favorito antes del partido.

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

donde $r$ y $c$ son el número de filas y columnas de la tabla. Va de 0 (sin asociación) a 1 (asociación perfecta). Se interpreta como despreciable si $V < 0{,}1$, pequeña entre 0,1 y 0,3, media entre 0,3 y 0,5 y grande si $V \geq 0{,}5$. No se aplica la corrección de Yates, para que la V sea idéntica en R y Python.

**Por qué se interpreta el tamaño del efecto y no el p-valor.** Con más de 150.000 filas, casi cualquier diferencia, por mínima que sea, resulta estadísticamente significativa ($p < 0{,}001$). El p-valor dice si una diferencia existe; el tamaño del efecto dice **si importa**. Para un entrenador, solo lo segundo es útil.

**Filas no independientes.** Cada partido aporta dos filas, una del ganador y otra del perdedor, que no son independientes entre sí. Por eso Mann-Whitney se complementa con la **prueba de rangos con signo de Wilcoxon** {cite:p}`wilcoxon1945` sobre la ventaja de cada pareja frente a su rival, calculada a nivel de partido.

## Análisis de solapamiento

Además del tamaño del efecto, se mide cuánto se superponen las distribuciones de ganadores y perdedores. El **solapamiento IQR** compara el 50 % central de cada grupo, es decir, el rango entre sus percentiles 25 y 75:

$$
\text{Solapamiento IQR} = \frac{\text{longitud de la intersección de } [P_{25}, P_{75}]_G \text{ y } [P_{25}, P_{75}]_P}{\text{longitud de su unión}}
$$

Vale 0 si los rangos centrales no se tocan y 1 si son idénticos. Un solapamiento alto significa que, aunque la diferencia sea significativa, una pareja típica ganadora y una perdedora se parecen mucho en esa variable; por lo tanto, la variable sirve poco para distinguirlas en la práctica.

## Análisis multivariado y redundancia

Para ver qué variables se mueven juntas se usa la **correlación de Spearman** ($\rho$) {cite:p}`spearman1904`, que mide la asociación monótona entre dos variables usando sus rangos. Va de −1 a 1 y no exige relación lineal ni normalidad.

Además se verifica que las variables conocidas antes del partido no sean **redundantes**, es decir, que ninguna sea casi una combinación de las demás. Si lo fueran, mirarlas juntas contaría dos veces la misma información. Se usa el **factor de inflación de la varianza** (VIF):

$$
VIF_j = \frac{1}{1 - R^2_j}
$$

donde $R^2_j$ es el coeficiente de determinación de una regresión de la variable $j$ sobre todas las demás. Un VIF menor o igual a 5 se considera aceptable, entre 5 y 10 requiere revisión y mayor que 10 indica una redundancia grave.

## Fases del procesamiento

El trabajo sigue cinco fases. Cada una produce un resultado que usa la siguiente (Tabla 3.1).

**Tabla 3.1.** Fases del procesamiento.

| Fase | Qué se hace | Resultado |
|---|---|---|
| 1. Datos crudos | Se lee el archivo original, que nunca se modifica | `vb_matches.csv`, 76.756 partidos |
| 2. Limpieza | Un único proceso aplica doce pasos de corrección (fechas, rankings, marcador, etapas, llaves y estadísticas) y registra cifras de control | Cifras de control comparables con R (capítulo 5) |
| 3. Archivos limpios | Se generan tres tablas con distinta unidad de análisis | Partido · equipo-partido · jugador-partido |
| 4. Análisis | Descripción de los datos y análisis exploratorio univariado, bivariado y multivariado | Capítulos 4 y 6 |
| 5. Conclusiones | Los hallazgos se traducen en recomendaciones para el entrenador | Capítulo 7 y dashboard |

La versión en R repite los mismos pasos de limpieza y debe obtener las mismas cifras de control. Todos los capítulos de análisis parten de los archivos limpios.

## Herramientas

**Tabla 3.2.** Herramientas y versiones utilizadas (libro en Python).

```{code-cell} ipython3
:tags: [remove-input]
import sys
import importlib.metadata as md
import pandas as pd
from great_tables import GT

def version(paquete):
    try:
        return md.version(paquete)
    except md.PackageNotFoundError:
        return "no instalado"

herramientas = pd.DataFrame(
    [("Lenguaje", "Python", sys.version.split()[0]),
     ("Lectura y limpieza", "pandas · numpy",
      f"{version('pandas')} · {version('numpy')}"),
     ("Pruebas estadísticas", "scipy · statsmodels",
      f"{version('scipy')} · {version('statsmodels')}"),
     ("Redundancia (VIF)", "scikit-learn", version("scikit-learn")),
     ("Gráficos", "plotly", version("plotly")),
     ("Tablas", "great_tables", version("great_tables")),
     ("Publicación", "Jupyter Book", version("jupyter-book"))],
    columns=["Tarea", "Librería", "Versión"])

GT(herramientas).opt_stylize(style=1, color="gray")
```

El libro se escribió en Python {cite:p}`python` con pandas {cite:p}`pandas`, NumPy {cite:p}`numpy`, SciPy {cite:p}`scipy`, statsmodels {cite:p}`statsmodels`, scikit-learn {cite:p}`sklearn`, Plotly {cite:p}`plotly` y great_tables {cite:p}`greattables`, y se publicó con Jupyter Book {cite:p}`jupyterbook`.

```{admonition} Limitaciones del análisis
:class: note
- **Faltante no aleatorio:** las estadísticas de juego dependen del circuito y describen casi solo a la AVP; no se imputan ni se usan para comparar circuitos.
- **Cobertura desigual en el tiempo:** 2000 es parcial, la AVP casi no tiene datos en 2011–2012 y 2019 llega hasta agosto, así que los conteos por año miden también la cobertura del conjunto de datos.
- **Ranking incompleto:** muchas parejas que vienen de clasificatoria solo tienen posición "Q", sin ranking numérico; los análisis de ranking se inclinan hacia el cuadro principal.
- **Supuestos sin confirmar con la fuente:** el significado del ranking compuesto ("17, Q2") y la corrección de 100 años en fechas de nacimiento.
- **Asociación no es causalidad:** los resultados muestran qué factores acompañan a la victoria, no qué la provoca.
- **Fuente secundaria:** los datos fueron recopilados por terceros a partir de los sitios de los circuitos; no son un registro oficial.
```