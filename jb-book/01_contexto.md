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

# Contexto

```{code-cell} ipython3
:tags: [remove-cell]
from py.utils import *

m = pd.read_csv(CLEAN + "vb_matches_limpio.csv", low_memory=False)
jg = pd.read_csv(CLEAN + "vb_jugadores.csv")

avp_anio = m[m.circuit == "AVP"].groupby("year").size()

# Probabilidad de que gane el mejor rankeado
r = m[m.rank_diff.notna()]
grupos = pd.cut(r.rank_diff.abs(), [0, 2, 5, 10, 20, np.inf])
fav_grupo = r.groupby(grupos, observed=True).es_upset.apply(lambda s: 100 * (1 - s.mean()))

pegar("c1_avp_2010", fmt(avp_anio[2010], 0))
pegar("c1_avp_2011", fmt(avp_anio[2011], 0))
pegar("c1_avp_2012", fmt(avp_anio[2012], 0))
pegar("c1_n_partidos", fmt(len(m), 0))
pegar("c1_n_fivb", fmt((m.circuit == "FIVB").sum(), 0))
pegar("c1_n_avp", fmt((m.circuit == "AVP").sum(), 0))
pegar("c1_n_paises", fmt(m.country.nunique(), 0))
pegar("c1_n_jugadores", fmt(jg.player_id.nunique(), 0))
pegar("c1_n_rank", fmt(len(r), 0))
pegar("c1_pct_fav", fmt_pct(100 * (1 - r.es_upset.mean())))
pegar("c1_fav_cerca", fmt_pct(fav_grupo.iloc[0]))
pegar("c1_fav_lejos", fmt_pct(fav_grupo.iloc[-1]))
pegar("c1_pct_stats", fmt_pct(100 * m.tiene_stats_completas.mean()))
pegar("c1_n_miles", fmt(len(m) // 1000 * 1000, 0))
```

## El voleibol de playa y los circuitos FIVB y AVP

El voleibol de playa se juega en parejas, sin cambios, sobre una cancha de arena de 16 × 8 metros. Los partidos se disputan al mejor de tres sets: los dos primeros a 21 puntos y el tercero, si hace falta, a 15. Es deporte olímpico desde Atlanta 1996 {cite:p}`wiki_beach`.

La competencia profesional se organiza en dos grandes circuitos. La **FIVB** (Federación Internacional de Voleibol) organiza el circuito mundial desde 1997, con torneos en todos los continentes {cite:p}`wiki_beach`. La **AVP** (Association of Volleyball Professionals) es el circuito profesional de Estados Unidos, fundado en 1983. En 2001 adoptó el sistema de puntuación por rally y la cancha reducida de la FIVB, de modo que desde entonces ambos circuitos juegan con las mismas reglas {cite:p}`wiki_avp`.

La historia de la AVP no fue continua. En agosto de 2010 suspendió sus operaciones y canceló los torneos que le quedaban; tras su venta, solo hubo un torneo en 2011 y dos en 2012, y la primera temporada completa bajo la nueva administración llegó en 2013 {cite:p}`wiki_avp`. Esa interrupción se ve directamente en los datos: la AVP registra {glue}`c1_avp_2010` partidos en 2010, apenas {glue}`c1_avp_2011` en 2011 y {glue}`c1_avp_2012` en 2012.

Este libro analiza el conjunto de datos `vb_matches`, recopilado por BigTimeStats a partir de los sitios oficiales de ambos circuitos {cite:p}`bigtimestats` y difundido por la comunidad TidyTuesday {cite:p}`tidytuesday2020`. Tras la limpieza (capítulo 5) contiene {glue}`c1_n_partidos` partidos jugados entre 2000 y 2019: {glue}`c1_n_fivb` de la FIVB y {glue}`c1_n_avp` de la AVP, en {glue}`c1_n_paises` países sede y con {glue}`c1_n_jugadores` jugadores distintos. Para cada partido se conoce quién ganó, el ranking, la edad, la estatura y el país de los cuatro jugadores, el marcador, la duración y, en una parte de los partidos, estadísticas de juego como ataques, aces, bloqueos y defensas.

## Pregunta de investigación y usuario del análisis

El usuario de este libro, y del dashboard que se construirá a partir de él, es un **entrenador de voleibol de playa**. Antes de cada torneo, ese entrenador toma decisiones con poca información: cómo preparar a su pareja frente a un rival concreto, con qué compañero conviene que juegue cada jugador y en qué aspectos del juego enfocar el entrenamiento.

La pregunta que guía todo el análisis es:

> **¿Qué factores, medibles antes del partido, se asocian con que una pareja gane, y cómo puede usarlos un entrenador para preparar partidos y armar parejas?**

De ella se derivan tres preguntas concretas que el libro responde con datos:

1. **Preparar un partido:** dada la diferencia de ranking con el rival, ¿qué tan probable es ganar? ¿En qué etapas del torneo son más frecuentes las sorpresas?
2. **Armar parejas:** ¿la estatura, la edad o la diferencia entre compañeros se asocian con ganar más?
3. **Priorizar el entrenamiento:** ¿qué acciones de juego (aces, bloqueos, ataques, defensas) separan más al ganador del perdedor?

## Planteamiento del problema

El ranking es la referencia más usada para anticipar un partido, y los datos confirman que informa: en los {glue}`c1_n_rank` partidos en que ambas parejas tienen ranking, la mejor rankeada gana el {glue}`c1_pct_fav` de las veces. Pero eso también significa que casi **uno de cada tres partidos lo gana la pareja peor rankeada**. Además, el ranking anticipa muy distinto según la distancia entre rivales: con 1 o 2 posiciones de diferencia el favorito gana el {glue}`c1_fav_cerca` de las veces, poco más que lanzar una moneda, mientras que con más de 20 posiciones gana el {glue}`c1_fav_lejos`.

Un entrenador que solo mira el ranking no sabe cuándo esa ventaja es real y cuándo es frágil, ni qué otros factores pueden inclinar el partido. A esto se suman tres dificultades de los datos disponibles:

- **Información dispersa y con errores silenciosos.** Rankings guardados como texto ("17, Q2"), fechas de nacimiento en el futuro, nombres de países duplicados y una llave de partido que se repite. Nada de esto produce un error al abrir el archivo, pero distorsiona cualquier conclusión si no se corrige.
- **Estadísticas de juego incompletas.** Solo el {glue}`c1_pct_stats` de los partidos tiene estadísticas completas, y casi todos son de la AVP. Sin tratar ese faltante, se corre el riesgo de generalizar a todo el deporte lo que solo describe a un circuito.
- **Cobertura desigual en el tiempo.** Años parciales y la interrupción de la AVP en 2011–2012 hacen que los conteos por año reflejen la cobertura del conjunto de datos tanto como la actividad del deporte.

El problema, entonces, es doble: **identificar qué factores medibles antes del partido se asocian con ganar, más allá del ranking**, y hacerlo sobre datos que primero hay que limpiar y delimitar con cuidado para no sacar conclusiones falsas.

## Justificación

Para un entrenador, saber qué tan sólida es una ventaja y qué factores la acompañan cambia decisiones concretas: cómo plantear un partido contra un rival mejor rankeado, qué perfil de compañero buscar para un jugador o qué acciones priorizar en la semana previa a un torneo. Hoy esas decisiones se toman sobre todo por experiencia e intuición.

Este análisis aporta tres cosas:

1. **Evidencia sobre veinte años de competencia.** Más de {glue}`c1_n_miles` partidos permiten distinguir patrones reales de casos anecdóticos.
2. **Límites claros.** El libro dice qué se puede concluir y qué no: por ejemplo, que las estadísticas de juego solo describen a la AVP, o que una asociación no prueba que un factor cause la victoria.
3. **Una base reproducible para el dashboard.** Cada cifra se calcula con el mismo código en Python y en R, de modo que el dashboard final para el entrenador se apoye en resultados verificados en dos herramientas distintas.