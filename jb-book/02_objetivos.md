# Objetivos

## Objetivo general

Identificar qué factores medibles antes de un partido de voleibol de playa se asocian con la victoria de una pareja y con qué fuerza lo hacen, y evaluar si permiten anticipar el resultado mejor que el ranking, mediante el análisis exploratorio, temporal y el modelado de más de 76.000 partidos de los circuitos FIVB y AVP (2000–2019), con el fin de ofrecer a un entrenador información confiable para preparar partidos y armar parejas.

## Objetivos específicos

1. Depurar y documentar el conjunto de datos, corrigiendo sus problemas de calidad (rankings en texto, fechas de nacimiento erróneas, llaves duplicadas y estadísticas imposibles) y delimitando la cobertura de cada variable por circuito, género y año, para que toda conclusión posterior se apoye en datos verificables (capítulos 4 y 5).

2. Describir la distribución de las variables del partido y de los jugadores, incluida la variable respuesta (ranking, edad, estatura, duración, sets, puntos y resultado), mediante estadísticos descriptivos y gráficos, para caracterizar el perfil típico de un partido y de una pareja profesional (sección 6.1).

3. Comparar a las parejas ganadoras con las perdedoras mediante pruebas no paramétricas (Mann-Whitney U, chi-cuadrado) y medidas de tamaño del efecto (correlación rango-biserial, V de Cramér), para determinar qué factores separan el resultado y con qué fuerza (sección 6.2).

4. Cuantificar cuándo el ranking anticipa el resultado y cuándo no, estimando la probabilidad de victoria del favorito según la diferencia de ranking y la tasa de sorpresas por etapa del torneo, circuito y género, y verificar que los factores previos al partido no sean redundantes entre sí (secciones 6.2 y 6.3).

5. Analizar la evolución de la competencia entre 2000 y 2019 (volumen de partidos, duración, edad, estatura y tasa de sorpresas), distinguiendo los cambios reales del deporte de los cambios de cobertura del conjunto de datos (capítulo 7).

6. Entrenar y comparar modelos de clasificación (regresión logística, Random Forest y XGBoost) que predigan el ganador con información previa al partido, y medir si superan la regla "gana el mejor rankeado" según AUC, accuracy y F1 (capítulo 8).

7. Traducir los hallazgos en recomendaciones para el entrenador, señalando qué puede usar, qué no se puede concluir con estos datos y qué información adicional haría falta, como base para el dashboard interactivo (capítulo 9).

**Tabla 2.1.** Objetivos específicos, decisión del entrenador que apoyan y capítulo donde se cumplen.

| Objetivo | Decisión del entrenador que apoya | Capítulo |
|---|---|---|
| 1. Depurar y documentar | Confiar en las cifras del análisis | 4 y 5 |
| 2. Describir variables | Conocer el perfil típico de una pareja y un partido | 6.1 |
| 3. Comparar ganadores y perdedores | Armar parejas y priorizar el entrenamiento | 6.2 |
| 4. Ranking, sorpresas y redundancia | Preparar un partido contra un rival concreto | 6.2 y 6.3 |
| 5. Evolución temporal | Saber si el juego cambió con los años | 7 |
| 6. Modelos de clasificación | Anticipar el resultado | 8 |
| 7. Recomendaciones | Tomar decisiones con límites claros | 9 |
