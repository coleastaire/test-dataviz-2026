# 🏐 Voleibol de playa: ¿qué hace ganar a una pareja?

Análisis exploratorio de **76.749 partidos** de los circuitos profesionales de voleibol de playa **FIVB** y **AVP** (2000–2019), pensado para un **entrenador**: qué tan sólida es una ventaja de ranking antes del partido, cómo armar parejas y qué priorizar en el entrenamiento.

**Autores:** Nicolle Ramírez y Roger Daza
**Curso:** Visualización de Datos y Toma de Decisiones · Universidad del Norte · 2026

| Entregable | Enlace |
|---|---|
| 📘 Libro en Python (Jupyter Book) | https://nicoleramirez2106-ui.github.io/dataviz-2026/ |
| 📊 Dashboard interactivo (Dash) | Se corre en su computador con las instrucciones de abajo |
| 💻 Código fuente | Este repositorio |

---

## Contenido del repositorio

```
dataviz-2026/
├── dashboard/                  ← Dashboard interactivo
│   ├── app.py                  ← páginas, gráficos e interacción (el archivo que se ejecuta)
│   ├── datos.py                ← carga de datos y pruebas estadísticas
│   ├── estilo.py               ← paleta de colores y formato de números
│   ├── prediccion.py           ← lee los modelos guardados (el dashboard no entrena nada)
│   ├── entrenar_modelos.py     ← entrena los modelos y los perfiles de juego (fuera del dashboard)
│   ├── preparar_datos.py       ← genera la carpeta datos/ (no hace falta correrlo)
│   ├── requirements.txt        ← librerías que necesita el dashboard
│   ├── requirements-pruebas.txt← librerías extra para correr las pruebas
│   ├── assets/estilo.css       ← diseño visual
│   ├── datos/                  ← datos ya preparados (5 MB), listos para usar
│   ├── modelos/                ← métricas, curvas, perfiles y modelo ya entrenados
│   └── tests/                  ← pruebas automáticas (pytest y navegador)
├── jb-book/                    ← Libro en Jupyter Book (capítulos 1 a 8)
│   ├── py/limpieza.py          ← proceso de limpieza (12 pasos)
│   ├── py/utils.py             ← funciones comunes del libro
│   └── requirements.txt        ← librerías que necesita el libro
├── data/raw/vb_matches.csv     ← datos originales (BigTimeStats vía TidyTuesday)
└── compartido/                 ← referencias (.bib) y cifras de control Python/R
```

---

## Cómo correr el dashboard

Funciona en **Windows, macOS y Linux**. Solo necesita **Python 3.10 o superior** ([descargar aquí](https://www.python.org/downloads/); en Windows, marque la casilla *"Add Python to PATH"* al instalarlo).

### 1. Descargar el repositorio

**Opción A, sin Git:** en esta página, botón verde **Code → Download ZIP**. Descomprima el archivo.

**Opción B, con Git:**
```bash
git clone https://github.com/nicoleramirez2106-ui/dataviz-2026.git
```

### 2. Abrir una terminal dentro de la carpeta

- **Windows:** abra la carpeta `dataviz-2026` en el Explorador, haga clic en la barra de direcciones, escriba `powershell` y presione Enter.
- **macOS:** clic derecho sobre la carpeta → *Nuevo terminal en la carpeta*.
- **Linux:** clic derecho dentro de la carpeta → *Abrir en terminal*.

### 3. Crear un entorno e instalar las librerías

Se hace **una sola vez**. El entorno virtual evita mezclar estas librerías con otras que ya tenga instaladas.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r dashboard/requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r dashboard/requirements.txt
```

Al activar el entorno, la terminal muestra `(.venv)` al inicio de la línea. La instalación tarda uno o dos minutos.

<details>
<summary>¿Usa Anaconda? Haga clic aquí</summary>

```bash
conda create -n voleibol python=3.11 -y
conda activate voleibol
python -m pip install -r dashboard/requirements.txt
```
</details>

### 4. Ejecutar

```bash
python dashboard/app.py
```

Cuando aparezca `Dashboard listo`, abra en el navegador: **http://127.0.0.1:8050**

Para apagarlo, vuelva a la terminal y presione `Ctrl + C`.

**Las siguientes veces** solo hay que abrir la terminal en la carpeta, activar el entorno (`.venv\Scripts\activate` en Windows o `source .venv/bin/activate` en macOS/Linux) y correr `python dashboard/app.py`.

### Problemas frecuentes

| Problema | Solución |
|---|---|
| `python` no se reconoce como comando | Python no está instalado o no está en el PATH. Reinstálelo marcando *"Add Python to PATH"*. En macOS/Linux use `python3`. |
| En Windows, al activar el entorno sale *"la ejecución de scripts está deshabilitada"* | Corra `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelva a activar el entorno. |
| `Port 8050 is in use` / el puerto está ocupado | Use otro puerto: `python dashboard/app.py 8051` y abra http://127.0.0.1:8051 |
| La página se ve sin colores ni diseño | El diseño (Bootstrap, íconos y fuente) se descarga de internet al abrir la página, salvo que exista una copia local en `dashboard/assets/vendor/`. Revise la conexión y recargue. |
| `No module named ...` | El entorno no está activado o falta instalar: repita el paso 3. |

---

## Qué tiene el dashboard

Cinco páginas, en el riel de la izquierda (en el celular, en la barra de abajo):

| Página | Contenido |
|---|---|
| **Inicio** | Una cancha: la pregunta de investigación de un lado de la red y el marcador 2000–2019 del otro. Accesos a cada sección, tres hallazgos, objetivos y el espacio del video del proyecto (*Próximamente*). |
| **Dashboard** | Marcador con los filtros actuales; partidos por año (haga clic en un año para ver su detalle), sorpresas por año, cobertura de estadísticas, partidos por circuito, género y etapa, y los problemas de calidad corregidos. |
| **Análisis** | Pestañas **Univariado**, **Bivariado** (ganador vs perdedor, poder discriminante, cuándo gana el favorito, sorpresas por etapa, V de Cramér), **Multivariado** (Spearman y VIF), **Estilos de juego (AVP)**, **Metodología** y **Conclusiones**. |
| **Predicción** | **Modelos:** regresión logística, Random Forest y Gradient Boosting frente a la regla "gana el mejor rankeado" (modelo recomendado, métricas, matriz de confusión, curvas ROC y precisión-exhaustividad, importancia de variables, umbral de decisión, comparación, conclusiones y limitaciones). **¿Quién gana?:** simulador con el modelo recomendado. **Calculadora histórica:** la calculadora del entrenador, con frecuencias observadas (no es el modelo). |
| **Equipo** | Pestaña reservada; todavía sin contenido. |

Los filtros rápidos de arriba (circuito, género, etapa y años) actualizan el marcador, el *Dashboard* y las pestañas univariada, bivariada y multivariada de *Análisis*. Las rutas antiguas (`/calculadora`, `/metodologia`, etc.) llevan a su nuevo lugar.

**Paleta fija** (definida en `estilo.py` y `assets/estilo.css`, la misma del libro): **Mar** `#0B3B5C` = ganador y marca, **Coral** `#E4572E` = perdedor y sorpresas, **Turquesa** `#1A8A9A` = FIVB, **Arena** `#D9A441` = AVP.

### Modelos y perfiles de juego (opcional: ya vienen entrenados)

El dashboard **no entrena nada al arrancar**: lee los resultados guardados en `dashboard/modelos/`. Para regenerarlos (tarda unos minutos):

```bash
python dashboard/entrenar_modelos.py
```

- **Modelo previo al partido.** Solo usa lo que se sabe antes de jugar: ventaja de ranking, edad, estatura, origen en clasificatoria, circuito, género y etapa. Las estadísticas de juego (kills, aces, bloqueos, defensas, errores) quedan fuera porque se producen durante el partido. De las dos filas de cada partido (ganador y perdedor) se toma una al azar con semilla fija; la división entrenamiento/prueba (75/25) es por partido, y todo el preprocesamiento se ajusta solo con el entrenamiento. El modelo recomendado se elige con validación cruzada, no con la prueba. La regla "gana el mejor rankeado" se evalúa solo en los partidos con ranking de las dos parejas, y se informa qué porcentaje cubre.
- **Perfiles de juego (AVP).** K-means con 4 grupos sobre kills, aces, bloqueos, defensas, ataques, errores de ataque y errores de saque por set, más el % de ataque, estandarizados. Los perfiles se superponen (ver la silueta en la pestaña) y sus tasas de victoria son asociaciones, no causas.
- Todo describe partidos de 2000 a 2019 y son **asociaciones observacionales, no relaciones causales**.

El modelo guardado (`modelo_recomendado.joblib`) depende de la versión de scikit-learn; si al abrir el dashboard aparece un aviso de que no se pudo cargar, vuelva a correr el script.

### Pruebas

```bash
python -m pip install -r dashboard/requirements-pruebas.txt
python -m playwright install chromium     # solo para la prueba en navegador
python -m pytest dashboard/tests
```

---

## Cómo reproducir el libro (opcional)

El libro ya está publicado en el enlace de arriba. Para construirlo desde cero:

```bash
python -m pip install -r jb-book/requirements.txt
cd jb-book
python -m py.limpieza        # genera data/clean/ a partir de data/raw/
cd ..
jupyter-book build jb-book   # genera jb-book/_build/html/
```

Abra `jb-book/_build/html/index.html` en el navegador. Si cambian los datos limpios, regenere también los datos del dashboard con `python dashboard/preparar_datos.py`.

---

## Datos y fuentes

- **Datos:** Vagnar, A. (2020). *AVP & FIVB beach volleyball match database*. BigTime Stats. Difundido por la comunidad TidyTuesday (2020-05-19).
- **Herramientas:** Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, joblib, Plotly, great_tables, Dash y Jupyter Book.
- Las estadísticas de juego (kills, aces, bloqueos, etc.) existen casi solo para la AVP; todos los resultados son asociaciones, no relaciones causales. Ver la sección *Limitaciones* del libro.