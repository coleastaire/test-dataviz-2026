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
│   ├── preparar_datos.py       ← genera la carpeta datos/ (no hace falta correrlo)
│   ├── requirements.txt        ← librerías que necesita el dashboard
│   ├── assets/estilo.css       ← diseño visual
│   └── datos/                  ← datos ya preparados (5 MB), listos para usar
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
| La página se ve sin colores ni diseño | El diseño (Bootstrap, íconos y fuente) se descarga de internet al abrir la página. Revise la conexión y recargue. |
| `No module named ...` | El entorno no está activado o falta instalar: repita el paso 3. |

---

## Qué tiene el dashboard

| Página | Contenido |
|---|---|
| **Inicio** | Pregunta de investigación, objetivos, cifras clave y tres hallazgos principales. |
| **Metodología** | Enfoque exploratorio, pruebas no paramétricas (Mann-Whitney, Wilcoxon, chi-cuadrado), tamaño del efecto, Spearman, VIF, fases del procesamiento, limitaciones y referencias. |
| **Datos y limpieza** | Partidos por circuito, género, etapa y año; cobertura de estadísticas de juego; problemas de calidad corregidos. |
| **Análisis exploratorio** | **Univariado:** distribución y estadísticos de 11 variables. **Bivariado:** ganador vs perdedor, poder discriminante, probabilidad de que gane el favorito, sorpresas por etapa y V de Cramér. **Multivariado:** correlación de Spearman y redundancia (VIF). |
| **Calculadora del entrenador** | Ingrese el ranking de su pareja y el del rival (y opcionalmente circuito, género, etapa y diferencia de estatura) para ver el porcentaje histórico de victorias en partidos similares. |
| **Conclusiones** | Diez hallazgos, recomendaciones para el entrenador, qué no se puede concluir y qué datos harían falta. |

Los filtros de la barra lateral (circuito, género y años) actualizan las páginas *Datos y limpieza* y *Análisis exploratorio*.

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
- **Herramientas:** Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, Plotly, great_tables, Dash y Jupyter Book.
- Las estadísticas de juego (kills, aces, bloqueos, etc.) existen casi solo para la AVP; todos los resultados son asociaciones, no relaciones causales. Ver la sección *Limitaciones* del libro.