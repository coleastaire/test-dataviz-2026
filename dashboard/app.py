"""Dashboard interactivo: Voleibol de playa, ¿qué hace ganar a una pareja?

Correr con:  python app.py   y abrir http://127.0.0.1:8050 en el navegador.
"""
import sys

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

import datos as D
from estilo import (ESCALA_CORR, ETAPAS, FUENTE, GENERO, GRIS, NEUTRO, PAL, PAL_ETAPA,
                    PAL_PODER, fmt, fmt_p, fmt_pct, vacia)

LIBRO = "https://nicoleramirez2106-ui.github.io/dataviz-2026/"
REPO = "https://github.com/nicoleramirez2106-ui/dataviz-2026"

app = Dash(__name__, title="Voleibol de playa · Dashboard",
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP,
                                 "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
           suppress_callback_exceptions=True)
server = app.server
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# ==========================================================================
# Piezas reutilizables
# ==========================================================================
def tarjeta(titulo, cuerpo, subtitulo=None, fuente=True):
    return dbc.Card([
        dbc.CardHeader([html.H6(titulo, className="tarjeta-titulo"),
                        html.Div(subtitulo, className="tarjeta-subtitulo") if subtitulo is not None else None]),
        dbc.CardBody(cuerpo),
        html.Div(FUENTE, className="tarjeta-fuente") if fuente else None,
    ], className="tarjeta mb-4")


def kpi(valor, etiqueta, nota=None, id_=None):
    extra = {"id": id_} if id_ else {}
    return html.Div([html.Div(etiqueta, className="kpi-etiqueta"),
                     html.Div(valor, className="kpi-valor", **extra),
                     html.Div(nota, className="kpi-nota") if nota else None], className="kpi")


def grafico(id_, alto=380):
    return dcc.Loading(dcc.Graph(id=id_, config=CONFIG, style={"height": f"{alto}px"}),
                       type="dot", color=NEUTRO)


def histograma(fig, s, ancho, color, nombre=None, opacidad=1.0):
    """Histograma calculado en Python: el navegador solo recibe las barras, no los 300.000 datos."""
    s = s.dropna()
    inicio = np.floor(s.min() / ancho) * ancho
    bordes = np.arange(inicio, s.max() + ancho * 1.001, ancho)
    n, bordes = np.histogram(s, bins=bordes)
    centros = (bordes[:-1] + bordes[1:]) / 2
    fig.add_bar(x=centros, y=n, width=ancho, name=nombre, marker=dict(color=color, line=dict(color="white", width=0.5)),
                opacity=opacidad, customdata=np.stack([bordes[:-1], bordes[1:]], axis=1), showlegend=nombre is not None,
                hovertemplate=(nombre + "<br>" if nombre else "") +
                "%{customdata[0]:.4~r} a %{customdata[1]:.4~r}<br>%{y:,} registros<extra></extra>")


def caja(fig, s, x, color, nombre, mostrar=True):
    """Diagrama de caja con los cuartiles ya calculados (más liviano que mandar todos los datos)."""
    s = s.dropna()
    q1, med, q3 = s.quantile([.25, .5, .75])
    iqr = q3 - q1
    bajo, alto = s[s >= q1 - 1.5 * iqr].min(), s[s <= q3 + 1.5 * iqr].max()
    fig.add_box(x=[x], q1=[q1], median=[med], q3=[q3], lowerfence=[bajo], upperfence=[alto],
                name=nombre, marker_color=color, offsetgroup=nombre, legendgroup=nombre, showlegend=mostrar,
                hoverinfo="y")


def entrenador(texto):
    return html.Div([html.I(className="bi bi-clipboard2-pulse me-2"),
                     html.Strong("Para el entrenador. "), texto], className="entrenador mb-4")


def encabezado(titulo, texto):
    return html.Div([html.H2(titulo, className="pagina-titulo"),
                     html.P(texto, className="pagina-intro")], className="mb-4")


# ==========================================================================
# Barra lateral: navegación y filtros generales
# ==========================================================================
PAGINAS = [("/", "Inicio", "bi-house-door"),
           ("/metodologia", "Metodología", "bi-journal-text"),
           ("/datos", "Datos y limpieza", "bi-database-check"),
           ("/exploratorio", "Análisis exploratorio", "bi-graph-up"),
           ("/calculadora", "Calculadora del entrenador", "bi-calculator"),
           ("/conclusiones", "Conclusiones", "bi-flag")]

barra = html.Div([
    html.Div([html.Span("🏐", className="marca-icono"),
              html.Div([html.Div("Voleibol de playa", className="marca-titulo"),
                        html.Div("¿Qué hace ganar a una pareja?", className="marca-sub")])],
             className="marca"),
    dbc.Nav([dbc.NavLink([html.I(className=f"bi {ico} me-2"), txt], href=ruta, active="exact")
             for ruta, txt, ico in PAGINAS], vertical=True, pills=True, className="menu"),
    html.Div([
        html.Div("Filtros", className="filtros-titulo"),
        html.Div("Aplican a Datos y Análisis exploratorio", className="filtros-nota"),
        html.Label("Circuito", className="filtro-etq"),
        dbc.RadioItems(id="f-circuito", value="Todos", inline=True, className="filtro-radio",
                       options=[{"label": x, "value": x} for x in ["Todos", "FIVB", "AVP"]]),
        html.Label("Género", className="filtro-etq"),
        dbc.RadioItems(id="f-genero", value="Todos", inline=True, className="filtro-radio",
                       options=[{"label": "Todos", "value": "Todos"},
                                {"label": "Hombres", "value": "M"},
                                {"label": "Mujeres", "value": "W"}]),
        html.Label("Años", className="filtro-etq"),
        dcc.RangeSlider(id="f-anios", min=D.ANIOS[0], max=D.ANIOS[1], step=1,
                        value=list(D.ANIOS), marks={a: str(a) for a in range(2000, 2020, 5)} | {2019: "2019"},
                        tooltip={"placement": "bottom", "always_visible": False}),
    ], className="filtros"),
    html.Div([html.A([html.I(className="bi bi-book me-1"), "Leer el libro"], href=LIBRO, target="_blank"),
              html.A([html.I(className="bi bi-github me-1"), "Repositorio"], href=REPO, target="_blank")],
             className="enlaces"),
], className="barra")

app.layout = html.Div([dcc.Location(id="url"), barra,
                       html.Main(id="contenido", className="contenido")])


# ==========================================================================
# Cifras fijas que se usan en Inicio y Conclusiones (todos los datos)
# ==========================================================================
C = D.CIFRAS
_r = D.P[D.P.rank_diff.notna()]
_grupos = pd.cut(_r.rank_diff.abs(), [0, 2, 5, 10, 20, np.inf])
_fav = _r.groupby(_grupos, observed=True).es_upset.apply(lambda s: 100 * (1 - s.mean()))
C["fav_cerca"], C["fav_lejos"] = _fav.iloc[0], _fav.iloc[-1]
_poder = D.tabla_poder("Todos", "Todos", D.ANIOS).set_index("var")
C["est10"] = 100 * (D.E[D.E.ventaja_estatura_cm >= 10].resultado == "Ganador").mean()
_perd = 100 * D.E.groupby("es_clasif").resultado.apply(lambda s: (s == "Perdedor").mean())
_vif, _n_vif = D.vif()


# ==========================================================================
# Página: Inicio
# ==========================================================================
def pagina_inicio():
    hallazgos = [
        ("Ranking", fmt_pct(C["pct_fav"]), "de los partidos los gana la pareja mejor rankeada. "
         "Casi uno de cada tres es una sorpresa."),
        ("Distancia con el rival", f"{fmt_pct(C['fav_cerca'], 0)} → {fmt_pct(C['fav_lejos'], 0)}",
         "gana el favorito con 1–2 posiciones de diferencia frente a más de 20."),
        ("Juego (AVP)", fmt(_poder.rbc["hitpct"]), "es la correlación rango-biserial del % de ataque: "
         "la variable que más separa al ganador del perdedor."),
    ]
    return html.Div([
        html.Div([
            html.Div("Visualización de Datos y Toma de Decisiones · Universidad del Norte · 2026",
                     className="hero-eyebrow"),
            html.H1("¿Qué hace ganar a una pareja de voleibol de playa?", className="hero-titulo"),
            html.P(f"{fmt(C['n_partidos'], 0)} partidos de los circuitos FIVB y AVP entre 2000 y 2019, "
                   "analizados para que un entrenador sepa qué tan sólida es una ventaja antes del "
                   "partido, cómo armar parejas y qué priorizar en el entrenamiento.", className="hero-texto"),
            html.Div([dbc.Button([html.I(className="bi bi-calculator me-2"), "Probar la calculadora"],
                                 href="/calculadora", className="boton-principal me-2"),
                      dbc.Button([html.I(className="bi bi-book me-2"), "Leer el libro completo"],
                                 href=LIBRO, target="_blank", outline=True, className="boton-secundario")]),
        ], className="hero mb-4"),
        html.Div([kpi(fmt(C["n_partidos"], 0), "Partidos"),
                  kpi(fmt(C["n_jugadores"], 0), "Jugadores distintos"),
                  kpi(fmt(C["n_paises"], 0), "Países sede"),
                  kpi(fmt(C["n_rank"], 0), "Partidos con ranking de ambas parejas"),
                  kpi(fmt_pct(C["pct_stats"]), "Partidos con estadísticas de juego", "casi todos de la AVP")],
                 className="kpis mb-4"),
        dbc.Row([
            dbc.Col(tarjeta("Pregunta de investigación", [
                html.P(html.Strong("¿Qué factores, medibles antes del partido, se asocian con que una "
                                   "pareja gane, y cómo puede usarlos un entrenador para preparar "
                                   "partidos y armar parejas?"), className="pregunta"),
                html.Ol([html.Li([html.Strong("Preparar un partido: "),
                                  "¿qué tan probable es ganar según la diferencia de ranking con el rival?"]),
                         html.Li([html.Strong("Armar parejas: "),
                                  "¿la estatura, la edad o la diferencia entre compañeros se asocian con ganar?"]),
                         html.Li([html.Strong("Priorizar el entrenamiento: "),
                                  "¿qué acciones de juego separan al ganador del perdedor?"])])],
                fuente=False), lg=7),
            dbc.Col(tarjeta("Objetivos", html.Ol([
                html.Li("Depurar y documentar el conjunto de datos."),
                html.Li("Describir la distribución de las variables del partido y de los jugadores."),
                html.Li("Comparar a las parejas ganadoras con las perdedoras."),
                html.Li("Cuantificar cuándo el ranking anticipa el resultado y cuándo no."),
                html.Li("Traducir los hallazgos en recomendaciones para el entrenador.")]), fuente=False), lg=5),
        ]),
        html.H5("Tres hallazgos para empezar", className="seccion-titulo"),
        dbc.Row([dbc.Col(html.Div([html.Div(etq, className="hallazgo-etq"),
                                   html.Div(val, className="hallazgo-valor"),
                                   html.Div(txt, className="hallazgo-texto")], className="hallazgo"), md=4)
                 for etq, val, txt in hallazgos], className="mb-4"),
    ])


# ==========================================================================
# Página: Metodología
# ==========================================================================
def pagina_metodologia():
    pruebas = r"""
**Mann-Whitney U.** Compara una variable numérica entre ganadores y perdedores sin suponer
normalidad. Su tamaño del efecto es la **correlación rango-biserial**:

$$RBC = \frac{2U}{n_1 n_2} - 1$$

Va de −1 a 1. Poder discriminante: **alto** si |RBC| ≥ 0,5; **moderado** entre 0,3 y 0,5; **bajo** si es menor a 0,3.

**Chi-cuadrado y V de Cramér.** Miden si dos variables categóricas están asociadas:

$$V = \sqrt{\frac{\chi^2}{n\,(\min(r, c) - 1)}}$$

Despreciable si V < 0,1; pequeña hasta 0,3; media hasta 0,5; grande desde 0,5.

**Wilcoxon de rangos con signo.** Cada partido aporta un ganador y un perdedor que no son
independientes; por eso se compara también a cada pareja con su propio rival.
"""
    otras = r"""
**Solapamiento IQR.** Cuánto se superponen los rangos centrales (P25–P75) de ganadores y
perdedores: 0 si no se tocan, 1 si son idénticos.

**Correlación de Spearman** ($\rho$). Asociación monótona entre dos variables, usando rangos.

**Factor de inflación de la varianza.** Verifica que las variables previas al partido no sean
redundantes:

$$VIF_j = \frac{1}{1 - R^2_j}$$

Aceptable si VIF ≤ 5.

**¿Por qué tamaño del efecto y no p-valor?** Con más de 150.000 filas casi todo es
"significativo". El p-valor dice si una diferencia existe; el tamaño del efecto dice si importa.
"""
    fases = pd.DataFrame([
        ("1. Datos crudos", "Archivo original, nunca se modifica", "76.756 partidos"),
        ("2. Limpieza", "12 pasos: fechas, rankings, marcador, etapas, llaves y estadísticas",
         "Cifras de control comparables con R"),
        ("3. Archivos limpios", "Tres tablas con distinta unidad de análisis",
         "Partido · equipo-partido · jugador-partido"),
        ("4. Análisis", "Exploratorio univariado, bivariado y multivariado", "Libro y dashboard"),
        ("5. Conclusiones", "Recomendaciones para el entrenador", "Calculadora y hallazgos")],
        columns=["Fase", "Qué se hace", "Resultado"])
    refs = ["Vagnar, A. (2020). AVP & FIVB beach volleyball match database. BigTime Stats.",
            "R for Data Science Online Learning Community (2020). TidyTuesday: Beach volleyball.",
            "Mann, H. B. y Whitney, D. R. (1947). The Annals of Mathematical Statistics, 18(1), 50–60.",
            "Kerby, D. S. (2014). The simple difference formula. Comprehensive Psychology, 3.",
            "Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2.ª ed.).",
            "Cramér, H. (1946). Mathematical methods of statistics. Princeton University Press.",
            "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6).",
            "Spearman, C. (1904). The American Journal of Psychology, 15(1), 72–101."]
    return html.Div([
        encabezado("Metodología", "Cómo se hizo el análisis y por qué se eligió cada técnica. "
                   "Todo se aplica igual en el libro de Python, en el de R y en este dashboard."),
        dbc.Row([
            dbc.Col(tarjeta("Enfoque", [
                html.P("Análisis exploratorio de datos (EDA) en tres niveles, sin modelos predictivos:"),
                html.Ul([html.Li([html.Strong("Univariado: "), "cada variable por separado."]),
                         html.Li([html.Strong("Bivariado: "), "cada variable frente al resultado."]),
                         html.Li([html.Strong("Multivariado: "), "varias variables a la vez."])]),
                html.P([html.Strong("Unidades de análisis: "), "partido, equipo-partido (una fila por "
                        "pareja) y jugador-partido."]),
                html.P([html.Strong("Favorito y sorpresa: "), "el favorito es la pareja con mejor ranking; "
                        "una sorpresa (upset) es un partido que gana la peor rankeada."], className="mb-0")],
                fuente=False), lg=4),
            dbc.Col(tarjeta("Pruebas no paramétricas", dcc.Markdown(pruebas, mathjax=True), fuente=False), lg=4),
            dbc.Col(tarjeta("Solapamiento, correlación y redundancia", dcc.Markdown(otras, mathjax=True),
                            fuente=False), lg=4),
        ]),
        dbc.Row([
            dbc.Col(tarjeta("Fases del procesamiento",
                            dbc.Table.from_dataframe(fases, striped=True, hover=True, className="tabla"),
                            fuente=False), lg=7),
            dbc.Col(tarjeta("Limitaciones", html.Ul([
                html.Li("Las estadísticas de juego describen casi solo a la AVP; no se imputan."),
                html.Li("Cobertura desigual en el tiempo: 2000 y 2019 son parciales y la AVP casi no tiene partidos en 2011–2012."),
                html.Li("Muchas parejas de clasificatoria no tienen ranking numérico."),
                html.Li("Asociación no es causalidad."),
                html.Li("Los datos fueron recopilados por terceros; no son un registro oficial.")]),
                fuente=False), lg=5),
        ]),
        tarjeta("Referencias principales", [html.Ul([html.Li(r) for r in refs]),
                                            html.A("Ver la lista completa en el libro →", href=LIBRO + "08_referencias.html",
                                                   target="_blank")], fuente=False),
    ])


# ==========================================================================
# Página: Datos y limpieza
# ==========================================================================
def pagina_datos():
    R = D.RES
    problemas = pd.DataFrame([
        ("Fechas de nacimiento en el futuro", f"{fmt(R['p1_partidos_bd_corregida'], 0)} partidos", "Restar 100 años y marcar"),
        ("Ranking guardado como texto (\"17, Q2\")", "Todos los partidos", "Separar ranking, posición Q y bandera"),
        ("Estadísticas de juego vacías", f"{fmt_pct(100 - C['pct_stats'])} de los partidos", "Bandera de cobertura; no se imputan"),
        ("Mismo jugador en ambos equipos", f"{fmt(R['p7_filas_invalidas'], 0)} partidos", "Eliminar las filas"),
        ("Estadísticas imposibles (kills > ataques)", f"{fmt(R['p8_celdas_invalidas'], 0)} celdas", "Anular y recalcular el % de ataque"),
        ("Forfeits y retiros", f"{fmt(R['p5_forfeit'], 0)} y {fmt(R['p5_retirado'], 0)}", "Banderas; forfeits fuera de duración"),
        ("Duración como texto HH:MM:SS", f"{fmt(R['p2_duracion_na'], 0)} vacías", "Convertir a minutos"),
        ("Edad que no cuadra con las fechas", f"{fmt(R['p9_edad_inconsistente'], 0)} partidos", "Marcar; se conservan")],
        columns=["Problema", "Alcance", "Decisión"])
    return html.Div([
        encabezado("Datos y limpieza", "De dónde vienen los datos, cómo se reparten y qué se corrigió "
                   "antes de analizarlos. Use los filtros de la izquierda para ver un circuito, un género o un periodo."),
        html.Div([kpi(fmt(R["p0_filas_crudas"], 0), "Partidos en el archivo original"),
                  kpi(fmt(R["p7_filas_invalidas"], 0), "Partidos eliminados", "imposibles: mismo jugador en ambos equipos"),
                  kpi(fmt(R["p7_filas_finales"], 0), "Partidos analizados"),
                  kpi("—", "Partidos con los filtros actuales", id_="d-n")], className="kpis mb-4"),
        dbc.Row([dbc.Col(tarjeta("Partidos por circuito y género", grafico("d-circuito", 340)), lg=6),
                 dbc.Col(tarjeta("Partidos por etapa del torneo", grafico("d-etapa", 340)), lg=6)]),
        dbc.Row([dbc.Col(tarjeta("Partidos por año y circuito", grafico("d-anio", 340),
                                 "La caída de la AVP en 2011–2012 coincide con su crisis financiera"), lg=6),
                 dbc.Col(tarjeta("Cobertura de estadísticas de juego por año", grafico("d-cobertura", 340),
                                 "% de partidos con estadísticas completas"), lg=6)]),
        tarjeta("Problemas de calidad encontrados y decisiones",
                dbc.Table.from_dataframe(problemas, striped=True, hover=True, className="tabla"), fuente=False),
        entrenador("Las estadísticas de juego (kills, aces, bloqueos) solo existen de forma confiable para la AVP. "
                   "Si su pareja compite en la FIVB, use el ranking, la edad, la estatura y la etapa, que están "
                   "disponibles para casi todos los partidos."),
    ])


@app.callback(Output("d-n", "children"), Output("d-circuito", "figure"), Output("d-etapa", "figure"),
              Output("d-anio", "figure"), Output("d-cobertura", "figure"),
              Input("f-circuito", "value"), Input("f-genero", "value"), Input("f-anios", "value"))
def actualizar_datos(circuito, genero, anios):
    d = D.filtrar(D.P, circuito, genero, anios)
    if d.empty:
        return "0", vacia(), vacia(), vacia(), vacia()

    t = d.groupby(["circuit", "gender"], observed=True).size().reset_index(name="n")
    t["circuit"] = pd.Categorical(t.circuit.astype(str), ["FIVB", "AVP"])
    t = t.sort_values("circuit")
    f1 = go.Figure()
    for g in ["M", "W"]:
        s = t[t.gender == g]
        f1.add_bar(x=s.circuit.astype(str), y=s.n, name=GENERO[g], marker_color=PAL[g],
                   text=[fmt(v, 0) for v in s.n], textposition="outside", cliponaxis=False,
                   hovertemplate="%{x} · " + GENERO[g] + "<br>%{y:,} partidos<extra></extra>")
    f1.update_layout(barmode="group", bargap=0.35, yaxis=dict(title="Partidos", tickformat=",d"))

    e = d.etapa.value_counts().reindex(ETAPAS).fillna(0)
    f2 = go.Figure(go.Bar(x=e.index, y=e.values, marker_color=[PAL_ETAPA[x] for x in e.index],
                          text=[fmt(v, 0) for v in e.values], textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>%{y:,} partidos<extra></extra>"))
    f2.update_layout(yaxis=dict(title="Partidos", tickformat=",d"), showlegend=False)

    a = d.groupby(["year", "circuit"], observed=True).size().reset_index(name="n")
    f3 = go.Figure()
    for c in ["FIVB", "AVP"]:
        s = a[a.circuit == c]
        if len(s):
            f3.add_scatter(x=s.year, y=s.n, name=c, mode="lines+markers", line=dict(color=PAL[c], width=2),
                           marker=dict(size=7), hovertemplate="%{x} · " + c + "<br>%{y:,} partidos<extra></extra>")
    f3.update_layout(yaxis=dict(title="Partidos", tickformat=",d"), hovermode="x unified")

    cob = d.groupby(["year", "circuit"], observed=True).tiene_stats_completas.mean().reset_index()
    f4 = go.Figure()
    for c in ["FIVB", "AVP"]:
        s = cob[cob.circuit == c]
        if len(s):
            f4.add_scatter(x=s.year, y=100 * s.tiene_stats_completas, name=c, mode="lines+markers",
                           line=dict(color=PAL[c], width=2), marker=dict(size=7),
                           hovertemplate="%{x} · " + c + "<br>%{y:.1f} % con estadísticas<extra></extra>")
    f4.update_layout(yaxis=dict(range=[0, 100], ticksuffix=" %"), hovermode="x unified")
    return fmt(len(d), 0), f1, f2, f3, f4


# ==========================================================================
# Página: Análisis exploratorio (univariado, bivariado y multivariado)
# ==========================================================================
def pagina_exploratorio():
    uni = html.Div([
        dbc.Row([dbc.Col([html.Label("Variable", className="control-etq"),
                          dcc.Dropdown(id="u-var", value="edad", clearable=False,
                                       options=[{"label": v[2], "value": k} for k, v in D.VARS_UNI.items()])],
                         md=5)], className="mb-3"),
        html.Div(id="u-kpis", className="kpis mb-3"),
        tarjeta("Distribución", grafico("u-fig", 400), html.Span(id="u-sub")),
        entrenador("La duración típica de un partido es de unos 42 minutos, pero uno de cada tres se va a "
                   "tercer set y dura cerca de 54: prepare la exigencia física para el partido largo."),
    ])
    bi = html.Div([
        dbc.Row([dbc.Col([html.Label("Variable", className="control-etq"),
                          dcc.Dropdown(id="b-var", value="rank_main", clearable=False,
                                       options=[{"label": v[0] + (" (AVP)" if v[2] else ""), "value": k}
                                                for k, v in D.VARS_BI.items()])], md=5)], className="mb-3"),
        html.Div(id="b-kpis", className="kpis mb-3"),
        dbc.Row([dbc.Col(tarjeta("Ganador vs perdedor", grafico("b-caja", 400), html.Span(id="b-sub")), lg=6),
                 dbc.Col(tarjeta("Poder discriminante de todas las variables", grafico("b-poder", 400),
                                 "|correlación rango-biserial|; líneas en 0,3 y 0,5"), lg=6)]),
        dbc.Row([dbc.Col(tarjeta("¿Cuándo gana el favorito?", grafico("b-fav", 360),
                                 "% de victorias del mejor rankeado según la diferencia de ranking"), lg=6),
                 dbc.Col(tarjeta("Sorpresas por etapa y circuito", grafico("b-upset", 360),
                                 "% de partidos que gana la pareja peor rankeada (etapas con 30 o más partidos)"), lg=6)]),
        tarjeta("Variables categóricas: chi-cuadrado y V de Cramér", html.Div(id="b-chi"), fuente=False),
        entrenador("Lo que más informa antes del partido es la distancia de ranking con el rival, no el ranking "
                   "propio. Durante el partido, en la AVP, separa al ganador la eficacia en ataque: convertir más "
                   "y fallar menos."),
    ])
    multi = html.Div([
        dbc.Row([dbc.Col([html.Label("Equipos", className="control-etq"),
                          dbc.RadioItems(id="m-res", value="Todos", inline=True,
                                         options=[{"label": x, "value": x} for x in ["Todos", "Ganador", "Perdedor"]])],
                         md=6)], className="mb-3"),
        dbc.Row([dbc.Col(tarjeta("Correlación de Spearman entre variables del equipo", grafico("m-spear", 520)), lg=7),
                 dbc.Col(tarjeta("Redundancia entre variables previas al partido (VIF)", [
                     dbc.Table.from_dataframe(_vif.assign(VIF=_vif.VIF.map(lambda v: fmt(v))),
                                              striped=True, hover=True, size="sm", className="tabla"),
                     html.Div(f"Calculado con los {fmt(_n_vif, 0)} partidos que tienen ranking, edad y estatura "
                              "de las dos parejas. VIF ≤ 5: no hay redundancia.", className="nota")], fuente=False), lg=5)]),
        entrenador("Las variables que se conocen antes del partido no se repiten entre sí: ranking, edad y estatura "
                   "pueden mirarse juntas. Los partidos se alargan cuando la pareja que pierde es buena."),
    ])
    return html.Div([
        encabezado("Análisis exploratorio", "Explore las variables una por una, compare ganadores con perdedores "
                   "y vea qué variables se mueven juntas. Los filtros de la izquierda aplican a todas las pestañas."),
        dbc.Tabs([dbc.Tab(uni, label="Univariado", tab_id="uni"),
                  dbc.Tab(bi, label="Bivariado", tab_id="bi"),
                  dbc.Tab(multi, label="Multivariado", tab_id="multi")],
                 active_tab="uni", className="pestanas mb-4"),
    ])


ANCHO = {"edad": 1, "estatura": 2.54, "ranking": 1, "duracion": 2, "puntos": 2, "kills": 1,
         "aces": 1, "blocks": 1, "digs": 1, "hitpct": 0.02}


@app.callback(Output("u-kpis", "children"), Output("u-fig", "figure"), Output("u-sub", "children"),
              Input("u-var", "value"), Input("f-circuito", "value"), Input("f-genero", "value"),
              Input("f-anios", "value"))
def actualizar_uni(clave, circuito, genero, anios):
    d, col = D.serie_uni(clave, circuito, genero, anios)
    _, _, etq, unidad, solo_stats = D.VARS_UNI[clave]
    s = d[col].dropna()
    if len(s) < 5:
        return [], vacia(), ""
    est = D.describir(s)
    dec = 3 if clave == "hitpct" else 1
    kpis = [kpi(fmt(est["n"], 0), "Observaciones"), kpi(fmt(est["Media"], dec), "Media"),
            kpi(fmt(est["Mediana"], dec), "Mediana"), kpi(fmt(est["DE"], dec), "Desv. estándar"),
            kpi(f"{fmt(est['P25'], dec)} – {fmt(est['P75'], dec)}", "50 % central (P25–P75)"),
            kpi(fmt(est["Asimetría"]), "Asimetría", "> 1: cola larga a la derecha" if est["Asimetría"] > 1 else None)]

    fig = go.Figure()
    if clave == "sets":
        v = s.value_counts().sort_index()
        fig.add_bar(x=[f"{int(k)} set" + ("" if k == 1 else "s") for k in v.index], y=v.values, marker_color=NEUTRO,
                    text=[fmt_pct(100 * x / v.sum()) for x in v.values], textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}<br>%{y:,} partidos<extra></extra>")
    elif clave == "estatura" and genero == "Todos":
        for g in ["M", "W"]:
            histograma(fig, d.loc[d.gender == g, col], ANCHO[clave], PAL[g], GENERO[g], 0.65)
        fig.update_layout(barmode="overlay")
    else:
        histograma(fig, s, ANCHO[clave], NEUTRO)
        fig.add_vline(x=est["Mediana"], line_dash="dash", line_color=GRIS,
                      annotation_text=f"Mediana: {fmt(est['Mediana'], dec)}", annotation_position="top right")
    fig.update_layout(xaxis_title=f"{etq} ({unidad})", yaxis=dict(title="Frecuencia", tickformat=",d"), bargap=0)
    sub = "Equipos con estadísticas completas (casi todos de la AVP)" if solo_stats else \
        ("Partidos jugados (sin forfeits)" if D.VARS_UNI[clave][0] == "P" else
         "Una fila por jugador en cada partido" if D.VARS_UNI[clave][0] == "J" else "Una fila por pareja en cada partido")
    return kpis, fig, sub


@app.callback(Output("b-kpis", "children"), Output("b-caja", "figure"), Output("b-sub", "children"),
              Output("b-poder", "figure"), Output("b-fav", "figure"), Output("b-upset", "figure"),
              Output("b-chi", "children"),
              Input("b-var", "value"), Input("f-circuito", "value"), Input("f-genero", "value"),
              Input("f-anios", "value"))
def actualizar_bi(var, circuito, genero, anios):
    etq, unidad, solo_stats = D.VARS_BI[var]
    e = D.filtrar(D.E, circuito, genero, anios)
    d = D.solo_avp(e) if solo_stats else e
    r = D.comparar(d, var)
    if r is None:
        msg = "No hay estadísticas de juego con estos filtros (solo existen para la AVP)" if solo_stats else None
        fig_caja = vacia(msg) if msg else vacia()
        kpis = []
    else:
        dec = 3 if var == "hitpct" else 0 if var == "rank_main" else 1
        kpis = [kpi(fmt(r["med_g"], dec), "Mediana ganadores"), kpi(fmt(r["med_p"], dec), "Mediana perdedores"),
                kpi(fmt(r["rbc"]), "Correlación rango-biserial"),
                kpi(html.Span(r["poder"], className=f"poder poder-{r['poder'].lower()}"), "Poder discriminante"),
                kpi(fmt(r["solap"]), "Solapamiento IQR"), kpi(fmt_p(r["p"]), "p-valor (Mann-Whitney)")]
        fig_caja = go.Figure()
        partir = var == "estatura_media_cm" and genero == "Todos"
        for res in ["Ganador", "Perdedor"]:
            dd = d[d.resultado == res]
            if partir:
                for g in ["M", "W"]:
                    caja(fig_caja, dd.loc[dd.gender == g, var], GENERO[g], PAL[res], res, g == "M")
            else:
                caja(fig_caja, dd[var], res, PAL[res], res, False)
        fig_caja.update_layout(boxmode="group", yaxis_title=f"{etq} ({unidad})")
    sub = ("Solo AVP con estadísticas completas" if solo_stats else "Una fila por pareja en cada partido") + \
          (" · la estatura se separa por género" if var == "estatura_media_cm" and genero == "Todos" else "")

    t = D.tabla_poder(circuito, genero, anios)
    if t.empty:
        fp = vacia()
    else:
        t = t.assign(a=t.rbc.abs()).sort_values("a")
        fp = go.Figure(go.Bar(x=t.a, y=t.Variable, orientation="h", marker_color=[PAL_PODER[x] for x in t.poder],
                              customdata=np.stack([t.rbc, t.poder, t.n], axis=1),
                              hovertemplate="%{y}<br>RBC: %{customdata[0]:.3f}<br>Poder: %{customdata[1]}"
                                            "<br>n: %{customdata[2]:,}<extra></extra>"))
        for x in [0.3, 0.5]:
            fp.add_vline(x=x, line_dash="dot", line_color=GRIS)
        fp.update_layout(xaxis=dict(range=[0, 1], title="|RBC|"), margin=dict(l=10, r=20, t=20, b=10))

    p = D.filtrar(D.P, circuito, genero, anios)
    r_ = p[p.rank_diff.notna()]
    if len(r_) < 30:
        ff, fu, chi = vacia(), vacia(), html.Div("No hay suficientes partidos con estos filtros.")
    else:
        etqs = ["1–2", "3–5", "6–10", "11–20", "21 o más"]
        g = pd.cut(r_.rank_diff.abs(), [0, 2, 5, 10, 20, np.inf], labels=etqs)
        fav = (r_.assign(g=g, f=r_.rank_diff > 0).groupby("g", observed=True)
               .agg(pct=("f", "mean"), n=("f", "size")).reset_index())
        fav["ic"] = 196 * np.sqrt(fav.pct * (1 - fav.pct) / fav.n)
        fav["pct"] = 100 * fav.pct
        ff = go.Figure(go.Scatter(x=fav.g, y=fav.pct, mode="lines+markers+text", text=[fmt_pct(v) for v in fav.pct],
                                  textposition="top center", line=dict(color=PAL["Ganador"], width=2),
                                  marker=dict(size=9), error_y=dict(type="data", array=fav.ic, color=PAL["Ganador"]),
                                  customdata=fav.n,
                                  hovertemplate="Diferencia: %{x}<br>Gana el favorito: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>"))
        ff.add_hline(y=50, line_dash="dash", line_color=GRIS, annotation_text="50 %: moneda al aire",
                     annotation_position="bottom right")
        ff.update_layout(yaxis=dict(range=[40, 100], ticksuffix=" %"), xaxis_title="Diferencia de ranking (posiciones)")

        ue = (r_.groupby(["etapa", "circuit"], observed=True)
              .agg(pct=("es_upset", "mean"), n=("es_upset", "size")).reset_index())
        ue = ue[ue.n >= 30]
        fu = go.Figure()
        for c in ["FIVB", "AVP"]:
            s = ue[ue.circuit == c].set_index("etapa").reindex(ETAPAS).reset_index()
            if s.n.notna().any():
                fu.add_bar(x=s.etapa, y=100 * s.pct, name=c, marker_color=PAL[c], customdata=s.n,
                           text=[fmt_pct(100 * v) if pd.notna(v) else "" for v in s.pct], textposition="outside",
                           cliponaxis=False, textfont=dict(size=11),
                           hovertemplate="%{x} · " + c + "<br>Sorpresas: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>")
        fu.update_layout(barmode="group", yaxis=dict(ticksuffix=" %"), bargap=0.3, uniformtext=dict(minsize=10, mode="show"))

        cruces = [("Sorpresa × circuito", r_.es_upset, r_.circuit), ("Sorpresa × género", r_.es_upset, r_.gender),
                  ("Sorpresa × etapa", r_.es_upset, r_.etapa),
                  ("Resultado × viene de clasificatoria", e.resultado, e.es_clasif)]
        filas = []
        for nombre, x, y in cruces:
            cv = D.chi_v(x, y)
            if cv:
                mag = "Despreciable" if cv["v"] < 0.1 else "Pequeña" if cv["v"] < 0.3 else "Media" if cv["v"] < 0.5 else "Grande"
                filas.append({"Cruce": nombre, "V de Cramér": fmt(cv["v"], 3), "p-valor": fmt_p(cv["p"]), "Magnitud": mag})
        chi = dbc.Table.from_dataframe(pd.DataFrame(filas), striped=True, hover=True, size="sm", className="tabla") \
            if filas else html.Div("Con estos filtros no hay cruces que comparar (cada variable queda con una sola categoría).")
    return kpis, fig_caja, sub, fp, ff, fu, chi


VARS_M = ["rank_main", "rank_rival", "edad_media", "estatura_media_cm", "dif_edad_pareja",
          "dif_estatura_pareja_cm", "duracion_min"]
ETQ_M = ["Ranking propio", "Ranking rival", "Edad media", "Estatura media", "Dif. edad pareja",
         "Dif. estatura pareja", "Duración"]


@app.callback(Output("m-spear", "figure"), Input("m-res", "value"), Input("f-circuito", "value"),
              Input("f-genero", "value"), Input("f-anios", "value"))
def actualizar_multi(res, circuito, genero, anios):
    d = D.filtrar(D.E, circuito, genero, anios)
    if res != "Todos":
        d = d[d.resultado == res]
    if len(d) < 30:
        return vacia()
    rho = d[VARS_M].corr(method="spearman").values.copy()
    rho[np.triu_indices_from(rho, k=1)] = np.nan
    fig = go.Figure(go.Heatmap(x=ETQ_M, y=ETQ_M, z=rho, colorscale=ESCALA_CORR, zmin=-1, zmax=1,
                               texttemplate="%{z:.2f}", colorbar=dict(title="rho"), xgap=2, ygap=2,
                               hovertemplate="%{y} – %{x}<br>rho = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-30)
    return fig


# ==========================================================================
# Página: Calculadora del entrenador
# ==========================================================================
def pagina_calculadora():
    def sel(id_, valor, ops):
        return dbc.Select(id=id_, value=valor, options=[{"label": l, "value": v} for v, l in ops])
    controles = tarjeta("Su partido", [
        dbc.Row([dbc.Col([html.Label("Ranking de su pareja", className="control-etq"),
                          dbc.Input(id="c-propio", type="number", min=1, max=64, step=1, value=8)]),
                 dbc.Col([html.Label("Ranking del rival", className="control-etq"),
                          dbc.Input(id="c-rival", type="number", min=1, max=64, step=1, value=12)])], className="mb-3"),
        html.Label("Circuito", className="control-etq"),
        sel("c-circuito", "Todos", [("Todos", "Todos"), ("FIVB", "FIVB"), ("AVP", "AVP")]),
        html.Label("Género", className="control-etq mt-3"),
        sel("c-genero", "Todos", [("Todos", "Todos"), ("M", "Hombres"), ("W", "Mujeres")]),
        html.Label("Etapa del torneo", className="control-etq mt-3"),
        sel("c-etapa", "Todas", [("Todas", "Todas")] + [(e, e) for e in ETAPAS]),
        html.Label("Diferencia de estatura media con el rival (cm)", className="control-etq mt-3"),
        dcc.Slider(id="c-estatura", min=-20, max=20, step=1, value=0,
                   marks={v: f"{v:+d}" if v else "0" for v in range(-20, 21, 10)},
                   tooltip={"placement": "top", "always_visible": False}),
        html.Div(id="c-estatura-txt", className="nota"),
    ], fuente=False)
    return html.Div([
        encabezado("Calculadora del entrenador", "Ingrese el ranking de su pareja y el del rival. La calculadora "
                   "busca los partidos históricos con una ventaja parecida y le muestra cuántos ganó la pareja "
                   "en su posición."),
        dbc.Row([
            dbc.Col(controles, lg=4),
            dbc.Col([
                html.Div(id="c-resultado", className="resultado mb-4"),
                tarjeta("% de victorias según la ventaja de ranking sobre el rival", grafico("c-curva", 330),
                        "El punto marca su partido · ventana móvil de ±2 posiciones"),
                tarjeta("% de victorias según la ventaja de estatura", grafico("c-est", 280),
                        "Mismos filtros de circuito, género y etapa · el punto marca su pareja"),
            ], lg=8),
        ]),
        html.Div("Son frecuencias históricas de 2000 a 2019, no una predicción: describen qué pasó en partidos "
                 "parecidos, no garantizan el resultado de uno nuevo.", className="nota mb-4"),
    ])


@app.callback(Output("c-estatura-txt", "children"), Input("c-estatura", "value"))
def texto_estatura(v):
    if not v:
        return "Seleccionado: misma estatura media que la pareja rival."
    return f"Seleccionado: su pareja es {abs(v)} cm más {'alta' if v > 0 else 'baja'} que la rival."


@app.callback(Output("c-resultado", "children"), Output("c-curva", "figure"), Output("c-est", "figure"),
              Input("c-propio", "value"), Input("c-rival", "value"), Input("c-circuito", "value"),
              Input("c-genero", "value"), Input("c-etapa", "value"), Input("c-estatura", "value"))
def calcular(propio, rival, circuito, genero, etapa, estatura):
    if not propio or not rival:
        return dbc.Alert("Ingrese los dos rankings (de 1 a 64).", color="warning"), vacia(), vacia()
    propio, rival = int(propio), int(rival)
    v = rival - propio                              # positivo: su pareja está mejor rankeada
    d = D.filtrar(D.E, circuito, genero, D.ANIOS, etapa)
    dr = d[d.ventaja_rank.notna()]
    if len(dr) < 30:
        return dbc.Alert("Hay muy pocos partidos con esos filtros. Pruebe con 'Todos'.", color="warning"), vacia(), vacia()

    ancho = 0 if abs(v) <= 2 else 1 if abs(v) <= 6 else 2 if abs(v) <= 15 else 4
    while True:
        sub = dr[dr.ventaja_rank.between(v - ancho, v + ancho)]
        if len(sub) >= 50 or ancho >= 10:
            break
        ancho += 1
    if len(sub) < 10:
        return dbc.Alert("No hay partidos históricos con esa diferencia de ranking y esos filtros.",
                         color="warning"), vacia(), vacia()
    p = (sub.resultado == "Ganador").mean()
    ic = 1.96 * np.sqrt(p * (1 - p) / len(sub))
    pct = 100 * p
    if pct >= 75:
        lectura, clase = "Ventaja clara: su pareja gana la gran mayoría de estos partidos.", "fuerte"
    elif pct >= 55:
        lectura, clase = "Ventaja real pero frágil: uno de cada tres o cuatro partidos así termina en sorpresa.", "media"
    elif pct > 45:
        lectura, clase = "Partido abierto: históricamente es casi una moneda al aire.", "abierto"
    else:
        lectura, clase = "Su pareja parte en desventaja, pero estos partidos también se ganan.", "baja"
    rango = "exactamente" if ancho == 0 else f"entre {v - ancho:+d} y {v + ancho:+d}"
    resultado = html.Div([
        html.Div("Probabilidad histórica de ganar", className="resultado-etq"),
        html.Div(fmt_pct(pct), className=f"resultado-valor resultado-{clase}"),
        dbc.Progress(value=pct, className="resultado-barra mb-2", color=None,
                     style={"--bs-progress-bar-bg": PAL["Ganador"]}),
        html.Div(lectura, className="resultado-lectura"),
        html.Div(f"Basado en {fmt(len(sub), 0)} partidos en que una pareja tenía una ventaja de ranking {rango} "
                 f"posiciones (la suya es {v:+d}). Intervalo de confianza del 95 %: "
                 f"{fmt_pct(max(0, pct - 100 * ic))} a {fmt_pct(min(100, pct + 100 * ic))}.", className="resultado-detalle"),
    ])

    cur = D.curva_victoria(dr, "ventaja_rank", 2)
    cur = cur[(cur.index >= -40) & (cur.index <= 40) & (cur.n >= 100)]
    f1 = go.Figure()
    f1.add_scatter(x=cur.index, y=cur.pct, mode="lines", line=dict(color=PAL["Ganador"], width=2), name="Histórico",
                   customdata=cur.n, hovertemplate="Ventaja %{x:+d}<br>Victorias: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>")
    f1.add_hline(y=50, line_dash="dash", line_color=GRIS)
    f1.add_vline(x=0, line_color="#D1D5DB")
    f1.add_scatter(x=[v], y=[pct], mode="markers", name="Su partido", showlegend=False,
                   marker=dict(size=14, color=PAL["Ganador"], line=dict(color="white", width=3)),
                   hovertemplate="Su partido<br>Ventaja %{x:+d}<br>%{y:.1f} %<extra></extra>")
    f1.update_layout(showlegend=False, yaxis=dict(range=[0, 100], ticksuffix=" %", title="% de victorias"),
                     xaxis_title="Ventaja de ranking (ranking del rival − ranking propio)")

    de = d[d.ventaja_estatura_cm.notna()]
    cortes = [-np.inf, -10, -5, -2, 2, 5, 10, np.inf]
    etqs = ["≤ −10", "−10 a −5", "−5 a −2", "−2 a 2", "2 a 5", "5 a 10", "≥ 10"]
    g = pd.cut(de.ventaja_estatura_cm, cortes, labels=etqs, right=False)
    t = de.assign(g=g, w=de.resultado == "Ganador").groupby("g", observed=True).agg(pct=("w", "mean"), n=("w", "size"))
    t["pct"] *= 100
    suyo = etqs[int(np.searchsorted(cortes[1:-1], estatura, side="right"))]
    f2 = go.Figure(go.Bar(x=t.index.astype(str), y=t.pct, customdata=t.n,
                          marker_color=[PAL["Ganador"] if k == suyo else "#C9D6E8" for k in t.index],
                          text=[fmt_pct(x) for x in t.pct], textposition="outside", cliponaxis=False,
                          hovertemplate="Ventaja %{x} cm<br>Victorias: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>"))
    f2.add_hline(y=50, line_dash="dash", line_color=GRIS, layer="below")
    f2.update_layout(yaxis=dict(range=[0, 80], ticksuffix=" %"), xaxis_title="Ventaja de estatura sobre el rival (cm)")
    return resultado, f1, f2


# ==========================================================================
# Página: Conclusiones
# ==========================================================================
def pagina_conclusiones():
    rho = D.E[VARS_M].corr(method="spearman").values
    rho_max = np.abs(rho[np.tril_indices_from(rho, k=-1)]).max()
    cv = D.chi_v(D.E.resultado, D.E.es_clasif)["v"]
    pct_fivb = 100 * (D.P.circuit == "FIVB").mean()
    datos_ = [
        ("El circuito define los datos", f"La FIVB tiene el {fmt_pct(pct_fivb, 2)} de los partidos, pero casi solo la AVP registra estadísticas de juego."),
        ("El faltante no es aleatorio", f"Solo el {fmt_pct(C['pct_stats'])} de los partidos tiene estadísticas completas; describen a la AVP de 2003 a 2019."),
        ("Los atípicos son legítimos", f"Jugadores muy altos, veteranos o partidos largos son reales. Solo se anularon {fmt(D.RES['p8_celdas_invalidas'], 0)} celdas imposibles."),
        ("Cobertura desigual en el tiempo", "La AVP casi desaparece en 2011–2012 por su crisis; 2000 y 2019 son años parciales."),
    ]
    ganar = [
        ("Una de cada tres es sorpresa", f"El mejor rankeado gana el {fmt_pct(C['pct_fav'])} de los {fmt(C['n_rank'], 0)} partidos con ranking."),
        ("Importa la distancia con el rival", f"El favorito gana el {fmt_pct(C['fav_cerca'])} con 1–2 posiciones de diferencia y el {fmt_pct(C['fav_lejos'])} con más de 20."),
        ("Edad y estatura pesan poco", f"Poder discriminante bajo. 10 cm o más de ventaja de estatura llevan la tasa de victoria al {fmt_pct(C['est10'])}."),
        ("Venir de clasificatoria pesa poco", f"Esas parejas pierden el {fmt_pct(_perd[1])} frente al {fmt_pct(_perd[0])}, con asociación despreciable (V = {fmt(cv, 3)})."),
        ("En la AVP gana la eficacia en ataque", f"% de ataque con poder alto (RBC = {fmt(_poder.rbc['hitpct'])}); le siguen errores de ataque, bloqueos, kills y aces."),
        ("Las variables no se repiten", f"Correlaciones de Spearman ≤ {fmt(rho_max)} y VIF ≤ {fmt(_vif.VIF.max())}."),
    ]

    def lista(items, inicio):
        return html.Div([html.Div([html.Div(str(i), className="num"),
                                   html.Div([html.Strong(t), html.Div(x, className="texto")])], className="hallazgo-item")
                         for i, (t, x) in enumerate(items, inicio)])
    recs = [("bi-trophy", "Preparar un partido", "Use la diferencia de ranking con el rival (calculadora). Con 1–2 posiciones el partido está abierto; solo con más de 20 el favorito es claro."),
            ("bi-stopwatch", "Exigencia física", f"Un partido típico dura {fmt(C['dur_med'], 0)} minutos, pero uno de cada tres se va a tercer set y se acerca a {fmt(C['dur_3'], 0)}."),
            ("bi-people", "Armar parejas", "Estatura y edad inclinan poco la balanza; no deben pesar más que el nivel reflejado en el ranking."),
            ("bi-bullseye", "Priorizar el entrenamiento", "Referencia AVP: convertir los ataques y reducir errores separa más al ganador que atacar más veces.")]
    return html.Div([
        encabezado("Conclusiones", "Lo que encontró el análisis y cómo traducirlo en decisiones."),
        html.Div([html.Div("Respuesta a la pregunta", className="hero-eyebrow"),
                  html.P("Antes del partido, el factor medible que más se asocia con ganar es la ventaja de ranking sobre "
                         "el rival; la estatura, la edad y el contexto aportan muy poco en comparación. Durante el partido, "
                         "al menos en la AVP, lo que separa al ganador es la eficacia en ataque.", className="respuesta")],
                 className="hero hero-compacto mb-4"),
        dbc.Row([dbc.Col(tarjeta("Sobre los datos", lista(datos_, 1), fuente=False), lg=5),
                 dbc.Col(tarjeta("Sobre qué hace ganar a una pareja", lista(ganar, 5), fuente=False), lg=7)]),
        html.H5("Recomendaciones para el entrenador", className="seccion-titulo"),
        dbc.Row([dbc.Col(html.Div([html.I(className=f"bi {ico} rec-icono"), html.H6(t), html.P(x)], className="rec"), md=6, lg=3)
                 for ico, t, x in recs], className="mb-4"),
        dbc.Row([
            dbc.Col(tarjeta("Qué no se puede concluir", html.Ul([
                html.Li("Que las estadísticas de juego describan a la FIVB."),
                html.Li("Que un factor cause la victoria: todos los resultados son asociaciones."),
                html.Li("Que las relaciones se mantengan iguales hoy: el análisis agrupa veinte años.")]), fuente=False), lg=6),
            dbc.Col(tarjeta("Qué datos harían falta", html.Ul([
                html.Li("Estadísticas de juego de la FIVB."),
                html.Li("Ranking numérico de las parejas de clasificatoria."),
                html.Li("Contexto: viento, lesiones, tiempo de la pareja junta, historial entre rivales.")]), fuente=False), lg=6),
        ]),
    ])


# ==========================================================================
# Navegación entre páginas
# ==========================================================================
RUTAS = {"/": pagina_inicio, "/metodologia": pagina_metodologia, "/datos": pagina_datos,
         "/exploratorio": pagina_exploratorio, "/calculadora": pagina_calculadora,
         "/conclusiones": pagina_conclusiones}


@app.callback(Output("contenido", "children"), Input("url", "pathname"))
def mostrar_pagina(ruta):
    return RUTAS.get(ruta, pagina_inicio)()


if __name__ == "__main__":
    # Puerto opcional: python app.py 8051 (por si el 8050 está ocupado)
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print(f"\nDashboard listo. Ábralo en el navegador: http://127.0.0.1:{puerto}\n(Para detenerlo: Ctrl + C)\n")
    app.run(debug=False, port=puerto)