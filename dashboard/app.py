"""Dashboard interactivo: Voleibol de playa, ¿qué hace ganar a una pareja?

Correr con:  python app.py   y abrir http://127.0.0.1:8050 en el navegador.

Cinco páginas: Inicio, Dashboard, Análisis, Predicción y Equipo. Los modelos de la
página Predicción y los perfiles de juego se entrenan aparte, con
entrenar_modelos.py; aquí solo se leen sus resultados guardados.
"""
import sys
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, dcc, html

import datos as D
import prediccion as M
from estilo import (ARENA, CORAL, ESCALA_CORR, ESCALA_SEC, ETAPAS, FUENTE, GENERO, GRIS, MAR, NEUTRO,
                    PAL, PAL_ETAPA, PAL_MODELO, PAL_PERFIL, PAL_PODER, TURQUESA, fmt, fmt_p, fmt_pct,
                    vacia)

LIBRO = "https://nicoleramirez2106-ui.github.io/dataviz-2026/"
REPO = "https://github.com/nicoleramirez2106-ui/dataviz-2026"

# Bootstrap e íconos: copia local en assets/vendor (funciona sin internet) o, si no está, la CDN
VENDOR = Path(__file__).resolve().parent / "assets" / "vendor"
if (VENDOR / "bootstrap.min.css").exists():
    HOJAS = ["/assets/vendor/bootstrap.min.css", "/assets/vendor/bootstrap-icons.min.css"]
else:
    HOJAS = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP,
             "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"]

app = Dash(__name__, title="Voleibol de playa · Dashboard", external_stylesheets=HOJAS,
           assets_ignore=r"^bootstrap.*\.css$", suppress_callback_exceptions=True,
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# ==========================================================================
# Piezas reutilizables
# ==========================================================================
def tarjeta(titulo, cuerpo, subtitulo=None, fuente=True, clase=""):
    return dbc.Card([
        dbc.CardHeader([html.H6(titulo, className="tarjeta-titulo"),
                        html.Div(subtitulo, className="tarjeta-subtitulo") if subtitulo is not None else None]),
        dbc.CardBody(cuerpo),
        html.Div(FUENTE, className="tarjeta-fuente") if fuente else None,
    ], className=f"tarjeta mb-4 {clase}")


def kpi(valor, etiqueta, nota=None, id_=None, tono=""):
    """Casilla de marcador: número grande con dígitos tabulares."""
    extra = {"id": id_} if id_ else {}
    return html.Div([html.Div(etiqueta, className="kpi-etiqueta"),
                     html.Div(valor, className="kpi-valor", **extra),
                     html.Div(nota, className="kpi-nota") if nota else None], className=f"kpi {tono}")


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


def encabezado(titulo, texto, saque=None):
    return html.Div([html.Div(saque, className="pagina-saque") if saque else None,
                     html.H2(titulo, className="pagina-titulo"),
                     html.P(texto, className="pagina-intro")], className="mb-4")


def chips(id_, valor, ops):
    """Botones de filtro rápidos (un clic, sin desplegables)."""
    return dbc.RadioItems(id=id_, value=valor, inline=True, className="chips",
                          options=[{"label": l, "value": v} for v, l in ops],
                          inputClassName="btn-check", labelClassName="chip",
                          labelCheckedClassName="chip-activo")


def sin_modelos():
    return dbc.Alert([html.I(className="bi bi-exclamation-triangle me-2"),
                      "Todavía no hay resultados guardados de los modelos. Desde la carpeta del repositorio, corra ",
                      html.Code("python dashboard/entrenar_modelos.py"), " y vuelva a abrir el dashboard."],
                     color="warning")


# ==========================================================================
# Estructura: riel de navegación (línea de banda), marcador y filtros rápidos
# ==========================================================================
PAGINAS = [("/", "Inicio", "bi-house-door"),
           ("/dashboard", "Dashboard", "bi-grid-1x2"),
           ("/analisis", "Análisis", "bi-graph-up"),
           ("/prediccion", "Predicción", "bi-bullseye"),
           ("/equipo", "Equipo", "bi-people")]

riel = html.Nav([
    dcc.Link([html.Span(className="balon", **{"aria-hidden": "true"}),
              html.Span("VP", className="riel-sigla")], href="/", className="riel-marca", title="Inicio"),
    dbc.Nav([dbc.NavLink([html.I(className=f"bi {ico}"), html.Span(txt)], href=ruta, active="exact",
                         className="riel-item", id=f"nav-{i}")
             for i, (ruta, txt, ico) in enumerate(PAGINAS)], vertical=True, className="riel-menu"),
    html.Div([html.A(html.I(className="bi bi-book"), href=LIBRO, target="_blank", title="Leer el libro"),
              html.A(html.I(className="bi bi-github"), href=REPO, target="_blank", title="Repositorio")],
             className="riel-enlaces"),
], className="riel", **{"aria-label": "Navegación principal"})

marcador = html.Header([
    html.Div([html.Div("Voleibol de playa", className="marcador-titulo"),
              html.Div("¿Qué hace ganar a una pareja?", className="marcador-sub")], className="marcador-marca"),
    html.Div(id="marcador-tablero", className="tablero", **{"aria-live": "polite"}),
], className="marcador")

filtros = html.Section([
    html.Div([html.Span("Circuito", className="filtro-etq"),
              chips("f-circuito", "Todos", [("Todos", "Todos"), ("FIVB", "FIVB"), ("AVP", "AVP")])],
             className="filtro-grupo"),
    html.Div([html.Span("Género", className="filtro-etq"),
              chips("f-genero", "Todos", [("Todos", "Todos"), ("M", "Hombres"), ("W", "Mujeres")])],
             className="filtro-grupo"),
    html.Div([html.Span("Etapa", className="filtro-etq"),
              chips("f-etapa", "Todas", [("Todas", "Todas")] + [(e, e) for e in ETAPAS])],
             className="filtro-grupo"),
    html.Div([html.Span("Años", className="filtro-etq"),
              dcc.RangeSlider(id="f-anios", min=D.ANIOS[0], max=D.ANIOS[1], step=1, value=list(D.ANIOS),
                              marks={a: str(a) for a in range(2000, 2020, 5)} | {2019: "2019"},
                              tooltip={"placement": "bottom", "always_visible": False})],
             className="filtro-grupo filtro-anios"),
    html.Button([html.I(className="bi bi-arrow-counterclockwise me-1"), "Restablecer"], id="f-reset",
                className="filtro-reset", n_clicks=0),
], id="filtros", className="filtros-banda")

app.layout = html.Div([
    dcc.Location(id="url"), riel,
    html.Div([marcador, filtros, html.Main(id="contenido", className="contenido")], className="principal"),
], className="app")

FILTROS = [Input("f-circuito", "value"), Input("f-genero", "value"), Input("f-anios", "value"),
           Input("f-etapa", "value")]


@app.callback(Output("f-circuito", "value"), Output("f-genero", "value"), Output("f-anios", "value"),
              Output("f-etapa", "value"), Input("f-reset", "n_clicks"), prevent_initial_call=True)
def restablecer(_):
    return "Todos", "Todos", list(D.ANIOS), "Todas"


@app.callback(Output("marcador-tablero", "children"), *FILTROS)
def actualizar_marcador(circuito, genero, anios, etapa):
    """Marcador superior: FIVB contra AVP y cuántas veces gana el favorito con los filtros actuales."""
    d = D.filtrar(D.P, circuito, genero, anios, etapa)
    n = d.circuit.astype(str).value_counts()
    r = d[d.rank_diff.notna()]
    fav = fmt_pct(100 * (1 - r.es_upset.mean()), 1) if len(r) else "—"
    return [html.Div([html.Span("FIVB", className="equipo equipo-fivb"),
                      html.Span(fmt(n.get("FIVB", 0), 0), className="digito"),
                      html.Span(":", className="dos-puntos"),
                      html.Span(fmt(n.get("AVP", 0), 0), className="digito"),
                      html.Span("AVP", className="equipo equipo-avp")], className="tablero-score",
                     title="Partidos con los filtros actuales"),
            html.Div([html.Span("Gana el favorito", className="tablero-etq"),
                      html.Span(fav, className="digito digito-chico")], className="tablero-dato"),
            html.Div([html.Span("Periodo", className="tablero-etq"),
                      html.Span(f"{anios[0]}–{anios[1]}", className="digito digito-chico")],
                     className="tablero-dato")]


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
_n_circ = D.P.circuit.astype(str).value_counts()


# ==========================================================================
# Página: Inicio (una cancha: pregunta de un lado de la red, marcador del otro)
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
    jugadas = [("/dashboard", "bi-grid-1x2", "Dashboard",
                "Partidos por año, circuito y etapa, sorpresas y cobertura de datos con filtros rápidos."),
               ("/analisis", "bi-graph-up", "Análisis",
                "Univariado, bivariado, multivariado, estilos de juego de la AVP, metodología y conclusiones."),
               ("/prediccion", "bi-bullseye", "Predicción",
                "Modelos previos al partido, simulador ¿Quién gana? y la calculadora histórica.")]
    return html.Div([
        html.Section([
            html.Div([
                html.Div("Visualización de Datos y Toma de Decisiones · Universidad del Norte · 2026",
                         className="hero-eyebrow"),
                html.H1("¿Qué hace ganar a una pareja de voleibol de playa?", className="hero-titulo"),
                html.P(f"{fmt(C['n_partidos'], 0)} partidos de los circuitos FIVB y AVP entre 2000 y 2019, "
                       "analizados para que un entrenador sepa qué tan sólida es una ventaja antes del "
                       "partido, cómo armar parejas y qué priorizar en el entrenamiento.", className="hero-texto"),
                html.Div("Nicolle Ramírez y Roger Daza", className="hero-autores"),
                html.Div([dcc.Link([html.I(className="bi bi-bullseye me-2"), "¿Quién gana? Simular un partido"],
                                   href="/prediccion", className="boton-principal"),
                          html.A([html.I(className="bi bi-book me-2"), "Leer el libro completo"],
                                 href=LIBRO, target="_blank", className="boton-secundario")],
                         className="hero-botones"),
            ], className="cancha-lado cancha-izq"),
            html.Div(html.Div(className="red-malla"), className="red", **{"aria-hidden": "true"}),
            html.Div([
                html.Div("Marcador 2000–2019", className="cancha-marcador-titulo"),
                html.Div([kpi(fmt_pct(C["pct_fav"]), "Gana el favorito", tono="tono-mar"),
                          kpi(fmt_pct(100 - C["pct_fav"]), "Sorpresas", tono="tono-coral"),
                          kpi(fmt(_n_circ.get("FIVB", 0), 0), "Partidos FIVB", tono="tono-turquesa"),
                          kpi(fmt(_n_circ.get("AVP", 0), 0), "Partidos AVP", tono="tono-arena")],
                         className="cancha-marcador"),
                html.Div(f"Favorito y sorpresa se calculan sobre los {fmt(C['n_rank'], 0)} partidos con "
                         "ranking de las dos parejas.", className="nota nota-clara"),
            ], className="cancha-lado cancha-der"),
        ], className="cancha mb-4"),
        html.Div([kpi(fmt(C["n_partidos"], 0), "Partidos"),
                  kpi(fmt(C["n_jugadores"], 0), "Jugadores distintos"),
                  kpi(fmt(C["n_paises"], 0), "Países sede"),
                  kpi(fmt(C["n_rank"], 0), "Partidos con ranking de ambas parejas"),
                  kpi(fmt_pct(C["pct_stats"]), "Partidos con estadísticas de juego", "casi todos de la AVP")],
                 className="kpis mb-4"),
        html.H5("Elija su jugada", className="seccion-titulo"),
        html.Div([dcc.Link([html.I(className=f"bi {ico} jugada-icono"), html.H6(t), html.P(x),
                            html.Span(["Ir ", html.I(className="bi bi-arrow-right")], className="jugada-ir")],
                           href=ruta, className="jugada")
                  for ruta, ico, t, x in jugadas] +
                 [html.Div([html.I(className="bi bi-play-btn jugada-icono"), html.H6("Video del proyecto"),
                            html.Div("Próximamente", className="proximamente")],
                           className="jugada jugada-video", id="video-proximamente")],
                 className="jugadas mb-4"),
        html.H5("Tres hallazgos para empezar", className="seccion-titulo"),
        html.Div([html.Div([html.Div(f"Set {i}", className="set-num"),
                            html.Div(etq, className="hallazgo-etq"),
                            html.Div(val, className="hallazgo-valor"),
                            html.Div(txt, className="hallazgo-texto")], className="hallazgo")
                  for i, (etq, val, txt) in enumerate(hallazgos, 1)], className="sets mb-4"),
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
                                  "¿qué acciones de juego separan al ganador del perdedor?"]),
                         html.Li([html.Strong("Anticipar el resultado: "),
                                  "con lo que se sabe antes del partido, ¿se puede predecir quién gana mejor "
                                  "que con la regla \"gana el mejor rankeado\"?"])])],
                fuente=False), lg=7),
            dbc.Col(tarjeta("Objetivos", html.Ol([
                html.Li("Depurar y documentar el conjunto de datos."),
                html.Li("Describir la distribución de las variables del partido y de los jugadores."),
                html.Li("Comparar a las parejas ganadoras con las perdedoras."),
                html.Li("Cuantificar cuándo el ranking anticipa el resultado y cuándo no."),
                html.Li("Traducir los hallazgos en recomendaciones para el entrenador.")]), fuente=False), lg=5),
        ]),
    ])


# ==========================================================================
# Página: Dashboard (antes "Datos y limpieza")
# ==========================================================================
def pagina_dashboard():
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
        encabezado("Dashboard", "Panorama de los partidos con los filtros de arriba: un clic en circuito, "
                   "género, etapa o años actualiza el marcador y todos los gráficos.", "Datos y limpieza"),
        html.Div(id="d-kpis", className="kpis mb-4"),
        dbc.Row([dbc.Col(tarjeta("Partidos por año y circuito", grafico("d-anio", 340),
                                 "La caída de la AVP en 2011–2012 coincide con su crisis financiera · "
                                 "haga clic en un año para ver su detalle"), lg=8),
                 dbc.Col(tarjeta("Detalle del año", html.Div(id="d-detalle"), fuente=False), lg=4)]),
        dbc.Row([dbc.Col(tarjeta("Sorpresas por año", grafico("d-sorpresas", 320),
                                 "% de partidos que gana la pareja peor rankeada (años con 30 o más partidos)"), lg=6),
                 dbc.Col(tarjeta("Cobertura de estadísticas de juego por año", grafico("d-cobertura", 320),
                                 "% de partidos con estadísticas completas"), lg=6)]),
        dbc.Row([dbc.Col(tarjeta("Partidos por circuito y género", grafico("d-circuito", 320)), lg=6),
                 dbc.Col(tarjeta("Partidos por etapa del torneo", grafico("d-etapa", 320)), lg=6)]),
        tarjeta("Problemas de calidad encontrados y decisiones", [
            html.Div([kpi(fmt(R["p0_filas_crudas"], 0), "Partidos en el archivo original"),
                      kpi(fmt(R["p7_filas_invalidas"], 0), "Partidos eliminados", "imposibles: mismo jugador en ambos equipos"),
                      kpi(fmt(R["p7_filas_finales"], 0), "Partidos analizados")], className="kpis mb-3"),
            dbc.Table.from_dataframe(problemas, striped=True, hover=True, className="tabla")], fuente=False),
        entrenador("Las estadísticas de juego (kills, aces, bloqueos) solo existen de forma confiable para la AVP. "
                   "Si su pareja compite en la FIVB, use el ranking, la edad, la estatura y la etapa, que están "
                   "disponibles para casi todos los partidos."),
    ])


@app.callback(Output("d-kpis", "children"), Output("d-circuito", "figure"), Output("d-etapa", "figure"),
              Output("d-anio", "figure"), Output("d-cobertura", "figure"), Output("d-sorpresas", "figure"),
              *FILTROS)
def actualizar_datos(circuito, genero, anios, etapa):
    d = D.filtrar(D.P, circuito, genero, anios, etapa)
    if d.empty:
        return [kpi("0", "Partidos con los filtros actuales")], vacia(), vacia(), vacia(), vacia(), vacia()
    r = d[d.rank_diff.notna()]
    jug = D.filtrar(D.J, circuito, genero, anios, etapa).player_id.nunique()
    kpis = [kpi(fmt(len(d), 0), "Partidos con los filtros actuales", id_="d-n"),
            kpi(fmt(jug, 0), "Jugadores distintos"),
            kpi(fmt_pct(100 * (1 - r.es_upset.mean())) if len(r) else "—", "Gana el favorito",
                f"{fmt(len(r), 0)} partidos con ranking de ambas", tono="tono-mar"),
            kpi(fmt_pct(100 * r.es_upset.mean()) if len(r) else "—", "Sorpresas", tono="tono-coral"),
            kpi(f"{fmt(d[d.es_forfeit == 0].duracion_min.median(), 0)} min", "Duración mediana"),
            kpi(fmt_pct(100 * d.tiene_stats_completas.mean()), "Con estadísticas de juego")]

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
            f3.add_bar(x=s.year, y=s.n, name=c, marker_color=PAL[c],
                       hovertemplate="%{x} · " + c + "<br>%{y:,} partidos<extra></extra>")
    f3.update_layout(barmode="stack", yaxis=dict(title="Partidos", tickformat=",d"), hovermode="x unified",
                     clickmode="event")

    cob = d.groupby(["year", "circuit"], observed=True).tiene_stats_completas.mean().reset_index()
    f4 = go.Figure()
    for c in ["FIVB", "AVP"]:
        s = cob[cob.circuit == c]
        if len(s):
            f4.add_scatter(x=s.year, y=100 * s.tiene_stats_completas, name=c, mode="lines+markers",
                           line=dict(color=PAL[c], width=2), marker=dict(size=7),
                           hovertemplate="%{x} · " + c + "<br>%{y:.1f} % con estadísticas<extra></extra>")
    f4.update_layout(yaxis=dict(range=[0, 100], ticksuffix=" %"), hovermode="x unified")

    # Sorpresas por año (capítulo 7 del libro): total en coral, circuitos en su color
    f5 = go.Figure()
    if len(r):
        for c, nombre, color, ancho, guion in [(None, "Todos", CORAL, 3, "solid"),
                                              ("FIVB", "FIVB", PAL["FIVB"], 1.5, "dot"),
                                              ("AVP", "AVP", PAL["AVP"], 1.5, "dot")]:
            s = r if c is None else r[r.circuit == c]
            u = s.groupby("year").es_upset.agg(["mean", "size"])
            u = u[u["size"] >= 30]
            if len(u):
                f5.add_scatter(x=u.index, y=100 * u["mean"], name=nombre, mode="lines+markers",
                               line=dict(color=color, width=ancho, dash=guion), marker=dict(size=6 if c else 8),
                               customdata=u["size"],
                               hovertemplate="%{x} · " + nombre + "<br>Sorpresas: %{y:.1f} %<br>n = %{customdata:,}<extra></extra>")
    f5.update_layout(yaxis=dict(ticksuffix=" %", rangemode="tozero"), hovermode="x unified")
    return kpis, f1, f2, f3, f4, (f5 if f5.data else vacia())


@app.callback(Output("d-detalle", "children"), Input("d-anio", "clickData"), *FILTROS)
def detalle_anio(click, circuito, genero, anios, etapa):
    d = D.filtrar(D.P, circuito, genero, anios, etapa)
    if d.empty:
        return html.Div("No hay partidos con estos filtros.", className="nota")
    anio = int(click["points"][0]["x"]) if click else int(d.year.max())
    y = d[d.year == anio]
    if y.empty:
        return html.Div(f"No hay partidos en {anio} con estos filtros.", className="nota")
    r = y[y.rank_diff.notna()]
    n = y.circuit.astype(str).value_counts()
    paises = y.country.astype(str).value_counts().head(3)
    return html.Div([
        html.Div(str(anio), className="detalle-anio"),
        html.Div([kpi(fmt(len(y), 0), "Partidos"),
                  kpi(fmt(n.get("FIVB", 0), 0), "FIVB", tono="tono-turquesa"),
                  kpi(fmt(n.get("AVP", 0), 0), "AVP", tono="tono-arena"),
                  kpi(fmt_pct(100 * r.es_upset.mean()) if len(r) >= 30 else "—", "Sorpresas", tono="tono-coral")],
                 className="kpis kpis-2 mb-2"),
        html.Div([html.Strong("Países sede con más partidos: "),
                  ", ".join(f"{p} ({fmt(v, 0)})" for p, v in paises.items())], className="nota"),
        html.Div("Mostrando el último año del filtro." if not click else "Año elegido en el gráfico.",
                 className="nota"),
    ])


# ==========================================================================
# Página: Análisis (univariado, bivariado, multivariado, estilos de juego,
# metodología y conclusiones)
# ==========================================================================
def contenido_uni():
    return html.Div([
        dbc.Row([dbc.Col([html.Label("Variable", className="control-etq"),
                          dcc.Dropdown(id="u-var", value="edad", clearable=False,
                                       options=[{"label": v[2], "value": k} for k, v in D.VARS_UNI.items()])],
                         md=5)], className="mb-3"),
        html.Div(id="u-kpis", className="kpis mb-3"),
        tarjeta("Distribución", grafico("u-fig", 400), html.Span(id="u-sub")),
        entrenador("La duración típica de un partido es de unos 42 minutos, pero uno de cada tres se va a "
                   "tercer set y dura cerca de 54: prepare la exigencia física para el partido largo."),
    ])


def contenido_bi():
    return html.Div([
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


def contenido_multi():
    return html.Div([
        dbc.Row([dbc.Col([html.Label("Equipos", className="control-etq"),
                          chips("m-res", "Todos", [(x, x) for x in ["Todos", "Ganador", "Perdedor"]])],
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


def contenido_perfiles():
    """Estilos de juego de la AVP: K-means sobre estadísticas por set (resultados guardados)."""
    if M.PERF is None:
        return sin_modelos()
    PF = M.PERF
    perf = PF["perfiles"]
    etq = PF["etiquetas"]
    sil = PF["silueta"]

    def rasgos(z):
        orden = sorted(z, key=lambda v: -abs(z[v]))[:3]
        return html.Ul([html.Li([html.I(className=f"bi bi-arrow-{'up' if z[v] > 0 else 'down'}-short"),
                                 f"{etq[v].capitalize()}: {fmt(z[v], 2)} DE"]) for v in orden],
                       className="perfil-rasgos")
    tarjetas = [html.Div([
        html.Div(f"Perfil {i + 1}", className="perfil-num"),
        html.H6(r["nombre"], className="perfil-nombre"),
        html.Div(fmt_pct(r["pct_victorias"]), className="perfil-valor"),
        html.Div(f"de victorias · IC 95 % ± {fmt(r['ic95'], 1)} pp", className="kpi-nota"),
        html.Div(f"{fmt(r['n'], 0)} filas ({fmt_pct(r['pct_filas'])})", className="kpi-nota"),
        rasgos(r["z"]),
    ], id={"type": "perfil-tarjeta", "index": i}, n_clicks=0, className="perfil-tarjeta perfil-activo",
        style={"--perfil": PAL_PERFIL[i]}, role="button", title="Clic para mostrar u ocultar en el radar")
        for i, r in enumerate(perf)]

    fig_v = go.Figure(go.Bar(
        x=[r["pct_victorias"] for r in perf], y=[f"P{i + 1}. {r['nombre']}" for i, r in enumerate(perf)],
        orientation="h", marker_color=PAL_PERFIL[:len(perf)],
        error_x=dict(type="data", array=[r["ic95"] for r in perf], color=GRIS),
        text=[fmt_pct(r["pct_victorias"]) for r in perf], textposition="outside", cliponaxis=False,
        customdata=[r["n"] for r in perf],
        hovertemplate="%{y}<br>Victorias: %{x:.1f} %<br>n = %{customdata:,}<extra></extra>"))
    fig_v.add_vline(x=50, line_dash="dash", line_color=GRIS)
    fig_v.update_layout(xaxis=dict(range=[0, 100], ticksuffix=" %"), yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=20, t=20, b=10))

    ks = sorted(PF["silueta_por_k"], key=int)
    fig_k = go.Figure(go.Scatter(x=[int(k) for k in ks], y=[PF["silueta_por_k"][k] for k in ks],
                                 mode="lines+markers", line=dict(color=MAR, width=2),
                                 marker=dict(size=[14 if int(k) == PF["k"] else 8 for k in ks],
                                             color=[ARENA if int(k) == PF["k"] else MAR for k in ks]),
                                 hovertemplate="k = %{x}<br>Silueta: %{y:.3f}<extra></extra>"))
    fig_k.update_layout(xaxis=dict(title="Número de perfiles (k)", dtick=1),
                        yaxis=dict(title="Silueta media", rangemode="tozero"))

    mu = PF["muestra"]
    fig_p = go.Figure()
    for i, r in enumerate(perf):
        sel = [j for j, o in enumerate(mu["orden"]) if o == i]
        fig_p.add_scattergl(x=[mu["x"][j] for j in sel], y=[mu["y"][j] for j in sel], mode="markers",
                            name=f"P{i + 1}", marker=dict(color=PAL_PERFIL[i], size=5, opacity=0.45),
                            hovertemplate=f"P{i + 1}. {r['nombre']}<extra></extra>")
    vx, vy = PF["pca_varianza"]
    fig_p.update_layout(xaxis_title=f"Componente 1 ({fmt_pct(100 * vx)} de la varianza)",
                        yaxis_title=f"Componente 2 ({fmt_pct(100 * vy)})")

    medias = pd.DataFrame([{"Perfil": f"P{i + 1}. {r['nombre']}",
                            **{etq[v].capitalize(): fmt(r["medias"][v], 3 if v == "hitpct" else 2)
                               for v in PF["variables"]}} for i, r in enumerate(perf)])
    return html.Div([
        html.P([f"K-means con k = {PF['k']} sobre {fmt(PF['filas'], 0)} filas pareja-partido de la AVP "
                f"({fmt(PF['partidos'], 0)} partidos, {PF['anios'][0]}–{PF['anios'][1]}). Variables por set y "
                "estandarizadas: ", ", ".join(etq[v] for v in PF["variables"]), ". ",
                html.Strong("Los filtros de arriba no aplican a esta pestaña.")], className="nota mb-3"),
        dbc.Alert([html.I(className="bi bi-info-circle me-2"),
                   html.Strong(f"Silueta media = {fmt(sil, 3)}. "),
                   f"Los perfiles se superponen mucho (una silueta cercana a 0 indica grupos poco separados; "
                   f"el {fmt_pct(PF['pct_silueta_negativa'])} de una muestra de filas queda más cerca de otro "
                   "perfil que del suyo). Son descripciones de estilos frecuentes, no tipos de pareja bien "
                   "definidos. Además, las estadísticas se producen durante el propio partido: el % de victorias "
                   "de cada perfil es una asociación observacional, no un efecto causal ni una predicción."],
                  color="light", className="aviso-solape"),
        html.Div(tarjetas, className="perfiles mb-4"),
        dbc.Row([dbc.Col(tarjeta("Radar de perfiles", grafico("p-radar", 420),
                                 "Centroides en desviaciones estándar respecto a la media de la AVP · "
                                 "clic en las tarjetas para elegir"), lg=6),
                 dbc.Col(tarjeta("% de victorias observado por perfil",
                                 dcc.Graph(figure=fig_v, config=CONFIG, style={"height": "420px"}),
                                 "Con intervalo de confianza del 95 %; línea en 50 %"), lg=6)]),
        dbc.Row([dbc.Col(tarjeta("Separación de los perfiles (PCA de una muestra de 2.000 filas)",
                                 dcc.Graph(figure=fig_p, config=CONFIG, style={"height": "380px"}),
                                 "Los colores se mezclan: hay solapamiento entre perfiles"), lg=7),
                 dbc.Col(tarjeta("Silueta según el número de perfiles",
                                 dcc.Graph(figure=fig_k, config=CONFIG, style={"height": "380px"}),
                                 f"El punto arena marca k = {PF['k']}, el que se describe"), lg=5)]),
        tarjeta("Promedio por set de cada perfil",
                dbc.Table.from_dataframe(medias, striped=True, hover=True, size="sm",
                                         className="tabla tabla-desplazable"), fuente=True),
        entrenador("Use los perfiles como lenguaje común para describir cómo juega una pareja, no como una receta: "
                   "las diferencias de victoria entre perfiles reflejan, sobre todo, la eficacia en ataque que ya "
                   "mostraba el análisis bivariado."),
    ])


@app.callback(Output({"type": "perfil-tarjeta", "index": ALL}, "className"), Output("p-radar", "figure"),
              Input({"type": "perfil-tarjeta", "index": ALL}, "n_clicks"))
def radar_perfiles(clics):
    PF = M.PERF
    if PF is None or not clics:
        return [], vacia()
    activos = [(c or 0) % 2 == 0 for c in clics]          # cada clic alterna mostrar / ocultar
    if not any(activos):
        activos = [True] * len(clics)
    etq = PF["etiquetas"]
    ejes = [etq[v] for v in PF["variables"]]
    fig = go.Figure()
    for i, r in enumerate(PF["perfiles"]):
        if activos[i]:
            z = [r["z"][v] for v in PF["variables"]]
            fig.add_scatterpolar(r=z + z[:1], theta=ejes + ejes[:1], name=f"P{i + 1}. {r['nombre']}",
                                 fill="toself", opacity=0.75, line=dict(color=PAL_PERFIL[i], width=2),
                                 hovertemplate="%{theta}: %{r:.2f} DE<extra>P" + str(i + 1) + "</extra>")
    lim = max(abs(r["z"][v]) for r in PF["perfiles"] for v in PF["variables"])
    fig.update_layout(polar=dict(radialaxis=dict(range=[-lim * 1.1, lim * 1.1], tickfont=dict(size=10)),
                                 bgcolor="rgba(0,0,0,0)"), legend=dict(y=-0.25, yanchor="top"),
                      margin=dict(l=30, r=30, t=30, b=10))
    clases = ["perfil-tarjeta" + (" perfil-activo" if a else "") for a in activos]
    return clases, fig


def contenido_metodologia():
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
        ("5. Modelado", "Modelo previo al partido y perfiles de juego, entrenados fuera del dashboard",
         "entrenar_modelos.py → dashboard/modelos/"),
        ("6. Conclusiones", "Recomendaciones para el entrenador", "Calculadora y hallazgos")],
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
        dbc.Row([
            dbc.Col(tarjeta("Enfoque", [
                html.P("Análisis exploratorio de datos (EDA) en tres niveles, complementado con un modelo "
                       "previo al partido (página Predicción) y perfiles de juego descriptivos:"),
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
                                            html.A("Ver la lista completa en el libro →", href=LIBRO + "10_referencias.html",
                                                   target="_blank")], fuente=False),
    ])


def contenido_conclusiones():
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
        html.Div([html.Div("Respuesta a la pregunta", className="hero-eyebrow"),
                  html.P("Antes del partido, el factor medible que más se asocia con ganar es la ventaja de ranking sobre "
                         "el rival; la estatura, la edad y el contexto aportan muy poco en comparación. Durante el partido, "
                         "al menos en la AVP, lo que separa al ganador es la eficacia en ataque.", className="respuesta")],
                 className="respuesta-caja mb-4"),
        dbc.Row([dbc.Col(tarjeta("Sobre los datos", lista(datos_, 1), fuente=False), lg=5),
                 dbc.Col(tarjeta("Sobre qué hace ganar a una pareja", lista(ganar, 5), fuente=False), lg=7)]),
        html.H5("Recomendaciones para el entrenador", className="seccion-titulo"),
        dbc.Row([dbc.Col(html.Div([html.I(className=f"bi {ico} rec-icono"), html.H6(t), html.P(x)], className="rec"), md=6, lg=3)
                 for ico, t, x in recs], className="mb-4 g-3"),
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


PESTANAS_ANALISIS = [("uni", "Univariado", contenido_uni), ("bi", "Bivariado", contenido_bi),
                     ("multi", "Multivariado", contenido_multi), ("perfiles", "Estilos de juego (AVP)", contenido_perfiles),
                     ("metodologia", "Metodología", contenido_metodologia),
                     ("conclusiones", "Conclusiones", contenido_conclusiones)]


def pagina_analisis(pestana="uni"):
    return html.Div([
        encabezado("Análisis", "Explore las variables una por una, compare ganadores con perdedores, vea qué "
                   "variables se mueven juntas y qué estilos de juego hay en la AVP. Los filtros de arriba aplican "
                   "a las pestañas univariada, bivariada y multivariada.", "Análisis exploratorio"),
        dbc.Tabs([dbc.Tab(f(), label=etq, tab_id=k) for k, etq, f in PESTANAS_ANALISIS],
                 active_tab=pestana, className="pestanas mb-4", id="a-pestanas"),
    ])


ANCHO = {"edad": 1, "estatura": 2.54, "ranking": 1, "duracion": 2, "puntos": 2, "kills": 1,
         "aces": 1, "blocks": 1, "digs": 1, "hitpct": 0.02}


@app.callback(Output("u-kpis", "children"), Output("u-fig", "figure"), Output("u-sub", "children"),
              Input("u-var", "value"), *FILTROS)
def actualizar_uni(clave, circuito, genero, anios, etapa):
    d, col = D.serie_uni(clave, circuito, genero, anios, etapa)
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
              Input("b-var", "value"), *FILTROS)
def actualizar_bi(var, circuito, genero, anios, etapa):
    etq, unidad, solo_stats = D.VARS_BI[var]
    e = D.filtrar(D.E, circuito, genero, anios, etapa)
    d = D.solo_avp(e) if solo_stats else e
    r = D.comparar(d, var)
    if r is None:
        msg = "No hay estadísticas de juego con estos filtros (solo existen para la AVP)" if solo_stats else None
        fig_caja = vacia(msg) if msg else vacia()
        kpis = []
    else:
        dec = 3 if var == "hitpct" else 0 if var == "rank_main" else 1
        kpis = [kpi(fmt(r["med_g"], dec), "Mediana ganadores", tono="tono-mar"),
                kpi(fmt(r["med_p"], dec), "Mediana perdedores", tono="tono-coral"),
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

    t = D.tabla_poder(circuito, genero, anios, etapa)
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

    p = D.filtrar(D.P, circuito, genero, anios, etapa)
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


@app.callback(Output("m-spear", "figure"), Input("m-res", "value"), *FILTROS)
def actualizar_multi(res, circuito, genero, anios, etapa):
    d = D.filtrar(D.E, circuito, genero, anios, etapa)
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
# Página: Predicción (modelos, simulador ¿Quién gana? y calculadora histórica)
# Todo lo que se muestra sale de dashboard/modelos/, generado por entrenar_modelos.py
# ==========================================================================
METRICAS = [("accuracy", "Accuracy"), ("precision", "Precisión"), ("recall", "Exhaustividad (recall)"),
            ("f1", "F1"), ("roc_auc", "ROC-AUC")]


def contenido_modelos():
    if M.MET is None or M.CUR is None:
        return sin_modelos()
    MET, rec = M.MET, M.MET["recomendado"]
    pol, regla = MET["politica"], MET["regla"]
    res = MET["modelos"]

    filas = [{"Modelo": n + (" ★" if n == rec else ""),
              "AUC validación cruzada": f"{fmt(res[n]['auc_cv_media'], 3)} ± {fmt(res[n]['auc_cv_de'], 3)}",
              "AUC prueba": fmt(res[n]["prueba"]["roc_auc"], 3),
              "Accuracy prueba": fmt(res[n]["prueba"]["accuracy"], 3),
              "Precisión": fmt(res[n]["prueba"]["precision"], 3), "Recall": fmt(res[n]["prueba"]["recall"], 3),
              "F1": fmt(res[n]["prueba"]["f1"], 3), "Brier": fmt(res[n]["prueba"]["brier"], 3),
              "AUC con ranking": fmt(res[n]["prueba_con_ranking"]["roc_auc"], 3),
              "Accuracy con ranking": fmt(res[n]["prueba_con_ranking"]["accuracy"], 3)} for n in M.NOMBRES]
    filas.append({"Modelo": "Regla \"gana el mejor rankeado\"", "AUC validación cruzada": "—", "AUC prueba": "—",
                  "Accuracy prueba": "—", "Precisión": fmt(regla["precision"], 3), "Recall": fmt(regla["recall"], 3),
                  "F1": fmt(regla["f1"], 3), "Brier": "—", "AUC con ranking": fmt(regla["roc_auc"], 3),
                  "Accuracy con ranking": fmt(regla["accuracy"], 3)})
    tabla = pd.DataFrame(filas)

    comp = go.Figure()
    for col, etq, clave in [("roc_auc", "ROC-AUC", "roc_auc"), ("accuracy", "Accuracy", "accuracy")]:
        comp.add_bar(x=M.NOMBRES + ["Regla del ranking"],
                     y=[res[n]["prueba_con_ranking"][clave] for n in M.NOMBRES] + [regla[clave]],
                     name=etq, marker_color=[PAL_MODELO[n] for n in M.NOMBRES + ["Regla del ranking"]],
                     marker_pattern_shape="" if col == "roc_auc" else "/", offsetgroup=col,
                     text=[fmt(v, 3) for v in [res[n]["prueba_con_ranking"][clave] for n in M.NOMBRES] + [regla[clave]]],
                     textposition="outside", cliponaxis=False,
                     hovertemplate="%{x}<br>" + etq + ": %{y:.3f}<extra></extra>")
    comp.update_layout(barmode="group", yaxis=dict(range=[0.5, 1], title="Métrica (subconjunto con ranking)"),
                       legend=dict(y=1.08))

    mejor_auc = max(res[n]["prueba"]["roc_auc"] for n in M.NOMBRES)
    gan = res[rec]["prueba_con_ranking"]["roc_auc"] - regla["roc_auc"]
    gan_acc = res[rec]["prueba_con_ranking"]["accuracy"] - regla["accuracy"]
    tm = MET["temporal"]
    top_imp = MET["importancias"][rec][0]
    conclusiones = [
        f"El modelo recomendado ({rec}) obtuvo en la prueba una ROC-AUC de {fmt(res[rec]['prueba']['roc_auc'], 3)} "
        f"y una accuracy de {fmt_pct(100 * res[rec]['prueba']['accuracy'])}. Las tres AUC de prueba quedan entre "
        f"{fmt(min(res[n]['prueba']['roc_auc'] for n in M.NOMBRES), 3)} y {fmt(mejor_auc, 3)}.",
        f"En los partidos con ranking de ambas parejas, frente a la regla \"gana el mejor rankeado\", el modelo "
        f"cambia la AUC en {fmt(gan, 3)} y la accuracy en {fmt(100 * gan_acc, 1)} puntos porcentuales.",
        f"La variable más importante (por permutación) es «{top_imp['etiqueta']}»: al desordenarla, la AUC cae "
        f"{fmt(top_imp['media'], 3)}.",
        f"Robustez temporal: entrenado con 2000–2016 y probado con 2017–2019, el mismo tipo de modelo obtiene AUC "
        f"{fmt(tm['prueba']['roc_auc'], 3)} y accuracy {fmt_pct(100 * tm['prueba']['accuracy'])} "
        f"(la regla, {fmt_pct(100 * tm['regla_accuracy'])} en sus {fmt(tm['regla_n'], 0)} partidos con ranking).",
    ]
    limitaciones = [
        "Solo usa información previa al partido: ventaja de ranking, edad, estatura, clasificatoria, circuito, género y "
        "etapa. No hay datos de forma reciente, lesiones, viento ni historial entre rivales.",
        f"El {fmt_pct(regla['partidos_sin_ranking_de_alguna_pareja_pct'])} de los partidos de prueba no tiene ranking "
        "numérico de alguna pareja; ahí el modelo se apoya en un indicador de ranking faltante.",
        "Los datos van de 2000 a 2019 y agrupan dos circuitos con reglas y coberturas distintas; las relaciones pueden "
        "haber cambiado.",
        "Son asociaciones observacionales: el modelo no dice que cambiar una variable cause la victoria.",
        "Las cifras son las observadas en esta división de prueba; con otros datos pueden variar.",
    ]
    return html.Div([
        html.Div([html.Div([html.I(className="bi bi-award"), " Modelo recomendado"], className="recomendado-etq"),
                  html.Div(rec, className="recomendado-nombre"),
                  html.P(MET["razon"], className="recomendado-razon")], className="recomendado mb-4"),
        html.Div([html.Span("Modelo", className="filtro-etq"),
                  chips("p-modelo", rec, [(n, n) for n in M.NOMBRES])], className="filtro-grupo mb-3"),
        html.Div(id="p-kpis", className="kpis mb-4"),
        dbc.Row([dbc.Col(tarjeta("Matriz de confusión (prueba, umbral 0,5)", grafico("p-cm", 340),
                                 "Filas: resultado real · columnas: predicción"), lg=5),
                 dbc.Col(tarjeta("Importancia de las variables", grafico("p-imp", 340),
                                 "Caída de la ROC-AUC en prueba al desordenar cada variable (5 repeticiones)"), lg=7)]),
        dbc.Row([
            dbc.Col(tarjeta("Curvas de desempeño", [
                html.Div([chips("p-curva", "roc", [("roc", "ROC"), ("pr", "Precisión-exhaustividad")]),
                          chips("p-pobl", "todos", [("todos", "Toda la prueba"),
                                                    ("con_ranking", "Solo con ranking de ambas")])],
                         className="controles-fila"),
                grafico("p-curvas", 380)],
                "La regla del ranking solo se puede trazar en los partidos con ranking de ambas parejas"), lg=6),
            dbc.Col(tarjeta("Probabilidad predicha según el resultado real", grafico("p-dist", 420),
                            "Distribución en la prueba: cuanto menos se mezclan, mejor separa el modelo"), lg=6)]),
        tarjeta("Umbral de decisión", [
            dcc.Slider(id="p-umbral", min=0.05, max=0.95, step=0.01, value=0.5,
                       marks={v: fmt(v, 1) for v in [0.1, 0.3, 0.5, 0.7, 0.9]},
                       tooltip={"placement": "top", "always_visible": True}),
            html.Div(id="p-umbral-txt", className="nota mb-2"),
            grafico("p-umbral-fig", 320)],
            "Con un umbral más alto el modelo solo dice \"gana\" cuando está más seguro: sube la precisión y baja la exhaustividad"),
        dbc.Row([dbc.Col(tarjeta("Comparación de modelos",
                                 dbc.Table.from_dataframe(tabla, striped=True, hover=True, size="sm",
                                                          className="tabla tabla-desplazable"),
                                 "★ recomendado. Las columnas \"con ranking\" usan el mismo subconjunto que la regla", fuente=False), lg=7),
                 dbc.Col(tarjeta("Modelos frente a la regla del ranking",
                                 dcc.Graph(figure=comp, config=CONFIG, style={"height": "360px"}),
                                 f"Subconjunto de prueba con ranking de ambas parejas y sin empate: "
                                 f"{fmt(regla['n'], 0)} partidos ({fmt_pct(regla['cobertura_prueba_pct'])} de la prueba)"), lg=5)]),
        dbc.Row([
            dbc.Col(tarjeta("Conclusiones", html.Ul([html.Li(x) for x in conclusiones]), fuente=False), lg=6),
            dbc.Col(tarjeta("Limitaciones", html.Ul([html.Li(x) for x in limitaciones]), fuente=False), lg=6)]),
        tarjeta("Cómo se entrenó (reproducibilidad)", html.Ul([
            html.Li([html.Strong("Alcance: "), pol["alcance"]]),
            html.Li([html.Strong("Unidad: "), pol["unidad"]]),
            html.Li([html.Strong("División: "), pol["division"],
                     f" Entrenamiento: {fmt(pol['n_entrenamiento'], 0)} partidos ({fmt_pct(pol['pct_gana_entrenamiento'])} con gana = 1); "
                     f"prueba: {fmt(pol['n_prueba'], 0)} ({fmt_pct(pol['pct_gana_prueba'])})."]),
            html.Li([html.Strong("Validación cruzada: "), pol["validacion_cruzada"]]),
            html.Li([html.Strong("Preprocesamiento: "), pol["preprocesamiento"]]),
            html.Li([html.Strong("Predictores: "), ", ".join(pol["etiquetas_predictores"][p] for p in pol["predictores"]),
                     ". Excluidas por fuga de información: ", ", ".join(pol["excluidas_por_fuga"]), "."]),
            html.Li([html.Strong("Descartes: "),
                     f"{fmt(pol['descartes']['partidos_forfeit'], 0)} forfeits; "
                     f"{fmt(pol['descartes']['partidos_sin_un_ganador_y_un_perdedor'], 0)} partidos sin exactamente un "
                     f"ganador y un perdedor; quedan {fmt(pol['descartes']['partidos_modelo'], 0)}."]),
            html.Li([html.Strong("Semilla y versiones: "),
                     f"semilla {pol['semilla']}; " + ", ".join(f"{k} {v}" for k, v in MET["versiones"].items()) +
                     f"; generado el {MET['generado']}."]),
            html.Li([html.Strong("Huella del conjunto de prueba (SHA-256 de los match_id): "),
                     html.Code(pol["huella_ids_prueba"][:16] + "…")]),
        ], className="lista-politica"), fuente=True),
    ])


@app.callback(Output("p-kpis", "children"), Output("p-cm", "figure"), Output("p-imp", "figure"),
              Output("p-dist", "figure"), Input("p-modelo", "value"))
def actualizar_modelo(nombre):
    r = M.MET["modelos"][nombre]
    kpis = [kpi(fmt(r["prueba"][k], 3), etq, tono="tono-mar" if k == "roc_auc" else "") for k, etq in METRICAS]
    kpis.append(kpi(f"{fmt(r['auc_cv_media'], 3)} ± {fmt(r['auc_cv_de'], 3)}", "AUC validación cruzada",
                    "5 pliegues en entrenamiento"))
    cm = np.array(r["matriz_confusion"]["valores"])
    total = cm.sum()
    fcm = go.Figure(go.Heatmap(z=cm, x=["Predice: gana", "Predice: pierde"], y=["Real: gana", "Real: pierde"],
                               colorscale=ESCALA_SEC, showscale=False, xgap=3, ygap=3,
                               text=[[f"{fmt(v, 0)}<br>{fmt_pct(100 * v / total)}" for v in fila] for fila in cm],
                               texttemplate="%{text}", textfont=dict(size=15),
                               hovertemplate="%{y} · %{x}<br>%{z:,} partidos<extra></extra>"))
    fcm.update_yaxes(autorange="reversed", showgrid=False)
    fcm.update_xaxes(showgrid=False, side="top")

    imp = pd.DataFrame(M.MET["importancias"][nombre]).sort_values("media")
    fimp = go.Figure(go.Bar(x=imp.media, y=imp.etiqueta, orientation="h", marker_color=PAL_MODELO[nombre],
                            error_x=dict(type="data", array=imp.de, color=GRIS),
                            hovertemplate="%{y}<br>Caída de AUC: %{x:.4f}<extra></extra>"))
    fimp.update_layout(xaxis_title="Caída de la ROC-AUC", margin=dict(l=10, r=20, t=20, b=10))

    dist = M.CUR[nombre]["distribucion"]
    b = np.array(dist["bordes"])
    c = (b[:-1] + b[1:]) / 2
    fd = go.Figure()
    for clave, etq, color in [("gana", "Real: gana", PAL["Ganador"]), ("pierde", "Real: pierde", PAL["Perdedor"])]:
        fd.add_bar(x=c, y=dist[clave], width=b[1] - b[0], name=etq, marker_color=color, opacity=0.65,
                   hovertemplate=etq + "<br>Probabilidad %{x:.3f}<br>%{y:,} partidos<extra></extra>")
    fd.add_vline(x=0.5, line_dash="dash", line_color=GRIS)
    fd.update_layout(barmode="overlay", bargap=0, xaxis=dict(range=[0, 1], title="Probabilidad predicha de ganar"),
                     yaxis=dict(title="Partidos de prueba", tickformat=",d"))
    return kpis, fcm, fimp, fd


@app.callback(Output("p-curvas", "figure"), Input("p-curva", "value"), Input("p-pobl", "value"),
              Input("p-modelo", "value"))
def actualizar_curvas(tipo, pobl, elegido):
    fig = go.Figure()
    nombres = M.NOMBRES + (["Regla del ranking"] if pobl == "con_ranking" else [])
    for n in nombres:
        cur = M.CUR[n][pobl][tipo]
        x, y = (cur["fpr"], cur["tpr"]) if tipo == "roc" else (cur["recall"], cur["precision"])
        fig.add_scatter(x=x, y=y, mode="lines", name=n,
                        line=dict(color=PAL_MODELO[n], width=4 if n == elegido else 2,
                                  dash="dash" if n == "Regla del ranking" else "solid"),
                        opacity=1 if n in (elegido, "Regla del ranking") else 0.6,
                        hovertemplate=n + "<br>%{x:.2f} · %{y:.2f}<extra></extra>")
    if tipo == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color="#C9D6E3", dash="dot"))
        fig.update_layout(xaxis=dict(range=[0, 1], title="Tasa de falsos positivos"),
                          yaxis=dict(range=[0, 1], title="Tasa de verdaderos positivos"))
    else:
        fig.update_layout(xaxis=dict(range=[0, 1], title="Exhaustividad (recall)"),
                          yaxis=dict(range=[0.4, 1], title="Precisión"))
    return fig


@app.callback(Output("p-umbral-fig", "figure"), Output("p-umbral-txt", "children"),
              Input("p-modelo", "value"), Input("p-umbral", "value"))
def actualizar_umbral(nombre, u):
    t = M.CUR[nombre]["umbrales"]
    fig = go.Figure()
    for clave, etq, color, guion in [("accuracy", "Accuracy", MAR, "solid"), ("precision", "Precisión", TURQUESA, "solid"),
                                     ("recall", "Exhaustividad", CORAL, "solid"), ("f1", "F1", ARENA, "dash")]:
        fig.add_scatter(x=t["umbral"], y=t[clave], name=etq, mode="lines", line=dict(color=color, width=2, dash=guion),
                        hovertemplate="Umbral %{x:.2f}<br>" + etq + ": %{y:.3f}<extra></extra>")
    fig.add_vline(x=u, line_color=GRIS)
    fig.update_layout(xaxis=dict(range=[0, 1], title="Umbral"), yaxis=dict(range=[0, 1]), hovermode="x unified")
    i = int(np.argmin(np.abs(np.array(t["umbral"]) - u)))
    prec = "no definida (no predice ningún \"gana\")" if t["precision"][i] is None else fmt(t["precision"][i], 3)
    txt = (f"Con umbral {fmt(t['umbral'][i], 2)}: accuracy {fmt(t['accuracy'][i], 3)}, precisión "
           f"{prec}, exhaustividad {fmt(t['recall'][i], 3)}, F1 {fmt(t['f1'][i], 3)}; el modelo "
           f"predice \"gana\" en el {fmt_pct(t['pct_pred_gana'][i])} de los partidos de prueba.")
    return fig, txt


def contenido_simulador():
    if M.MODELO is None or M.MET is None:
        return sin_modelos()

    def pareja(lado, titulo, rank, edad, est):
        return html.Div([
            html.Div(titulo, className="lado-titulo"),
            html.Label("Ranking (vacío = sin ranking numérico)", className="control-etq"),
            dbc.Input(id=f"s-rank-{lado}", type="number", min=1, max=300, step=1, value=rank),
            html.Label("Edad media de la pareja (años)", className="control-etq mt-2"),
            dbc.Input(id=f"s-edad-{lado}", type="number", min=16, max=50, step=0.5, value=edad),
            html.Label("Estatura media de la pareja (cm)", className="control-etq mt-2"),
            dbc.Input(id=f"s-est-{lado}", type="number", min=150, max=215, step=1, value=est),
            dbc.Switch(id=f"s-clasif-{lado}", label="Viene de clasificatoria", value=False, className="mt-2"),
        ], className=f"lado lado-{lado}")
    return html.Div([
        html.P(["Simulador del modelo recomendado (", html.Strong(M.MET["recomendado"]), "). Solo usa información "
                "previa al partido. La probabilidad se calcula desde los dos lados y se promedia, así que la de su pareja "
                "y la del rival suman 100 %."], className="nota mb-3"),
        html.Div([pareja("a", "Su pareja", 8, 28, 190), html.Div(className="red red-chica", **{"aria-hidden": "true"}),
                  pareja("b", "Rival", 12, 28, 188)], className="simulador-cancha mb-3"),
        dbc.Row([dbc.Col([html.Span("Circuito", className="filtro-etq"),
                          chips("s-circuito", "FIVB", [("FIVB", "FIVB"), ("AVP", "AVP")])], md=3),
                 dbc.Col([html.Span("Género", className="filtro-etq"),
                          chips("s-genero", "M", [("M", "Hombres"), ("W", "Mujeres")])], md=3),
                 dbc.Col([html.Span("Etapa", className="filtro-etq"),
                          chips("s-etapa", "Cuadro principal", [(e, e) for e in ETAPAS])], md=6)], className="mb-3"),
        html.Div(id="s-resultado", className="resultado mb-4"),
    ])


@app.callback(Output("s-resultado", "children"),
              Input("s-rank-a", "value"), Input("s-rank-b", "value"), Input("s-edad-a", "value"),
              Input("s-edad-b", "value"), Input("s-est-a", "value"), Input("s-est-b", "value"),
              Input("s-clasif-a", "value"), Input("s-clasif-b", "value"), Input("s-circuito", "value"),
              Input("s-genero", "value"), Input("s-etapa", "value"))
def simular(ra, rb, ea, eb, sa, sb, ca, cb, circuito, genero, etapa):
    if None in (ea, eb, sa, sb):
        return dbc.Alert("Ingrese la edad y la estatura media de las dos parejas.", color="warning")
    p = M.probabilidad(rank_propio=ra, rank_rival=rb, edad_propia=float(ea), edad_rival=float(eb),
                       est_propia=float(sa), est_rival=float(sb), clasif_propia=bool(ca), clasif_rival=bool(cb),
                       circuito=circuito, genero=genero, etapa=etapa)
    pct = 100 * p
    if pct >= 75:
        lectura = (f"Favorito claro según el modelo, aunque le asigna a la sorpresa un {fmt_pct(100 - pct, 0)} "
                   "de probabilidad.")
    elif pct >= 55:
        lectura = "Ligera ventaja: el modelo inclina la balanza, pero el partido está lejos de decidido."
    elif pct > 45:
        lectura = "Partido abierto: el modelo no separa a las dos parejas."
    else:
        lectura = "El modelo pone a su pareja en desventaja; estos partidos también se ganan."
    r = M.MET["modelos"][M.MET["recomendado"]]["prueba"]
    aviso = [] if ra and rb else [html.Div("Una de las parejas no tiene ranking numérico: el modelo usa el indicador "
                                           "de ranking faltante, con menos información.", className="resultado-detalle")]
    return html.Div([
        html.Div("Probabilidad estimada de ganar", className="resultado-etq"),
        html.Div([html.Div([html.Span("Su pareja", className="score-etq"),
                            html.Span(fmt_pct(pct, 0), className="digito score-a", id="s-pct")]),
                  html.Span("–", className="score-guion"),
                  html.Div([html.Span(fmt_pct(100 - pct, 0), className="digito score-b"),
                            html.Span("Rival", className="score-etq")])], className="score"),
        html.Div([html.Div(style={"width": f"{pct}%"}, className="barra-a"),
                  html.Div(style={"width": f"{100 - pct}%"}, className="barra-b")], className="barra-duelo mb-2"),
        html.Div(lectura, className="resultado-lectura"),
        *aviso,
        html.Div(f"Modelo: {M.MET['recomendado']}, con ROC-AUC {fmt(r['roc_auc'], 3)} y accuracy "
                 f"{fmt_pct(100 * r['accuracy'])} en {fmt(r['n'], 0)} partidos de prueba (2000–2019). Es una "
                 "estimación a partir de asociaciones históricas, no una garantía.", className="resultado-detalle"),
    ])


def contenido_calculadora():
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
        dbc.Alert([html.I(className="bi bi-clock-history me-2"), html.Strong("Calculadora del entrenador. "),
                   "Ingrese el ranking de su pareja y el del rival. La calculadora busca los partidos históricos con una "
                   "ventaja parecida y le muestra cuántos ganó la pareja en su posición. No es el modelo predictivo: "
                   "son frecuencias observadas, sin ajustar por otras variables."], color="light", className="aviso-solape"),
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


def pagina_prediccion(pestana="modelos"):
    return html.Div([
        encabezado("Predicción", "¿Se puede anticipar quién gana con lo que se sabe antes del partido? Compare tres "
                   "modelos con la regla \"gana el mejor rankeado\", simule un partido o consulte la calculadora "
                   "histórica.", "Modelado"),
        dbc.Tabs([dbc.Tab(contenido_modelos(), label="Modelos", tab_id="modelos"),
                  dbc.Tab(contenido_simulador(), label="¿Quién gana?", tab_id="simulador"),
                  dbc.Tab(contenido_calculadora(), label="Calculadora histórica", tab_id="calculadora")],
                 active_tab=pestana, className="pestanas mb-4", id="p-pestanas"),
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
# Página: Equipo (por ahora solo la pestaña, sin contenido)
# ==========================================================================
def pagina_equipo():
    return html.Div(id="pagina-equipo", className="pagina-vacia")


# ==========================================================================
# Navegación entre páginas (las rutas antiguas llevan a su nuevo lugar)
# ==========================================================================
RUTAS = {"/": pagina_inicio, "/dashboard": pagina_dashboard, "/analisis": pagina_analisis,
         "/prediccion": pagina_prediccion, "/equipo": pagina_equipo,
         "/datos": pagina_dashboard,
         "/exploratorio": pagina_analisis,
         "/metodologia": lambda: pagina_analisis("metodologia"),
         "/conclusiones": lambda: pagina_analisis("conclusiones"),
         "/calculadora": lambda: pagina_prediccion("calculadora")}
CON_FILTROS = {"/dashboard", "/datos", "/analisis", "/exploratorio"}


@app.callback(Output("contenido", "children"), Output("filtros", "className"), Input("url", "pathname"))
def mostrar_pagina(ruta):
    clase = "filtros-banda" + ("" if ruta in CON_FILTROS else " oculto")
    return RUTAS.get(ruta, pagina_inicio)(), clase


if __name__ == "__main__":
    # Puerto opcional: python app.py 8051 (por si el 8050 está ocupado)
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print(f"\nDashboard listo. Ábralo en el navegador: http://127.0.0.1:{puerto}\n(Para detenerlo: Ctrl + C)\n")
    app.run(debug=False, port=puerto)
