"""Las cinco páginas se arman, todos los callbacks están conectados y responden con filtros normales y extremos."""
import json

import plotly.graph_objects as go
import pytest

import app as A
import estilo
import prediccion as M


def ids(componente, salida=None):
    """Todos los id de un árbol de componentes de Dash."""
    salida = set() if salida is None else salida
    i = getattr(componente, "id", None)
    if i is not None:
        salida.add(json.dumps(i, sort_keys=True) if isinstance(i, dict) else i)
    hijos = getattr(componente, "children", None)
    for h in (hijos if isinstance(hijos, (list, tuple)) else [hijos]):
        if hasattr(h, "to_plotly_json"):
            ids(h, salida)
    for prop in ("label",):
        v = getattr(componente, prop, None)
        if hasattr(v, "to_plotly_json"):
            ids(v, salida)
    return salida


def test_cinco_paginas_en_el_menu():
    assert [t for _, t, _ in A.PAGINAS] == ["Inicio", "Dashboard", "Análisis", "Predicción", "Equipo"]
    for ruta in A.RUTAS:
        contenido, clase = A.mostrar_pagina(ruta)
        assert contenido is not None
        assert ("oculto" in clase) == (ruta not in A.CON_FILTROS)


def test_equipo_sin_contenido_y_video_proximamente():
    eq = A.pagina_equipo()
    assert eq.children is None
    inicio = str(A.pagina_inicio())
    assert "Próximamente" in inicio and "video-proximamente" in inicio
    assert "<video" not in inicio and "Video(" not in inicio


def test_callbacks_conectados_a_componentes_existentes():
    presentes = ids(A.app.layout)
    for ruta in ["/", "/dashboard", "/analisis", "/prediccion", "/equipo"]:
        ids(A.RUTAS[ruta](), presentes)
    simples = {p for p in presentes if not p.startswith("{")}
    for cb in A.app.callback_map.values():
        for dep in cb["inputs"]:
            i = dep["id"]
            if isinstance(i, str) and not i.startswith("{"):
                assert i in simples, f"Input sin componente: {i}"
    for clave in A.app.callback_map:
        for parte in clave.strip(".").split("..."):
            i = parte.split(".")[0]
            if i and not i.startswith("{"):
                assert i in simples, f"Output sin componente: {i}"


TODOS = ("Todos", "Todos", [2000, 2019], "Todas")
EXTREMOS = [TODOS, ("AVP", "W", [2011, 2012], "Clasificatoria"), ("FIVB", "M", [2019, 2019], "Finales"),
            ("AVP", "Todos", [2000, 2000], "Fase de grupos")]


@pytest.mark.parametrize("filtros", EXTREMOS)
def test_callbacks_con_filtros(filtros):
    A.actualizar_marcador(*filtros)
    kpis, *figs = A.actualizar_datos(*filtros)
    assert all(isinstance(f, go.Figure) for f in figs)
    A.detalle_anio(None, *filtros)
    A.detalle_anio({"points": [{"x": 2005}]}, *filtros)
    for clave in A.D.VARS_UNI:
        A.actualizar_uni(clave, *filtros)
    for var in ["rank_main", "hitpct", "estatura_media_cm"]:
        A.actualizar_bi(var, *filtros)
    for res in ["Todos", "Ganador", "Perdedor"]:
        A.actualizar_multi(res, *filtros)


def test_restablecer_filtros():
    assert A.restablecer(1) == ("Todos", "Todos", list(A.D.ANIOS), "Todas")


def test_callbacks_de_prediccion():
    for n in M.NOMBRES:
        kpis, cm, imp, dist = A.actualizar_modelo(n)
        assert len(kpis) == 6
        for tipo in ["roc", "pr"]:
            for pobl in ["todos", "con_ranking"]:
                fig = A.actualizar_curvas(tipo, pobl, n)
                nombres = {t.name for t in fig.data}
                assert ("Regla del ranking" in nombres) == (pobl == "con_ranking")
        for u in [0.05, 0.5, 0.95]:
            A.actualizar_umbral(n, u)


def test_simulador_simetrico_y_sensato():
    base = dict(edad_propia=28.0, edad_rival=28.0, est_propia=190.0, est_rival=190.0, clasif_propia=False,
                clasif_rival=False, circuito="FIVB", genero="M", etapa="Cuadro principal")
    p = M.probabilidad(rank_propio=3, rank_rival=25, **base)
    q = M.probabilidad(rank_propio=25, rank_rival=3, **base)
    assert p + q == pytest.approx(1)
    assert p > 0.5 > q
    assert M.probabilidad(rank_propio=10, rank_rival=10, **base) == pytest.approx(0.5)
    assert 0 < M.probabilidad(rank_propio=None, rank_rival=10, **base) < 1
    A.simular(8, 12, 28, 28, 190, 188, False, False, "FIVB", "M", "Cuadro principal")
    A.simular(None, None, 28, 28, 190, 188, True, False, "AVP", "W", "Clasificatoria")
    A.simular(8, 12, None, 28, 190, 188, False, False, "AVP", "W", "Finales")


def test_calculadora_historica():
    A.texto_estatura(0), A.texto_estatura(-5)
    res, f1, f2 = A.calcular(8, 12, "Todos", "Todos", "Todas", 0)
    assert "Probabilidad histórica" in str(res)
    A.calcular(None, 12, "Todos", "Todos", "Todas", 0)
    A.calcular(1, 64, "AVP", "W", "Finales", 15)


def test_radar_perfiles():
    n = len(M.PERF["perfiles"])
    clases, fig = A.radar_perfiles([0] * n)
    assert len(fig.data) == n and all("perfil-activo" in c for c in clases)
    clases, fig = A.radar_perfiles([1] + [0] * (n - 1))
    assert len(fig.data) == n - 1 and "perfil-activo" not in clases[0]


def test_paleta_fija():
    assert (estilo.MAR, estilo.CORAL, estilo.TURQUESA, estilo.ARENA) == ("#0B3B5C", "#E4572E", "#1A8A9A", "#D9A441")
    assert estilo.PAL["Ganador"] == estilo.MAR and estilo.PAL["Perdedor"] == estilo.CORAL
    assert estilo.PAL["FIVB"] == estilo.TURQUESA and estilo.PAL["AVP"] == estilo.ARENA
    css = (A.Path(A.__file__).parent / "assets" / "estilo.css").read_text(encoding="utf-8")
    for nombre, color in [("mar", "#0B3B5C"), ("coral", "#E4572E"), ("turquesa", "#1A8A9A"), ("arena", "#D9A441")]:
        assert f"--{nombre}: {color}" in css


def test_servidor_responde():
    c = A.server.test_client()
    for ruta in ["/", "/dashboard", "/_dash-layout", "/_dash-dependencies", "/assets/estilo.css"]:
        assert c.get(ruta).status_code == 200, ruta
