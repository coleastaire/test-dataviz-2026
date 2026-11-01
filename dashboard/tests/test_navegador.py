"""Prueba de punta a punta en un navegador real (Chromium vía Playwright).

Recorre las cinco páginas en escritorio y en pantalla angosta, usa los filtros, las
pestañas, las tarjetas clicables y el simulador, y guarda capturas si se define
la variable de entorno CAPTURAS (carpeta de salida).
"""
import os
import threading
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

import app as A  # noqa: E402

PUERTO = 8765
URL = f"http://127.0.0.1:{PUERTO}"
CAPTURAS = Path(os.environ["CAPTURAS"]) if os.environ.get("CAPTURAS") else None
PAGINAS = [("/", "inicio"), ("/dashboard", "dashboard"), ("/analisis", "analisis"),
           ("/prediccion", "prediccion"), ("/equipo", "equipo")]


@pytest.fixture(scope="module")
def servidor():
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", PUERTO, A.server, threaded=True)
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()
    yield URL
    srv.shutdown()


@pytest.fixture(scope="module")
def navegador():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def esperar(page):
    page.wait_for_load_state("networkidle")
    page.wait_for_function("() => !document.querySelector('._dash-loading') && "
                           "document.querySelector('#contenido') && document.querySelector('#contenido').children.length >= 0")
    page.wait_for_timeout(700)


def capturar(page, nombre, completa=True):
    if CAPTURAS:
        CAPTURAS.mkdir(parents=True, exist_ok=True)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(200)
        page.screenshot(path=str(CAPTURAS / f"{nombre}.png"), full_page=completa)


def nueva(navegador, ancho, alto):
    ctx = navegador.new_context(viewport={"width": ancho, "height": alto}, device_scale_factor=1)
    page = ctx.new_page()
    errores = []
    page.on("console", lambda m: errores.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errores.append(str(e)))
    return page, errores


def propios(errores):
    """Errores de la app (se ignoran recursos externos que no cargan sin internet)."""
    return [e for e in errores if "Failed to load resource" not in e and "net::ERR" not in e]


def test_escritorio_paginas_y_navegacion(servidor, navegador):
    page, errores = nueva(navegador, 1440, 900)
    for ruta, nombre in PAGINAS:
        page.goto(servidor + ruta)
        esperar(page)
        capturar(page, f"escritorio-{nombre}")
    # Navegación con el riel
    page.goto(servidor + "/")
    esperar(page)
    page.click("#nav-1")
    page.wait_for_url("**/dashboard")
    esperar(page)
    assert page.is_visible("#filtros")
    assert page.inner_text("#nav-1").strip() == "Dashboard"
    assert "active" in page.get_attribute("#nav-1", "class")
    assert not propios(errores), errores


def test_filtros_rapidos_y_clic_en_grafico(servidor, navegador):
    page, errores = nueva(navegador, 1440, 900)
    page.goto(servidor + "/dashboard")
    esperar(page)
    antes = page.inner_text("#d-n")
    page.click("#f-circuito label:has-text('AVP')")
    page.wait_for_function(f"() => document.querySelector('#d-n') && document.querySelector('#d-n').innerText !== '{antes}'")
    despues = page.inner_text("#d-n")
    assert despues != antes
    tablero = page.inner_text("#marcador-tablero")
    assert "FIVB\n0\n" in tablero or "FIVB 0" in tablero.replace("\n", " ")
    # Clic en un año del gráfico: el detalle muestra ese año
    page.evaluate("""() => { const gd = document.querySelector('#d-anio .js-plotly-plot');
                             gd.emit('plotly_click', {points: [{x: 2005, curveNumber: 0, pointNumber: 0}]}); }""")
    page.wait_for_function("() => document.querySelector('#d-detalle .detalle-anio')?.innerText === '2005'")
    capturar(page, "escritorio-dashboard-filtro-avp-2005")
    page.click("#f-reset")
    page.wait_for_function(f"() => document.querySelector('#d-n').innerText === '{antes}'")
    assert not propios(errores), errores


def test_analisis_pestanas_y_perfiles(servidor, navegador):
    page, errores = nueva(navegador, 1440, 900)
    page.goto(servidor + "/analisis")
    esperar(page)
    for pestana in ["Bivariado", "Multivariado", "Metodología", "Conclusiones", "Estilos de juego (AVP)"]:
        page.click(f".pestanas .nav-link:has-text('{pestana}')")
        page.wait_for_timeout(500)
    tarjeta = page.locator(".perfil-tarjeta").first
    assert "perfil-activo" in tarjeta.get_attribute("class")
    tarjeta.click()
    page.wait_for_function("() => !document.querySelector('.perfil-tarjeta').classList.contains('perfil-activo')")
    page.wait_for_timeout(500)
    capturar(page, "escritorio-analisis-estilos-de-juego")
    assert not propios(errores), errores


def test_prediccion_modelos_simulador_calculadora(servidor, navegador):
    page, errores = nueva(navegador, 1440, 900)
    page.goto(servidor + "/prediccion")
    esperar(page)
    page.click("#p-curva label:has-text('Precisión-exhaustividad')")
    page.click("#p-pobl label:has-text('Solo con ranking de ambas')")
    page.wait_for_timeout(800)
    assert page.locator("#p-curvas .legendtext:has-text('Regla del ranking')").count() == 1
    capturar(page, "escritorio-prediccion-modelos")
    page.click(".pestanas .nav-link:has-text('¿Quién gana?')")
    page.wait_for_selector("#s-pct")
    antes = page.inner_text("#s-pct")
    page.fill("#s-rank-a", "1")
    page.fill("#s-rank-b", "40")
    page.wait_for_function(f"() => document.querySelector('#s-pct').innerText !== '{antes}'")
    assert int(page.inner_text("#s-pct").split()[0]) > 50
    capturar(page, "escritorio-prediccion-quien-gana")
    page.click(".pestanas .nav-link:has-text('Calculadora histórica')")
    page.wait_for_selector("#c-resultado .resultado-valor")
    capturar(page, "escritorio-prediccion-calculadora")
    assert not propios(errores), errores


def test_pantalla_angosta(servidor, navegador):
    page, errores = nueva(navegador, 390, 844)
    for ruta, nombre in PAGINAS:
        page.goto(servidor + ruta)
        esperar(page)
        ancho = page.evaluate("() => document.documentElement.scrollWidth")
        assert ancho <= 392, f"{ruta}: desborde horizontal ({ancho}px)"
        caja = page.locator(".riel").bounding_box()
        assert caja["y"] > 700 and caja["width"] >= 380, "el riel debe pasar abajo como barra de pestañas"
        capturar(page, f"angosta-{nombre}", completa=False)      # lo que ve el teléfono (con la barra abajo)
        capturar(page, f"angosta-{nombre}-completa")
    page.click("#nav-3")
    page.wait_for_url("**/prediccion")
    assert not propios(errores), errores
