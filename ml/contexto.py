"""
Motor de contexto de partido — "información que entiende el fútbol".

Genera viñetas en español describiendo el contexto de un E1 vs E2 a partir de:
  • Forma reciente (últimos N partidos): puntos, goles a favor/en contra, racha.
  • Estilo de juego: equipo de muchos córners, agresivo (faltas/tarjetas),
    dominante en posesión, volumen de tiros.
  • Cara a cara (H2H) histórico.
  • (best-effort) API gratuita TheSportsDB para resultados recientes "en vivo".

Diseñado para que el front muestre por qué el modelo ve el partido como lo ve,
y como base para futuras features de accuracy (estilo → córners/tarjetas).

NOTA: el dataset combinado codifica Es_Local_E1=1 siempre (no distingue sede
neutral), así que NO se infiere ventaja local; el contexto es de forma/estilo.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "selecciones_combinado.csv"

_DF_CACHE: pd.DataFrame | None = None
N = 5  # ventana de forma


def _df() -> pd.DataFrame:
    global _DF_CACHE
    if _DF_CACHE is None:
        d = pd.read_csv(DATASET)
        d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
        _DF_CACHE = d.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    return _DF_CACHE


def _team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    return df[(df["Equipo1"] == team) | (df["Equipo2"] == team)]


def _perfil(df: pd.DataFrame, team: str) -> dict | None:
    """Stats medias del equipo en sus últimos N partidos (lado correcto E1/E2)."""
    m = _team_matches(df, team).tail(N)
    if m.empty:
        return None
    gf = ga = corners = faltas = tarj = tiros = poses = 0.0
    pts = []
    cnt = 0
    for _, r in m.iterrows():
        e1 = r["Equipo1"] == team
        suf = "E1" if e1 else "E2"
        osuf = "E2" if e1 else "E1"
        g = r.get(f"EQUIPO1_GOLES") if e1 else r.get(f"EQUIPO2_GOLES")
        og = r.get(f"EQUIPO2_GOLES") if e1 else r.get(f"EQUIPO1_GOLES")
        if pd.isna(g) or pd.isna(og):
            continue
        cnt += 1
        gf += g; ga += og
        pts.append(3 if g > og else (1 if g == og else 0))
        def num(c):
            v = r.get(c); return float(v) if pd.notna(v) else 0.0
        corners += num(f"Saques_de_esquina_sacados_{suf}")
        faltas += num(f"Faltas_cometidas_{suf}")
        ta = r.get(f"Tarjetas_amarillas_{suf}"); tr = r.get(f"Tarjetas_rojas_{suf}")
        tarj += (float(ta) if pd.notna(ta) else 0) + (float(tr) if pd.notna(tr) else 0)
        tiros += num(f"Disparos_totales_{suf}")
        poses += num(f"Posesion_{suf}")
    if cnt == 0:
        return None
    return {
        "n": cnt, "gf": gf / cnt, "ga": ga / cnt, "pts": sum(pts),
        "ppp": sum(pts) / cnt, "corners": corners / cnt, "faltas": faltas / cnt,
        "tarjetas": tarj / cnt, "tiros": tiros / cnt,
        "posesion": poses / cnt if poses > 0 else None,
        "racha": pts,
    }


def _racha_txt(pts: list) -> str:
    s = "".join("V" if p == 3 else ("E" if p == 1 else "D") for p in pts)
    return s


def _h2h(df: pd.DataFrame, e1: str, e2: str) -> dict | None:
    m = df[((df["Equipo1"] == e1) & (df["Equipo2"] == e2)) |
           ((df["Equipo1"] == e2) & (df["Equipo2"] == e1))].tail(6)
    if m.empty:
        return None
    w1 = w2 = dr = 0
    for _, r in m.iterrows():
        g1, g2 = r.get("EQUIPO1_GOLES"), r.get("EQUIPO2_GOLES")
        if pd.isna(g1) or pd.isna(g2):
            continue
        a = r["Equipo1"]
        gana = a if g1 > g2 else (None if g1 == g2 else r["Equipo2"])
        if gana == e1: w1 += 1
        elif gana == e2: w2 += 1
        else: dr += 1
    return {"e1": w1, "e2": w2, "empates": dr, "n": w1 + w2 + dr}


def _estilo_bullets(team: str, p: dict) -> list[str]:
    out = []
    # córners
    if p["corners"] >= 6: out.append(f"{team} genera muchos córners ({p['corners']:.1f} por partido), buen perfil para Más córners.")
    elif p["corners"] <= 3.5: out.append(f"{team} genera pocos córners ({p['corners']:.1f} por partido).")
    # agresividad
    if p["tarjetas"] >= 2.3: out.append(f"{team} es propenso a tarjetas ({p['tarjetas']:.1f} por partido) y comete {p['faltas']:.0f} faltas, ojo con Más amarillas.")
    elif p["tarjetas"] <= 1.2: out.append(f"{team} es disciplinado ({p['tarjetas']:.1f} tarjetas por partido).")
    # posesión / tiros
    if p["posesion"] and p["posesion"] >= 55: out.append(f"{team} domina la posesión ({p['posesion']:.0f}%) y dispara {p['tiros']:.0f} veces, suele llevar el juego.")
    elif p["posesion"] and p["posesion"] <= 45: out.append(f"{team} cede la posesión ({p['posesion']:.0f}%), juega más a la contra.")
    return out


def _forma_bullet(team: str, p: dict) -> str:
    v = p["racha"].count(3); e = p["racha"].count(1); d = p["racha"].count(0)
    tono = "en gran forma" if p["ppp"] >= 2.2 else ("irregular" if p["ppp"] >= 1.0 else "en mala forma")
    return (f"{team} llega {tono}: {p['pts']} pts en los últimos {p['n']} "
            f"({v} ganados, {e} empates, {d} perdidos), "
            f"marca {p['gf']:.1f} y recibe {p['ga']:.1f} goles por partido.")


# Nombres de selección ES → EN (como los lista TheSportsDB)
_ES_EN = {
    "Brasil": "Brazil", "España": "Spain", "Alemania": "Germany", "Francia": "France",
    "Inglaterra": "England", "Italia": "Italy", "Países Bajos": "Netherlands",
    "Bélgica": "Belgium", "Croacia": "Croatia", "Suiza": "Switzerland", "Japón": "Japan",
    "México": "Mexico", "Estados Unidos": "USA", "Corea del Sur": "South Korea",
    "Marruecos": "Morocco", "Senegal": "Senegal", "Ghana": "Ghana", "Túnez": "Tunisia",
    "Camerún": "Cameroon", "Nigeria": "Nigeria", "Egipto": "Egypt", "Argelia": "Algeria",
    "Costa de Marfil": "Ivory Coast", "Sudáfrica": "South Africa", "Catar": "Qatar",
    "Arabia Saudita": "Saudi Arabia", "Australia": "Australia", "Canadá": "Canada",
    "Costa Rica": "Costa Rica", "Dinamarca": "Denmark", "Polonia": "Poland",
    "Portugal": "Portugal", "Uruguay": "Uruguay", "Colombia": "Colombia", "Chile": "Chile",
    "Perú": "Peru", "Ecuador": "Ecuador", "Paraguay": "Paraguay", "Argentina": "Argentina",
    "Escocia": "Scotland", "Gales": "Wales", "Irlanda": "Ireland", "Noruega": "Norway",
    "Suecia": "Sweden", "Serbia": "Serbia", "Austria": "Austria", "Turquía": "Turkey",
    "Ucrania": "Ukraine", "Grecia": "Greece", "Hungría": "Hungary", "Rumania": "Romania",
}


def _api_thesportsdb(team: str, timeout: float = 4.0) -> list[str]:
    """Best-effort: últimos resultados desde TheSportsDB (API gratis, key de prueba '3').

    Mapea el nombre ES→EN y solo conserva eventos donde la selección aparece
    realmente como local o visitante (evita falsos positivos de clubes)."""
    import json, urllib.parse, urllib.request
    out = []
    en = _ES_EN.get(team, team)
    try:
        q = urllib.parse.quote(en)
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={q}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        teams = data.get("teams") or []
        # priorizar selección nacional (liga internacional / sin estadio de club típico)
        cand = [t for t in teams if t.get("strSport") == "Soccer"]
        tsel = next((t for t in cand if "National" in (t.get("strLeague") or "")
                     or (t.get("strTeam") or "").lower() == en.lower()), None)
        tid = (tsel or (cand[0] if cand else {})).get("idTeam")
        if not tid:
            return out
        url2 = f"https://www.thesportsdb.com/api/v1/json/3/eventslast.php?id={tid}"
        with urllib.request.urlopen(url2, timeout=timeout) as r:
            data2 = json.loads(r.read().decode())
        for ev in (data2.get("results") or []):
            h, a = ev.get("strHomeTeam"), ev.get("strAwayTeam")
            hs, as_ = ev.get("intHomeScore"), ev.get("intAwayScore")
            if not (h and a and hs is not None):
                continue
            if en.lower() not in (h.lower(), a.lower()):   # filtra clubes con nombre parecido
                continue
            out.append(f"{h} {hs}-{as_} {a} ({ev.get('dateEvent','')})")
            if len(out) >= 3:
                break
    except Exception:
        pass
    return out


def contexto_partido(e1: str, e2: str, usar_api: bool = True) -> dict:
    """Devuelve contexto estructurado + viñetas en español para E1 vs E2."""
    df = _df()
    p1, p2 = _perfil(df, e1), _perfil(df, e2)
    bullets: list[str] = []

    if p1: bullets.append(_forma_bullet(e1, p1))
    if p2: bullets.append(_forma_bullet(e2, p2))

    # comparativa de forma
    if p1 and p2:
        if p1["ppp"] - p2["ppp"] >= 0.8:
            bullets.append(f"{e1} llega claramente mejor que {e2} en forma reciente.")
        elif p2["ppp"] - p1["ppp"] >= 0.8:
            bullets.append(f"{e2} llega claramente mejor que {e1} en forma reciente.")

    if p1: bullets += _estilo_bullets(e1, p1)
    if p2: bullets += _estilo_bullets(e2, p2)

    h2h = _h2h(df, e1, e2)
    if h2h and h2h["n"] > 0:
        bullets.append(f"Cara a cara (últimos {h2h['n']}): {e1} ganó {h2h['e1']}, {h2h['empates']} empates, {e2} ganó {h2h['e2']}.")

    api_bullets = []
    if usar_api:
        for t in (e1, e2):
            res = _api_thesportsdb(t)
            if res:
                api_bullets.append(f"Últimos de {t}:")
                api_bullets += res

    return {
        "equipo1": e1, "equipo2": e2,
        "perfil_e1": p1, "perfil_e2": p2, "h2h": h2h,
        "contexto": bullets,
        "contexto_api": api_bullets,
        "fuente_api": "TheSportsDB (gratis)" if api_bullets else None,
    }


if __name__ == "__main__":
    import json, sys
    a = sys.argv[1] if len(sys.argv) > 1 else "Brasil"
    b = sys.argv[2] if len(sys.argv) > 2 else "Argentina"
    print(json.dumps(contexto_partido(a, b, usar_api=True), indent=2, ensure_ascii=False, default=str))
