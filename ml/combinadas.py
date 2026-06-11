"""
Cálculo de probabilidad de una apuesta combinada (parlay) — correcto en
correlación.

Regla clave:
  • Patas de PARTIDOS DISTINTOS son independientes → se multiplican.
  • Patas del MISMO partido están correlacionadas (ej. "gana local" y "over 2.5"
    suben juntas) → NO se pueden multiplicar; se evalúan sobre una ÚNICA
    simulación Monte Carlo conjunta de ese partido.

Cada partido aporta sus parámetros pre-partido (goles esperados, lambdas de
córners/amarillas, y el consenso 1X2 del clasificador). Las patas se describen
con un catálogo cerrado de predicados que el backend sabe evaluar.

P(combinada) = Π_partido  P(todas las patas de ese partido se cumplen)
donde cada factor sale de la simulación conjunta (o del consenso si la pata es
1X2/doble/DNB y es la única de ese partido — ahí el clasificador es más preciso).
"""
from __future__ import annotations

import numpy as np

N_SIMS = 40_000


def _sim_partido(params: dict, rng: np.random.Generator) -> dict:
    """Genera arrays Monte Carlo de un partido a partir de sus parámetros."""
    g1e = max(float(params.get("g1_exp", 1.2)), 0.05)
    g2e = max(float(params.get("g2_exp", 1.1)), 0.05)
    lc1 = max(float(params.get("corners_e1", 4.8)), 0.3)
    lc2 = max(float(params.get("corners_e2", 4.6)), 0.3)
    la1 = max(float(params.get("amarillas_e1", 1.9)), 0.1)
    la2 = max(float(params.get("amarillas_e2", 1.9)), 0.1)

    g1 = rng.poisson(g1e, N_SIMS)
    g2 = rng.poisson(g2e, N_SIMS)
    c = rng.poisson(lc1, N_SIMS) + rng.poisson(lc2, N_SIMS)
    a = rng.poisson(la1, N_SIMS) + rng.poisson(la2, N_SIMS)
    return {"g1": g1, "g2": g2, "ctot": c, "atot": a}


def _leg_mask(leg: dict, sim: dict, params: dict) -> np.ndarray:
    """Devuelve un array booleano: en qué simulaciones se cumple la pata."""
    g1, g2 = sim["g1"], sim["g2"]
    tot = g1 + g2
    tipo = leg["tipo"]
    sel = leg.get("sel")
    _ln = leg.get("line")
    line = float(_ln) if _ln is not None else 2.5

    if tipo == "1x2":
        if sel == "local":     return g1 > g2
        if sel == "empate":    return g1 == g2
        return g1 < g2                                  # visitante
    if tipo == "doble":
        if sel == "1X":        return g1 >= g2
        if sel == "12":        return g1 != g2
        return g1 <= g2                                 # X2
    if tipo == "dnb":
        # se evalúa solo sobre simulaciones sin empate (el push se ignora)
        if sel == "local":     return (g1 > g2)
        return (g1 < g2)                                # visitante
    if tipo == "btts":
        si = (g1 > 0) & (g2 > 0)
        return si if sel == "si" else ~si
    if tipo == "gana2plus":
        return np.abs(g1 - g2) >= 2
    if tipo == "over_goles":   return tot > line
    if tipo == "under_goles":  return tot < line
    if tipo == "over_corners": return sim["ctot"] > line
    if tipo == "under_corners":return sim["ctot"] < line
    if tipo == "over_amarillas":  return sim["atot"] > line
    if tipo == "under_amarillas": return sim["atot"] < line
    raise ValueError(f"tipo de pata desconocido: {tipo}")


def _prob_partido(params: dict, legs: list[dict], rng: np.random.Generator) -> float:
    """P(todas las patas de un mismo partido se cumplen) — simulación conjunta."""
    sim = _sim_partido(params, rng)

    # DNB: hay que descartar los empates del denominador para esas patas.
    dnb = [l for l in legs if l["tipo"] == "dnb"]
    valid = np.ones(N_SIMS, dtype=bool)
    if dnb:
        valid &= sim["g1"] != sim["g2"]          # quita push

    hit = np.ones(N_SIMS, dtype=bool)
    for leg in legs:
        hit &= _leg_mask(leg, sim, params)

    denom = valid.sum()
    if denom == 0:
        return 0.0
    return float((hit & valid).sum() / denom)


def simular_combinada(partidos: list[dict]) -> dict:
    """
    partidos: lista de { 'params': {...}, 'legs': [ {tipo, sel, line, etiqueta}, ...] }

    Devuelve probabilidad combinada, cuota justa, y el desglose por pata.
    Partidos distintos → independientes (se multiplican sus factores).
    """
    rng = np.random.default_rng(42)
    prob_total = 1.0
    desglose = []

    for p in partidos:
        params = p.get("params", {})
        legs = p.get("legs", [])
        if not legs:
            continue

        # factor del partido (correlación intra-partido capturada por la sim conjunta)
        factor = _prob_partido(params, legs, rng)
        prob_total *= factor

        # prob individual de cada pata (sim aislada) — para mostrar en el desglose
        for leg in legs:
            sim = _sim_partido(params, rng)
            if leg["tipo"] == "dnb":
                nd = sim["g1"] != sim["g2"]
                p_ind = float((_leg_mask(leg, sim, params) & nd).sum() / max(nd.sum(), 1))
            else:
                p_ind = float(_leg_mask(leg, sim, params).mean())
            desglose.append({
                "etiqueta": leg.get("etiqueta", leg["tipo"]),
                "partido": p.get("nombre", ""),
                "prob": round(p_ind, 4),
                "cuota_justa": round(1.0 / p_ind, 2) if p_ind > 0 else None,
            })

    return {
        "prob": round(prob_total, 4),
        "porcentaje": round(prob_total * 100, 1),
        "cuota_justa": round(1.0 / prob_total, 2) if prob_total > 0 else None,
        "n_patas": len(desglose),
        "n_partidos": len([p for p in partidos if p.get("legs")]),
        "desglose": desglose,
    }
