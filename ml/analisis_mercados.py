"""
Backtest de predictibilidad/rentabilidad de mercados de apuestas comunes.

Objetivo: responder "¿cuáles mercados son más fáciles de predecir y más
rentables?" de forma honesta y sin leakage temporal.

Método (walk-forward, cronológico):
  1. Ordena selecciones_combinado.csv por fecha.
  2. Para cada equipo mantiene un historial EXPANDING (solo partidos pasados)
     de cada estadística: goles for/against, corners for/against, tarjetas,
     disparos, tiros a puerta, faltas, fueras de juego, paradas.
  3. Para cada partido con historial suficiente estima el lambda esperado de
     cada mercado con el blend clásico ataque×defensa:
         lam_equipo = (media_for_equipo + media_against_rival) / 2
  4. Con Poisson calcula la probabilidad de cada lado del mercado, recomienda
     el lado más probable y lo compara con el resultado real.
  5. Reporta por mercado:
         acc        accuracy del pick recomendado (todos los partidos)
         acc_conf   accuracy cuando la confianza del modelo >= UMBRAL
         cov_conf   % de partidos que alcanzan esa confianza (cuántas apuestas)
         edge       acc_conf - break_even(prob media de los picks confiados)

"fácil de predecir" = acc_conf alto.   "rentable" = edge positivo y estable.
NOTA: el ROI real exige las cuotas de la casa; aquí 'edge' usa como break-even
la probabilidad implícita justa, así que mide cuánto le ganamos a una casa
que cotizara sin margen. Es el mejor proxy disponible sin scrapear cuotas.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "selecciones_combinado.csv"
OUT_JSON = ROOT / "data" / "analisis_mercados.json"

MIN_HIST = 3          # partidos previos mínimos por equipo para predecir
CONF = 0.58           # umbral de confianza para los picks "rentables"

# stat E1 / stat E2 que alimentan cada mercado (for del equipo)
STAT = {
    "goles":      ("EQUIPO1_GOLES", "EQUIPO2_GOLES"),
    "corners":    ("Saques_de_esquina_sacados_E1", "Saques_de_esquina_sacados_E2"),
    "tarjetas":   ("_TARJ_E1", "_TARJ_E2"),       # amarillas+rojas, derivado
    "disparos":   ("Disparos_totales_E1", "Disparos_totales_E2"),
    "tiros_p":    ("Disparos_a_puerta_E1", "Disparos_a_puerta_E2"),
    "faltas":     ("Faltas_cometidas_E1", "Faltas_cometidas_E2"),
    "fueras":     ("Fueras_de_juego_E1", "Fueras_de_juego_E2"),
    "paradas":    ("Paradas_E1", "Paradas_E2"),
}

# líneas O/U a evaluar por mercado de conteo total
LINES = {
    "goles":    [1.5, 2.5, 3.5],
    "corners":  [7.5, 8.5, 9.5, 10.5],
    "tarjetas": [2.5, 3.5, 4.5, 5.5],
    "disparos": [19.5, 21.5, 23.5, 25.5],
    "tiros_p":  [6.5, 7.5, 8.5, 9.5],
    "faltas":   [20.5, 22.5, 24.5, 26.5],
    "fueras":   [2.5, 3.5, 4.5],
    "paradas":  [5.5, 6.5, 7.5],
}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    # tarjetas = amarillas + rojas
    for s in ("1", "2"):
        a = pd.to_numeric(df.get(f"Tarjetas_amarillas_E{s}"), errors="coerce")
        r = pd.to_numeric(df.get(f"Tarjetas_rojas_E{s}"), errors="coerce")
        df[f"_TARJ_E{s}"] = a.fillna(0) + r.fillna(0)
        df.loc[a.isna() & r.isna(), f"_TARJ_E{s}"] = np.nan
    return df


def expand_mean(hist: list) -> float | None:
    vals = [v for v in hist if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else None


def over_prob(lam: float, line: float) -> float:
    """P(total > line) con Poisson(lam). line es X.5 => P(X >= ceil(line))."""
    k = int(np.floor(line))           # line 7.5 -> k=7 -> P(X>=8)=1-cdf(7)
    return float(1.0 - poisson.cdf(k, lam))


def main():
    df = load()
    n = len(df)

    # historiales por equipo: for y against de cada stat
    hist_for = defaultdict(lambda: defaultdict(list))
    hist_ag = defaultdict(lambda: defaultdict(list))

    # acumuladores de resultados por mercado
    res = defaultdict(lambda: {"hit": 0, "tot": 0,
                               "chit": 0, "ctot": 0, "cprob": []})

    def num(row, col):
        if col not in row:
            return None
        v = row[col]
        return float(v) if pd.notna(v) else None

    for _, row in df.iterrows():
        e1, e2 = row["Equipo1"], row["Equipo2"]

        # ¿hay historial mínimo para ambos?
        ready = (len(hist_for[e1]["goles"]) >= MIN_HIST and
                 len(hist_for[e2]["goles"]) >= MIN_HIST)

        if ready:
            # lambdas esperados ataque x defensa para cada mercado
            lam = {}
            for mk, (c1, c2) in STAT.items():
                f1 = expand_mean(hist_for[e1][mk]); a1 = expand_mean(hist_ag[e1][mk])
                f2 = expand_mean(hist_for[e2][mk]); a2 = expand_mean(hist_ag[e2][mk])
                if None in (f1, a1, f2, a2):
                    lam[mk] = None
                    continue
                lam[mk] = ((f1 + a2) / 2.0, (f2 + a1) / 2.0)  # (E1, E2)

            real = {mk: (num(row, c1), num(row, c2)) for mk, (c1, c2) in STAT.items()}

            # ── 1X2, doble oportunidad, DNB, BTTS, resultado 2+ (de goles) ──
            if lam["goles"] and None not in real["goles"]:
                l1, l2 = lam["goles"]
                rng = np.random.default_rng(0)
                g1 = rng.poisson(max(l1, .05), 12000)
                g2 = rng.poisson(max(l2, .05), 12000)
                pw = float((g1 > g2).mean()); pd_ = float((g1 == g2).mean())
                pl = float((g1 < g2).mean())
                rg1, rg2 = real["goles"]
                real_res = "W" if rg1 > rg2 else ("D" if rg1 == rg2 else "L")

                _bucket(res["1X2 (resultado final)"],
                        max(("W", pw), ("D", pd_), ("L", pl), key=lambda x: x[1]),
                        real_res, CONF)

                # doble oportunidad: 1X / 12 / X2
                _bucket(res["Doble oportunidad"],
                        max(("1X", pw + pd_), ("12", pw + pl), ("X2", pd_ + pl),
                            key=lambda x: x[1]),
                        ("1X" if real_res in ("W", "D") else "_",  # se evalúa abajo
                         ), CONF, multi=("1X", "12", "X2"), real_res=real_res)

                # apuesta sin empate (DNB) — se anula el empate
                if real_res != "D":
                    s = pw + pl
                    _bucket(res["Apuesta sin empate (DNB)"],
                            max(("W", pw / s), ("L", pl / s), key=lambda x: x[1]),
                            real_res, CONF)

                # BTTS
                pb = float(((g1 > 0) & (g2 > 0)).mean())
                real_btts = "si" if (rg1 > 0 and rg2 > 0) else "no"
                _bucket(res["Ambos marcan (BTTS)"],
                        ("si", pb) if pb >= .5 else ("no", 1 - pb),
                        real_btts, CONF)

                # resultado final 2+ (gana algún equipo por 2+)
                p2 = float((np.abs(g1 - g2) >= 2).mean())
                real_2 = "si" if abs(rg1 - rg2) >= 2 else "no"
                _bucket(res["Gana por 2+ goles"],
                        ("si", p2) if p2 >= .5 else ("no", 1 - p2),
                        real_2, CONF)

            # ── mercados O/U de conteo total ──
            for mk, lines in LINES.items():
                if not lam.get(mk) or None in real[mk]:
                    continue
                lt = lam[mk][0] + lam[mk][1]
                rt = real[mk][0] + real[mk][1]
                for ln in lines:
                    p_over = over_prob(lt, ln)
                    side = ("Over", p_over) if p_over >= .5 else ("Under", 1 - p_over)
                    real_side = "Over" if rt > ln else "Under"
                    _bucket(res[f"{_nice(mk)} O/U {ln}"], side, real_side, CONF)

        # ── actualizar historiales SOLO con el partido ya jugado ──
        for mk, (c1, c2) in STAT.items():
            v1, v2 = num(row, c1), num(row, c2)
            if v1 is not None:
                hist_for[e1][mk].append(v1); hist_ag[e2][mk].append(v1)
            if v2 is not None:
                hist_for[e2][mk].append(v2); hist_ag[e1][mk].append(v2)

    _report(res)


def _bucket(acc, pick, real, conf, multi=None, real_res=None):
    """Registra un acierto. pick=(label, prob). real=label real (o tupla)."""
    label, prob = pick
    if multi:  # doble oportunidad: real_res W/D/L -> qué dobles se cumplen
        real_ok = {
            "1X": real_res in ("W", "D"),
            "12": real_res in ("W", "L"),
            "X2": real_res in ("D", "L"),
        }
        ok = real_ok[label]
    else:
        ok = (label == real)
    acc["tot"] += 1
    acc["hit"] += int(ok)
    if prob >= conf:
        acc["ctot"] += 1
        acc["chit"] += int(ok)
        acc["cprob"].append(prob)


def _nice(mk):
    return {
        "goles": "Total goles", "corners": "Córners", "tarjetas": "Tarjetas",
        "disparos": "Disparos", "tiros_p": "Tiros a puerta", "faltas": "Faltas",
        "fueras": "Fueras de juego", "paradas": "Paradas portero",
    }[mk]


def _report(res):
    rows = []
    for mk, a in res.items():
        if a["tot"] < 50:
            continue
        acc = a["hit"] / a["tot"]
        accc = a["chit"] / a["ctot"] if a["ctot"] else 0.0
        cov = a["ctot"] / a["tot"]
        be = float(np.mean(a["cprob"])) if a["cprob"] else 0.0  # break-even justo
        edge = accc - be
        rows.append({
            "mercado": mk, "n": a["tot"], "acc": round(acc, 4),
            "acc_conf": round(accc, 4), "cov_conf": round(cov, 3),
            "break_even": round(be, 4), "edge": round(edge, 4),
            "n_conf": a["ctot"],
        })

    rows.sort(key=lambda r: (r["acc_conf"], r["edge"]), reverse=True)

    print("\n" + "=" * 92)
    print(f"{'MERCADO':34s} {'n':>5} {'acc':>6} {'accConf':>8} {'cov':>6} {'edge':>7} {'nConf':>6}")
    print("-" * 92)
    for r in rows:
        print(f"{r['mercado']:34s} {r['n']:5d} {r['acc']*100:5.1f}% "
              f"{r['acc_conf']*100:6.1f}% {r['cov_conf']*100:5.0f}% "
              f"{r['edge']*100:+5.1f}% {r['n_conf']:6d}")
    print("=" * 92)
    print(f"Umbral confianza={CONF:.2f}  (acc_conf = aciertos cuando el modelo "
          f"da prob >= {CONF:.0%})")
    print("Ordenado por acc_conf (más fácil de predecir) y luego edge (más rentable).")

    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 Guardado: {OUT_JSON}")


if __name__ == "__main__":
    main()
