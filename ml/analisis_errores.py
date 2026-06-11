"""
Análisis de en qué falla el modelo 1X2 (selecciones) — walk-forward, sin leakage.

Usa el mismo motor que analisis_mercados.py (medias expanding ataque×defensa +
Poisson) para reconstruir la predicción 1X2 cronológicamente y diseccionar los
errores:

  1. Matriz de confusión (pred W/D/L  vs  real W/D/L).
  2. Accuracy por resultado real  → ¿se come los empates?
  3. Accuracy según quién es favorito y si juega de local.
  4. Calibración: en los partidos donde el modelo dijo "favorito ≥X%",
     ¿con qué frecuencia acertó? → detecta sobreestimación del gran favorito.

Esto sustenta los ajustes posteriores (cap de goleada, etc.).
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "selecciones_combinado.csv"
MIN_HIST = 4


def load():
    df = pd.read_csv(DATASET)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    return df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)


def main():
    df = load()
    gf = defaultdict(list)   # goles a favor
    ga = defaultdict(list)   # goles en contra
    es_local_hist = {}       # no usado aún

    rng = np.random.default_rng(0)

    conf = defaultdict(int)              # (pred, real) -> n
    by_real = defaultdict(lambda: [0, 0])   # real -> [hit, tot]
    by_scn = defaultdict(lambda: [0, 0])    # escenario -> [hit, tot]
    calib = defaultdict(lambda: [0, 0])     # bucket prob fav -> [hit, tot]
    fav_strength = []                       # (prob_fav, acierto_fav)

    n_eval = 0
    for _, r in df.iterrows():
        e1, e2 = r["Equipo1"], r["Equipo2"]
        g1r, g2r = r.get("EQUIPO1_GOLES"), r.get("EQUIPO2_GOLES")
        local_e1 = int(r.get("Es_Local_E1", 0) or 0)

        if len(gf[e1]) >= MIN_HIST and len(gf[e2]) >= MIN_HIST and pd.notna(g1r) and pd.notna(g2r):
            # lambdas ataque x defensa
            l1 = (np.mean(gf[e1]) + np.mean(ga[e2])) / 2
            l2 = (np.mean(gf[e2]) + np.mean(ga[e1])) / 2
            # ventaja de localía leve
            if local_e1 == 1: l1 *= 1.10
            g1 = rng.poisson(max(l1, .05), 8000)
            g2 = rng.poisson(max(l2, .05), 8000)
            pw = (g1 > g2).mean(); pd_ = (g1 == g2).mean(); pl = (g1 < g2).mean()

            pred = max((("W", pw), ("D", pd_), ("L", pl)), key=lambda x: x[1])
            real = "W" if g1r > g2r else ("D" if g1r == g2r else "L")
            ok = pred[0] == real

            conf[(pred[0], real)] += 1
            by_real[real][1] += 1; by_real[real][0] += int(ok)
            n_eval += 1

            # escenario: ¿el favorito (mayor prob no-empate) es local o visitante?
            fav_is_e1 = pw >= pl
            fav_local = (fav_is_e1 and local_e1 == 1) or ((not fav_is_e1) and local_e1 == 0)
            scn = "favorito_local" if fav_local else "favorito_visitante"
            # solo cuenta cuando el modelo eligió al favorito (no empate)
            if pred[0] in ("W", "L"):
                fav_real_ok = (fav_is_e1 and real == "W") or ((not fav_is_e1) and real == "L")
                by_scn[scn][1] += 1; by_scn[scn][0] += int(fav_real_ok)
                p_fav = max(pw, pl)
                fav_strength.append((p_fav, fav_real_ok))
                b = f"{int(p_fav*10)*10}-{int(p_fav*10)*10+10}%"
                calib[b][1] += 1; calib[b][0] += int(fav_real_ok)

        if pd.notna(g1r) and pd.notna(g2r):
            gf[e1].append(float(g1r)); ga[e1].append(float(g2r))
            gf[e2].append(float(g2r)); ga[e2].append(float(g1r))

    acc = sum(by_real[k][0] for k in by_real) / max(n_eval, 1)
    print(f"\n=== ANÁLISIS DE FALLOS 1X2  (n={n_eval}, acc global={acc*100:.1f}%) ===\n")

    print("Matriz de confusión (filas=predicho, cols=real):")
    print(f"{'':>8}{'W':>7}{'D':>7}{'L':>7}")
    for p in ("W", "D", "L"):
        print(f"{p:>8}" + "".join(f"{conf[(p,r)]:>7}" for r in ("W", "D", "L")))

    print("\nAccuracy por resultado REAL:")
    for k in ("W", "D", "L"):
        h, t = by_real[k]
        print(f"  {k}: {h}/{t} = {100*h/max(t,1):.1f}%")

    print("\nAccuracy del favorito según escenario:")
    for k, (h, t) in by_scn.items():
        print(f"  {k}: {h}/{t} = {100*h/max(t,1):.1f}%")

    print("\nCalibración del favorito (prob asignada → acierto real):")
    print("  (si 'real%' < 'prob%' el modelo SOBREESTIMA al favorito)")
    for b in sorted(calib):
        h, t = calib[b]
        if t < 20: continue
        print(f"  modelo dijo {b:>8}:  acertó {100*h/t:5.1f}%   (n={t})")


if __name__ == "__main__":
    main()
