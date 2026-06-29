"""
Genera el pick del día del Mundial en LOCAL y lo publica en la base (Neon),
para que la web (Render free, que no puede cargar el modelo) lo muestre.

Uso:
    DATABASE_URL="postgresql://...neon..." \
        python scripts/publicar_pick_mundial.py "México" "Sudáfrica" --fase "Grupos" --hora "14:00" --nruns 15

Escribe en la tabla EstadoApp (clave 'featured_pick') con el MISMO formato que
arma el admin del frontend (incluye valores priorizados: 1X2, Doble Oportunidad,
DNB + córners/amarillas). No reentrena en Render: solo guarda el resultado.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pick_linea(dist, lo=0.62, hi=0.82):
    """Replica _pickLinea() del front: mejor Más/Menos en la banda [lo,hi]."""
    if not dist:
        return None
    cands = []
    for k, v in dist.items():
        if not (isinstance(k, str) and k.startswith("over_")) or not isinstance(v, (int, float)):
            continue
        try:
            line = float(k.replace("over_", "").replace("_", "."))
        except ValueError:
            continue
        cands.append(("Más de", line, float(v)))
        cands.append(("Menos de", line, 1.0 - float(v)))
    f = [c for c in cands if lo <= c[2] <= hi]
    f.sort(key=lambda c: -c[2])
    return f[0] if f else None


def _auto_valores(mercados, consenso, e1, e2):
    """Replica autoValores() del front (priorización 1X2 → Doble → DNB → córners/amarillas)."""
    items = []
    w = (consenso or {}).get("win", 0) or 0
    d = (consenso or {}).get("draw", 0) or 0
    l = (consenso or {}).get("loss", 0) or 0
    n1, n2 = e1.split(" ")[0], e2.split(" ")[0]

    # 1X2
    if w >= d and w >= l:
        items.append({"nombre": f"1X2 · {e1}", "descripcion": "Victoria " + n1, "porcentaje": round(w * 100)})
    elif l >= w and l >= d:
        items.append({"nombre": f"1X2 · {e2}", "descripcion": "Victoria " + n2, "porcentaje": round(l * 100)})
    else:
        items.append({"nombre": "1X2 · Empate", "descripcion": "Partido igualado", "porcentaje": round(d * 100)})

    # Doble oportunidad
    dobles = sorted([
        (f"Doble · {n1} o empate", w + d),
        (f"Doble · {n1} o {n2}", w + l),
        (f"Doble · empate o {n2}", d + l),
    ], key=lambda x: -x[1])[0]
    if dobles[1] < 0.93:
        items.append({"nombre": dobles[0], "descripcion": "Doble oportunidad", "porcentaje": round(dobles[1] * 100)})

    # Apuesta sin empate (DNB)
    wl = w + l
    if wl > 0:
        dnb_w, dnb_l = w / wl, l / wl
        if dnb_w >= dnb_l:
            items.append({"nombre": f"Sin empate · {n1}", "descripcion": "Apuesta sin empate (DNB)", "porcentaje": round(dnb_w * 100)})
        else:
            items.append({"nombre": f"Sin empate · {n2}", "descripcion": "Apuesta sin empate (DNB)", "porcentaje": round(dnb_l * 100)})

    # Córners / amarillas (secundarios)
    if mercados:
        cu = _pick_linea(mercados.get("corners"))
        if cu:
            items.append({"nombre": f"Córners {cu[0]} {cu[1]}", "descripcion": "Mejor línea córners", "porcentaje": round(cu[2] * 100)})
        au = _pick_linea(mercados.get("amarillas"))
        if au:
            items.append({"nombre": f"Amarillas {au[0]} {au[1]}", "descripcion": "Mejor línea amarillas", "porcentaje": round(au[2] * 100)})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("equipo1")
    ap.add_argument("equipo2")
    ap.add_argument("--fase", default="Grupos")
    ap.add_argument("--hora", default="14:00")
    ap.add_argument("--competencia", default="Mundial")
    ap.add_argument("--nruns", type=int, default=15)
    args = ap.parse_args()

    if not os.getenv("DATABASE_URL"):
        sys.exit("ERROR: falta DATABASE_URL (la connection string de Neon) en el entorno.")

    import joblib
    from ml.pipeline_mundial import predecir_partido_mundial

    cache = ROOT / "data" / "_mundial_model_cache.pkl"
    if not cache.exists():
        sys.exit(f"ERROR: no existe el caché {cache}. Entrená el Mundial primero en local.")

    print(f"⏳ Cargando modelo y prediciendo {args.equipo1} vs {args.equipo2} (n_runs={args.nruns})…")
    obj = joblib.load(cache)
    with contextlib.redirect_stdout(io.StringIO()):
        salida = predecir_partido_mundial(args.equipo1, args.equipo2, obj,
                                          n_runs=args.nruns, fase=args.fase)
    if not salida:
        sys.exit("ERROR: la predicción no devolvió datos.")

    consenso = salida.get("consenso") or {}
    mercados = salida.get("mercados")

    # Predicción indexada por equipos → la web la sirve sin cargar el modelo.
    pred_cache = {
        "consenso": consenso,
        "mercados": mercados,
        "modelos": salida.get("modelos", []),
        "goles": salida.get("goles", []),
        "ultimos_e1": salida.get("ultimos_e1", []),
        "ultimos_e2": salida.get("ultimos_e2", []),
        "stats_equipos": salida.get("stats_equipos"),
    }

    body = {
        "equipo1": args.equipo1, "equipo2": args.equipo2,
        "hora": args.hora, "fase": args.fase, "competencia": args.competencia,
        "prob_win": consenso.get("win", 0), "prob_draw": consenso.get("draw", 0),
        "prob_loss": consenso.get("loss", 0),
        "valores": _auto_valores(mercados, consenso, args.equipo1, args.equipo2),
        "mercados": mercados,
        "modelos": salida.get("modelos", []),
        "goles": salida.get("goles", []),
        "ultimos_e1": salida.get("ultimos_e1", []),
        "ultimos_e2": salida.get("ultimos_e2", []),
        "stats_equipos": salida.get("stats_equipos"),
    }

    # Escribir en Neon (tabla estadoapp, clave featured_pick) — upsert.
    from sqlalchemy import create_engine, text
    url = os.environ["DATABASE_URL"]
    for pre in ("postgres://", "postgresql://"):
        if url.startswith(pre):
            url = url.replace(pre, "postgresql+psycopg2://", 1)
            break
    eng = create_engine(url, pool_pre_ping=True)
    # numpy-safe: convierte np.float64/int64 a tipos nativos.
    _np = lambda o: o.item() if hasattr(o, "item") else str(o)
    payload = json.dumps(body, ensure_ascii=False, default=_np)
    pred_payload = json.dumps(pred_cache, ensure_ascii=False, default=_np)
    pred_key = f"predcache:{args.equipo1}|{args.equipo2}"
    with eng.begin() as c:
        for clave, val in (("featured_pick", payload), (pred_key, pred_payload)):
            c.execute(text("""
                INSERT INTO estadoapp (clave, valor) VALUES (:k, :v)
                ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor
            """), {"k": clave, "v": val})

    print(f"✅ Publicado en Neon: {args.equipo1} vs {args.equipo2}")
    print(f"   1X2  W {body['prob_win']:.0%} / D {body['prob_draw']:.0%} / L {body['prob_loss']:.0%}")
    print("   Valores:", " | ".join(f"{v['nombre']} {v['porcentaje']}%" for v in body["valores"][:3]))


if __name__ == "__main__":
    main()
