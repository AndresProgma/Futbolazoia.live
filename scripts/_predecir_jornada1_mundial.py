"""Predice la jornada 1 del Mundial 2026 (24 partidos, 11-17 jun) ANTES de
mirar resultados. Guarda predicciones en data/_pred_jornada1.json."""
from __future__ import annotations
import contextlib, io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from ml.pipeline_mundial import predecir_partido_mundial

# (fecha, equipo_local_web, equipo_visita_web) en nombres del dataset
PARTIDOS = [
    ("2026-06-11", "México", "Sudáfrica"),
    ("2026-06-11", "Corea del Sur", "República Checa"),
    ("2026-06-12", "Canadá", "Bosnia"),
    ("2026-06-12", "USA", "Paraguay"),
    ("2026-06-13", "Catar", "Suiza"),
    ("2026-06-13", "Brasil", "Marruecos"),
    ("2026-06-13", "Escocia", "Haití"),
    ("2026-06-13", "Australia", "Turquía"),
    ("2026-06-14", "Alemania", "Curazao"),
    ("2026-06-14", "Países Bajos", "Japón"),
    ("2026-06-14", "Costa de Marfil", "Ecuador"),
    ("2026-06-14", "Suecia", "Túnez"),
    ("2026-06-15", "España", "Cabo Verde"),
    ("2026-06-15", "Bélgica", "Egipto"),
    ("2026-06-15", "Arabia Saudita", "Uruguay"),
    ("2026-06-15", "Irán", "Nueva Zelanda"),
    ("2026-06-16", "Francia", "Senegal"),
    ("2026-06-16", "Noruega", "Iraq"),
    ("2026-06-16", "Argentina", "Argelia"),
    ("2026-06-16", "Austria", "Jordania"),
    ("2026-06-17", "Portugal", "RD Congo"),
    ("2026-06-17", "Inglaterra", "Croacia"),
    ("2026-06-17", "Ghana", "Panamá"),
    ("2026-06-17", "Colombia", "Uzbekistán"),
]

def main():
    cache = ROOT / "data" / "_mundial_model_cache.pkl"
    print("Cargando modelo…", file=sys.stderr)
    obj = joblib.load(cache)
    out = []
    for fecha, e1, e2 in PARTIDOS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                s = predecir_partido_mundial(e1, e2, obj, n_runs=15, fase="Grupos")
            c = (s or {}).get("consenso") or {}
            w, d, l = float(c.get("win", 0)), float(c.get("draw", 0)), float(c.get("loss", 0))
            if w >= d and w >= l:
                pred = "1"   # gana local
            elif l >= w and l >= d:
                pred = "2"   # gana visita
            else:
                pred = "X"   # empate
            rec = {"fecha": fecha, "e1": e1, "e2": e2,
                   "win": round(w, 3), "draw": round(d, 3), "loss": round(l, 3),
                   "pred": pred}
        except Exception as ex:
            rec = {"fecha": fecha, "e1": e1, "e2": e2, "error": str(ex)[:120]}
        out.append(rec)
        if "error" in rec:
            print(f"  {e1} vs {e2}: ERROR {rec['error']}", file=sys.stderr)
        else:
            print(f"  {fecha} {e1} vs {e2} -> {rec['pred']}  "
                  f"(W{rec['win']:.0%}/X{rec['draw']:.0%}/L{rec['loss']:.0%})", file=sys.stderr)

    dest = ROOT / "data" / "_pred_jornada1.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {dest}", file=sys.stderr)

if __name__ == "__main__":
    main()
