"""Regenera predicciones de mercados (goles/córners/amarillas) de la jornada 1."""
from __future__ import annotations
import contextlib, io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from ml.pipeline_mundial import predecir_partido_mundial

PARTIDOS = [
    ("México","Sudáfrica"),("Corea del Sur","República Checa"),("Canadá","Bosnia"),
    ("USA","Paraguay"),("Catar","Suiza"),("Brasil","Marruecos"),("Escocia","Haití"),
    ("Australia","Turquía"),("Alemania","Curazao"),("Países Bajos","Japón"),
    ("Costa de Marfil","Ecuador"),("Suecia","Túnez"),("España","Cabo Verde"),
    ("Bélgica","Egipto"),("Arabia Saudita","Uruguay"),("Irán","Nueva Zelanda"),
    ("Francia","Senegal"),("Noruega","Iraq"),("Argentina","Argelia"),
    ("Austria","Jordania"),("Portugal","RD Congo"),("Inglaterra","Croacia"),
    ("Ghana","Panamá"),("Colombia","Uzbekistán"),
]

def linea_ou(dist, lo=0.55, hi=0.85):
    """Mejor Más/Menos en banda [lo,hi] sobre over_X_5."""
    if not dist: return None
    cands=[]
    for k,v in dist.items():
        if isinstance(k,str) and k.startswith("over_") and isinstance(v,(int,float)):
            try: line=float(k.replace("over_","").replace("_","."))
            except ValueError: continue
            cands.append(("Más de",line,float(v))); cands.append(("Menos de",line,1.0-float(v)))
    f=[c for c in cands if lo<=c[2]<=hi]; f.sort(key=lambda c:-c[2])
    return f[0] if f else None

def main():
    obj=joblib.load(ROOT/"data"/"_mundial_model_cache.pkl")
    out=[]
    for e1,e2 in PARTIDOS:
        with contextlib.redirect_stdout(io.StringIO()):
            s=predecir_partido_mundial(e1,e2,obj,n_runs=15,fase="Grupos")
        m=s.get("mercados") or {}
        corners=m.get("corners") or {}; amar=m.get("amarillas") or {}; tg=m.get("total_goles") or {}
        cl=linea_ou(corners); al=linea_ou(amar); gl=linea_ou(tg)
        rec={"e1":e1,"e2":e2,
             "goles_esp":round(float(tg.get("esperado",0)),2),
             "corners_esp":round(float(corners.get("esperado",0)),2),
             "amarillas_esp":round(float(amar.get("esperado",0)),2),
             "goles_ou":(gl[0],gl[1],round(gl[2],2)) if gl else None,
             "corners_ou":(cl[0],cl[1],round(cl[2],2)) if cl else None,
             "amarillas_ou":(al[0],al[1],round(al[2],2)) if al else None}
        out.append(rec)
        g=rec["goles_ou"]; print(f'{e1} vs {e2}: G≈{rec["goles_esp"]} C≈{rec["corners_esp"]} A≈{rec["amarillas_esp"]}'
              f'  | {("Goles "+g[0]+" "+str(g[1])+f" {g[2]:.0%}") if g else "-"}', file=sys.stderr)
    (ROOT/"data"/"_pred_mercados_j1.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("guardado data/_pred_mercados_j1.json", file=sys.stderr)

if __name__=="__main__":
    main()
