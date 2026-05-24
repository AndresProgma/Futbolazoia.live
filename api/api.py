"""
API REST — Predicción de Fútbol, Champions League.

Recursos:
  /partidos     — GET, GET{id}, DELETE (soft)  — datos del Excel
  /evaluaciones — POST, GET, GET{id}, DELETE (soft)  — corre main()
  /predicciones — POST, GET, GET{id}, DELETE (soft)  — corre predecir_partido()

ORM: SQLModel + SQLite (futbol.db)
Soft delete: campo activo=False en vez de borrar el registro

Arrancar:
    pip install sqlmodel
    uvicorn api:app --reload
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Field, Session, SQLModel, create_engine, select, func

from ml.knime_workflow_converter import main as run_pipeline, predecir_partido
from ml.predecir_v2 import predecir_partido_v2

# Paths relativos al proyecto raíz (un nivel arriba de api/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "creando_dataset_modificado.xlsx"
DATASET = os.getenv("DATASET_PATH", str(_DEFAULT_DATASET))

# CSVs de contexto (PL) para enriquecer ELO/forma/xG del pipeline UCL
_PL_CONTEXT = [
    str(p) for p in [
        _PROJECT_ROOT / "data" / "premier_2024-25_enriquecido.csv",
        _PROJECT_ROOT / "data" / "premier_2025-26_enriquecido.csv",
    ]
    if p.exists()
]
DATABASE_URL  = os.getenv("DATABASE_URL", f"sqlite:///{_PROJECT_ROOT / 'data' / 'futbol.db'}")
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")

engine = create_engine(DATABASE_URL)

# Pipeline results guardados en memoria (demasiado pesados para DB)
_resultados_pipeline: dict[int, dict] = {}
# Estado de entrenamiento: "training" | "ready" | "error:<msg>"
_training_status: dict[int, str] = {}

# Modelo PL separado (entrenado solo con datos de Premier League, sin contexto UCL)
_PL_KEY = -1          # sentinel: evaluacion_id=-1 → modelo PL
_pl_status: str = "idle"   # "idle" | "training" | "ready" | "error:<msg>"


# ---------------------------------------------------------------------------
# Modelos ORM — cada clase es una tabla en SQLite
# ---------------------------------------------------------------------------

class Partido(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    fase: str
    goles_e1: int
    goles_e2: int
    fecha: Optional[str] = None
    activo: bool = True


class Evaluacion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filepath: str
    resultados: str       # JSON con accuracy por modelo
    cv_resultados: str    # JSON con cross-validation
    activo: bool = True


class Prediccion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    evaluacion_id: int = Field(foreign_key="evaluacion.id")
    n_runs: int
    output: str           # Salida completa de predecir_partido()
    activo: bool = True


class PrediccionTrack(SQLModel, table=True):
    """Predicción guardada para track record público — se compara con el
    resultado real cuando el partido se juegue (auto-resuelta)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    equipo1: str
    equipo2: str
    fecha_partido: Optional[str] = None       # fecha esperada del partido (YYYY-MM-DD)
    fecha_prediccion: str                     # cuándo se hizo la predicción
    evaluacion_id: int = Field(foreign_key="evaluacion.id")
    pred_consenso: str                        # Win / Draw / Loss
    prob_win: float
    prob_draw: float
    prob_loss: float
    elo_e1: float
    elo_e2: float
    fase: str = "Liga"
    competencia: Optional[str] = Field(default="UCL")
    # Auto-resuelto cuando llega el resultado
    resultado_real: Optional[str] = None      # Win / Draw / Loss desde la perspectiva E1
    g1_real: Optional[int] = None
    g2_real: Optional[int] = None
    acierto: Optional[bool] = None
    activo: bool = True


# ---------------------------------------------------------------------------
# Schemas de entrada (no son tablas)
# ---------------------------------------------------------------------------

class PartidoUpdate(SQLModel):
    fase: Optional[str] = None
    goles_e1: Optional[int] = None
    goles_e2: Optional[int] = None
    fecha: Optional[str] = None


class EvaluacionCreate(SQLModel):
    filepath: str = DATASET


class EvaluacionUpdate(SQLModel):
    filepath: str = DATASET


class PrediccionCreate(SQLModel):
    equipo1: str
    equipo2: str
    evaluacion_id: int
    n_runs: int = 20


class PrediccionUpdate(SQLModel):
    equipo1: Optional[str] = None
    equipo2: Optional[str] = None
    n_runs: Optional[int] = None


# ---------------------------------------------------------------------------
# Startup: crear tablas y cargar Excel
# ---------------------------------------------------------------------------

def _cargar_excel(session: Session) -> int:
    """Sincroniza la tabla Partido con el Excel.
    Agrega los partidos que faltan en la DB (incremental). Devuelve el número de
    filas insertadas. Seguro de llamar múltiples veces."""
    try:
        df = pd.read_excel(DATASET)
    except FileNotFoundError:
        return 0

    ids_en_db = set(session.exec(select(Partido.id).where(Partido.activo == True)).all())

    nuevos = 0
    for _, row in df.iterrows():
        pid = int(row["Partido_id"]) if pd.notna(row.get("Partido_id")) else None
        if pid is not None and pid in ids_en_db:
            continue  # ya existe
        goles_e1 = row.get("EQUIPO1_GOLES")
        goles_e2 = row.get("EQUIPO2_GOLES")
        partido = Partido(
            id=pid,
            equipo1=str(row.get("Equipo1", "")),
            equipo2=str(row.get("Equipo2", "")),
            fase=str(row.get("Fase", "")),
            goles_e1=int(goles_e1) if pd.notna(goles_e1) else 0,
            goles_e2=int(goles_e2) if pd.notna(goles_e2) else 0,
            fecha=str(row["Fecha"]) if pd.notna(row.get("Fecha")) else None,
        )
        session.add(partido)
        nuevos += 1

    if nuevos:
        session.commit()
    return nuevos


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    # Migración: agregar columna competencia si no existe (SQLite)
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE predicciontrack ADD COLUMN competencia VARCHAR DEFAULT 'UCL'"))
            conn.commit()
        except Exception:
            pass
    with Session(engine) as session:
        _cargar_excel(session)
    # Entrenar modelo PL en background al arrancar
    import threading
    threading.Thread(target=_run_pl_pipeline_background, daemon=True).start()
    yield


app = FastAPI(title="Predicción Fútbol UCL", lifespan=lifespan)

# CORS — permite que el frontend en otro host (Vercel, Netlify, dominio propio)
# llame a este API. Configurable vía env var ALLOWED_ORIGINS (lista separada por comas).
_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir dashboard estático desde /static y / (raíz redirige a index.html)
_STATIC_DIR = _PROJECT_ROOT / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard_root():
        return FileResponse(_STATIC_DIR / "index.html")


def get_session():
    with Session(engine) as session:
        yield session


def _run_pl_pipeline_background() -> None:
    """Combina los CSVs de PL y entrena un modelo solo con datos de PL (sin contexto UCL)."""
    global _pl_status
    _pl_status = "training"
    try:
        pl_files = [
            _PROJECT_ROOT / "data" / "premier_2024-25_enriquecido.csv",
            _PROJECT_ROOT / "data" / "premier_2025-26_enriquecido.csv",
        ]
        dfs = [pd.read_csv(p) for p in pl_files if p.exists()]
        if not dfs:
            _pl_status = "error:no hay CSVs de PL"
            return
        combined = pd.concat(dfs, ignore_index=True)
        tmp_path = _PROJECT_ROOT / "data" / "_pl_combined.csv"
        combined.to_csv(tmp_path, index=False)
        results = run_pipeline(str(tmp_path), context_filepaths=None)
        _resultados_pipeline[_PL_KEY] = results
        _pl_status = "ready"
        print("✓ Modelo Premier League entrenado y listo")
    except Exception as exc:
        _pl_status = f"error:{exc}"
        print(f"⚠ Error entrenando modelo PL: {exc}")


def _run_pipeline_background(evaluacion_id: int, filepath: str) -> None:
    """Entrena el pipeline en background y actualiza la DB cuando termina.
    Después genera automáticamente el record histórico honesto (predecir_v2)."""
    try:
        results = run_pipeline(filepath, context_filepaths=_PL_CONTEXT)
        with Session(engine) as s:
            ev = s.get(Evaluacion, evaluacion_id)
            if ev:
                ev.resultados    = results["results"].to_json(orient="records")
                ev.cv_resultados = results["cv_results"].to_json(orient="records")
                s.add(ev)
                s.commit()
        _resultados_pipeline[evaluacion_id] = results
        _training_status[evaluacion_id] = "ready"
    except Exception as exc:
        _training_status[evaluacion_id] = f"error:{exc}"
        return

    # Generar record histórico honesto con predecir_v2 (automático, sin botón)
    _generar_record_background(n_partidos=35)


def _get_or_run_pipeline(evaluacion_id: int, session: Session) -> dict:
    """Devuelve los resultados en memoria, o reejecuta el pipeline si se reinició el server.
    evaluacion_id == _PL_KEY (-1) → usa el modelo Premier League."""
    if evaluacion_id == _PL_KEY:
        if _PL_KEY not in _resultados_pipeline:
            if _pl_status == "training":
                raise HTTPException(status_code=503, detail="Modelo Premier League aún entrenando, intenta en unos minutos")
            raise HTTPException(status_code=503, detail=f"Modelo Premier League no disponible: {_pl_status}")
        return _resultados_pipeline[_PL_KEY]
    if evaluacion_id in _resultados_pipeline:
        return _resultados_pipeline[evaluacion_id]
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail=f"Evaluación {evaluacion_id} no encontrada")
    results = run_pipeline(ev.filepath, context_filepaths=_PL_CONTEXT)
    _resultados_pipeline[evaluacion_id] = results
    return results


def _resolve_evaluacion_id(competencia: Optional[str], evaluacion_id: Optional[int], session: Session) -> int:
    """Resuelve qué pipeline usar: PL si la competencia es Premier, si no UCL."""
    if competencia and "premier" in competencia.lower():
        return _PL_KEY
    if evaluacion_id is not None:
        return evaluacion_id
    # Fallback: usar la evaluación UCL más reciente
    ev = session.exec(select(Evaluacion).where(Evaluacion.activo == True).order_by(Evaluacion.id.desc())).first()
    if not ev:
        raise HTTPException(status_code=404, detail="No hay evaluaciones disponibles")
    return ev.id


def _resolver_predicciones_para_partido(partido: Partido, session: Session) -> int:
    """Cuando un partido tiene resultado, marca las predicciones tracked que
    coincidan (mismos equipos en cualquier orden) como acertadas o falladas."""
    tracks = session.exec(
        select(PrediccionTrack)
        .where(PrediccionTrack.activo == True)
        .where(PrediccionTrack.resultado_real == None)
    ).all()

    e1_lo, e2_lo = partido.equipo1.lower(), partido.equipo2.lower()
    resueltas = 0
    for t in tracks:
        t1_lo, t2_lo = t.equipo1.lower(), t.equipo2.lower()
        # Coinciden si los dos equipos son los mismos (en cualquier orden)
        mismo_orden    = (t1_lo == e1_lo and t2_lo == e2_lo)
        orden_invertido = (t1_lo == e2_lo and t2_lo == e1_lo)
        if not (mismo_orden or orden_invertido):
            continue
        if mismo_orden:
            g1, g2 = partido.goles_e1, partido.goles_e2
        else:
            g1, g2 = partido.goles_e2, partido.goles_e1
        real = 'Win' if g1 > g2 else ('Loss' if g1 < g2 else 'Draw')
        t.resultado_real = real
        t.g1_real = g1
        t.g2_real = g2
        t.acierto = (t.pred_consenso == real)
        session.add(t)
        resueltas += 1
    if resueltas:
        session.commit()
    return resueltas


# ---------------------------------------------------------------------------
# /partidos — solo lectura + soft delete
# ---------------------------------------------------------------------------

@app.get("/partidos", response_model=list[Partido])
def listar_partidos(
    equipo1: Optional[str] = None,
    equipo2: Optional[str] = None,
    session: Session = Depends(get_session),
):
    partidos = session.exec(select(Partido).where(Partido.activo == True)).all()

    if equipo1 and equipo2:
        e1, e2 = equipo1.lower(), equipo2.lower()
        partidos = [
            p for p in partidos
            if (e1 in p.equipo1.lower() and e2 in p.equipo2.lower())
            or (e2 in p.equipo1.lower() and e1 in p.equipo2.lower())
        ]
    elif equipo1:
        e1 = equipo1.lower()
        partidos = [p for p in partidos if e1 in p.equipo1.lower() or e1 in p.equipo2.lower()]
    elif equipo2:
        e2 = equipo2.lower()
        partidos = [p for p in partidos if e2 in p.equipo1.lower() or e2 in p.equipo2.lower()]

    return partidos


@app.get("/partidos/{partido_id}", response_model=Partido)
def obtener_partido(partido_id: int, session: Session = Depends(get_session)):
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    return partido


@app.put("/partidos/{partido_id}", response_model=Partido)
def actualizar_partido(partido_id: int, data: PartidoUpdate, session: Session = Depends(get_session)):
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    if data.fase is not None:
        partido.fase = data.fase
    if data.goles_e1 is not None:
        partido.goles_e1 = data.goles_e1
    if data.goles_e2 is not None:
        partido.goles_e2 = data.goles_e2
    if data.fecha is not None:
        partido.fecha = data.fecha
    session.add(partido)
    session.commit()
    session.refresh(partido)
    # Si llegó el resultado real, resolver predicciones tracked
    if data.goles_e1 is not None or data.goles_e2 is not None:
        _resolver_predicciones_para_partido(partido, session)
    return partido


@app.delete("/partidos/{partido_id}", status_code=204)
def desactivar_partido(partido_id: int, session: Session = Depends(get_session)):
    """Soft delete: marca activo=False para mantener historial."""
    partido = session.get(Partido, partido_id)
    if not partido or not partido.activo:
        raise HTTPException(status_code=404, detail="Partido no encontrado")
    partido.activo = False
    session.add(partido)
    session.commit()


# ---------------------------------------------------------------------------
# /evaluaciones — CRUD + soft delete
# ---------------------------------------------------------------------------

@app.post("/evaluaciones", response_model=Evaluacion, status_code=201)
def crear_evaluacion(data: EvaluacionCreate, background_tasks: BackgroundTasks,
                     session: Session = Depends(get_session)):
    """Inicia el pipeline en background y devuelve la evaluación inmediatamente."""
    if not Path(data.filepath).exists():
        raise HTTPException(status_code=422, detail=f"Archivo no encontrado: {data.filepath}")

    evaluacion = Evaluacion(filepath=data.filepath, resultados="[]", cv_resultados="[]")
    session.add(evaluacion)
    session.commit()
    session.refresh(evaluacion)

    _training_status[evaluacion.id] = "training"
    background_tasks.add_task(_run_pipeline_background, evaluacion.id, data.filepath)
    return evaluacion


@app.get("/evaluaciones", response_model=list[Evaluacion])
def listar_evaluaciones(session: Session = Depends(get_session)):
    return session.exec(select(Evaluacion).where(Evaluacion.activo == True)).all()


@app.get("/evaluaciones/{evaluacion_id}", response_model=Evaluacion)
def obtener_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    return ev


@app.put("/evaluaciones/{evaluacion_id}", response_model=Evaluacion)
def actualizar_evaluacion(evaluacion_id: int, data: EvaluacionUpdate, session: Session = Depends(get_session)):
    """Re-ejecuta el pipeline con un nuevo filepath y actualiza los resultados."""
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    try:
        results = run_pipeline(data.filepath)
    except FileNotFoundError:
        raise HTTPException(status_code=422, detail=f"Archivo no encontrado: {data.filepath}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    ev.filepath = data.filepath
    ev.resultados = results["results"].to_json(orient="records")
    ev.cv_resultados = results["cv_results"].to_json(orient="records")
    session.add(ev)
    session.commit()
    session.refresh(ev)
    _resultados_pipeline[evaluacion_id] = results
    return ev


@app.delete("/evaluaciones/{evaluacion_id}", status_code=204)
def desactivar_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    """Soft delete: el historial de accuracy queda en la DB con activo=False."""
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    ev.activo = False
    session.add(ev)
    session.commit()
    _resultados_pipeline.pop(evaluacion_id, None)


# ---------------------------------------------------------------------------
# /predicciones — CRUD + soft delete
# ---------------------------------------------------------------------------

@app.post("/predicciones", response_model=Prediccion, status_code=201)
def crear_prediccion(data: PrediccionCreate, session: Session = Depends(get_session)):
    """Predice un partido usando el modelo de la evaluacion indicada."""
    ev = session.get(Evaluacion, data.evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail=f"Evaluación {data.evaluacion_id} no encontrada")

    results = _get_or_run_pipeline(data.evaluacion_id, session)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        predecir_partido(data.equipo1, data.equipo2, results, n_runs=data.n_runs)

    prediccion = Prediccion(
        equipo1=data.equipo1,
        equipo2=data.equipo2,
        evaluacion_id=data.evaluacion_id,
        n_runs=data.n_runs,
        output=buf.getvalue(),
    )
    session.add(prediccion)
    session.commit()
    session.refresh(prediccion)
    return prediccion


@app.get("/predicciones", response_model=list[Prediccion])
def listar_predicciones(session: Session = Depends(get_session)):
    return session.exec(select(Prediccion).where(Prediccion.activo == True)).all()


@app.get("/predicciones/{prediccion_id}", response_model=Prediccion)
def obtener_prediccion(prediccion_id: int, session: Session = Depends(get_session)):
    pred = session.get(Prediccion, prediccion_id)
    if not pred or not pred.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    return pred


@app.put("/predicciones/{prediccion_id}", response_model=Prediccion)
def actualizar_prediccion(prediccion_id: int, data: PrediccionUpdate, session: Session = Depends(get_session)):
    """Re-ejecuta predecir_partido con los nuevos parámetros y actualiza el resultado."""
    pred = session.get(Prediccion, prediccion_id)
    if not pred or not pred.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    if pred.evaluacion_id not in _resultados_pipeline:
        raise HTTPException(
            status_code=409,
            detail="El pipeline no está en memoria — vuelve a ejecutar POST /evaluaciones",
        )
    if data.equipo1 is not None:
        pred.equipo1 = data.equipo1
    if data.equipo2 is not None:
        pred.equipo2 = data.equipo2
    if data.n_runs is not None:
        pred.n_runs = data.n_runs
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        predecir_partido(pred.equipo1, pred.equipo2, _resultados_pipeline[pred.evaluacion_id], n_runs=pred.n_runs)
    pred.output = buf.getvalue()
    session.add(pred)
    session.commit()
    session.refresh(pred)
    return pred


@app.delete("/predicciones/{prediccion_id}", status_code=204)
def desactivar_prediccion(prediccion_id: int, session: Session = Depends(get_session)):
    """Soft delete: la predicción queda en la DB con activo=False."""
    pred = session.get(Prediccion, prediccion_id)
    if not pred or not pred.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    pred.activo = False
    session.add(pred)
    session.commit()


# ---------------------------------------------------------------------------
# Endpoints "dashboard" — devuelven JSON estructurado para el frontend
# ---------------------------------------------------------------------------

@app.get("/api/evaluaciones/{evaluacion_id}/status")
def estado_entrenamiento(evaluacion_id: int):
    """Devuelve el estado del entrenamiento: training | ready | error:<msg>"""
    status = _training_status.get(evaluacion_id, "ready")
    return {"id": evaluacion_id, "status": status}


@app.get("/api/equipos")
def listar_equipos(session: Session = Depends(get_session)):
    """Lista única de equipos del dataset UCL + PL (para los dropdowns del predictor)."""
    partidos = session.exec(select(Partido).where(Partido.activo == True)).all()
    ucl_equipos = sorted({p.equipo1 for p in partidos} | {p.equipo2 for p in partidos})

    pl_equipos: set = set()
    for csv_path in _PL_CONTEXT:
        try:
            df_pl = pd.read_csv(csv_path, usecols=["Equipo1", "Equipo2"])
            pl_equipos |= set(df_pl["Equipo1"].dropna()) | set(df_pl["Equipo2"].dropna())
        except Exception:
            pass

    todos = sorted(set(ucl_equipos) | pl_equipos)
    return {"equipos": todos, "ucl": ucl_equipos, "pl": sorted(pl_equipos), "total": len(todos)}


@app.get("/api/evaluaciones/{evaluacion_id}/elos")
def ranking_elo(evaluacion_id: int, session: Session = Depends(get_session)):
    """Ranking ELO de todos los equipos según la evaluación indicada."""
    results = _get_or_run_pipeline(evaluacion_id, session)
    team_elos = results.get("team_elos", {})
    ranking = sorted(
        ({"equipo": t, "elo": round(float(e), 1)} for t, e in team_elos.items()),
        key=lambda x: -x["elo"],
    )
    return {"ranking": ranking, "total": len(ranking)}


@app.get("/api/evaluaciones/{evaluacion_id}/metricas")
def metricas_evaluacion(evaluacion_id: int, session: Session = Depends(get_session)):
    """Resultados de modelos: CV (walk-forward) + test cronológico."""
    ev = session.get(Evaluacion, evaluacion_id)
    if not ev or not ev.activo:
        raise HTTPException(status_code=404, detail="Evaluación no encontrada")
    n_partidos = session.exec(
        select(func.count(Partido.id)).where(Partido.activo == True)
    ).one()
    return {
        "evaluacion_id": evaluacion_id,
        "n_partidos": n_partidos,                  # total usado en el walk-forward
        "test": json.loads(ev.resultados),         # accuracy + F1 macro en test único
        "cv": json.loads(ev.cv_resultados),        # CV walk-forward
    }


@app.get("/api/evaluaciones/{evaluacion_id}/feature-importance")
def feature_importance(evaluacion_id: int, top: int = 15, session: Session = Depends(get_session)):
    """Top features del Random Forest (las que el modelo más usa)."""
    results = _get_or_run_pipeline(evaluacion_id, session)
    fi = results.get("feature_importance", []) or []
    return {"top_features": fi[:top], "total": len(fi)}


@app.get("/api/evaluaciones/{evaluacion_id}/predicciones-test")
def predicciones_test(evaluacion_id: int, session: Session = Depends(get_session)):
    """Predicciones del test cronológico — usa record_historico.json si existe."""
    if RECORD_HISTORICO_FILE.exists():
        data = json.loads(RECORD_HISTORICO_FILE.read_text(encoding="utf-8"))
        return {"predicciones": data, "tipo": "v2_honesto"}
    results = _get_or_run_pipeline(evaluacion_id, session)
    pred_df = results.get("predictions")
    if pred_df is None:
        raise HTTPException(status_code=500, detail="No hay predicciones de test")
    return {"predicciones": json.loads(pred_df.to_json(orient="records")), "tipo": "test_split"}


class PrediccionRapida(SQLModel):
    equipo1: str
    equipo2: str
    evaluacion_id: Optional[int] = None
    n_runs: int = 20
    fase: str = "Liga"
    competencia: Optional[str] = "UCL"


@app.post("/api/predecir")
def predecir_rapido(data: PrediccionRapida, session: Session = Depends(get_session)):
    """
    Predicción estructurada en JSON (no guarda en DB).
    Devuelve probabilidades por modelo, consenso, marcadores y ELO.
    """
    eid = _resolve_evaluacion_id(data.competencia, data.evaluacion_id, session)
    results = _get_or_run_pipeline(eid, session)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        salida = predecir_partido(
            data.equipo1, data.equipo2, results,
            n_runs=data.n_runs, fase=data.fase,
        )
    if salida is None:
        raise HTTPException(status_code=500, detail="predecir_partido() no devolvió datos")
    salida["log"] = buf.getvalue()
    return salida


@app.get("/api/health")
def health():
    return {"status": "ok", "evaluaciones_en_memoria": list(_resultados_pipeline.keys())}


# ---------------------------------------------------------------------------
# Cuotas reales — The Odds API
# ---------------------------------------------------------------------------

_SPORT_KEY = {
    "ucl":            "soccer_uefa_champs_league",
    "premier league": "soccer_epl",
    "laliga":         "soccer_spain_la_liga",
    "la liga":        "soccer_spain_la_liga",
    "bundesliga":     "soccer_germany_bundesliga",
    "serie a":        "soccer_italy_serie_a",
    "ligue 1":        "soccer_france_ligue_one",
}
_odds_cache: dict = {}   # clave → (timestamp, datos)
_ODDS_TTL = 600          # segundos (10 min)


def _sport_key(competencia: str) -> str:
    return _SPORT_KEY.get(competencia.lower(), "soccer_uefa_champs_league")


def _best_match(name: str, candidates: list[str]) -> Optional[str]:
    """Devuelve el candidato más parecido al nombre dado (fuzzy)."""
    from difflib import get_close_matches
    hits = get_close_matches(name.lower(), [c.lower() for c in candidates], n=1, cutoff=0.6)
    if not hits:
        return None
    return candidates[[c.lower() for c in candidates].index(hits[0])]


@app.get("/api/odds")
def get_odds(equipo1: str, equipo2: str, competencia: str = "UCL"):
    """
    Devuelve cuotas reales de The Odds API para el partido dado.
    Requiere ODDS_API_KEY en .env. Cachea 10 minutos.
    """
    import time, urllib.request, urllib.parse

    if not ODDS_API_KEY:
        return {"disponible": False, "motivo": "ODDS_API_KEY no configurada"}

    sport = _sport_key(competencia)
    cache_key = sport
    now = time.time()

    # Usar caché si está fresca
    if cache_key in _odds_cache:
        ts, data = _odds_cache[cache_key]
        if now - ts < _ODDS_TTL:
            eventos = data
        else:
            eventos = None
    else:
        eventos = None

    if eventos is None:
        try:
            params = urllib.parse.urlencode({
                "apiKey":      ODDS_API_KEY,
                "regions":     "eu",
                "markets":     "h2h",
                "oddsFormat":  "decimal",
                "dateFormat":  "iso",
            })
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?{params}"
            with urllib.request.urlopen(url, timeout=8) as resp:
                eventos = json.loads(resp.read())
            _odds_cache[cache_key] = (now, eventos)
        except Exception as exc:
            return {"disponible": False, "motivo": str(exc)}

    # Buscar el partido por fuzzy match de nombres de equipos
    nombres_api = [(e["home_team"], e["away_team"], e) for e in eventos]
    todos_nombres = list({n for h, a, _ in nombres_api for n in [h, a]})

    home_match = _best_match(equipo1, todos_nombres)
    away_match = _best_match(equipo2, todos_nombres)

    evento = None
    for home, away, ev in nombres_api:
        if home_match and away_match:
            if home.lower() == home_match.lower() and away.lower() == away_match.lower():
                evento = ev
                break
        # También probar orden inverso
        if home_match and away_match:
            if home.lower() == away_match.lower() and away.lower() == home_match.lower():
                evento = ev
                break

    if not evento:
        return {"disponible": False, "motivo": f"Partido no encontrado en The Odds API ({equipo1} vs {equipo2})"}

    # Extraer cuotas promedio entre bookmakers
    cuotas_win, cuotas_draw, cuotas_loss = [], [], []
    for bm in evento.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            for outcome in market["outcomes"]:
                n = outcome["name"].lower()
                p = outcome["price"]
                if n == evento["home_team"].lower():
                    cuotas_win.append(p)
                elif n == evento["away_team"].lower():
                    cuotas_loss.append(p)
                elif n == "draw":
                    cuotas_draw.append(p)

    avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else None
    return {
        "disponible":   True,
        "home_team":    evento["home_team"],
        "away_team":    evento["away_team"],
        "commence":     evento.get("commence_time", ""),
        "cuota_win":    avg(cuotas_win),
        "cuota_draw":   avg(cuotas_draw),
        "cuota_loss":   avg(cuotas_loss),
        "bookmakers":   len(evento.get("bookmakers", [])),
    }


@app.get("/api/pl/status")
def pl_status():
    """Estado del modelo Premier League."""
    return {"status": _pl_status, "listo": _PL_KEY in _resultados_pipeline}


@app.post("/api/pl/entrenar", status_code=202)
def pl_entrenar():
    """Re-entrena el modelo Premier League en background."""
    import threading
    if _pl_status == "training":
        return {"mensaje": "Ya está entrenando"}
    threading.Thread(target=_run_pl_pipeline_background, daemon=True).start()
    return {"mensaje": "Entrenamiento iniciado"}


@app.post("/api/sync")
def sync_dataset(session: Session = Depends(get_session)):
    """Sincroniza la tabla Partido con el Excel: agrega los partidos nuevos.
    Llamar después de agregar partidos con agregar_partido.py para reflejar
    los cambios en el dashboard sin reiniciar el servidor."""
    nuevos = _cargar_excel(session)
    total = session.exec(select(func.count(Partido.id)).where(Partido.activo == True)).one()
    return {"nuevos_agregados": nuevos, "total_partidos": total}


# ---------------------------------------------------------------------------
# Track record — predicciones públicas vs resultados reales (monetización)
# ---------------------------------------------------------------------------

class PrediccionTrackCreate(SQLModel):
    equipo1: str
    equipo2: str
    fecha_partido: Optional[str] = None
    evaluacion_id: Optional[int] = None
    n_runs: int = 20
    fase: str = "Liga"
    competencia: Optional[str] = "UCL"


@app.post("/api/track", response_model=PrediccionTrack, status_code=201)
def crear_prediccion_track(data: PrediccionTrackCreate, session: Session = Depends(get_session)):
    """Predice un partido futuro y lo guarda para tracking. Si el partido ya
    está en la DB con resultado, se autoresuelve inmediatamente."""
    from datetime import date
    eid = _resolve_evaluacion_id(data.competencia, data.evaluacion_id, session)
    results = _get_or_run_pipeline(eid, session)
    salida = predecir_partido(
        data.equipo1, data.equipo2, results,
        n_runs=data.n_runs, fase=data.fase,
    )
    if salida is None:
        raise HTTPException(status_code=500, detail="predecir_partido() no devolvió datos")

    track = PrediccionTrack(
        equipo1=data.equipo1,
        equipo2=data.equipo2,
        fecha_partido=data.fecha_partido,
        fecha_prediccion=str(date.today()),
        evaluacion_id=data.evaluacion_id,
        pred_consenso=salida["consenso"]["pred"],
        prob_win=salida["consenso"]["win"],
        prob_draw=salida["consenso"]["draw"],
        prob_loss=salida["consenso"]["loss"],
        elo_e1=salida["elo_e1"],
        elo_e2=salida["elo_e2"],
        fase=data.fase,
        competencia=data.competencia or "UCL",
    )
    session.add(track)
    session.commit()
    session.refresh(track)

    # Si el partido ya existe en /partidos con resultado, autorresolverlo
    partidos = session.exec(select(Partido).where(Partido.activo == True)).all()
    e1_lo, e2_lo = data.equipo1.lower(), data.equipo2.lower()
    for p in partidos:
        p1_lo, p2_lo = p.equipo1.lower(), p.equipo2.lower()
        if {p1_lo, p2_lo} == {e1_lo, e2_lo} and (p.goles_e1 or p.goles_e2):
            _resolver_predicciones_para_partido(p, session)
            session.refresh(track)
            break

    return track


@app.get("/api/track", response_model=list[PrediccionTrack])
def listar_track(session: Session = Depends(get_session)):
    return session.exec(
        select(PrediccionTrack).where(PrediccionTrack.activo == True)
    ).all()


@app.delete("/api/track/{track_id}", status_code=204)
def desactivar_track(track_id: int, session: Session = Depends(get_session)):
    t = session.get(PrediccionTrack, track_id)
    if not t or not t.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    t.activo = False
    session.add(t)
    session.commit()


class ResolverTrackBody(SQLModel):
    resultado_real: str
    g1_real: int = 0
    g2_real: int = 0


@app.post("/api/track/{track_id}/resolver", response_model=PrediccionTrack)
def resolver_manual_track(track_id: int, data: ResolverTrackBody, session: Session = Depends(get_session)):
    """Resolver manualmente una predicción del track record (admin)."""
    t = session.get(PrediccionTrack, track_id)
    if not t or not t.activo:
        raise HTTPException(status_code=404, detail="Predicción no encontrada")
    t.resultado_real = data.resultado_real
    t.g1_real = data.g1_real
    t.g2_real = data.g2_real
    t.acierto = (data.resultado_real == t.pred_consenso)
    session.add(t)
    session.commit()
    session.refresh(t)
    return t


@app.get("/api/track/stats")
def stats_track_record(session: Session = Depends(get_session)):
    """Estadísticas agregadas del track record público (lo que verá la audiencia)."""
    tracks = session.exec(
        select(PrediccionTrack).where(PrediccionTrack.activo == True)
    ).all()
    resueltas = [t for t in tracks if t.resultado_real is not None]
    pendientes = [t for t in tracks if t.resultado_real is None]
    aciertos = [t for t in resueltas if t.acierto]
    total = len(resueltas)
    accuracy = (len(aciertos) / total) if total else 0.0

    # Accuracy por clase predicha
    por_clase = {}
    for cls in ('Win', 'Draw', 'Loss'):
        sub = [t for t in resueltas if t.pred_consenso == cls]
        ok = [t for t in sub if t.acierto]
        por_clase[cls] = {
            'total': len(sub),
            'aciertos': len(ok),
            'accuracy': round((len(ok) / len(sub)) if sub else 0.0, 4),
        }

    return {
        'total_predicciones': len(tracks),
        'resueltas': total,
        'pendientes': len(pendientes),
        'aciertos': len(aciertos),
        'accuracy_global': round(accuracy, 4),
        'por_clase_predicha': por_clase,
    }


# ---------------------------------------------------------------------------
# Featured Pick del Día (admin lo configura, público lo ve)
# ---------------------------------------------------------------------------

FEATURED_PICK_FILE  = _PROJECT_ROOT / "data" / "featured_pick.json"
RECORD_HISTORICO_FILE = _PROJECT_ROOT / "data" / "record_historico.json"

# Estado del generador de record: "idle" | "running" | "done" | "error:<msg>"
_record_status: dict = {"status": "idle", "progreso": 0, "total": 0}


@app.get("/admin")
def admin_page():
    return FileResponse("static/admin.html")


@app.get("/api/featured-pick")
def get_featured_pick():
    """Retorna el partido destacado del día configurado por el admin."""
    if FEATURED_PICK_FILE.exists():
        return json.loads(FEATURED_PICK_FILE.read_text(encoding="utf-8"))
    return {}


class ValorBet(SQLModel):
    nombre: str
    descripcion: str = ""
    porcentaje: int


class FeaturedPickBody(SQLModel):
    equipo1: str
    equipo2: str
    hora: str = "21:00"
    fase: str = "UCL"
    competencia: Optional[str] = "UCL"
    prob_win: float
    prob_draw: float
    prob_loss: float
    valores: list[ValorBet] = []
    mercados: Optional[dict] = None
    modelos: Optional[list] = None
    goles: Optional[list] = None
    ultimos_e1: Optional[list] = None
    ultimos_e2: Optional[list] = None
    stats_equipos: Optional[dict] = None


@app.delete("/api/admin/featured-pick", status_code=204)
def delete_featured_pick():
    """Elimina el pick del día."""
    if FEATURED_PICK_FILE.exists():
        FEATURED_PICK_FILE.unlink()


@app.post("/api/admin/featured-pick")
def set_featured_pick(data: FeaturedPickBody):
    """Guarda el partido destacado del día (solo admin via frontend)."""
    FEATURED_PICK_FILE.write_text(
        json.dumps(data.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


# ---------------------------------------------------------------------------
# Record histórico honesto (predecir_v2 para los últimos N partidos)
# ---------------------------------------------------------------------------

def _generar_record_background(n_partidos: int) -> None:
    """Corre predecir_partido_v2 para los últimos n_partidos del dataset."""
    global _record_status
    _record_status = {"status": "running", "progreso": 0, "total": n_partidos}
    try:
        df = pd.read_excel(DATASET)
        if "Fecha" in df.columns:
            df["_f"] = pd.to_datetime(df["Fecha"], errors="coerce")
            df = df.sort_values("_f").drop(columns=["_f"]).reset_index(drop=True)

        ultimos = df.tail(n_partidos).reset_index(drop=True)
        resultados = []

        for i, (_, row) in enumerate(ultimos.iterrows()):
            e1   = str(row["Equipo1"])
            e2   = str(row["Equipo2"])
            fecha = str(row.get("Fecha", ""))[:10] if pd.notna(row.get("Fecha")) else None
            fase  = str(row.get("Fase", "Liga"))

            try:
                pred = predecir_partido_v2(
                    e1, e2, fecha=fecha, fase=fase, n_runs=10
                )
                g1r = int(row["EQUIPO1_GOLES"]) if pd.notna(row.get("EQUIPO1_GOLES")) else None
                g2r = int(row["EQUIPO2_GOLES"]) if pd.notna(row.get("EQUIPO2_GOLES")) else None
                resultado_real = pred.get("resultado_real") or (
                    "Win" if (g1r is not None and g2r is not None and g1r > g2r) else
                    "Loss" if (g1r is not None and g2r is not None and g1r < g2r) else
                    "Draw" if (g1r is not None and g2r is not None) else None
                )
                resultados.append({
                    "equipo1":        e1,
                    "equipo2":        e2,
                    "fecha":          fecha or "—",
                    "fase":           fase,
                    "goles_real":     f"{g1r}–{g2r}" if g1r is not None else "—",
                    "resultado_real": resultado_real,
                    "consenso":       pred["consenso"],
                    "modelos":        pred.get("modelos", []),
                    "acierto":        pred.get("acierto_consenso"),
                })
            except Exception as exc:
                resultados.append({
                    "equipo1": e1, "equipo2": e2, "fecha": fecha or "—",
                    "error": str(exc),
                })

            _record_status["progreso"] = i + 1

        RECORD_HISTORICO_FILE.write_text(
            json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _record_status["status"] = "done"
    except Exception as exc:
        _record_status["status"] = f"error:{exc}"


@app.get("/api/admin/record-status")
def record_status_endpoint():
    return _record_status


@app.get("/api/record")
def record_publico(session: Session = Depends(get_session)):
    """Record histórico: primero el archivo v2 (honesto), luego el test split."""
    if RECORD_HISTORICO_FILE.exists():
        data = json.loads(RECORD_HISTORICO_FILE.read_text(encoding="utf-8"))
        return {"tipo": "v2_honesto", "predicciones": data}
    # Fallback: test split del último entrenamiento
    evals = session.exec(select(Evaluacion).where(Evaluacion.activo == True)).all()
    if not evals:
        return {"tipo": "sin_datos", "predicciones": []}
    ev = max(evals, key=lambda e: e.id)
    results = _get_or_run_pipeline(ev.id, session)
    pred_df = results.get("predictions")
    if pred_df is None:
        return {"tipo": "sin_datos", "predicciones": []}
    return {"tipo": "test_split", "predicciones": json.loads(pred_df.to_json(orient="records"))}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
