# Futbolazoia.live

> Predicción de partidos de fútbol con Machine Learning, dashboard web interactivo y API REST.
> Vivo en **<https://futbolazoia.live>**.

Sistema completo de predicción para tres dominios distintos:

- **UEFA Champions League** (modelo principal)
- **Premier League** (modelo independiente)
- **Selecciones nacionales** — Mundial 2026, Eurocopa, Copa América, Eliminatorias, AFCON, Gold Cup, Asian Cup, Nations League

Cada dominio scrapea sus propias fuentes (UEFA.com y SofaScore vía Playwright), calcula features pre-partido sin leakage (ELO, forma rolling, head-to-head, xG sintético, coef. UEFA/FIFA), entrena un ensemble de 6 clasificadores + Dixon-Coles y expone todo en un dashboard responsive con track record público auto-resuelto.

---

## Tabla de contenidos

1. [Highlights](#highlights)
2. [Cómo funciona el pipeline ML](#cómo-funciona-el-pipeline-ml)
3. [Arquitectura del proyecto](#arquitectura-del-proyecto)
4. [Estructura de directorios](#estructura-de-directorios)
5. [Detalle de cada módulo](#detalle-de-cada-módulo)
6. [Stack tecnológico](#stack-tecnológico)
7. [Setup local](#setup-local)
8. [Endpoints de la API](#endpoints-de-la-api)
9. [Flujo de datos end-to-end](#flujo-de-datos-end-to-end)
10. [Métricas actuales](#métricas-actuales)
11. [Deploy](#deploy)
12. [Roadmap / Ideas pendientes](#roadmap--ideas-pendientes)

---

## Highlights

- **3 modelos independientes** ejecutándose en paralelo en el mismo proceso: UCL (Excel), Premier League (CSV) y Selecciones (CSV combinado con 2.337 partidos).
- **Ensemble de 6 clasificadores** (Random Forest, Gradient Boosting, Logistic Regression, SVM, XGBoost, KNN) + **Dixon-Coles bivariate Poisson** como modelo base estadístico.
- **Calibración isotónica** (`CalibratedClassifierCV`, cv=3) sobre cada clasificador → las probabilidades reflejan frecuencias reales (P(Win)=70% ≈ 70% de aciertos empíricos).
- **Walk-forward Cross-Validation** con `TimeSeriesSplit` (no random shuffle): respeta el orden temporal real.
- **Features sin leakage**: ELO incremental, forma últimos 5 partidos, H2H últimos 3, días de descanso, xG sintético rolling, coeficiente UEFA/ranking FIFA.
- **Predicción honesta (`predecir_v2`)**: filtra el dataset para usar solo partidos jugados ANTES de la fecha del partido a predecir — simula el modelo "como si fuera hoy".
- **Track record público auto-resuelto**: guardás una predicción y se compara automáticamente con el resultado real cuando el partido se juega.
- **Featured pick diario** con histórico de aciertos.
- **Dashboard responsive** sin build step (HTML + Tailwind CDN + Chart.js).
- **API REST documentada** con SQLModel + SQLite, soft delete, filtros por equipo.
- **Deploy listo**: Dockerfile multi-stage + Render Blueprint con dominio custom y healthcheck.

---

## Cómo funciona el pipeline ML

El cerebro del proyecto está en `ml/knime_workflow_converter.py` (~1.992 líneas). Ejecuta este flujo por cada dominio:

### 1. Carga y limpieza

Lee Excel/CSV con `pandas`, ordena por fecha y limpia valores faltantes con la mediana de la columna (`handle_missing_values`).

### 2. Cálculo de features pre-partido (sin leakage)

Cada función recorre los partidos en orden cronológico y, para cada uno, calcula features usando **solo lo que ocurrió antes**:

| Función | Qué calcula |
|---|---|
| `compute_elo_features` | ELO incremental tipo Elo (K=30, ventaja local=60 pts, bonus por margen de goles). Mantiene `team_elos` actualizado tras cada partido. |
| `compute_form_features` | Forma últimos 5: victorias, empates, derrotas, goles a favor/contra, puntos acumulados. |
| `compute_h2h_features` | Head-to-head: resultados de los últimos 3 enfrentamientos directos entre los dos equipos. |
| `compute_uefa_coef_features` | Tabla estática del coeficiente UEFA (~50 clubes). Crítico para "cold start" de equipos sin historial reciente. |
| `compute_xg_features` | xG sintético rolling de 5 partidos: `xG = 0.05·Disparos + 0.20·Disparos_a_puerta + 0.55·Oportunidades_claras`. Pesos calibrados para que xG promedio ≈ goles promedio. |
| `compute_media11_rolling_features` | Promedio rolling del rating de los 11 titulares (cuando hay lineups disponibles). |
| `compute_market_rolling_features` | Rolling de mercados derivados (más/menos 2.5 goles, BTTS, corners, tarjetas) para predicciones especiales. |

### 3. Selección de columnas (anti-leakage)

`select_columns` descarta **todas las stats post-partido** que no estarían disponibles antes de jugar (posesión real, disparos a puerta del partido, etc.). Solo quedan las features derivadas pre-partido.

### 4. Entrenamiento del ensemble

`train_models` arma un `Pipeline` para cada clasificador con:

1. **`SelectKBest(f_classif, k=20)`** — selecciona las 20 mejores features por ANOVA-F dentro de cada fold de CV (evita curse-of-dimensionality y selección de features con leakage).
2. **`class_weight={0:1.5, 1:1, 2:1}`** — boost a la clase Draw para que el modelo no colapse prediciendo solo Win/Loss (la clase Draw es ~20% del dataset).
3. **`CalibratedClassifierCV(method='isotonic', cv=3)`** — calibración isotónica que mapea las probas crudas (a menudo sobre-confiadas en RF/SVM/XGB) a la frecuencia empírica real.

Los 6 clasificadores son: **Random Forest, Gradient Boosting, Logistic Regression, SVM (RBF), XGBoost, KNN**.

### 5. Modelo Dixon-Coles paralelo

`ml/dixon_coles.py` implementa el modelo bivariate Poisson clásico de Dixon-Coles (1997). Cada equipo tiene fuerza de ataque α y defensa β; los goles esperados son λ = α_E1 · β_E2 · γ (home advantage). Incluye corrección τ para 0-0, 0-1, 1-0, 1-1 que captura la correlación negativa en scores bajos. MLE con SLSQP.

Sus predicciones se promedian con las del ensemble para el consenso final.

### 6. Validación temporal honesta

`cross_validate_models` corre `TimeSeriesSplit` (walk-forward, no random) con 3 folds. Esto simula el caso real: entrenás con lo viejo, predecís lo nuevo.

### 7. Regresores de marcador

`train_regressors` entrena dos `XGBRegressor` y dos `RandomForestRegressor` para predecir goles esperados de cada equipo. Se usa para el marcador estimado y para alimentar la simulación de mercados (Poisson sobre los goles esperados).

### 8. Predicción de un partido nuevo

`predecir_partido(equipo1, equipo2, n_runs=20)` entrena cada clasificador **20 veces con seeds distintos** y promedia las probabilidades. El consenso final es el `argmax` del promedio. Devuelve un dict estructurado con:

- Probas por modelo (Win/Draw/Loss)
- Consenso del ensemble
- Marcador estimado (mediana de los regresores) + desvío estándar
- ELOs actuales de ambos equipos
- Probabilidades por mercado (más/menos 2.5 goles, BTTS, corners, etc.)

### 9. Predicción honesta (`predecir_v2.py`)

Variante crítica: antes de predecir, filtra el dataset para usar **solo los partidos jugados ANTES de la fecha del partido a predecir**. Esto simula lo que el modelo realmente vería en producción y elimina cualquier sospecha de leakage de información futura.

---

## Arquitectura del proyecto

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CLIENTE (Browser)                                │
│                  static/index.html + js/                             │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ HTTPS
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI (api/api.py)                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Endpoints REST  •  Static files  •  CORS  •  Healthcheck      │  │
│  └────────┬─────────────────┬─────────────────┬─────────────────┬─┘  │
│           │                 │                 │                 │    │
│           ▼                 ▼                 ▼                 ▼    │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  ┌─────────┐  │
│  │ Pipeline UCL│   │ Pipeline PL  │   │ Pipeline Mund│  │ SQLite  │  │
│  │  (Excel)    │   │  (CSV)       │   │  (CSV)       │  │ futbol  │  │
│  └─────┬───────┘   └──────┬───────┘   └──────┬───────┘  │  .db    │  │
│        │                  │                  │           └─────────┘  │
│        └──────────────────┴──────────────────┘                       │
│                            │                                         │
│                            ▼                                         │
│              ml/knime_workflow_converter.py                          │
│              (ensemble + Dixon-Coles)                                │
└──────────────────────────────────────────────────────────────────────┘

         ▲ datos                                       datos ▲
         │                                                   │
┌────────┴────────┐  ┌──────────────────┐  ┌────────────────┴──┐
│ scraper_uefa.py │  │ scraper_premier  │  │ scraper_seleccion │
│ (Playwright)    │  │ (SofaScore)      │  │ (SofaScore)       │
└─────────────────┘  └──────────────────┘  └───────────────────┘
```

---

## Estructura de directorios

```
Futbolazoia.live/
│
├── api/                          # API REST con FastAPI
│   └── api.py                    # ~1.234 líneas — endpoints + ORM + lifecycle
│
├── ml/                           # Núcleo de Machine Learning
│   ├── knime_workflow_converter.py   # Pipeline completo (~1.992 líneas)
│   ├── dixon_coles.py                # Modelo bivariate Poisson
│   ├── predecir_v2.py                # Predicción honesta (filtra por fecha)
│   ├── pipeline_mundial.py           # Pipeline para selecciones nacionales
│   └── _generar_record.py            # Genera record_historico.json
│
├── scraper/                      # Scrapers Playwright
│   ├── scraper_uefa.py               # UEFA.com (UCL)
│   ├── scraper_premier.py            # SofaScore (Premier League)
│   ├── scraper_selecciones.py        # SofaScore (todas las selecciones)
│   ├── enriquecer_premier.py         # Enriquece CSV PL con stats faltantes
│   └── enriquecer_lineups.py         # Agrega ratings de los 11 titulares
│
├── scripts/                      # Utilidades CLI
│   ├── agregar_partido.py            # Agrega partidos al dataset (3 modos)
│   ├── predecir_y_agregar.py         # Predict-before-add (track record honesto)
│   ├── completar_temporada.py        # Backfill de una temporada completa
│   ├── scraper_temporada_v2.py       # Scrape masivo por temporada
│   ├── buscar_torneos_ss.py          # Descubre torneos en SofaScore
│   └── descubrir_torneos_ss.py       # Lista temporadas disponibles
│
├── static/                       # Dashboard web (sin build step)
│   ├── index.html                    # ~2.511 líneas — markup principal
│   ├── admin.html                    # Panel de administración
│   ├── css/styles.css                # Estilos custom
│   └── js/
│       ├── api.js                    # Cliente HTTP centralizado
│       └── app.js                    # Lógica del dashboard
│
├── data/                         # Datasets y bases de datos
│   ├── creando_dataset_modificado.xlsx       # Dataset principal UCL
│   ├── premier_2024-25_enriquecido.csv       # PL temporada 24-25
│   ├── premier_2025-26_enriquecido.csv       # PL temporada 25-26
│   ├── selecciones_combinado.csv             # 2.337 partidos de selecciones
│   ├── selecciones_<torneo>_<año>.csv        # CSVs individuales por torneo
│   ├── record_historico.json                 # Track record histórico
│   ├── track_record_predictions.csv          # Predicciones predict-before-add
│   ├── featured_pick.json                    # Pick destacado del día
│   └── futbol.db                             # SQLite (se crea al arrancar)
│
├── experimentos/                 # Experimentación ML reproducible
│   ├── harness.py                    # Framework de evaluación
│   ├── RESUMEN.md                    # Bitácora de experimentos
│   ├── baseline.json                 # Métricas baseline
│   ├── v14_produccion_final.json     # Config actual en producción
│   ├── eval_*.py                     # Scripts de evaluación por hipótesis
│   └── resultados.csv                # Resultados acumulados
│
├── Dockerfile                    # Imagen Python 3.12-slim multi-stage
├── render.yaml                   # Blueprint de Render.com
├── requirements.txt              # Dependencias Python
└── README.md
```

---

## Detalle de cada módulo

### `api/api.py` — FastAPI + SQLModel (~1.234 líneas)

API REST que expone los tres pipelines ML como servicios web y sirve el dashboard estático.

**4 tablas SQLite** (todas con campo `activo: bool` para soft delete):

| Tabla | Función |
|---|---|
| `Partido` | Autocarga desde Excel al arranque si está vacía. Sincronización incremental. |
| `Evaluacion` | Cada corrida del pipeline guarda accuracy + CV results en JSON. |
| `Prediccion` | Historial de predicciones one-off (captura el stdout de `predecir_partido`). |
| `PrediccionTrack` | Predicciones públicas con auto-resolución cuando llega el resultado real. |

**Cache en memoria**: `_resultados_pipeline: dict[int, dict]` guarda los modelos entrenados (son demasiado pesados para serializar a DB). Si el server se reinicia (cold start de Render), `_get_or_run_pipeline()` los reentrena automáticamente.

**Entrenamiento en background**: al arrancar, el `lifespan` lanza dos threads daemon que entrenan los modelos de PL y Mundial en paralelo, sin bloquear el startup. Los estados (`_pl_status`, `_mundial_status`) se exponen vía `/api/pl/status` y `/api/mundial/estado`.

**Auto-resolución del track record** (la pieza más interesante): cuando un `PUT /partidos/{id}` recibe goles nuevos, el endpoint busca tracks pendientes con esos dos equipos (en cualquier orden), invierte los goles si el orden está al revés, y marca `acierto = (pred_consenso == resultado_real)`.

**CORS configurable** vía env var `ALLOWED_ORIGINS` (lista separada por comas, o `*`).

### `ml/knime_workflow_converter.py` — Pipeline ML (~1.992 líneas)

El cerebro. Originalmente fue una conversión de un workflow KNIME (de ahí el nombre, ya legacy), hoy es un pipeline ML completo en Python puro. Ver sección [Cómo funciona el pipeline ML](#cómo-funciona-el-pipeline-ml).

Outputs CSV generados al correr `main()`:
- `model_results.csv` — accuracy + F1 macro por modelo en test cronológico
- `predictions.csv` — test set vs predicción de cada modelo
- `processed_data.csv` — dataset con todas las features derivadas

### `ml/dixon_coles.py` — Modelo estadístico clásico

Implementación del modelo Dixon-Coles 1997. MLE con SLSQP, restricción `mean(α)=1` para identificabilidad. Probabilidades de Win/Draw/Loss + distribución completa de goles.

Su importancia: cuando un equipo tiene poca historia (cold start), los modelos ML sobre-confían. Dixon-Coles, al ser paramétrico y trabajar con ataque/defensa explícitos, es más robusto en esos casos.

### `ml/predecir_v2.py` — Predicción honesta

Wrapper sobre `predecir_partido` que **filtra el dataset por fecha** antes de entrenar. Garantiza que el modelo no ve partidos futuros (ni siquiera del mismo día). Es la versión que se usa en producción para el endpoint de predicción.

### `ml/pipeline_mundial.py` — Adaptación para selecciones

Reutiliza el pipeline UCL pero adaptado a selecciones nacionales:
- Dataset: `selecciones_combinado.csv` (2.337 partidos × 143 columnas)
- Ranking FIFA en lugar de coeficiente UEFA (~150 selecciones mapeadas)
- Sede neutral (mayoría de partidos de torneos internacionales)
- Cobertura: World Cup, Eurocopa, Copa América, Eliminatorias (UEFA y CONMEBOL), AFCON, Asian Cup, Gold Cup, Nations League (UEFA y CONCACAF)

### `scraper/scraper_uefa.py` — Playwright sobre UEFA.com

Scrapea stats de partidos UCL desde `es.uefa.com`. Dos modos:
- `obtener_info_partido(url)` — un partido específico
- `listar_partidos_por_fecha(fecha)` — todos los partidos de una jornada

Extrae stats de los `pk-list-stat-item` del DOM y fecha/marcador del JSON embebido en `__NEXT_DATA__`. Mapea 70+ alias de nombres de equipos al esquema canónico (ej. "Atlético de Madrid" → "Atleti", "Bayern München" → "Bayern").

### `scraper/scraper_premier.py` — SofaScore con sesión persistente

Mantiene **un solo browser abierto** para toda la sesión; cada llamada a la API interna de SofaScore se hace desde el contexto del browser. Esto evita Cloudflare, que bloquea requests directos.

Cobertura: ~90-100 de las 140 columnas del dataset, dependiendo de la riqueza de stats disponibles para cada partido (algunos partidos viejos no tienen ciertos campos).

### `scraper/scraper_selecciones.py` — Multi-torneo

10 competiciones soportadas, todas vía SofaScore. CLI con `--torneos`, `--temporadas`, `--scrape`, `--partido`, `--combinar`. El comando `--combinar` une todos los CSVs individuales (`selecciones_*.csv`) en `selecciones_combinado.csv`.

### `scripts/agregar_partido.py` — CLI principal de ingesta

3 modos de uso:

```bash
python scripts/agregar_partido.py                              # interactivo (manual)
python scripts/agregar_partido.py --url <URL>                  # scrape de 1 URL
python scripts/agregar_partido.py --fecha 2026-03-10 --fase Octavos  # jornada completa
```

El modo `--fecha` lanza un **subprocess por partido** para evitar que Chromium acumule recursos y se cuelgue. Dedup por `(Equipo1, Equipo2, Fecha)` normalizada.

### `scripts/predecir_y_agregar.py` — Track record honesto

Para cada partido de una fecha:
1. **Predice** con el dataset tal como está AHORA (antes de agregar el match).
2. **Guarda** la predicción a `track_record_predictions.csv`.
3. **Agrega** el partido al dataset.

Entrena el modelo UNA sola vez por fecha (todos los partidos de esa jornada se predicen con el mismo estado del dataset, antes de agregar ninguno). Esto garantiza que el track record refleja predicciones realmente "ciegas".

### `static/` — Dashboard sin build step

Frontend vanilla. Tailwind y Chart.js se cargan por CDN.

- **`index.html`** — markup principal. Secciones: KPIs, ranking ELO, métricas CV, predicciones de test, predictor interactivo, track record público, historial de partidos, featured pick.
- **`admin.html`** — panel de administración (gestión de featured pick, debugging).
- **`js/api.js`** — wrapper de `fetch` con manejo de errores y URL base configurable (`window.API_BASE_URL`).
- **`js/app.js`** — lógica del dashboard: carga datos al arrancar, renderiza tablas, dibuja gráficos con Chart.js, maneja el form del predictor.

### `experimentos/` — Bitácora reproducible

Cada hipótesis testeada vive en su propio script (`eval_*.py`) con su JSON de configuración. `harness.py` corre el pipeline con la config y guarda métricas a `resultados.csv`. `RESUMEN.md` es la bitácora narrativa con justificación de cada decisión (por qué k=20 y no k=25, por qué `class_weight={0:1.5, 1:1, 2:1}`, por qué calibración isotónica, etc.).

---

## Stack tecnológico

| Capa | Tecnologías |
|---|---|
| **Backend** | Python 3.12 · FastAPI · SQLModel · Uvicorn · SQLite (default) / Postgres-compatible |
| **ML** | scikit-learn 1.4+ · XGBoost 2.0+ · pandas 2.0+ · numpy · scipy (Dixon-Coles MLE) |
| **Scraping** | Playwright · Chromium headless · sesión persistente para Cloudflare |
| **Frontend** | HTML5 · Tailwind CSS (CDN) · Chart.js (CDN) · vanilla JS (sin framework) |
| **Storage** | SQLite (operacional) · Excel/CSV (datasets) · JSON (record histórico) |
| **Deploy** | Docker multi-stage · Render Blueprint · dominio custom · healthcheck interno |

---

## Setup local

### Requisitos

- Python 3.12+
- (Opcional) Chromium para scraping local

### Instalación

```bash
git clone https://github.com/AndresProgma/Futbolazoia.live.git
cd Futbolazoia.live

pip install -r requirements.txt
python -m playwright install chromium      # solo si vas a scrapear
```

### Arrancar el servidor

```bash
uvicorn api.api:app --reload
```

Abrí <http://localhost:8000> para el dashboard. Documentación interactiva de la API en <http://localhost:8000/docs>.

### Variables de entorno (opcionales)

| Variable | Default | Descripción |
|---|---|---|
| `PORT` | `8000` | Puerto del servidor |
| `ALLOWED_ORIGINS` | `*` | CORS — lista separada por comas o `*` |
| `DATABASE_URL` | `sqlite:///data/futbol.db` | URL de la base de datos |
| `DATASET_PATH` | `data/creando_dataset_modificado.xlsx` | Path al dataset principal |
| `ODDS_API_KEY` | _(vacía)_ | API key de the-odds-api (opcional, para odds en vivo) |

### Correr solo el pipeline (sin server)

```bash
python -m ml.knime_workflow_converter
```

### Correr experimentos

```bash
cd experimentos
python eval_mercados_top10.py     # ejemplo: eval de mercados especiales
python harness.py v14_produccion_final.json    # corre una config específica
```

---

## Endpoints de la API

### Recursos principales

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Dashboard web (sirve `static/index.html`) |
| `GET` | `/admin` | Panel de administración |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/api/health` | Health check + evaluaciones en memoria |

### Partidos (UCL)

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/partidos` | Listar (filtros `?equipo1=&equipo2=`) |
| `GET` | `/partidos/{id}` | Detalle |
| `PUT` | `/partidos/{id}` | Actualizar goles/fecha → **dispara autoresolución de tracks** |
| `DELETE` | `/partidos/{id}` | Soft delete |

### Evaluaciones del pipeline

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/evaluaciones` | Reentrena el pipeline completo (UCL) |
| `GET` | `/evaluaciones` · `/evaluaciones/{id}` | Listar / detalle |
| `GET` | `/api/evaluaciones/{id}/status` | Estado del entrenamiento |
| `GET` | `/api/evaluaciones/{id}/elos` | Ranking ELO |
| `GET` | `/api/evaluaciones/{id}/metricas` | Accuracy + F1 (test y CV) |
| `GET` | `/api/evaluaciones/{id}/feature-importance` | Top features del RF |
| `GET` | `/api/evaluaciones/{id}/predicciones-test` | Test set vs predicciones por modelo |

### Predicciones

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/predecir` | Predicción rápida (JSON, no guarda) |
| `POST` | `/api/track` | Predecir + guardar al track record público |
| `GET` | `/api/track` | Listar track record |
| `GET` | `/api/track/stats` | Estadísticas agregadas (% aciertos) |
| `POST` | `/api/track/{id}/resolver` | Forzar resolución manual |
| `DELETE` | `/api/track/{id}` | Soft delete |

### Premier League

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/pl/status` | Estado del modelo PL (idle/training/ready/error) |
| `POST` | `/api/pl/entrenar` | Lanzar entrenamiento manual |

### Selecciones (Mundial 2026, etc.)

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/mundial/estado` | Estado del modelo |
| `POST` | `/api/mundial/entrenar` | Lanzar entrenamiento |
| `GET` | `/api/mundial/equipos` | Lista de selecciones |
| `POST` | `/api/mundial/predecir` | Predecir partido de selecciones |
| `GET` | `/api/mundial/elos` | Ranking ELO de selecciones |
| `GET` | `/api/mundial/metricas` | Métricas del modelo |

### Otros

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/equipos` | Equipos UCL |
| `GET` | `/api/odds` | Odds en vivo (requiere `ODDS_API_KEY`) |
| `GET` | `/api/featured-pick` | Pick destacado del día |
| `POST` | `/api/admin/featured-pick` | Configurar pick (admin) |
| `GET` | `/api/record` | Record histórico completo |
| `POST` | `/api/sync` | Forzar sincronización Excel ↔ DB |

---

## Flujo de datos end-to-end

### 1. Ingesta de un partido nuevo

```
Usuario corre:                 Pipeline interno:
─────────────────              ──────────────────
agregar_partido.py     →       scraper_uefa.py / scraper_premier.py
  --fecha 2026-03-10           (Playwright → JSON stats)
                                      ↓
                               Normalización de equipos (aliases)
                                      ↓
                               Dedup por (E1, E2, Fecha)
                                      ↓
                               Append a Excel/CSV
                                      ↓
                               POST /api/sync → DB sincronizada
```

### 2. Predicción con track record

```
Frontend                    API                       ML Pipeline
─────────                   ────                      ────────────
Click "Predecir"   →    POST /api/track       →     predecir_partido_v2()
                                                          ↓
                                                    Filtra dataset por fecha
                                                          ↓
                                                    Entrena ensemble + DC
                                                          ↓
                                                    Predice 20 corridas
                                                          ↓
                                                    Promedia probas
                        ←   Devuelve JSON      ←     Consenso + ELOs
                            con consenso
                                ↓
                        Guarda PrediccionTrack
                        (pendiente de resolución)
```

### 3. Auto-resolución del track

```
Cuando llega el resultado real:

PUT /partidos/{id}      →      Busca tracks pendientes con
{ goles_e1, goles_e2 }         (E1, E2) o (E2, E1)
                                     ↓
                               Invierte goles si orden invertido
                                     ↓
                               Calcula resultado real (W/D/L desde E1)
                                     ↓
                               acierto = (pred_consenso == real)
                                     ↓
                               UPDATE PrediccionTrack
                                     ↓
                               Stats agregadas actualizadas
```

---

## Métricas actuales

Validación walk-forward CV (`TimeSeriesSplit`, 5 folds) sobre **189 partidos UCL 2025-26**:

| Modelo | CV F1 mean | Config v14 |
|---|---:|---|
| Logistic Regression | **72.26%** | k=20, db=1.5, isotónica |
| Gradient Boosting | **73.55%** | k=20, db=1.5, isotónica |
| Random Forest | 69.03% | k=20, db=1.5, isotónica |
| SVM (RBF) | 68.39% | k=20, db=1.5, isotónica |
| XGBoost | 66.45% | k=20, db=1.5, isotónica |
| KNN | 63.87% | k=20, db=1.5, isotónica |
| **Ensemble avg** | **68.93%** | — |

**Track record predict-before-add** (acumulado, modelo anterior): 90 predicciones / 47 aciertos = **52.8%**.

Con la config v14 actual se espera ~68-72% en futuras predicciones reales — al nivel de las casas de apuestas profesionales.

### Por qué cada decisión funcionó

- **Calibración isotónica**: sube ~3pp en LR y ~2pp en XGB. Las probas crudas de RF/SVM/XGB están sobre-confiadas; isotónica las mapea a la frecuencia empírica real.
- **`class_weight={0:1.5, 1:1, 2:1}`**: la clase Draw es ~20% del dataset. Con `'balanced'`, sklearn empuja demasiado a predecir empates. 1.5 es el sweet spot.
- **K=20 features (de ~160)**: con 189 muestras, k=25 ya hace curse-of-dimensionality. k=20 → SelectKBest filtra más agresivo → modelos menos overfitted.
- **xG sintético**: GB sube 3pp. Pesos calibrados sobre disparos/disparos-a-puerta/oportunidades-claras.

Ver `experimentos/RESUMEN.md` para la bitácora completa.

---

## Deploy

### Render.com (lo que está en producción)

1. Push del repo a GitHub.
2. En Render: **New → Blueprint** → conectar el repo.
3. Render detecta `render.yaml` y crea el servicio automáticamente.
4. Para que SQLite persista entre deploys, descomentar el bloque `disk:` en `render.yaml`:
   ```yaml
   disk:
     name: data
     mountPath: /app/data
     sizeGB: 1
   ```

Variables de entorno (`PORT`, `ALLOWED_ORIGINS`, `DATABASE_URL`, `DATASET_PATH`) se configuran en el dashboard de Render. Defaults OK para empezar.

### Cualquier Docker host (Railway, Fly.io, VPS, etc.)

```bash
docker build -t futbolazoia .
docker run -p 8000:8000 -e ALLOWED_ORIGINS="*" futbolazoia
```

El Dockerfile es multi-stage con Python 3.12-slim. Healthcheck interno apunta a `/api/health` cada 30s.

---

## Roadmap / Ideas pendientes

Del archivo `ideas.txt`:

- **Bookmaker odds históricas**: la feature de mayor impacto en literatura. APIs gratis (the-odds-api) solo dan odds en vivo. Pendiente: scraper de Oddsportal o API paga.
- **xG/xGA real** (Understat/FBRef): construir scraper alternativo, actualmente se usa xG sintético como proxy.
- **Forma en liga doméstica** para equipos UCL: requiere mapear cada equipo a su liga y scrapear su contexto.
- **Backfill UCL 2024-25**: UEFA cerró acceso fácil a temporadas pasadas. Posible workaround vía SofaScore.
- **Modelo de mercados especializados**: corners, tarjetas, BTTS con regresores dedicados (no Poisson sobre goles).
- **Notificaciones push** cuando se resuelve una predicción del track.
- **Comparativa pública**: ranking de usuarios que envíen sus predicciones.

---

## Licencia

Proyecto personal. Datos de UEFA.com y SofaScore se usan solo con fines educativos / de portafolio.

---

**Autor**: [@AndresProgma](https://github.com/AndresProgma) · **Dominio**: <https://futbolazoia.live>
