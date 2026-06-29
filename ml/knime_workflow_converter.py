"""
KNIME Workflow to Python Converter
================================
Convierte un complejo workflow de KNIME sobre datos deportivos a Python puro.

Estructura:
- Carga de datos Excel
- Filtrado y selección de columnas
- Limpieza de valores faltantes
- Creación de variables derivadas (Rule Engine)
- Filtrado de filas
- Agregaciones (GroupBy)
- Joins de tablas
- Modelos de ML (Random Forest, Gradient Boosting, Linear Regression)
- Evaluación de modelos (Numeric Scorer)
- Visualizaciones

Librerías principales:
- pandas: manipulación de datos
- scikit-learn: machine learning
- numpy: operaciones numéricas
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, f1_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

from ml.dixon_coles import DixonColesModel

K_FEATURES = 20  # top features según ANOVA-F dentro de cada fold (evita curse of dimensionality)
FORMA_DECAY = 0.75      # factor de decaimiento exponencial por partido (0.75 → 25% descuento por partido más antiguo)
ELO_SEASON_DECAY = 0.90 # acercar ELO a ELO_BASE un 10% después de off-season (>90 días sin jugar)
ELO_SEASON_GAP_DAYS = 90


def _dc_proba_in_class_order(dc, equipo1, equipo2, is_home_e1, classes):
    """Reordena las probas de DC ([W, D, L]) al orden de `classes` del LabelEncoder
    (típicamente ['Draw', 'Loss', 'Win'])."""
    p = dc.predict_proba(equipo1, equipo2, is_home_e1=is_home_e1)
    out = np.zeros(len(classes))
    for i, c in enumerate(classes):
        if c == 'Win':
            out[i] = p[0]
        elif c == 'Draw':
            out[i] = p[1]
        elif c == 'Loss':
            out[i] = p[2]
    return out

# Configuración v14 (mejor ensemble en walk-forward CV con 189 partidos):
#   - features extra (xG sintético + UEFA coef) ya integrados en compute_*
#   - clase Draw recibe peso 1.5× → modelos pueden predecir empates sin colapsar
#   - todos los clasificadores envueltos en CalibratedClassifierCV (isotónica, cv=3)
#     → probabilidades calibradas: P(Win)=70% ≈ frecuencia real de aciertos 70%
#   - K=20 features (de ~160) → óptimo en CV walk-forward, evita curse-of-dim
CALIBRAR_DEFECTO = True
DRAW_WEIGHT_BOOST = 1.5   # peso de la clase Draw (probado 2.0/2.5: solo +0.05-0.5p F1, ruido)

# Peso extra para los partidos del Mundial 2026 ya jugados (Partido_id con prefijo
# 'SEL-WORLD_CUP-2026'). Cada uno cuenta como N en el entrenamiento de los modelos
# de PREDICCIÓN (full_models, post-CV) → equivale a sample_weight=N sin corromper
# las features pre-partido (ELO/forma se calculan una sola vez) ni romper la
# validación walk-forward (la CV usa el dataset original). 1 = sin peso extra.
WC2026_PREFIX = 'SEL-WORLD_CUP-2026'
WC2026_PESO   = 6


def _class_weight():
    """Pesos de clase para LabelEncoder (0=Draw, 1=Loss, 2=Win — orden alfabético)."""
    return {0: DRAW_WEIGHT_BOOST, 1: 1.0, 2: 1.0}


def _wrap_calibrado(pipe, calibrar=None):
    if calibrar is None:
        calibrar = CALIBRAR_DEFECTO
    if calibrar:
        return CalibratedClassifierCV(pipe, method='isotonic', cv=3)
    return pipe


def build_classifiers(seed=42, n_features=None, calibrar=None):
    """Pipelines con SelectKBest + clasificador + (opcional) calibración isotónica.

    Cambios v4 vs v0:
      - class_weight={0:2, 1:1, 2:1}  (Draw boost para no colapsar empates)
      - CalibratedClassifierCV(isotónica, cv=3) envuelve cada clasificador
    """
    k = min(K_FEATURES, n_features) if n_features else K_FEATURES
    sk = lambda: SelectKBest(f_classif, k=k)
    cw = _class_weight()
    clfs = {
        'Random Forest':       _wrap_calibrado(Pipeline([('sk', sk()),
                                         ('rf', RandomForestClassifier(
                                             n_estimators=200, max_depth=5, min_samples_leaf=3,
                                             class_weight=cw,
                                             random_state=seed, n_jobs=-1))]), calibrar),
        'Gradient Boosting':   _wrap_calibrado(Pipeline([('sk', sk()),
                                         ('gb', GradientBoostingClassifier(
                                             n_estimators=100, max_depth=3, min_samples_leaf=3,
                                             random_state=seed))]), calibrar),
        'Logistic Regression': _wrap_calibrado(Pipeline([('sc', StandardScaler()), ('sk', sk()),
                                         ('lr', LogisticRegression(
                                             max_iter=2000, C=0.3, class_weight=cw,
                                             random_state=seed))]), calibrar),
        'SVM':                 _wrap_calibrado(Pipeline([('sc', StandardScaler()), ('sk', sk()),
                                         ('svm', SVC(
                                             kernel='rbf', C=0.5, class_weight=cw,
                                             probability=True, random_state=seed))]), calibrar),
        'XGBoost':             _wrap_calibrado(Pipeline([('sk', sk()),
                                         ('xgb', XGBClassifier(
                                             n_estimators=100, max_depth=3, learning_rate=0.1,
                                             reg_lambda=1.0, random_state=seed,
                                             eval_metric='mlogloss', verbosity=0))]), calibrar),
        'KNN':                 _wrap_calibrado(Pipeline([('sc', StandardScaler()), ('sk', sk()),
                                         ('knn', KNeighborsClassifier(
                                             n_neighbors=5, weights='distance'))]), calibrar),
    }
    if _HAS_LGBM:
        clfs['LightGBM'] = _wrap_calibrado(Pipeline([('sk', sk()),
                                         ('lgbm', _LGBMClassifier(
                                             n_estimators=300, max_depth=4, learning_rate=0.05,
                                             num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                             class_weight=cw,
                                             random_state=seed, verbose=-1))]), calibrar)
    return clfs


ELO_BASE = 1500
ELO_K = 30           # qué tan rápido reacciona (estándar football: 20-32)
ELO_HOME_ADV = 60    # ventaja de local en puntos ELO (ClubElo usa ~60)


def compute_elo_features(df):
    """
    Calcula ELO incremental recorriendo partidos en orden cronológico.
    Para cada fila agrega ELO_E1, ELO_E2 y Diff_ELO con el rating ANTES
    del partido (sin leakage). Luego actualiza los ratings con el resultado.

    Bonus por margen de goles: 1x si ≤1 gol, 1.5x si 2, 1.75x+0.1*extra si ≥3.
    Devuelve también el dict de ELO finales para usar en predicciones futuras.
    """
    print("📈 Calculando ELO incremental...")
    df = df.copy()
    elos = {}
    _elo_last_date = {}  # equipo → última fecha jugada (para decay por off-season)
    elo_e1_list, elo_e2_list = [], []

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']

        # Off-season ELO decay: si un equipo lleva >ELO_SEASON_GAP_DAYS sin jugar
        # su ELO se acerca a ELO_BASE un (1-ELO_SEASON_DECAY)*100% (ej. 10%)
        fecha_raw = row.get('Fecha')
        fecha_dt = pd.to_datetime(fecha_raw, errors='coerce') if fecha_raw is not None else pd.NaT
        if pd.notna(fecha_dt):
            for equipo in (e1, e2):
                prev = _elo_last_date.get(equipo)
                if prev is not None and (fecha_dt - prev).days > ELO_SEASON_GAP_DAYS:
                    cur = elos.get(equipo, ELO_BASE)
                    elos[equipo] = ELO_BASE + (cur - ELO_BASE) * ELO_SEASON_DECAY

        elo_e1 = elos.get(e1, ELO_BASE)
        elo_e2 = elos.get(e2, ELO_BASE)
        elo_e1_list.append(elo_e1)
        elo_e2_list.append(elo_e2)

        g1, g2 = row.get('EQUIPO1_GOLES'), row.get('EQUIPO2_GOLES')
        if pd.isna(g1) or pd.isna(g2):
            continue  # partido sin jugar todavía → no actualiza

        es_local = int(row.get('Es_Local_E1', 1)) if pd.notna(row.get('Es_Local_E1')) else 1
        adv = ELO_HOME_ADV * es_local

        # Probabilidad esperada de victoria de E1
        expected_e1 = 1 / (1 + 10 ** ((elo_e2 - elo_e1 - adv) / 400))

        # Resultado real (1 / 0.5 / 0)
        s1 = 1.0 if g1 > g2 else (0.0 if g1 < g2 else 0.5)

        # Multiplicador por margen de goles
        gd = abs(g1 - g2)
        margin = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) * 0.1)

        delta = ELO_K * margin * (s1 - expected_e1)
        elos[e1] = elo_e1 + delta
        elos[e2] = elo_e2 - delta
        if pd.notna(fecha_dt):
            _elo_last_date[e1] = fecha_dt
            _elo_last_date[e2] = fecha_dt

    df['ELO_E1'] = elo_e1_list
    df['ELO_E2'] = elo_e2_list
    df['Diff_ELO'] = df['ELO_E1'] - df['ELO_E2']

    top5 = sorted(elos.items(), key=lambda x: -x[1])[:5]
    bot5 = sorted(elos.items(), key=lambda x: x[1])[:5]
    print(f"   ✓ ELO calculado para {len(elos)} equipos")
    print(f"   Top 5: " + ' · '.join(f'{t} {int(e)}' for t, e in top5))
    print(f"   Bot 5: " + ' · '.join(f'{t} {int(e)}' for t, e in bot5))

    return df, elos


# ============================================================================
# Coeficiente UEFA aproximado por club (5-year, ~temporada 2024-25)
# ============================================================================

UEFA_COEF = {
    'Real Madrid': 144.0, 'Man City': 145.0, 'Bayern Munchen': 138.0,
    'Liverpool': 132.0, 'Paris': 122.5, 'Inter': 119.5,
    'BDortmund': 109.0, 'Leverkursen': 107.0, 'Atleti': 107.0,
    'Juventus': 100.5, 'Napoli': 100.0, 'Barcelona': 99.5,
    'Arsenal': 98.0, 'Benfica': 94.0, 'Atlanta': 90.0,
    'Chelsea': 88.0, 'Milan': 88.0, 'Sporting CP': 84.0,
    'Tottenham': 81.0, 'PSV': 71.0, 'Villareal': 70.0,
    'Marseille': 65.0, 'Ajax': 64.0, 'Athletic Club': 53.0,
    'Frankfurt': 52.0, 'Slavia Praha': 51.5, 'Club Brugge': 51.5,
    'Galatasaray': 51.0, 'Newcastle': 50.0, 'Olympiacos': 47.5,
    'Bodo': 49.0, 'Copengagen': 49.0, 'Qarabag': 41.0,
    'Union SG': 41.0, 'Monaco': 30.0, 'Pafos': 12.0,
    'Kairat Almaty': 8.0,
    # 2024-25 únicos (no en 2025-26)
    'Leipzig': 87.0, 'Stuttgart': 50.0, 'Sturm Graz': 35.0,
    'Salzburg': 55.0, 'Brest': 25.0, 'Bologna': 35.0,
    'Aston Villa': 50.0, 'Lille': 60.0, 'Crvena Zvezda': 40.0,
    'YB': 35.0, 'Celtic': 50.0, 'Feyenoord': 75.0,
    'Dinamo Zagreb': 50.0, 'Sparta': 40.0, 'Girona': 30.0,
    'Slovan Bratislava': 20.0, 'Shakhtar': 70.0,
    # 2023-24 grupos
    'Sevilla': 92.0, 'Lazio': 70.0, 'Lens': 25.0,
    'Antwerp': 25.0, 'Galatasaray SK': 51.0, 'Manchester United': 95.0,
    'Man United': 95.0, 'RBL': 87.0, 'Young Boys': 35.0,
    'Braga': 65.0, 'Real Sociedad': 65.0, 'Union Berlin': 45.0,
    'Berlin': 45.0, 'Royal Antwerp': 25.0, 'Real Madrid CF': 144.0,
    'Real Betis': 50.0,
}

UEFA_COEF_DEFAULT = 25.0   # equipo desconocido → underdog plausible


def compute_uefa_coef_features(df):
    """Anexa Coef_UEFA_E1, Coef_UEFA_E2 y Diff_Coef_UEFA.

    Es una señal estática del 'nivel histórico europeo' del club. Útil
    sobre todo para cold-start (Pafos, Kairat, Qarabag) donde el ELO está
    en su default de 1500 al no tener historia en el dataset."""
    print("🏆 Calculando coeficiente UEFA por club...")
    df = df.copy()
    df['Coef_UEFA_E1']  = df['Equipo1'].map(UEFA_COEF).fillna(UEFA_COEF_DEFAULT)
    df['Coef_UEFA_E2']  = df['Equipo2'].map(UEFA_COEF).fillna(UEFA_COEF_DEFAULT)
    df['Diff_Coef_UEFA'] = df['Coef_UEFA_E1'] - df['Coef_UEFA_E2']

    desconocidos = set(df.loc[~df['Equipo1'].isin(UEFA_COEF), 'Equipo1']) | \
                   set(df.loc[~df['Equipo2'].isin(UEFA_COEF), 'Equipo2'])
    if desconocidos:
        print(f"   ⚠️  {len(desconocidos)} equipos sin coeficiente UEFA, usando default {UEFA_COEF_DEFAULT}: {sorted(desconocidos)[:10]}")
    return df


# ============================================================================
# xG sintético pre-partido (promedio histórico del equipo en partidos previos)
# ============================================================================

# Pesos calibrados aproximadamente para que xG promedio ≈ goles promedio
XG_W_SHOTS = 0.05   # disparos totales
XG_W_SOT   = 0.20   # disparos a puerta
XG_W_CC    = 0.55   # oportunidades claras
XG_WINDOW  = 5      # promedio de últimos N partidos (más receptivo a forma actual)
MARKET_WINDOW = 5   # rolling de stats específicas por mercado (corners, tarjetas, etc.)


def _xg_partido(shots, sot, clear):
    """xG estimado desde stats del propio partido."""
    return (XG_W_SHOTS * (shots or 0)
            + XG_W_SOT * (sot or 0)
            + XG_W_CC  * (clear or 0))


MEDIA11_WINDOW = 5   # partidos previos para el rolling de ratings

def compute_media11_rolling_features(df):
    """Rating medio del once titular: promedio rolling de los últimos
    MEDIA11_WINDOW partidos ANTERIORES de cada equipo. Sin leakage.

    Los ratings de SofaScore son post-partido (reflejan la performance real),
    por lo que usar el rating del partido actual sería leakage directo.
    Esta función toma el historial previo de cada equipo y calcula su
    rating medio esperado antes del partido.

    Anexa: media11_E1_rolling, media11_E2_rolling, Diff_media11_rolling.
    """
    print("⭐ Calculando rating medio del once (rolling pre-partido)...")
    df = df.copy()
    historial = {}   # equipo → list de ratings

    r1, r2 = [], []
    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']
        h1 = (historial.get(e1) or [])[-MEDIA11_WINDOW:]
        h2 = (historial.get(e2) or [])[-MEDIA11_WINDOW:]
        r1.append(np.mean(h1) if h1 else np.nan)
        r2.append(np.mean(h2) if h2 else np.nan)

        # Agregar el rating de ESTE partido al historial (para partidos futuros)
        m1 = row.get('media11_titular_E1')
        m2 = row.get('media11_titular_E2')
        if pd.notna(m1):
            historial.setdefault(e1, []).append(float(m1))
        if pd.notna(m2):
            historial.setdefault(e2, []).append(float(m2))

    df['media11_E1_rolling']   = r1
    df['media11_E2_rolling']   = r2
    df['Diff_media11_rolling'] = df['media11_E1_rolling'].fillna(0) - df['media11_E2_rolling'].fillna(0)

    print(f"   ✓ Rating rolling calculado (ventana={MEDIA11_WINDOW}) para {len(historial)} equipos")
    return df, historial


def compute_xg_features(df):
    """xG / xGA sintético pre-partido: promedio rolling de los últimos
    XG_WINDOW partidos del equipo (como E1 o E2). Sin leakage.

    Anexa: xG_E1_rolling, xGA_E1_rolling, xG_E2_rolling, xGA_E2_rolling,
            Diff_xG_rolling. Si el equipo no tiene historial → NaN
            (handle_missing_values lo imputará luego con la media)."""
    print("🎯 Calculando xG sintético pre-partido...")
    df = df.copy()
    historial = {}   # equipo → list of {xg, xga}

    xg1, xga1, xg2, xga2 = [], [], [], []

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']

        # Rolling xG/xGA del equipo desde su historial previo
        h1 = (historial.get(e1) or [])[-XG_WINDOW:]
        h2 = (historial.get(e2) or [])[-XG_WINDOW:]
        xg1.append(np.mean([h['xg']  for h in h1]) if h1 else np.nan)
        xga1.append(np.mean([h['xga'] for h in h1]) if h1 else np.nan)
        xg2.append(np.mean([h['xg']  for h in h2]) if h2 else np.nan)
        xga2.append(np.mean([h['xga'] for h in h2]) if h2 else np.nan)

        # Calcular xG de ESTE partido y agregar al historial (sin leakage:
        # las features rolling ya fueron asignadas antes, esto es para los
        # partidos futuros).
        m_xg1 = _xg_partido(row.get('Disparos_totales_E1'),
                            row.get('Disparos_a_puerta_E1'),
                            row.get('Oportunidades_claras_E1'))
        m_xg2 = _xg_partido(row.get('Disparos_totales_E2'),
                            row.get('Disparos_a_puerta_E2'),
                            row.get('Oportunidades_claras_E2'))
        if not pd.isna(m_xg1):
            historial.setdefault(e1, []).append({'xg': m_xg1, 'xga': m_xg2})
        if not pd.isna(m_xg2):
            historial.setdefault(e2, []).append({'xg': m_xg2, 'xga': m_xg1})

    df['xG_E1_rolling']  = xg1
    df['xGA_E1_rolling'] = xga1
    df['xG_E2_rolling']  = xg2
    df['xGA_E2_rolling'] = xga2
    df['Diff_xG_rolling'] = df['xG_E1_rolling'].fillna(0) - df['xG_E2_rolling'].fillna(0)

    print(f"   ✓ xG rolling calculado (ventana={XG_WINDOW}) para {len(historial)} equipos")
    return df, historial


# ============================================================================
# Features pre-partido avanzadas: forma reciente, descanso, head-to-head
# ============================================================================

FORMA_WINDOW = 5      # últimos N partidos
FORMA_MAX_MESES = 24  # ventana máxima en meses — descarta partidos muy viejos
H2H_WINDOW = 5        # últimos N enfrentamientos directos


def compute_form_features(df):
    """
    Para cada partido calcula la forma reciente de ambos equipos basada en
    los últimos FORMA_WINDOW partidos (jugados como local O visitante).
    Usa SOLO partidos anteriores (sin leakage).

    Agrega columnas: Forma_W_E1, Forma_D_E1, Forma_L_E1, Forma_GF_E1,
    Forma_GC_E1, Forma_Pts_E1 (+ equivalentes _E2),
    Diff_Forma_Pts, Diff_Forma_GD, Dias_Descanso_E1/_E2.

    Devuelve también un dict por equipo con su historial completo, para que
    predecir_partido() pueda reconstruir las features de partidos futuros.
    """
    print("📊 Calculando forma reciente y descanso...")
    df = df.copy()
    if 'Fecha' in df.columns:
        df['_fecha_dt'] = pd.to_datetime(df['Fecha'], errors='coerce')
    else:
        df['_fecha_dt'] = pd.NaT

    # Historial por equipo: lista de dicts con {fecha, gf, gc, resultado}
    historial = {}

    cols = ['Forma_W_E1', 'Forma_D_E1', 'Forma_L_E1', 'Forma_GF_E1', 'Forma_GC_E1', 'Forma_Pts_E1',
            'Forma_W_E2', 'Forma_D_E2', 'Forma_L_E2', 'Forma_GF_E2', 'Forma_GC_E2', 'Forma_Pts_E2',
            'Dias_Descanso_E1', 'Dias_Descanso_E2',
            'Diff_Forma_Pts', 'Diff_Forma_GD',
            # Nuevas: forma con decaimiento exponencial + racha de victorias
            'Forma_Exp_Pts_E1', 'Forma_Exp_Pts_E2', 'Diff_Forma_Exp_Pts',
            'WinStreak_E1', 'WinStreak_E2']
    rows = {c: [] for c in cols}

    def resumen(hist, fecha_ref=None):
        recientes = hist
        if fecha_ref is not None and pd.notna(fecha_ref):
            cutoff = fecha_ref - pd.Timedelta(days=30 * FORMA_MAX_MESES)
            recientes = [h for h in hist if pd.notna(h.get('fecha')) and h['fecha'] >= cutoff]
        last = recientes[-FORMA_WINDOW:] if recientes else []
        w = sum(1 for h in last if h['res'] == 'W')
        d = sum(1 for h in last if h['res'] == 'D')
        l = sum(1 for h in last if h['res'] == 'L')
        gf = sum(h['gf'] for h in last) if last else 0
        gc = sum(h['gc'] for h in last) if last else 0
        pts = w * 3 + d
        return w, d, l, gf, gc, pts

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']
        fecha = row['_fecha_dt']

        h1 = historial.get(e1, [])
        h2 = historial.get(e2, [])

        # Forma actual (antes del partido) — solo partidos dentro de FORMA_MAX_MESES
        w1, d1, l1, gf1, gc1, pts1 = resumen(h1, fecha)
        w2, d2, l2, gf2, gc2, pts2 = resumen(h2, fecha)
        rows['Forma_W_E1'].append(w1); rows['Forma_D_E1'].append(d1); rows['Forma_L_E1'].append(l1)
        rows['Forma_GF_E1'].append(gf1); rows['Forma_GC_E1'].append(gc1); rows['Forma_Pts_E1'].append(pts1)
        rows['Forma_W_E2'].append(w2); rows['Forma_D_E2'].append(d2); rows['Forma_L_E2'].append(l2)
        rows['Forma_GF_E2'].append(gf2); rows['Forma_GC_E2'].append(gc2); rows['Forma_Pts_E2'].append(pts2)
        rows['Diff_Forma_Pts'].append(pts1 - pts2)
        rows['Diff_Forma_GD'].append((gf1 - gc1) - (gf2 - gc2))

        # Días de descanso (desde último partido)
        for equipo, key in [(e1, 'Dias_Descanso_E1'), (e2, 'Dias_Descanso_E2')]:
            h = historial.get(equipo, [])
            if h and pd.notna(fecha) and h[-1]['fecha'] is not pd.NaT and pd.notna(h[-1]['fecha']):
                dias = (fecha - h[-1]['fecha']).days
                rows[key].append(max(0, int(dias)))
            else:
                rows[key].append(7)  # default razonable (semana)

        # Forma con decaimiento exponencial y racha actual (filtrada por fecha)
        def _filtrar(hist):
            if fecha is not None and pd.notna(fecha):
                cutoff = fecha - pd.Timedelta(days=30 * FORMA_MAX_MESES)
                return [h for h in hist if pd.notna(h.get('fecha')) and h['fecha'] >= cutoff]
            return hist
        h1f, h2f = _filtrar(h1), _filtrar(h2)
        ep1 = _exp_form_pts(h1f)
        ep2 = _exp_form_pts(h2f)
        rows['Forma_Exp_Pts_E1'].append(ep1)
        rows['Forma_Exp_Pts_E2'].append(ep2)
        rows['Diff_Forma_Exp_Pts'].append(ep1 - ep2)
        rows['WinStreak_E1'].append(_win_streak(h1f))
        rows['WinStreak_E2'].append(_win_streak(h2f))

        # Actualizar historial con el resultado real
        g1, g2 = row.get('EQUIPO1_GOLES'), row.get('EQUIPO2_GOLES')
        if pd.notna(g1) and pd.notna(g2):
            res1 = 'W' if g1 > g2 else ('L' if g1 < g2 else 'D')
            res2 = 'W' if g2 > g1 else ('L' if g2 < g1 else 'D')
            historial.setdefault(e1, []).append({'fecha': fecha, 'gf': int(g1), 'gc': int(g2), 'res': res1})
            historial.setdefault(e2, []).append({'fecha': fecha, 'gf': int(g2), 'gc': int(g1), 'res': res2})

    for c, vals in rows.items():
        df[c] = vals
    df = df.drop(columns=['_fecha_dt'])

    print(f"   ✓ Forma calculada para {len(historial)} equipos (ventana={FORMA_WINDOW}, decay={FORMA_DECAY})")
    return df, historial


def compute_market_rolling_features(df):
    """Rolling pre-partido de stats específicas por mercado (corners, tarjetas,
    disparos a puerta, faltas) por equipo. Sin leakage temporal.

    Agrega features por equipo:
      Corners_<E>_for_ult5         corners hechos por el equipo en últ. 5
      Corners_<E>_against_ult5     corners concedidos
      Tarjetas_<E>_ult5            amarillas + rojas recibidas
      Disparos_apuerta_<E>_for_ult5
      Disparos_apuerta_<E>_against_ult5
      Faltas_<E>_ult5

    Y diffs E1−E2 sobre las for_*:
      Diff_Corners_for, Diff_Tarjetas, Diff_Disparos_apuerta, Diff_Faltas

    Si un equipo no tiene historial previo, la feature queda NaN (la imputa
    handle_missing_values luego con la media de la columna)."""
    print("🎲 Calculando features rolling de mercado (corners/tarjetas/disparos/faltas)...")
    df = df.copy()
    historial = {}  # equipo -> list of dicts con stats por partido

    base_cols = [
        'Corners_E1_for_ult5', 'Corners_E1_against_ult5',
        'Corners_E2_for_ult5', 'Corners_E2_against_ult5',
        'Tarjetas_E1_ult5', 'Tarjetas_E2_ult5',
        'Disparos_apuerta_E1_for_ult5', 'Disparos_apuerta_E1_against_ult5',
        'Disparos_apuerta_E2_for_ult5', 'Disparos_apuerta_E2_against_ult5',
        'Faltas_E1_ult5', 'Faltas_E2_ult5',
        'Diff_Corners_for', 'Diff_Tarjetas',
        'Diff_Disparos_apuerta', 'Diff_Faltas',
    ]
    rows = {c: [] for c in base_cols}

    def roll(hist, key):
        if not hist:
            return np.nan
        last = hist[-MARKET_WINDOW:]
        vals = [h[key] for h in last if pd.notna(h[key])]
        return float(np.mean(vals)) if vals else np.nan

    def num(row, c):
        v = row.get(c)
        return float(v) if pd.notna(v) else np.nan

    def safe_diff(a, b):
        return float(a - b) if pd.notna(a) and pd.notna(b) else np.nan

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']
        h1 = historial.get(e1, [])
        h2 = historial.get(e2, [])

        c1f, c1a = roll(h1, 'corners_for'), roll(h1, 'corners_against')
        c2f, c2a = roll(h2, 'corners_for'), roll(h2, 'corners_against')
        t1, t2 = roll(h1, 'tarjetas'), roll(h2, 'tarjetas')
        d1f, d1a = roll(h1, 'disparos_apuerta_for'), roll(h1, 'disparos_apuerta_against')
        d2f, d2a = roll(h2, 'disparos_apuerta_for'), roll(h2, 'disparos_apuerta_against')
        f1, f2 = roll(h1, 'faltas'), roll(h2, 'faltas')

        rows['Corners_E1_for_ult5'].append(c1f)
        rows['Corners_E1_against_ult5'].append(c1a)
        rows['Corners_E2_for_ult5'].append(c2f)
        rows['Corners_E2_against_ult5'].append(c2a)
        rows['Tarjetas_E1_ult5'].append(t1)
        rows['Tarjetas_E2_ult5'].append(t2)
        rows['Disparos_apuerta_E1_for_ult5'].append(d1f)
        rows['Disparos_apuerta_E1_against_ult5'].append(d1a)
        rows['Disparos_apuerta_E2_for_ult5'].append(d2f)
        rows['Disparos_apuerta_E2_against_ult5'].append(d2a)
        rows['Faltas_E1_ult5'].append(f1)
        rows['Faltas_E2_ult5'].append(f2)
        rows['Diff_Corners_for'].append(safe_diff(c1f, c2f))
        rows['Diff_Tarjetas'].append(safe_diff(t1, t2))
        rows['Diff_Disparos_apuerta'].append(safe_diff(d1f, d2f))
        rows['Diff_Faltas'].append(safe_diff(f1, f2))

        # Solo actualizar historial si el partido se jugó (tiene goles).
        if pd.notna(row.get('EQUIPO1_GOLES')) and pd.notna(row.get('EQUIPO2_GOLES')):
            ta1 = num(row, 'Tarjetas_amarillas_E1')
            tr1 = num(row, 'Tarjetas_rojas_E1')
            ta2 = num(row, 'Tarjetas_amarillas_E2')
            tr2 = num(row, 'Tarjetas_rojas_E2')
            tarj_e1 = (ta1 if pd.notna(ta1) else 0) + (tr1 if pd.notna(tr1) else 0)
            tarj_e2 = (ta2 if pd.notna(ta2) else 0) + (tr2 if pd.notna(tr2) else 0)

            historial.setdefault(e1, []).append({
                'corners_for':     num(row, 'Saques_de_esquina_sacados_E1'),
                'corners_against': num(row, 'Saques_de_esquina_sacados_E2'),
                'tarjetas':        tarj_e1 if (pd.notna(ta1) or pd.notna(tr1)) else np.nan,
                'disparos_apuerta_for':     num(row, 'Disparos_a_puerta_E1'),
                'disparos_apuerta_against': num(row, 'Disparos_a_puerta_E2'),
                'faltas':          num(row, 'Faltas_cometidas_E1'),
            })
            historial.setdefault(e2, []).append({
                'corners_for':     num(row, 'Saques_de_esquina_sacados_E2'),
                'corners_against': num(row, 'Saques_de_esquina_sacados_E1'),
                'tarjetas':        tarj_e2 if (pd.notna(ta2) or pd.notna(tr2)) else np.nan,
                'disparos_apuerta_for':     num(row, 'Disparos_a_puerta_E2'),
                'disparos_apuerta_against': num(row, 'Disparos_a_puerta_E1'),
                'faltas':          num(row, 'Faltas_cometidas_E2'),
            })

    for c, vals in rows.items():
        df[c] = vals

    print(f"   ✓ Rolling de mercado calculado para {len(historial)} equipos (ventana={MARKET_WINDOW})")
    return df, historial


def compute_h2h_features(df):
    """
    Para cada partido cuenta los últimos H2H_WINDOW enfrentamientos directos
    entre los dos equipos (en cualquier orden de local/visitante). Sin leakage.

    Agrega: H2H_W_E1, H2H_D, H2H_L_E1, H2H_GF_E1, H2H_GC_E1 (todos sobre los
    últimos H2H_WINDOW partidos entre ambos).

    Devuelve dict {(e1,e2): [partidos]} ordenado para reconstruir en predicción.
    """
    print("🤝 Calculando head-to-head...")
    df = df.copy()
    h2h_log = {}   # key: frozenset({e1,e2}) → lista de partidos en orden

    cols = ['H2H_W_E1', 'H2H_D', 'H2H_L_E1', 'H2H_GF_E1', 'H2H_GC_E1', 'H2H_N']
    rows = {c: [] for c in cols}

    for _, row in df.iterrows():
        e1, e2 = row['Equipo1'], row['Equipo2']
        key = frozenset({e1, e2})
        previos = h2h_log.get(key, [])[-H2H_WINDOW:]
        w, d, l, gf, gc = 0, 0, 0, 0, 0
        for p in previos:
            # Resultado desde la perspectiva de E1
            if p['equipo1'] == e1:
                g_propios, g_rival = p['g1'], p['g2']
            else:
                g_propios, g_rival = p['g2'], p['g1']
            gf += g_propios
            gc += g_rival
            if g_propios > g_rival: w += 1
            elif g_propios < g_rival: l += 1
            else: d += 1
        rows['H2H_W_E1'].append(w); rows['H2H_D'].append(d); rows['H2H_L_E1'].append(l)
        rows['H2H_GF_E1'].append(gf); rows['H2H_GC_E1'].append(gc); rows['H2H_N'].append(len(previos))

        # Registrar este partido para futuros H2H
        g1, g2 = row.get('EQUIPO1_GOLES'), row.get('EQUIPO2_GOLES')
        if pd.notna(g1) and pd.notna(g2):
            h2h_log.setdefault(key, []).append({
                'equipo1': e1, 'equipo2': e2,
                'g1': int(g1), 'g2': int(g2),
            })

    for c, vals in rows.items():
        df[c] = vals
    print(f"   ✓ H2H calculado para {len(h2h_log)} pares de equipos (ventana={H2H_WINDOW})")
    return df, h2h_log


def _exp_form_pts(hist_list, window=FORMA_WINDOW, decay=FORMA_DECAY):
    """Puntos de forma con decaimiento exponencial (partidos recientes pesan más).
    Retorna un valor normalizado [0, 3] — comparable entre equipos con distinto
    número de partidos en la ventana."""
    last = (hist_list or [])[-window:]
    if not last:
        return 0.0
    n = len(last)
    weights = [decay**(n-1-i) for i in range(n)]  # índice 0=más antiguo, n-1=más reciente
    tw = sum(weights)
    return sum(w * (3 if h['res']=='W' else (1 if h['res']=='D' else 0)) for w, h in zip(weights, last)) / tw


def _win_streak(hist_list):
    """Número de victorias consecutivas hasta el último partido (racha actual)."""
    streak = 0
    for entry in reversed(hist_list or []):
        if entry['res'] == 'W':
            streak += 1
        else:
            break
    return streak


def resumen_forma(hist_equipo, window=FORMA_WINDOW, fecha_ref=None):
    """Helper para predecir_partido: resume el historial de un equipo."""
    recientes = hist_equipo or []
    if recientes and fecha_ref is not None and pd.notna(fecha_ref):
        cutoff = fecha_ref - pd.Timedelta(days=30 * FORMA_MAX_MESES)
        recientes = [h for h in recientes if pd.notna(h.get('fecha')) and h['fecha'] >= cutoff]
    last = recientes[-window:]
    w = sum(1 for h in last if h['res'] == 'W')
    d = sum(1 for h in last if h['res'] == 'D')
    l = sum(1 for h in last if h['res'] == 'L')
    gf = sum(h['gf'] for h in last) if last else 0
    gc = sum(h['gc'] for h in last) if last else 0
    return {
        'w': w, 'd': d, 'l': l, 'gf': gf, 'gc': gc, 'pts': w * 3 + d,
        'exp_pts': _exp_form_pts(recientes),
        'streak':  _win_streak(recientes),
    }


def resumen_h2h(h2h_log, e1, e2, window=H2H_WINDOW):
    """Helper para predecir_partido: resume el head-to-head entre dos equipos."""
    key = frozenset({e1, e2})
    previos = (h2h_log.get(key) or [])[-window:]
    w, d, l, gf, gc = 0, 0, 0, 0, 0
    for p in previos:
        if p['equipo1'] == e1:
            g_propios, g_rival = p['g1'], p['g2']
        else:
            g_propios, g_rival = p['g2'], p['g1']
        gf += g_propios; gc += g_rival
        if g_propios > g_rival: w += 1
        elif g_propios < g_rival: l += 1
        else: d += 1
    return {'w': w, 'd': d, 'l': l, 'gf': gf, 'gc': gc, 'n': len(previos)}


import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGAR DATOS (Excel Reader #4)
# ============================================================================
def load_data(filepath):
    """Carga archivo Excel o CSV con datos de partidos deportivos"""
    filepath = str(filepath)
    print(f"📂 Cargando datos de {filepath}...")
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    print(f"   ✓ {df.shape[0]} filas, {df.shape[1]} columnas cargadas")
    return df


# ============================================================================
# 2. SELECCIONAR COLUMNAS (Column Filter #5)
# ============================================================================
def select_columns(df):
    """Filtra y mantiene solo las columnas relevantes"""
    print("🔍 Filtrando columnas relevantes...")
    
    # ⚠️ Solo features PRE-PARTIDO (sin leakage): nada de stats del propio
    # partido (disparos, posesión, pases, oportunidades, etc.) que solo se
    # conocen DESPUÉS del pitido final. Eso inflaba CV a 70%+ artificialmente.
    columns_to_keep = [
        'Equipo1', 'Equipo2', 'Es_Local_E1', 'EQUIPO1_GOLES', 'EQUIPO2_GOLES',
        # Coeficiente UEFA estático (compute_uefa_coef_features)
        'Coef_UEFA_E1', 'Coef_UEFA_E2', 'Diff_Coef_UEFA',
        # ELO pre-partido (calculado en compute_elo_features)
        'ELO_E1', 'ELO_E2', 'Diff_ELO',
        # Forma últimos 5 partidos (calculado en compute_form_features)
        'Forma_W_E1', 'Forma_D_E1', 'Forma_L_E1', 'Forma_GF_E1', 'Forma_GC_E1', 'Forma_Pts_E1',
        'Forma_W_E2', 'Forma_D_E2', 'Forma_L_E2', 'Forma_GF_E2', 'Forma_GC_E2', 'Forma_Pts_E2',
        'Diff_Forma_Pts', 'Diff_Forma_GD',
        'Dias_Descanso_E1', 'Dias_Descanso_E2',
        # Head-to-head últimos 3 enfrentamientos (compute_h2h_features)
        'H2H_W_E1', 'H2H_D', 'H2H_L_E1', 'H2H_GF_E1', 'H2H_GC_E1', 'H2H_N',
        # xG sintético rolling (compute_xg_features) — promedio de partidos previos
        'xG_E1_rolling', 'xGA_E1_rolling', 'xG_E2_rolling', 'xGA_E2_rolling', 'Diff_xG_rolling',
        # Rating medio del once (rolling pre-partido, sin leakage)
        'media11_E1_rolling', 'media11_E2_rolling', 'Diff_media11_rolling',
        # Forma con decaimiento exponencial + racha de victorias
        'Forma_Exp_Pts_E1', 'Forma_Exp_Pts_E2', 'Diff_Forma_Exp_Pts',
        'WinStreak_E1', 'WinStreak_E2',
        # Rolling pre-partido por mercado (compute_market_rolling_features)
        'Corners_E1_for_ult5', 'Corners_E1_against_ult5',
        'Corners_E2_for_ult5', 'Corners_E2_against_ult5',
        'Tarjetas_E1_ult5', 'Tarjetas_E2_ult5',
        'Disparos_apuerta_E1_for_ult5', 'Disparos_apuerta_E1_against_ult5',
        'Disparos_apuerta_E2_for_ult5', 'Disparos_apuerta_E2_against_ult5',
        'Faltas_E1_ult5', 'Faltas_E2_ult5',
        'Diff_Corners_for', 'Diff_Tarjetas',
        'Diff_Disparos_apuerta', 'Diff_Faltas',
    ]
    
    # Mantener solo columnas que existen en el dataframe
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df_filtered = df[available_cols].copy()
    
    print(f"   ✓ {len(available_cols)} columnas seleccionadas")
    return df_filtered


# ============================================================================
# 3. LIMPIEZA: VALORES FALTANTES (Missing Value #33, #41, #62, #71, #91, #94, #116, #125)
# ============================================================================
def handle_missing_values(df, strategy='mean'):
    """
    Maneja valores faltantes en el dataset
    
    Args:
        df: DataFrame
        strategy: 'mean', 'median', 'forward_fill', 'drop'
    """
    print(f"🧹 Limpiando valores faltantes (estrategia: {strategy})...")
    
    initial_nulls = df.isnull().sum().sum()
    
    if strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'drop':
        df = df.dropna()
    
    final_nulls = df.isnull().sum().sum()
    print(f"   ✓ Valores faltantes: {initial_nulls} → {final_nulls}")
    
    return df


# ============================================================================
# 4. CREAR VARIABLES DERIVADAS (Rule Engine #18)
# ============================================================================
def create_derived_variables(df):
    """
    Crea nuevas variables basadas en reglas (similar a Rule Engine de KNIME)
    """
    print("📐 Creando variables derivadas...")
    
    df = df.copy()
    
    # Eficiencia de tiros
    if 'Disparos_a_puerta_E1' in df.columns and 'EQUIPO1_GOLES' in df.columns:
        df['Eficiencia_Tiros_E1'] = df.apply(
            lambda x: (x['EQUIPO1_GOLES'] / x['Disparos_a_puerta_E1'] 
                      if x['Disparos_a_puerta_E1'] > 0 else 0), 
            axis=1
        )
    
    if 'Disparos_a_puerta_E2' in df.columns and 'EQUIPO2_GOLES' in df.columns:
        df['Eficiencia_Tiros_E2'] = df.apply(
            lambda x: (x['EQUIPO2_GOLES'] / x['Disparos_a_puerta_E2'] 
                      if x['Disparos_a_puerta_E2'] > 0 else 0), 
            axis=1
        )
    
    # Diferencia de goles
    if 'EQUIPO1_GOLES' in df.columns and 'EQUIPO2_GOLES' in df.columns:
        df['Diferencia_Goles'] = df['EQUIPO1_GOLES'] - df['EQUIPO2_GOLES']

    # Resultado (Win/Draw/Loss para Equipo1)
    if 'EQUIPO1_GOLES' in df.columns and 'EQUIPO2_GOLES' in df.columns:
        df['Resultado_E1'] = df.apply(
            lambda x: 'Win' if x['EQUIPO1_GOLES'] > x['EQUIPO2_GOLES'] 
                      else ('Draw' if x['EQUIPO1_GOLES'] == x['EQUIPO2_GOLES'] else 'Loss'),
            axis=1
        )
    
    print(f"   ✓ {df.shape[1]} columnas después de derivadas")
    
    return df


# ============================================================================
# 5. FILTRAR FILAS (Row Filter #21, #23, #34, #37, #51, #55, #63, #74, #96, #97, #98, #104, #105, #109, #117, #128)
# ============================================================================
def filter_rows(df):
    """Filtra filas según criterios"""
    print("📋 Aplicando filtros de filas...")
    
    initial_rows = len(df)
    df_filtered = df.copy()
    
    # Ejemplo: eliminar filas donde falten datos críticos
    if 'Equipo1' in df_filtered.columns and 'Equipo2' in df_filtered.columns:
        df_filtered = df_filtered.dropna(subset=['Equipo1', 'Equipo2'])
    
    # Ejemplo: filtrar solo partidos con datos completos de goles
    if 'EQUIPO1_GOLES' in df_filtered.columns and 'EQUIPO2_GOLES' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['EQUIPO1_GOLES'].notna()) & 
            (df_filtered['EQUIPO2_GOLES'].notna())
        ]
    
    # Ejemplo: eliminar duplicados
    df_filtered = df_filtered.drop_duplicates()
    
    final_rows = len(df_filtered)
    print(f"   ✓ Filas filtradas: {initial_rows} → {final_rows}")
    
    return df_filtered


# ============================================================================
# 6. AGREGACIONES (GroupBy #20, #26, #32, #35, #36, #40, #53, #54, #61, #67, #70, #72, #84, #89, #92, #93, #100, #103, #107, #108, #115, #121, #124, #126)
# ============================================================================
def aggregate_by_team(df):
    """Agrega estadísticas por equipo"""
    print("📊 Agregando estadísticas por equipo...")
    
    team_stats = {}
    
    # Estadísticas cuando el equipo juega como Equipo1
    if 'Equipo1' in df.columns:
        agg_e1 = {'EQUIPO1_GOLES': ['mean', 'sum', 'std'], 'EQUIPO2_GOLES': 'mean'}
        if 'Disparos_a_puerta_E1' in df.columns:
            agg_e1['Disparos_a_puerta_E1'] = 'mean'
        if 'Pases_realizados_E1' in df.columns:
            agg_e1['Pases_realizados_E1'] = 'mean'
        stats_e1 = df.groupby('Equipo1').agg(agg_e1).round(2)
        team_stats['as_home'] = stats_e1

    # Estadísticas cuando el equipo juega como Equipo2
    if 'Equipo2' in df.columns:
        agg_e2 = {'EQUIPO2_GOLES': ['mean', 'sum', 'std'], 'EQUIPO1_GOLES': 'mean'}
        if 'Disparos_a_puerta_E2' in df.columns:
            agg_e2['Disparos_a_puerta_E2'] = 'mean'
        if 'Pases_realizados_E2' in df.columns:
            agg_e2['Pases_realizados_E2'] = 'mean'
        stats_e2 = df.groupby('Equipo2').agg(agg_e2).round(2)
        team_stats['as_away'] = stats_e2
    
    print(f"   ✓ Estadísticas de {len(team_stats)} grupos calculadas")
    
    return team_stats


# ============================================================================
# 7. JOINS (Joiner #42, #73, #95, #127)
# ============================================================================
def join_team_stats(df, team_stats):
    """Une estadísticas de equipos al dataset principal.

    Promedio de goles del equipo SOLO con partidos previos (expanding mean) —
    así la feature está disponible pre-partido sin leakage temporal. Antes se
    usaba groupby+transform sobre todo el df, que incluía partidos futuros.
    """
    print("🔗 Realizando joins de estadísticas de equipos...")

    df_enriched = df.copy()

    if {'Equipo1', 'Equipo2', 'EQUIPO1_GOLES', 'EQUIPO2_GOLES'}.issubset(df.columns):
        gf_hist = {}  # equipo -> lista de goles anotados como E1 o E2 (en orden cronológico)
        prom_e1, prom_e2 = [], []

        for _, row in df.iterrows():
            e1, e2 = row['Equipo1'], row['Equipo2']
            h1 = gf_hist.get(e1, [])
            h2 = gf_hist.get(e2, [])
            prom_e1.append(float(np.mean(h1)) if h1 else np.nan)
            prom_e2.append(float(np.mean(h2)) if h2 else np.nan)

            # Actualizar historial con el resultado real (solo partidos jugados)
            g1, g2 = row.get('EQUIPO1_GOLES'), row.get('EQUIPO2_GOLES')
            if pd.notna(g1):
                gf_hist.setdefault(e1, []).append(float(g1))
            if pd.notna(g2):
                gf_hist.setdefault(e2, []).append(float(g2))

        df_enriched['Equipo1_PromedioGoles'] = prom_e1
        df_enriched['Equipo2_PromedioGoles'] = prom_e2
        print(f"   ✓ Promedio de goles pre-partido (expanding) para {len(gf_hist)} equipos")
    else:
        print("   ⚠ faltan columnas para calcular promedio de goles (skipped)")

    return df_enriched


# ============================================================================
# 8. PREPARAR DATOS PARA MODELADO
# ============================================================================
def prepare_for_modeling(df):
    """Prepara datos para modelos ML"""
    print("🤖 Preparando datos para modelado...")
    
    df_model = df.copy()
    
    # Codificar variables categóricas
    le_dict = {}
    categorical_cols = df_model.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col not in ['Equipo1', 'Equipo2']:  # Excluir variables de identificación
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # También codificar Equipo1 y Equipo2
    if 'Equipo1' in df_model.columns:
        le_e1 = LabelEncoder()
        df_model['Equipo1_Encoded'] = le_e1.fit_transform(df_model['Equipo1'].astype(str))
        le_dict['Equipo1'] = le_e1
    
    if 'Equipo2' in df_model.columns:
        le_e2 = LabelEncoder()
        df_model['Equipo2_Encoded'] = le_e2.fit_transform(df_model['Equipo2'].astype(str))
        le_dict['Equipo2'] = le_e2
    
    # Rellenar NaNs restantes con 0
    df_model = df_model.fillna(0)
    
    print(f"   ✓ Datos preparados: {df_model.shape[1]} features")
    
    return df_model, le_dict


# ============================================================================
# 9. ENTRENAR MODELOS — CLASIFICACIÓN (Win / Draw / Loss)
# ============================================================================
def train_models(X_train, y_train, X_test, y_test, df_train=None, df_test=None, le_res=None):
    """Entrena los clasificadores balanceados sobre el split cronológico.

    df_train/df_test/le_res: opcionales. Si se pasan, también entrena Dixon-Coles
    sobre df_train y agrega sus predicciones (int-encoded vía le_res) al dict.
    """
    print("🎓 Entrenando modelos de clasificación...")

    models = build_classifiers(seed=42, n_features=X_train.shape[1])
    predictions = {}
    for name, clf in models.items():
        print(f"   - Entrenando {name}...")
        clf.fit(X_train, y_train)
        predictions[name] = clf.predict(X_test)

    if df_train is not None and df_test is not None and le_res is not None:
        print("   - Entrenando Dixon-Coles...")
        dc = DixonColesModel(max_goals=10).fit(df_train)
        models['Dixon-Coles'] = dc
        preds_dc = []
        for _, r in df_test.iterrows():
            e1, e2 = str(r['Equipo1']), str(r['Equipo2'])
            is_home = bool(r.get('Es_Local_E1', 1))
            if e1 not in dc.params_['attack'] or e2 not in dc.params_['attack']:
                preds_dc.append('Win')  # fallback
            else:
                preds_dc.append(dc.predict(e1, e2, is_home_e1=is_home))
        predictions['Dixon-Coles'] = le_res.transform(preds_dc)

    print(f"   ✓ {len(models)} modelos entrenados")
    return models, predictions


# ============================================================================
# 10. EVALUAR MODELOS
# ============================================================================
def evaluate_models(models, predictions, y_test, class_labels=None):
    """Evalúa y compara clasificadores (accuracy + macro-F1)."""
    print("\n" + "="*60)
    print("RESULTADOS DE MODELOS  (Win / Draw / Loss)")
    print("="*60)

    results_rows = []

    for model_name, y_pred in predictions.items():
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"\n{model_name}  —  Accuracy: {acc:.2%}  |  Macro-F1: {f1m:.2%}")
        print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))
        results_rows.append({'Model': model_name, 'Accuracy': round(acc, 4), 'Macro_F1': round(f1m, 4)})

    print("\nRANKING (por Macro-F1, mejor para clases desbalanceadas):")
    for r in sorted(results_rows, key=lambda x: x['Macro_F1'], reverse=True):
        print(f"  {r['Model']:<22}  acc {r['Accuracy']:.2%}   F1 {r['Macro_F1']:.2%}")

    print("="*60)
    return pd.DataFrame(results_rows)


# ============================================================================
# 10B. CROSS-VALIDATION — accuracy estable con pocos datos
# ============================================================================
def cross_validate_models(X, y, df_dc=None, n_splits=3, random_state=42):
    """
    Walk-forward validation: respeta el orden cronológico del dataset.
    Cada fold entrena solo con partidos ANTERIORES a los del test.
    n_splits=3 → test folds de ~20 partidos (con 81 totales) — menos ruido de muestreo
    que n_splits=5 (que daba folds de ~13 y varianza inflada artificialmente).
    Requisito: X e y deben venir ordenados por Fecha ascendente.

    df_dc: opcional, DataFrame con columnas Equipo1/Equipo2/EQUIPO1_GOLES/EQUIPO2_GOLES/
        Es_Local_E1 alineado por índice con X. Si se pasa, evalúa también Dixon-Coles.
    """
    n = len(X)
    test_size = n // (n_splits + 1)
    print("\n" + "="*60)
    print(f"VALIDACIÓN TEMPORAL WALK-FORWARD ({n_splits} folds, ~{test_size} partidos por test)")
    print("="*60)

    cv = TimeSeriesSplit(n_splits=n_splits)
    classifiers = build_classifiers(seed=random_state, n_features=X.shape[1])

    rows = []
    for name, clf in classifiers.items():
        out = cross_validate(
            clf, X, y, cv=cv,
            scoring={'acc': 'accuracy', 'f1m': 'f1_macro'},
            n_jobs=1,
        )
        acc_m, acc_s = out['test_acc'].mean(), out['test_acc'].std()
        f1_m, f1_s   = out['test_f1m'].mean(), out['test_f1m'].std()
        acc_last = out['test_acc'][-1]   # fold con más historial = mejor proxy del rendimiento futuro
        f1_last  = out['test_f1m'][-1]
        rows.append({'Model': name, 'CV Mean': acc_m, 'CV Std': acc_s,
                     'F1 Mean': f1_m, 'F1 Std': f1_s,
                     'Acc Last': acc_last, 'F1 Last': f1_last})
        folds_str = ' '.join(f'{s:.0%}' for s in out['test_acc'])
        print(f"  {name:<22}  acc {acc_m:.2%}±{acc_s:.2%}  F1 {f1_m:.2%}±{f1_s:.2%}  "
              f"último fold: {acc_last:.0%}  (folds: {folds_str})")

    # Dixon-Coles walk-forward (mismas splits, fit por fold)
    if df_dc is not None:
        accs, f1s = [], []
        for tr_idx, te_idx in cv.split(X):
            train_df = df_dc.iloc[tr_idx]
            test_df = df_dc.iloc[te_idx]
            dc = DixonColesModel(max_goals=10).fit(train_df)
            preds, reals = [], []
            for _, r in test_df.iterrows():
                e1, e2 = str(r['Equipo1']), str(r['Equipo2'])
                is_home = bool(r.get('Es_Local_E1', 1))
                g1, g2 = int(r['EQUIPO1_GOLES']), int(r['EQUIPO2_GOLES'])
                real = 'Win' if g1 > g2 else ('Loss' if g1 < g2 else 'Draw')
                reals.append(real)
                if e1 not in dc.params_['attack'] or e2 not in dc.params_['attack']:
                    preds.append('Win')
                else:
                    preds.append(dc.predict(e1, e2, is_home_e1=is_home))
            accs.append(accuracy_score(reals, preds))
            f1s.append(f1_score(reals, preds, labels=['Win', 'Draw', 'Loss'],
                                average='macro', zero_division=0))
        acc_arr = np.array(accs); f1_arr = np.array(f1s)
        rows.append({'Model': 'Dixon-Coles',
                     'CV Mean': acc_arr.mean(), 'CV Std': acc_arr.std(),
                     'F1 Mean': f1_arr.mean(), 'F1 Std': f1_arr.std(),
                     'Acc Last': acc_arr[-1], 'F1 Last': f1_arr[-1]})
        folds_str = ' '.join(f'{s:.0%}' for s in acc_arr)
        print(f"  {'Dixon-Coles':<22}  acc {acc_arr.mean():.2%}±{acc_arr.std():.2%}  "
              f"F1 {f1_arr.mean():.2%}±{f1_arr.std():.2%}  "
              f"último fold: {acc_arr[-1]:.0%}  (folds: {folds_str})")

    print("\nRANKING (por Macro-F1 del último fold — entrenó con más historial):")
    for r in sorted(rows, key=lambda x: x['F1 Last'], reverse=True):
        print(f"  {r['Model']:<22}  acc {r['Acc Last']:.2%}   F1 {r['F1 Last']:.2%}   "
              f"(CV medio: {r['F1 Mean']:.2%}±{r['F1 Std']:.2%})")
    print("="*60)

    return pd.DataFrame(rows)


# ============================================================================
# 10C-STACKER. Meta-learner OOF sobre predicciones de los modelos base
# ============================================================================
def train_stacker(X, y, n_splits=3, random_state=42):
    """Entrena un meta-learner (LogisticRegression) sobre predicciones
    out-of-fold (OOF) de todos los modelos base usando TimeSeriesSplit.

    El stacker aprende la combinación óptima de los modelos base sin leakage:
    para cada fold, los modelos se entrenan en el pasado y predicen el futuro.

    Retorna (meta, model_names_ordered, n_classes) o None si hay muy pocos datos.
    """
    from sklearn.linear_model import LogisticRegression as _LRmeta
    from sklearn.base import clone

    cv = TimeSeriesSplit(n_splits=n_splits)
    classifiers = build_classifiers(seed=random_state, n_features=X.shape[1])
    model_names_stk = list(classifiers.keys())
    n_classes = 3
    n = len(X)

    oof = np.full((n, len(model_names_stk) * n_classes), np.nan)

    print("🧩 Entrenando stacker OOF...")
    for mi, (name, clf) in enumerate(classifiers.items()):
        for tr_idx, te_idx in cv.split(X):
            c = clone(clf)
            c.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[te_idx, mi*n_classes:(mi+1)*n_classes] = c.predict_proba(X.iloc[te_idx])

    valid = ~np.isnan(oof).any(axis=1)
    if valid.sum() < 30:
        print("   ⚠️  Pocos datos para stacker — saltando")
        return None

    meta = _LRmeta(max_iter=2000, C=0.5, multi_class='multinomial', random_state=random_state)
    meta.fit(oof[valid], y.values[valid])

    # Evaluar en el último fold (mejor proxy del rendimiento real)
    splits = list(cv.split(X))
    last_tr, last_te = splits[-1]
    valid_te = ~np.isnan(oof[last_te]).any(axis=1)
    if valid_te.any():
        stk_pred = meta.predict(oof[last_te][valid_te])
        stk_acc = accuracy_score(y.values[last_te[valid_te]], stk_pred)
        stk_f1  = f1_score(y.values[last_te[valid_te]], stk_pred, average='macro', zero_division=0)
        print(f"   ✓ Stacker último fold: acc {stk_acc:.2%}  F1 {stk_f1:.2%}  "
              f"(entrenado sobre {valid.sum()}/{n} partidos OOF)")
    return meta, model_names_stk, n_classes


# ============================================================================
# 10C. ENTRENAR REGRESORES — predice goles de cada equipo
# ============================================================================
def train_regressors(X_train, y1_train, y2_train, X_test, y1_test, y2_test):
    """Entrena modelos para predecir goles de Equipo1 y Equipo2 por separado"""
    print("\n⚽ Entrenando regresores de goles...")

    regs = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':       XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                      random_state=42, verbosity=0),
    }

    regressors = {}
    for name, reg in regs.items():
        reg.fit(X_train, y1_train)
        reg2 = type(reg)(**reg.get_params())
        reg2.fit(X_train, y2_train)

        mae1 = mean_absolute_error(y1_test, reg.predict(X_test))
        mae2 = mean_absolute_error(y2_test, reg2.predict(X_test))
        print(f"   {name:<16} MAE goles E1: {mae1:.2f}  |  MAE goles E2: {mae2:.2f}")

        regressors[name] = {'E1': reg, 'E2': reg2}

    return regressors


# ============================================================================
# FLUJO PRINCIPAL
# ============================================================================
def main(filepath, test_size=0.2, random_state=42, context_filepaths=None):
    """
    Pipeline completo: carga, limpia, transforma y predice el resultado
    del partido (Win / Draw / Loss para Equipo1).

    Args:
        filepath:          Ruta al archivo Excel con datos UCL
        test_size:         Proporción de datos de prueba (default 0.2 = 20 %)
        random_state:      Semilla para reproducibilidad
        context_filepaths: Lista de CSVs de otras ligas (ej. Premier League)
                           usados SOLO para enriquecer las features de ELO,
                           forma y xG rolling. El modelo entrena y evalúa
                           únicamente con los partidos UCL del filepath.
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "="*60)
    print("⚽ PREDICCIÓN DE RESULTADO — CHAMPIONS LEAGUE")
    print("="*60 + "\n")

    # 1. Carga principal (UCL)
    df_ucl = load_data(filepath)
    df_ucl['_es_ucl'] = True

    # 1b. Contexto adicional (PL u otras ligas) — opcional
    if context_filepaths:
        dfs_ctx = []
        for ctx_path in context_filepaths:
            try:
                df_ctx = pd.read_csv(ctx_path)
                df_ctx['_es_ucl'] = False
                dfs_ctx.append(df_ctx)
                print(f"   📋 Contexto cargado: {ctx_path} ({len(df_ctx)} partidos)")
            except Exception as e:
                print(f"   ⚠ No se pudo cargar contexto {ctx_path}: {e}")
        if dfs_ctx:
            df = pd.concat([df_ucl] + dfs_ctx, ignore_index=True)
            print(f"   ✓ Dataset combinado: {len(df_ucl)} UCL + {len(df)-len(df_ucl)} contexto = {len(df)} total")
        else:
            df = df_ucl
    else:
        df = df_ucl

    # Orden cronológico — crítico para que la validación temporal funcione.
    if 'Fecha' in df.columns:
        df['_fecha_orden'] = pd.to_datetime(df['Fecha'], errors='coerce')
        sort_cols = ['_fecha_orden']
        if 'Partido_id' in df.columns:
            sort_cols.append('Partido_id')
        df = df.sort_values(by=sort_cols, na_position='last')
        df = df.drop(columns=['_fecha_orden']).reset_index(drop=True)
    elif 'Partido_id' in df.columns:
        df = df.sort_values('Partido_id').reset_index(drop=True)
    print(f"   ✓ Dataset ordenado cronológicamente (más antiguo → más reciente)")

    # Sede neutral: las finales se juegan en cancha neutral → Es_Local_E1 = 0
    if 'Fase' in df.columns and 'Es_Local_E1' in df.columns:
        mask_final = df['Fase'].astype(str).str.strip().str.lower() == 'final'
        if mask_final.any():
            df.loc[mask_final, 'Es_Local_E1'] = 0
            print(f"   ✓ {int(mask_final.sum())} partido(s) de Final marcados como sede neutral")

    # Features pre-partido — se calculan sobre el dataset completo (UCL + contexto)
    # para que los historiales de ELO/forma/xG incluyan partidos de otras ligas.
    df = compute_uefa_coef_features(df)
    df, team_elos = compute_elo_features(df)
    df, team_historial = compute_form_features(df)
    df, h2h_log = compute_h2h_features(df)
    df, xg_historial = compute_xg_features(df)
    df, media11_historial = compute_media11_rolling_features(df)

    # Filtrar: a partir de aquí solo trabaja con partidos UCL
    if context_filepaths and '_es_ucl' in df.columns:
        n_antes = len(df)
        df = df[df['_es_ucl'] == True].copy().reset_index(drop=True)
        print(f"   ✓ Filtrado a UCL: {n_antes} → {len(df)} partidos (contexto excluido del modelo)")

    df_raw_stats = df.copy()   # conservar stats brutas (corners, amarillas, etc.)
    df = select_columns(df)
    df = handle_missing_values(df, strategy='mean')
    df = create_derived_variables(df)
    df = filter_rows(df)
    team_stats = aggregate_by_team(df)
    df = join_team_stats(df, team_stats)

    # 8. Preparar para modelado
    df_model, le_dict = prepare_for_modeling(df)

    # 9. Separar features y target
    print("\n📌 Separando features y target...")

    if 'Resultado_E1' not in df_model.columns:
        print("⚠️  ERROR: columna 'Resultado_E1' no encontrada")
        return

    y = df_model['Resultado_E1']  # 0=Draw, 1=Loss, 2=Win  (orden LabelEncoder)
    class_labels = le_dict['Resultado_E1'].classes_  # ['Draw', 'Loss', 'Win']

    # Excluir identificadores, el target y columnas que revelan el resultado
    # (son estadísticas POST-partido directamente derivadas de los goles)
    exclude_cols = [
        'Partido_id', 'Fase', 'Equipo1', 'Equipo2',
        'EQUIPO1_GOLES', 'EQUIPO2_GOLES', 'Resultado_E1', 'Diferencia_Goles',
        'Goles_dentro_area_E1', 'Goles_Fuera_Area_E1',
        'Goles_dentro_area_E2', 'Goles_Fuera_Area_E2',
        'goles_encajados_E1', 'goles_encajados_E2',
        'Goles_encajados_propia_puerta_E1', 'Goles_encajados_propia_puerta_E2',
        'Porterias_a_cero_E1', 'Porterias_a_cero_E2',
        'Eficiencia_Tiros_E1', 'Eficiencia_Tiros_E2',
    ]
    X = df_model.drop(columns=[c for c in exclude_cols if c in df_model.columns])

    print(f"   ✓ Features: {X.shape[1]} | Partidos: {y.shape[0]}")
    print(f"   Distribución clases: {dict(zip(class_labels, [int((y==i).sum()) for i in range(len(class_labels))]))}")

    # 10. Split CRONOLÓGICO: últimos 35 partidos como test, el resto entrena.
    n = len(X)
    N_TEST = 35
    n_train = max(n - N_TEST, int(n * 0.5))   # mínimo 50 % para entrenamiento
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    print(f"\n🔀 Split cronológico: {len(X_train)} train (antiguos) — {len(X_test)} test (últimos {len(X_test)} partidos)")

    # df paralelo a X (mismo índice) para Dixon-Coles
    dc_cols = ['Equipo1', 'Equipo2', 'EQUIPO1_GOLES', 'EQUIPO2_GOLES', 'Es_Local_E1']
    df_dc = df.loc[X.index, [c for c in dc_cols if c in df.columns]].copy()
    if 'Fecha' in df.columns:
        df_dc['Fecha'] = df.loc[X.index, 'Fecha'].values
    df_dc_train = df_dc.iloc[:n_train]
    df_dc_test  = df_dc.iloc[n_train:]

    # 11. Cross-validation antes de entrenar el modelo final (incluye Dixon-Coles)
    cv_results = cross_validate_models(X, y, df_dc=df_dc)

    # 12. Entrenar clasificadores (resultado) y regresores (goles)
    models, predictions = train_models(
        X_train, y_train, X_test, y_test,
        df_train=df_dc_train, df_test=df_dc_test, le_res=le_dict['Resultado_E1'],
    )

    y1_train = df_model.loc[X_train.index, 'EQUIPO1_GOLES']
    y2_train = df_model.loc[X_train.index, 'EQUIPO2_GOLES']
    y1_test  = df_model.loc[X_test.index,  'EQUIPO1_GOLES']
    y2_test  = df_model.loc[X_test.index,  'EQUIPO2_GOLES']
    regressors = train_regressors(X_train, y1_train, y2_train, X_test, y1_test, y2_test)

    # 13. Evaluar modelos (split único)
    results_df = evaluate_models(models, predictions, y_test, class_labels)

    # 13. Guardar resultados junto a los partidos reales
    idx_test = X_test.index
    predictions_df = df.loc[idx_test, ['Equipo1', 'Equipo2']].copy()
    predictions_df['Resultado_Real'] = df.loc[idx_test, 'Resultado_E1'].values
    cl = list(class_labels)  # ['Draw', 'Loss', 'Win']
    win_i  = cl.index('Win')  if 'Win'  in cl else None
    draw_i = cl.index('Draw') if 'Draw' in cl else None
    loss_i = cl.index('Loss') if 'Loss' in cl else None
    for model_name, y_pred in predictions.items():
        predictions_df[model_name] = le_dict['Resultado_E1'].inverse_transform(y_pred)
        if model_name == 'Dixon-Coles':
            dc = models[model_name]
            probas = []
            for _, r in df_dc_test.iterrows():
                e1, e2 = str(r['Equipo1']), str(r['Equipo2'])
                is_home = bool(r.get('Es_Local_E1', 1))
                if e1 not in dc.params_['attack'] or e2 not in dc.params_['attack']:
                    probas.append(np.array([1/3]*len(cl)))
                else:
                    probas.append(_dc_proba_in_class_order(dc, e1, e2, is_home, cl))
            if win_i  is not None: predictions_df[f'{model_name}__win']  = [round(float(p[win_i])*100,  1) for p in probas]
            if draw_i is not None: predictions_df[f'{model_name}__draw'] = [round(float(p[draw_i])*100, 1) for p in probas]
            if loss_i is not None: predictions_df[f'{model_name}__loss'] = [round(float(p[loss_i])*100, 1) for p in probas]
        elif model_name in models:
            probas = models[model_name].predict_proba(X_test)
            if win_i  is not None: predictions_df[f'{model_name}__win']  = [round(float(p[win_i])*100,  1) for p in probas]
            if draw_i is not None: predictions_df[f'{model_name}__draw'] = [round(float(p[draw_i])*100, 1) for p in probas]
            if loss_i is not None: predictions_df[f'{model_name}__loss'] = [round(float(p[loss_i])*100, 1) for p in probas]

    output_dir = r"C:\Users\fehgb\OneDrive\Desktop\prediccion futybol"
    results_df.to_csv(rf"{output_dir}\model_results.csv", index=False)
    predictions_df.to_csv(rf"{output_dir}\predictions.csv", index=False)
    df.to_csv(rf"{output_dir}\processed_data.csv", index=False)

    print(f"\n✅ Archivos guardados en: {output_dir}")
    print("\n📋 Predicciones vs resultado real:")
    print(predictions_df.to_string(index=False))

    print("\n" + "="*60)
    print("✨ PIPELINE COMPLETADO")
    print("="*60 + "\n")

    feature_importance = []  # se calcula después de full_models

    # Entrenar N_PRED_SEEDS semillas por clasificador/regresor sobre el dataset
    # completo. predecir_partido() promedia sus predicciones (ensemble por
    # diversidad de semilla) sin necesidad de re-entrenar nada en cada llamada.
    N_PRED_SEEDS = 5
    print(f"⚡ Entrenando modelos de predicción ({N_PRED_SEEDS} semillas × dataset completo)...")

    # Peso extra al Mundial 2026: replicar esas filas (ya con sus features
    # calculadas) sólo para el entrenamiento de los modelos de predicción.
    # Equivale a sample_weight=WC2026_PESO sin tocar pipelines ni la CV.
    # select_columns elimina Partido_id, pero df_raw_stats (copiado antes) lo
    # conserva con las mismas etiquetas de índice que X → lo usamos para
    # localizar los partidos del Mundial 2026.
    if WC2026_PESO > 1 and 'Partido_id' in df_raw_stats.columns:
        _es_wc26 = df_raw_stats['Partido_id'].astype(str).str.startswith(WC2026_PREFIX)
        _idx_wc26 = X.index[_es_wc26.loc[X.index].values]
    else:
        _idx_wc26 = X.index[[]]
    if len(_idx_wc26):
        _rep = [X] + [X.loc[_idx_wc26]] * (WC2026_PESO - 1)
        X_fit = pd.concat(_rep, ignore_index=True)
        y_fit = pd.concat([y] + [y.loc[_idx_wc26]] * (WC2026_PESO - 1), ignore_index=True)
        df_dc_fit = pd.concat([df_dc] + [df_dc.loc[_idx_wc26]] * (WC2026_PESO - 1),
                              ignore_index=True)
        print(f"   ⚖️  Mundial 2026: {len(_idx_wc26)} partidos × peso {WC2026_PESO} "
              f"→ +{len(_idx_wc26) * (WC2026_PESO - 1)} filas de entrenamiento")
    else:
        X_fit, y_fit, df_dc_fit = X, y, df_dc

    full_models: dict = {}
    for seed in range(N_PRED_SEEDS):
        for name, clf in build_classifiers(seed=seed, n_features=len(X.columns)).items():
            clf.fit(X_fit, y_fit)
            full_models.setdefault(name, []).append(clf)

    # Dixon-Coles: una sola "semilla" (no es estocástico) entrenada sobre todo el dataset
    dc_full = DixonColesModel(max_goals=10).fit(df_dc_fit)
    full_models['Dixon-Coles'] = [dc_full]

    # Feature importance: usar el primer RF ya fiteado sobre el dataset completo
    try:
        rf_fitted = full_models.get('Random Forest', [None])[0]
        if rf_fitted is not None:
            # Unwrap CalibratedClassifierCV → Pipeline fiteado del primer fold
            if hasattr(rf_fitted, 'calibrated_classifiers_') and rf_fitted.calibrated_classifiers_:
                inner = rf_fitted.calibrated_classifiers_[0].estimator
            elif hasattr(rf_fitted, 'estimator'):
                inner = rf_fitted.estimator
            else:
                inner = rf_fitted
            sk_step = inner.named_steps.get('sk')
            rf_step = inner.named_steps.get('rf')
            if sk_step is not None and rf_step is not None:
                mask = sk_step.get_support()
                selected_features = [c for c, keep in zip(X.columns, mask) if keep]
                importances = rf_step.feature_importances_
                feature_importance = sorted(
                    [{'feature': f, 'importance': float(imp)}
                     for f, imp in zip(selected_features, importances)],
                    key=lambda x: -x['importance'],
                )
    except Exception as e:
        print(f"   ⚠️  No se pudo calcular feature importance: {e}")

    y1_full = df_model['EQUIPO1_GOLES']
    y2_full = df_model['EQUIPO2_GOLES']
    if len(_idx_wc26):
        y1_fit = pd.concat([y1_full] + [y1_full.loc[_idx_wc26]] * (WC2026_PESO - 1), ignore_index=True)
        y2_fit = pd.concat([y2_full] + [y2_full.loc[_idx_wc26]] * (WC2026_PESO - 1), ignore_index=True)
    else:
        y1_fit, y2_fit = y1_full, y2_full
    full_regressors: dict = {}
    for seed in range(N_PRED_SEEDS):
        r_rf1  = RandomForestRegressor(n_estimators=100, random_state=seed,        n_jobs=-1)
        r_rf2  = RandomForestRegressor(n_estimators=100, random_state=seed + 1000, n_jobs=-1)
        r_xgb1 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=seed,        verbosity=0)
        r_xgb2 = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                               random_state=seed + 1000, verbosity=0)
        r_rf1.fit(X_fit, y1_fit); r_rf2.fit(X_fit, y2_fit)
        r_xgb1.fit(X_fit, y1_fit); r_xgb2.fit(X_fit, y2_fit)
        full_regressors.setdefault('Random Forest', []).append((r_rf1, r_rf2))
        full_regressors.setdefault('XGBoost',       []).append((r_xgb1, r_xgb2))
    print(f"   ✓ {N_PRED_SEEDS} semillas × {len(full_models)} clasificadores + {len(full_regressors)} regresores listos")

    # Regresores de mercado: corners, amarillas + stats por equipo
    print("\n📐 Regresores de mercado...")
    full_market_regressors: dict = {}

    def _train_market_reg(key, y_raw, label):
        """Entrena RF sobre filas con dato REAL (descarta partidos sin stats,
        p.ej. Mundial 2026 sin captura, que si no contaminarían el test con
        targets constantes), evalúa vs baseline y guarda si Acc > 55%."""
        valid = y_raw.notna().values
        Xv, yv = X[valid], y_raw[valid]
        nv = len(yv)
        if nv < 60:
            print(f"   {label:25s} → datos insuficientes ({nv} con stat)")
            return
        # Split cronológico sobre las filas válidas (últimas 35 como test)
        nt = max(nv - 35, int(nv * 0.5))
        Xtr_, Xte_ = Xv.iloc[:nt], Xv.iloc[nt:]
        ytr_, yte_ = yv.iloc[:nt], yv.iloc[nt:]
        if len(yte_) == 0:
            return
        baseline_pred = np.full(len(yte_), ytr_.mean())
        rf_ = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        rf_.fit(Xtr_, ytr_)
        ml_pred = rf_.predict(Xte_)
        acc = float((np.abs(yte_.values - ml_pred) < np.abs(yte_.values - baseline_pred)).mean())
        mae_b = float(np.abs(yte_.values - baseline_pred).mean())
        mae_ml = float(np.abs(yte_.values - ml_pred).mean())
        ok = acc > 0.55
        print(f"   {label:25s} → Acc {acc*100:.1f}%  MAE base {mae_b:.2f} → ML {mae_ml:.2f}  ({nv} part.)  {'✅ usar ML' if ok else '❌ usar media'}")
        if ok:
            rf_full_ = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
            rf_full_.fit(Xv, yv)
            full_market_regressors[key] = rf_full_

    def _col(c):
        if c not in df_raw_stats.columns:
            return None
        return pd.to_numeric(df_raw_stats.loc[X.index, c], errors='coerce')

    # Totales (corners + amarillas) — para Poisson Monte Carlo.
    # Series CRUDA (con NaN donde no hay stats); _train_market_reg descarta esas filas.
    for key, c1, c2, lbl in [
        ('corners',   'Saques_de_esquina_sacados_E1', 'Saques_de_esquina_sacados_E2', 'corners total'),
        ('amarillas', 'Tarjetas_amarillas_E1',         'Tarjetas_amarillas_E2',         'amarillas total'),
    ]:
        s1, s2 = _col(c1), _col(c2)
        if s1 is None or s2 is None:
            print(f"   ⚠️  {key}: columnas no encontradas")
            continue
        _train_market_reg(key, s1 + s2, lbl)

    # Por equipo (disparos, tiros a puerta, posesión) — para stats card
    for key_e1, key_e2, c1, c2, lbl in [
        ('disparos_e1',     'disparos_e2',     'Disparos_totales_E1',  'Disparos_totales_E2',  'disparos'),
        ('tiros_puerta_e1', 'tiros_puerta_e2', 'Disparos_a_puerta_E1', 'Disparos_a_puerta_E2',  'tiros a puerta'),
        ('posesion_e1',     'posesion_e2',     'Posesion_E1',          'Posesion_E2',           'posesión'),
    ]:
        s1, s2 = _col(c1), _col(c2)
        if s1 is None or s2 is None:
            print(f"   ⚠️  {lbl}: columnas no encontradas")
            continue
        _train_market_reg(key_e1, s1, f'{lbl} E1')
        _train_market_reg(key_e2, s2, f'{lbl} E2')

    # Stacker OOF — entrena sobre el dataset completo con walk-forward CV
    stacker_info = train_stacker(X, y, n_splits=3, random_state=random_state)

    return {
        'df': df,
        'df_raw_stats': df_raw_stats,
        'df_model': df_model,
        'models': models,
        'regressors': regressors,
        'le_dict': le_dict,
        'feature_cols': list(X.columns),
        'results': results_df,
        'predictions': predictions_df,
        'cv_results': cv_results,
        'team_elos': team_elos,
        'team_historial': team_historial,
        'h2h_log': h2h_log,
        'xg_historial': xg_historial,
        'media11_historial': media11_historial,
        'feature_importance': feature_importance,
        'full_models': full_models,
        'full_regressors': full_regressors,
        'full_market_regressors': full_market_regressors,
        'stacker_info': stacker_info,
    }


# ============================================================================
# PREDICCIÓN DE PARTIDO FUTURO  (ensemble de n_runs corridas)
# ============================================================================
def ultimos_partidos(df: pd.DataFrame, equipo: str, n: int = 3) -> list:
    """Últimos N partidos del equipo (como local o visitante).
    Usa el índice original del df para el orden cronológico — main() ya lo ordena por fecha."""
    tiene_fecha = 'Fecha' in df.columns

    cols_local = ['Equipo2', 'EQUIPO1_GOLES', 'EQUIPO2_GOLES']
    cols_visit = ['Equipo1', 'EQUIPO2_GOLES', 'EQUIPO1_GOLES']
    if tiene_fecha:
        cols_local = ['Fecha'] + cols_local
        cols_visit = ['Fecha'] + cols_visit

    local = df[df['Equipo1'] == equipo][cols_local].copy()
    local.columns = (['fecha'] if tiene_fecha else []) + ['rival', 'gf', 'gc']
    local['es_local'] = True

    visit = df[df['Equipo2'] == equipo][cols_visit].copy()
    visit.columns = (['fecha'] if tiene_fecha else []) + ['rival', 'gf', 'gc']
    visit['es_local'] = False

    # Filtrar solo partidos con resultado (goles no nulos) y ordenar más reciente primero
    todos = pd.concat([local, visit])
    todos = todos[todos['gf'].notna() & todos['gc'].notna()]
    todos = todos.sort_index(ascending=False)

    result = []
    for _, r in todos.head(n).iterrows():
        gf = int(r['gf']) if pd.notna(r['gf']) else 0
        gc = int(r['gc']) if pd.notna(r['gc']) else 0
        fecha_val = r.get('fecha', None)
        fecha_str = str(fecha_val)[:10] if fecha_val is not None and pd.notna(fecha_val) and str(fecha_val) not in ('nan', 'NaT', '—') else '—'
        result.append({
            'fecha':    fecha_str,
            'rival':    str(r['rival']),
            'gf': gf, 'gc': gc,
            'res':      'W' if gf > gc else ('L' if gf < gc else 'D'),
            'es_local': bool(r['es_local']),
        })
    return result


def _distribution_summary(samples: np.ndarray) -> dict:
    """
    Analiza la distribución de un mercado discreto (corners, amarillas, goles).
    Devuelve:
      - esperado       : media simulada
      - mas_probable   : moda (entero más probable) y su % individual
      - top5           : top 5 valores con su probabilidad
      - rango_60 / 80  : el rango más estrecho que cubre 60% / 80% de la probabilidad (HPD)
      - over_lines     : probabilidades Over X.5 para todas las líneas relevantes
    """
    samples = samples.astype(int)
    counts  = np.bincount(samples)
    probs   = counts / counts.sum()

    # Moda
    mode = int(np.argmax(probs))
    mode_prob = float(probs[mode])

    # Top 5
    top_idx = np.argsort(-probs)[:5]
    top5 = [{'valor': int(i), 'prob': round(float(probs[i]), 4)} for i in top_idx if probs[i] > 0]

    # HPD: rango más estrecho que cubre target% (acumular los más probables)
    def hpd(target):
        order = np.argsort(-probs)
        cum = 0.0
        chosen = []
        for i in order:
            chosen.append(int(i))
            cum += probs[i]
            if cum >= target:
                break
        return int(min(chosen)), int(max(chosen)), round(float(cum), 4)

    r60_lo, r60_hi, p60 = hpd(0.60)
    r80_lo, r80_hi, p80 = hpd(0.80)

    # Over X.5 lines (desde 0.5 hasta el max+1)
    max_val = int(samples.max())
    over_lines = {}
    for k in range(max_val + 1):
        over_lines[f'over_{k}_5'] = round(float((samples > k + 0.5).mean()), 4)

    return {
        'esperado'       : round(float(samples.mean()), 1),
        'mas_probable'   : mode,
        'mas_probable_p' : round(mode_prob, 4),
        'top5'           : top5,
        'rango_60'       : {'lo': r60_lo, 'hi': r60_hi, 'prob': p60},
        'rango_80'       : {'lo': r80_lo, 'hi': r80_hi, 'prob': p80},
        **over_lines,
    }


def simular_mercados(g1_exp: float, g2_exp: float,
                     prob_win: float, prob_draw: float, prob_loss: float,
                     equipo1: str, equipo2: str,
                     lam_corners_e1: float = 4.5, lam_corners_e2: float = 4.5,
                     lam_amarillas_e1: float = 1.8, lam_amarillas_e2: float = 1.8,
                     n_sims: int = 20_000) -> dict:
    """
    Simulación Monte Carlo (Poisson) para estimar probabilidades de todos
    los mercados de apuestas comunes a partir de los goles esperados.

    g1_exp / g2_exp: goles esperados promedio de los regresores.
    prob_win/draw/loss: consenso del clasificador (más preciso para 1X2 y DNB).
    lam_corners_e1/e2: promedio histórico de corners por partido de cada equipo.
    lam_amarillas_e1/e2: promedio histórico de amarillas por partido.
    """
    rng = np.random.default_rng(42)

    lam1 = max(g1_exp, 0.05)
    lam2 = max(g2_exp, 0.05)

    # Simular partido completo — goles
    g1 = rng.poisson(lam1, n_sims)
    g2 = rng.poisson(lam2, n_sims)

    # Simular corners
    lc1 = max(lam_corners_e1, 0.5)
    lc2 = max(lam_corners_e2, 0.5)
    c1 = rng.poisson(lc1, n_sims)
    c2 = rng.poisson(lc2, n_sims)
    ctotal = c1 + c2

    # Simular amarillas
    la1 = max(lam_amarillas_e1, 0.1)
    la2 = max(lam_amarillas_e2, 0.1)
    a1 = rng.poisson(la1, n_sims)
    a2 = rng.poisson(la2, n_sims)
    atotal = a1 + a2
    total = g1 + g2
    diff  = g1.astype(int) - g2.astype(int)

    # Simular cada mitad (45 % / 55 % de los goles esperados)
    g1_ht = rng.poisson(lam1 * 0.45, n_sims)
    g2_ht = rng.poisson(lam2 * 0.45, n_sims)
    g1_2h = rng.poisson(lam1 * 0.55, n_sims)
    g2_2h = rng.poisson(lam2 * 0.55, n_sims)
    diff_ht = g1_ht.astype(int) - g2_ht.astype(int)
    diff_ft = (g1_ht + g1_2h).astype(int) - (g2_ht + g2_2h).astype(int)

    def res(d): return np.where(d > 0, 'W', np.where(d < 0, 'L', 'D'))
    ht_r = res(diff_ht)
    ft_r = res(diff_ft)

    # DNB: usar consenso del clasificador (más preciso que la simulación)
    wd_sum = prob_win + prob_loss
    dnb_e1 = prob_win  / wd_sum if wd_sum > 0 else 0.5
    dnb_e2 = prob_loss / wd_sum if wd_sum > 0 else 0.5

    # Resultado correcto top-10
    score_counts = Counter(zip(g1.clip(0, 7).tolist(), g2.clip(0, 7).tolist()))
    top_scores = sorted(
        [{'score': f"{s1}-{s2}", 'g1': int(s1), 'g2': int(s2), 'prob': round(cnt / n_sims, 4)}
         for (s1, s2), cnt in score_counts.items()],
        key=lambda x: -x['prob']
    )[:10]

    # HT/FT
    ht_ft = {}
    for ht in ('W', 'D', 'L'):
        for ft in ('W', 'D', 'L'):
            ht_ft[f"{ht}/{ft}"] = round(float(((ht_r == ht) & (ft_r == ft)).mean()), 4)

    return {
        # Ambos marcan
        'btts': {
            'si':  round(float(((g1 > 0) & (g2 > 0)).mean()), 4),
            'no':  round(float(((g1 == 0) | (g2 == 0)).mean()), 4),
        },
        # Resultado correcto (top 10)
        'resultado_correcto': top_scores,
        # Draw No Bet
        'dnb': {
            equipo1: round(dnb_e1, 4),
            equipo2: round(dnb_e2, 4),
        },
        # Doble oportunidad
        'doble_oportunidad': {
            f"1X (gana {equipo1} o empate)": round(prob_win + prob_draw, 4),
            f"X2 (empate o gana {equipo2})": round(prob_draw + prob_loss, 4),
            f"12 (no empate)":               round(prob_win + prob_loss, 4),
        },
        # Goles de cada equipo
        'goles_e1': {
            'nombre': equipo1,
            'xg': round(float(g1.mean()), 2),
            'over_0_5': round(float((g1 > 0.5).mean()), 4),
            'over_1_5': round(float((g1 > 1.5).mean()), 4),
            'over_2_5': round(float((g1 > 2.5).mean()), 4),
        },
        'goles_e2': {
            'nombre': equipo2,
            'xg': round(float(g2.mean()), 2),
            'over_0_5': round(float((g2 > 0.5).mean()), 4),
            'over_1_5': round(float((g2 > 1.5).mean()), 4),
            'over_2_5': round(float((g2 > 2.5).mean()), 4),
        },
        # Total de goles
        'total_goles': {
            'xg': round(float(total.mean()), 2),
            'over_0_5': round(float((total > 0.5).mean()), 4),
            'over_1_5': round(float((total > 1.5).mean()), 4),
            'over_2_5': round(float((total > 2.5).mean()), 4),
            'over_3_5': round(float((total > 3.5).mean()), 4),
            'over_4_5': round(float((total > 4.5).mean()), 4),
        },
        # Hándicap europeo
        'handicap': {
            f'{equipo1} -1':   round(float((diff > 1).mean()), 4),
            f'{equipo1} -1.5': round(float((diff > 1.5).mean()), 4),
            f'{equipo1} -2':   round(float((diff > 2).mean()), 4),
            f'+1 {equipo2}':   round(float((diff >= -1).mean()), 4),
            f'+1.5 {equipo2}': round(float((diff >= -1).mean()), 4),
            f'{equipo2} -1':   round(float((diff < -1).mean()), 4),
            f'{equipo2} -1.5': round(float((diff < -1.5).mean()), 4),
        },
        # Victoria + ambos marcan
        'win_btts': {
            equipo1: round(float(((diff > 0) & (g1 > 0) & (g2 > 0)).mean()), 4),
            equipo2: round(float(((diff < 0) & (g1 > 0) & (g2 > 0)).mean()), 4),
        },
        # Gol en ambas mitades
        'gol_ambas_mitades': round(float(
            ((g1_ht + g2_ht > 0) & (g1_2h + g2_2h > 0)).mean()
        ), 4),
        # Descanso / Tiempo reglamentario
        'ht_ft': ht_ft,
        # Corners — distribución completa (total + por equipo)
        'corners':    _distribution_summary(ctotal),
        'corners_e1': _distribution_summary(c1),
        'corners_e2': _distribution_summary(c2),
        # Amarillas — total + por equipo
        'amarillas':    _distribution_summary(atotal),
        'amarillas_e1': _distribution_summary(a1),
        'amarillas_e2': _distribution_summary(a2),
        # Goles — total + por equipo
        'goles_dist':    _distribution_summary(g1.astype(int) + g2.astype(int)),
        'goles_e1_dist': _distribution_summary(g1.astype(int)),
        'goles_e2_dist': _distribution_summary(g2.astype(int)),
    }


def predecir_partido(equipo1, equipo2, results, n_runs=20, fase='Liga'):
    """
    Predice el resultado entrenando cada modelo n_runs veces con distintas
    semillas sobre el dataset completo y promediando las probabilidades.
    """
    df        = results['df']
    df_model  = results['df_model']
    le_dict   = results['le_dict']
    feat_cols = results['feature_cols']

    # --- Aviso si el partido ya existe en el historial ---
    partidos_previos = df[(df['Equipo1'] == equipo1) & (df['Equipo2'] == equipo2)]
    if not partidos_previos.empty:
        for _, p in partidos_previos.iterrows():
            g1 = int(p['EQUIPO1_GOLES'])
            g2 = int(p['EQUIPO2_GOLES'])
            fecha = p.get('Fecha', '')
            fecha_str = f" ({fecha})" if pd.notna(fecha) and fecha else ""
            print(f"⚠️  Este partido ya está en el dataset{fecha_str}: {equipo1} {g1}–{g2} {equipo2}  →  la predicción puede estar sesgada")

    # --- Perfiles: promedio de cada stat numérica por equipo ---
    e1_stat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.endswith('_E1')]
    e2_stat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.endswith('_E2')]
    perfil_e1 = df.groupby('Equipo1')[e1_stat_cols].mean()
    perfil_e2 = df.groupby('Equipo2')[e2_stat_cols].mean()

    global_e1 = df[e1_stat_cols].mean()
    global_e2 = df[e2_stat_cols].mean()

    if equipo1 in perfil_e1.index:
        stats_e1 = perfil_e1.loc[equipo1]
    elif equipo1 in perfil_e2.index:
        stats_e1 = perfil_e2.loc[equipo1].rename(lambda c: c.replace('_E2', '_E1'))
    else:
        stats_e1 = global_e1

    if equipo2 in perfil_e2.index:
        stats_e2 = perfil_e2.loc[equipo2]
    elif equipo2 in perfil_e1.index:
        stats_e2 = perfil_e1.loc[equipo2].rename(lambda c: c.replace('_E1', '_E2'))
    else:
        stats_e2 = global_e2

    # --- Construir fila de features ---
    # Final = sede neutral → Es_Local_E1 = 0; resto de fases = local real (Equipo1)
    es_final = str(fase).strip().lower() == 'final'
    if es_final:
        print(f"⚖️  Fase: Final → sede neutral (Es_Local_E1 = 0)")
    row = {'Es_Local_E1': 0 if es_final else 1}

    # ELO actual de cada equipo (post-último partido jugado)
    team_elos = results.get('team_elos', {})
    elo_e1 = team_elos.get(equipo1, ELO_BASE)
    elo_e2 = team_elos.get(equipo2, ELO_BASE)
    print(f"   ELO actual: {equipo1} {int(elo_e1)}  vs  {equipo2} {int(elo_e2)}  (Δ {int(elo_e1-elo_e2):+d})")

    # Forma reciente y H2H usando los historiales acumulados durante main()
    historial = results.get('team_historial', {})
    h2h_log = results.get('h2h_log', {})
    fecha_ref = pd.Timestamp.now()
    forma_e1 = resumen_forma(historial.get(equipo1, []), fecha_ref=fecha_ref)
    forma_e2 = resumen_forma(historial.get(equipo2, []), fecha_ref=fecha_ref)
    h2h = resumen_h2h(h2h_log, equipo1, equipo2)
    print(f"   Forma últ.{FORMA_WINDOW}: {equipo1} {forma_e1['w']}W-{forma_e1['d']}D-{forma_e1['l']}L  ·  {equipo2} {forma_e2['w']}W-{forma_e2['d']}D-{forma_e2['l']}L")
    if h2h['n'] > 0:
        print(f"   H2H últ.{h2h['n']}: {equipo1} {h2h['w']}-{h2h['d']}-{h2h['l']}  (GF {h2h['gf']}, GC {h2h['gc']})")

    # xG sintético rolling (últimos XG_WINDOW partidos)
    xg_historial = results.get('xg_historial', {})
    h_xg_e1 = (xg_historial.get(equipo1) or [])[-XG_WINDOW:]
    h_xg_e2 = (xg_historial.get(equipo2) or [])[-XG_WINDOW:]
    xg_e1_roll  = float(np.mean([h['xg']  for h in h_xg_e1])) if h_xg_e1 else float(df_model.get('xG_E1_rolling', pd.Series([np.nan])).mean())
    xga_e1_roll = float(np.mean([h['xga'] for h in h_xg_e1])) if h_xg_e1 else float(df_model.get('xGA_E1_rolling', pd.Series([np.nan])).mean())
    xg_e2_roll  = float(np.mean([h['xg']  for h in h_xg_e2])) if h_xg_e2 else float(df_model.get('xG_E2_rolling', pd.Series([np.nan])).mean())
    xga_e2_roll = float(np.mean([h['xga'] for h in h_xg_e2])) if h_xg_e2 else float(df_model.get('xGA_E2_rolling', pd.Series([np.nan])).mean())
    if h_xg_e1 or h_xg_e2:
        print(f"   xG últ.{XG_WINDOW}: {equipo1} {xg_e1_roll:.2f} (xGA {xga_e1_roll:.2f})  ·  {equipo2} {xg_e2_roll:.2f} (xGA {xga_e2_roll:.2f})")

    # Rating medio del once (rolling últimos MEDIA11_WINDOW partidos, con contexto PL)
    m11_hist = results.get('media11_historial', {})
    h_m11_e1 = (m11_hist.get(equipo1) or [])[-MEDIA11_WINDOW:]
    h_m11_e2 = (m11_hist.get(equipo2) or [])[-MEDIA11_WINDOW:]
    m11_e1 = float(np.mean(h_m11_e1)) if h_m11_e1 else float(df_model.get('media11_E1_rolling', pd.Series([np.nan])).mean())
    m11_e2 = float(np.mean(h_m11_e2)) if h_m11_e2 else float(df_model.get('media11_E2_rolling', pd.Series([np.nan])).mean())

    # Coeficiente UEFA
    coef_e1 = UEFA_COEF.get(equipo1, UEFA_COEF_DEFAULT)
    coef_e2 = UEFA_COEF.get(equipo2, UEFA_COEF_DEFAULT)
    print(f"   Coef UEFA: {equipo1} {coef_e1:.1f}  ·  {equipo2} {coef_e2:.1f}  (Δ {coef_e1 - coef_e2:+.1f})")

    print(f"   Forma exp: {equipo1} {forma_e1['exp_pts']:.2f}pts  racha {forma_e1['streak']}W  ·  "
          f"{equipo2} {forma_e2['exp_pts']:.2f}pts  racha {forma_e2['streak']}W")

    forma_features = {
        'Forma_W_E1': forma_e1['w'], 'Forma_D_E1': forma_e1['d'], 'Forma_L_E1': forma_e1['l'],
        'Forma_GF_E1': forma_e1['gf'], 'Forma_GC_E1': forma_e1['gc'], 'Forma_Pts_E1': forma_e1['pts'],
        'Forma_W_E2': forma_e2['w'], 'Forma_D_E2': forma_e2['d'], 'Forma_L_E2': forma_e2['l'],
        'Forma_GF_E2': forma_e2['gf'], 'Forma_GC_E2': forma_e2['gc'], 'Forma_Pts_E2': forma_e2['pts'],
        'Diff_Forma_Pts': forma_e1['pts'] - forma_e2['pts'],
        'Diff_Forma_GD':  (forma_e1['gf'] - forma_e1['gc']) - (forma_e2['gf'] - forma_e2['gc']),
        'Dias_Descanso_E1': 7, 'Dias_Descanso_E2': 7,  # default razonable para partidos futuros
        'H2H_W_E1': h2h['w'], 'H2H_D': h2h['d'], 'H2H_L_E1': h2h['l'],
        'H2H_GF_E1': h2h['gf'], 'H2H_GC_E1': h2h['gc'], 'H2H_N': h2h['n'],
        'xG_E1_rolling': xg_e1_roll, 'xGA_E1_rolling': xga_e1_roll,
        'xG_E2_rolling': xg_e2_roll, 'xGA_E2_rolling': xga_e2_roll,
        'Diff_xG_rolling': xg_e1_roll - xg_e2_roll,
        'media11_E1_rolling': m11_e1, 'media11_E2_rolling': m11_e2,
        'Diff_media11_rolling': m11_e1 - m11_e2,
        'Coef_UEFA_E1': coef_e1, 'Coef_UEFA_E2': coef_e2,
        'Diff_Coef_UEFA': coef_e1 - coef_e2,
        # Nuevas: forma con decaimiento exponencial + racha
        'Forma_Exp_Pts_E1': forma_e1['exp_pts'], 'Forma_Exp_Pts_E2': forma_e2['exp_pts'],
        'Diff_Forma_Exp_Pts': forma_e1['exp_pts'] - forma_e2['exp_pts'],
        'WinStreak_E1': forma_e1['streak'], 'WinStreak_E2': forma_e2['streak'],
    }

    # Features de mercado rolling para el row de predicción (team-specific)
    _dfr_pred = results.get('df_raw_stats')
    def _pred_roll(eq, c_local, c_visita, fallback):
        if _dfr_pred is None:
            return fallback
        try:
            v1 = pd.to_numeric(_dfr_pred[_dfr_pred['Equipo1'] == eq][c_local],  errors='coerce').dropna().tail(MARKET_WINDOW)
            v2 = pd.to_numeric(_dfr_pred[_dfr_pred['Equipo2'] == eq][c_visita], errors='coerce').dropna().tail(MARKET_WINDOW)
            vals = list(v1) + list(v2)
            return float(np.mean(vals)) if vals else fallback
        except Exception:
            return fallback
    _gc_r = _pred_roll(equipo1, 'Saques_de_esquina_sacados_E1', 'Saques_de_esquina_sacados_E2', 4.5)
    _ga_r = float(pd.to_numeric(_dfr_pred['Tarjetas_amarillas_E1'], errors='coerce').mean()) if _dfr_pred is not None else 1.8
    market_features = {
        'Corners_E1_for_ult5':     _pred_roll(equipo1, 'Saques_de_esquina_sacados_E1', 'Saques_de_esquina_sacados_E2', _gc_r),
        'Corners_E1_against_ult5': _pred_roll(equipo1, 'Saques_de_esquina_sacados_E2', 'Saques_de_esquina_sacados_E1', _gc_r),
        'Corners_E2_for_ult5':     _pred_roll(equipo2, 'Saques_de_esquina_sacados_E2', 'Saques_de_esquina_sacados_E1', _gc_r),
        'Corners_E2_against_ult5': _pred_roll(equipo2, 'Saques_de_esquina_sacados_E1', 'Saques_de_esquina_sacados_E2', _gc_r),
        'Tarjetas_E1_ult5':        _pred_roll(equipo1, 'Tarjetas_amarillas_E1', 'Tarjetas_amarillas_E2', _ga_r),
        'Tarjetas_E2_ult5':        _pred_roll(equipo2, 'Tarjetas_amarillas_E2', 'Tarjetas_amarillas_E1', _ga_r),
        'Diff_Corners_for':        _pred_roll(equipo1, 'Saques_de_esquina_sacados_E1', 'Saques_de_esquina_sacados_E2', _gc_r) -
                                   _pred_roll(equipo2, 'Saques_de_esquina_sacados_E2', 'Saques_de_esquina_sacados_E1', _gc_r),
        'Diff_Tarjetas':           _pred_roll(equipo1, 'Tarjetas_amarillas_E1', 'Tarjetas_amarillas_E2', _ga_r) -
                                   _pred_roll(equipo2, 'Tarjetas_amarillas_E2', 'Tarjetas_amarillas_E1', _ga_r),
    }

    for col in feat_cols:
        if col == 'ELO_E1':
            row[col] = elo_e1
        elif col == 'ELO_E2':
            row[col] = elo_e2
        elif col == 'Diff_ELO':
            row[col] = elo_e1 - elo_e2
        elif col in forma_features:
            row[col] = forma_features[col]
        elif col in market_features:
            row[col] = market_features[col]
        elif col in stats_e1.index:
            row[col] = stats_e1[col]
        elif col in stats_e2.index:
            row[col] = stats_e2[col]
        elif col == 'Equipo1_Encoded':
            classes = le_dict['Equipo1'].classes_
            row[col] = int(le_dict['Equipo1'].transform([equipo1])[0]) if equipo1 in classes else 0
        elif col == 'Equipo2_Encoded':
            classes = le_dict['Equipo2'].classes_
            row[col] = int(le_dict['Equipo2'].transform([equipo2])[0]) if equipo2 in classes else 0
        elif col == 'Equipo1_PromedioGoles':
            mask = df['Equipo1'] == equipo1
            row[col] = df.loc[mask, 'EQUIPO1_GOLES'].mean() if mask.any() else df['EQUIPO1_GOLES'].mean()
        elif col == 'Equipo2_PromedioGoles':
            mask = df['Equipo2'] == equipo2
            row[col] = df.loc[mask, 'EQUIPO2_GOLES'].mean() if mask.any() else df['EQUIPO2_GOLES'].mean()
        else:
            row[col] = df_model[col].mean()

    X_pred = pd.DataFrame([row])[feat_cols]

    # --- Dataset completo para entrenar (sin split) ---
    X_full  = df_model[feat_cols]
    y_full  = df_model['Resultado_E1']
    y1_full = df_model['EQUIPO1_GOLES']
    y2_full = df_model['EQUIPO2_GOLES']
    le_res  = le_dict['Resultado_E1']
    classes = list(le_res.classes_)   # ['Draw', 'Loss', 'Win']

    n_feat = len(feat_cols)
    def make_clfs(seed):
        return build_classifiers(seed=seed, n_features=n_feat)

    # Ordenar modelos por F1 del último fold (entrenó con más historial → mejor proxy)
    full_models = results.get('full_models')
    model_names = list(make_clfs(0).keys())
    if full_models and 'Dixon-Coles' in full_models:
        model_names.append('Dixon-Coles')
    cv_df = results.get('cv_results')
    if cv_df is not None:
        for col in ('F1 Last', 'F1 Mean', 'CV Mean'):
            if col in cv_df.columns:
                cv_order = cv_df.set_index('Model')[col].to_dict()
                model_names = sorted(model_names, key=lambda n: cv_order.get(n, 0), reverse=True)
                break

    probas_acum = {name: [] for name in model_names}

    print(f"\n{'='*58}")
    print(f"🔮  {equipo1}  vs  {equipo2}")
    print(f"{'='*58}")

    if full_models:
        n_seeds = max(len(v) for v in full_models.values())
        print(f"   Prediciendo con {n_seeds} semillas pre-entrenadas...", end='', flush=True)
        for name in model_names:
            if name in full_models:
                if name == 'Dixon-Coles':
                    dc = full_models[name][0]
                    is_home_e1 = bool(row.get('Es_Local_E1', 1))
                    if equipo1 in dc.params_['attack'] and equipo2 in dc.params_['attack']:
                        probas_acum[name].append(_dc_proba_in_class_order(dc, equipo1, equipo2, is_home_e1, classes))
                    else:
                        probas_acum[name].append(np.array([1/len(classes)] * len(classes)))
                else:
                    for clf in full_models[name]:
                        probas_acum[name].append(clf.predict_proba(X_pred)[0])
        print(" listo.")
    else:
        print(f"   Entrenando ensemble ({n_runs} corridas)...", end='', flush=True)
        for seed in range(n_runs):
            for name, clf in make_clfs(seed).items():
                clf.fit(X_full, y_full)
                probas_acum[name].append(clf.predict_proba(X_pred)[0])
        print(" listo.")

    # --- Stacker: añadir como modelo extra si está disponible ---
    stacker_info = results.get('stacker_info')
    if stacker_info is not None:
        meta, stk_model_names, stk_n_classes = stacker_info
        stk_input = []
        for sname in stk_model_names:
            if sname in probas_acum and probas_acum[sname]:
                stk_input.extend(np.mean(probas_acum[sname], axis=0).tolist())
            else:
                stk_input.extend([1/3] * stk_n_classes)
        try:
            stk_proba = meta.predict_proba([stk_input])[0]
            probas_acum['Stacker'] = [stk_proba]
            model_names = model_names + ['Stacker']
        except Exception:
            pass

    # --- Resultados clasificadores ---
    win_idx  = classes.index('Win')  if 'Win'  in classes else None
    draw_idx = classes.index('Draw') if 'Draw' in classes else None
    loss_idx = classes.index('Loss') if 'Loss' in classes else None

    print(f"\n  {'Modelo':<22}  Pred     Win    Draw    Loss")
    print(f"  {'-'*55}")
    modelos_out = []
    for name in model_names:
        avg  = np.mean(probas_acum[name], axis=0)
        pred = classes[np.argmax(avg)]
        w = float(avg[win_idx])  if win_idx  is not None else 0.0
        d = float(avg[draw_idx]) if draw_idx is not None else 0.0
        l = float(avg[loss_idx]) if loss_idx is not None else 0.0
        print(f"  {name:<22} → {pred:<6}  {w:>5.1%}  {d:>5.1%}  {l:>5.1%}")
        modelos_out.append({'modelo': name, 'pred': pred, 'win': w, 'draw': d, 'loss': l})

    # --- CONSENSO: promedio ponderado por F1 del último fold CV ---
    # Los modelos con mejor CV F1 (últimas splits, más cercanas al futuro) pesan más.
    cv_df = results.get('cv_results')
    if cv_df is not None:
        f1_col = 'F1 Last' if 'F1 Last' in cv_df.columns else ('F1 Mean' if 'F1 Mean' in cv_df.columns else None)
        if f1_col:
            scores = cv_df.set_index('Model')[f1_col].to_dict()
            # Stacker hereda la mediana de los modelos base (conservador: no infla su peso)
            if 'Stacker' not in scores and model_names:
                base_scores = [float(scores.get(n, 0.3)) for n in model_names if n != 'Stacker']
                scores['Stacker'] = float(np.median(base_scores)) if base_scores else 0.3
            w_arr = np.array([max(float(scores.get(n, 0.3)), 0.01) for n in model_names])
            # Softmax suave (temperatura baja = pesos más uniformes, alta = favorece al mejor)
            w_arr = np.exp((w_arr - w_arr.max()) * 8)
            w_arr /= w_arr.sum()
        else:
            w_arr = np.ones(len(model_names)) / len(model_names)
    else:
        w_arr = np.ones(len(model_names)) / len(model_names)

    todas = np.average([np.mean(probas_acum[n], axis=0) for n in model_names], axis=0, weights=w_arr)
    pred_global = classes[np.argmax(todas)]
    w_c = float(todas[win_idx])  if win_idx  is not None else 0.0
    d_c = float(todas[draw_idx]) if draw_idx is not None else 0.0
    l_c = float(todas[loss_idx]) if loss_idx is not None else 0.0
    print(f"  {'-'*55}")
    print(f"  {'CONSENSO (ponderado)':<22} → {pred_global:<6}  {w_c:>5.1%}  {d_c:>5.1%}  {l_c:>5.1%}")

    # --- Regresores de goles (ensemble) ---
    def make_regs(seed):
        return {
            'Random Forest': (RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
                              RandomForestRegressor(n_estimators=100, random_state=seed+1000, n_jobs=-1)),
            'XGBoost':       (XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                           random_state=seed, verbosity=0),
                              XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                           random_state=seed+1000, verbosity=0)),
        }

    full_regressors = results.get('full_regressors')
    print(f"\n  {'Modelo':<16}  Marcador predicho")
    print(f"  {'-'*42}")
    goles_out = []
    for reg_name in (full_regressors or make_regs(0)).keys():
        if full_regressors and reg_name in full_regressors:
            g1_preds, g2_preds = [], []
            for r1, r2 in full_regressors[reg_name]:
                g1_preds.append(float(r1.predict(X_pred)[0]))
                g2_preds.append(float(r2.predict(X_pred)[0]))
            g1 = int(max(0, round(np.mean(g1_preds))))
            g2 = int(max(0, round(np.mean(g2_preds))))
            s1, s2 = float(np.std(g1_preds)), float(np.std(g2_preds))
        else:
            g1_preds, g2_preds = [], []
            for seed in range(n_runs):
                r1, r2 = make_regs(seed)[reg_name]
                r1.fit(X_full, y1_full)
                r2.fit(X_full, y2_full)
                g1_preds.append(r1.predict(X_pred)[0])
                g2_preds.append(r2.predict(X_pred)[0])
            g1 = int(max(0, round(np.mean(g1_preds))))
            g2 = int(max(0, round(np.mean(g2_preds))))
            s1, s2 = float(np.std(g1_preds)), float(np.std(g2_preds))
        print(f"  {reg_name:<16}  {equipo1} {g1} – {g2} {equipo2}  (±{s1:.1f} / ±{s2:.1f})")
        goles_out.append({'modelo': reg_name, 'g1': g1, 'g2': g2, 'std1': s1, 'std2': s2})

    print()
    equipos_conocidos = sorted(set(df['Equipo1'].tolist() + df['Equipo2'].tolist()))
    desconocidos = []
    for e in [equipo1, equipo2]:
        if e not in equipos_conocidos:
            print(f"  ⚠️  '{e}' no está en el historial — se usaron promedios globales")
            desconocidos.append(e)

    # --- Mercados de apuestas (Monte Carlo Poisson) ---
    # Usar el promedio de goles esperados de todos los regresores
    g1_exp = np.mean([g['g1'] for g in goles_out]) if goles_out else 1.2
    g2_exp = np.mean([g['g2'] for g in goles_out]) if goles_out else 1.0

    # Stats brutas (corners, amarillas) desde el df pre-select_columns
    _dfr = results.get('df_raw_stats')

    def _team_avg(equipo, col_e1, col_e2, fallback=0.0):
        if _dfr is None or col_e1 not in _dfr.columns:
            return fallback
        try:
            v1 = pd.to_numeric(_dfr[_dfr['Equipo1'] == equipo][col_e1], errors='coerce').dropna()
            v2 = pd.to_numeric(_dfr[_dfr['Equipo2'] == equipo][col_e2], errors='coerce').dropna()
            vals = list(v1) + list(v2)
            return float(np.mean(vals)) if vals else fallback
        except Exception:
            return fallback

    def _global(col, fallback):
        if _dfr is None or col not in _dfr.columns:
            return fallback
        v = pd.to_numeric(_dfr[col], errors='coerce').mean()
        return float(v) if pd.notna(v) else fallback

    _gc = _global('Saques_de_esquina_sacados_E1', 4.5)
    _ga = _global('Tarjetas_amarillas_E1', 1.8)

    # Lambdas base (Poisson histórico)
    lam_c1_base = _team_avg(equipo1, 'Saques_de_esquina_sacados_E1', 'Saques_de_esquina_sacados_E2', _gc)
    lam_c2_base = _team_avg(equipo2, 'Saques_de_esquina_sacados_E2', 'Saques_de_esquina_sacados_E1', _gc)
    lam_a1_base = _team_avg(equipo1, 'Tarjetas_amarillas_E1', 'Tarjetas_amarillas_E2', _ga)
    lam_a2_base = _team_avg(equipo2, 'Tarjetas_amarillas_E2', 'Tarjetas_amarillas_E1', _ga)

    # Si hay regresores ML, usarlos para el total y repartir proporcionalmente
    mkt_regs = results.get('full_market_regressors', {})
    if mkt_regs and 'corners' in mkt_regs and 'amarillas' in mkt_regs:
        try:
            # Corners: ML mejor (MAE -0.14). Amarillas: Poisson mejor → se mantiene base
            total_c_ml = max(float(mkt_regs['corners'].predict(X_pred)[0]), 1.0)
            c_ratio = lam_c1_base / (lam_c1_base + lam_c2_base) if (lam_c1_base + lam_c2_base) > 0 else 0.5
            lam_corners_e1 = total_c_ml * c_ratio
            lam_corners_e2 = total_c_ml * (1 - c_ratio)
            lam_amarillas_e1, lam_amarillas_e2 = lam_a1_base, lam_a2_base
            print(f"   🤖 ML corners: {total_c_ml:.1f} ({lam_corners_e1:.1f}+{lam_corners_e2:.1f})  Poisson amarillas: {lam_amarillas_e1:.1f}+{lam_amarillas_e2:.1f}")
        except Exception as e:
            print(f"   ⚠️  ML mercados falló ({e}), usando Poisson base")
            lam_corners_e1, lam_corners_e2   = lam_c1_base, lam_c2_base
            lam_amarillas_e1, lam_amarillas_e2 = lam_a1_base, lam_a2_base
    else:
        lam_corners_e1, lam_corners_e2   = lam_c1_base, lam_c2_base
        lam_amarillas_e1, lam_amarillas_e2 = lam_a1_base, lam_a2_base

    # ── Stats completos de cada equipo (para tabla comparativa) ─────
    def _team_full_stats(equipo):
        n = 0
        if _dfr is not None:
            n = int((_dfr['Equipo1'] == equipo).sum() + (_dfr['Equipo2'] == equipo).sum())
        return {
            'nombre':           equipo,
            'partidos':         n,
            'goles_for':        round(_team_avg(equipo, 'EQUIPO1_GOLES', 'EQUIPO2_GOLES', 0), 2),
            'goles_against':    round(_team_avg(equipo, 'EQUIPO2_GOLES', 'EQUIPO1_GOLES', 0), 2),
            'corners_for':      round(_team_avg(equipo, 'Saques_de_esquina_sacados_E1', 'Saques_de_esquina_sacados_E2', 0), 1),
            'corners_against':  round(_team_avg(equipo, 'Saques_de_esquina_sacados_E2', 'Saques_de_esquina_sacados_E1', 0), 1),
            'tiros':            round(_team_avg(equipo, 'Disparos_totales_E1', 'Disparos_totales_E2', 0), 1),
            'tiros_puerta':     round(_team_avg(equipo, 'Disparos_a_puerta_E1', 'Disparos_a_puerta_E2', 0), 1),
            'amarillas':        round(_team_avg(equipo, 'Tarjetas_amarillas_E1', 'Tarjetas_amarillas_E2', 0), 2),
            'faltas':           round(_team_avg(equipo, 'Faltas_cometidas_E1', 'Faltas_cometidas_E2', 0), 1),
            'posesion':         round(_team_avg(equipo, 'Posesion_E1', 'Posesion_E2', 0), 1),
            'precision_pase':   round(_team_avg(equipo, 'Precision_pase_E1', 'Precision_pase_E2', 0), 1),
            'oportunidades':    round(_team_avg(equipo, 'Oportunidades_claras_E1', 'Oportunidades_claras_E2', 0), 1),
        }

    stats_equipos = {
        'e1': _team_full_stats(equipo1),
        'e2': _team_full_stats(equipo2),
    }

    # Sobreescribir stats con ML si el regresor superó 55% de accuracy
    for se_key, ml_key_e1, ml_key_e2 in [
        ('tiros',        'disparos_e1',     'disparos_e2'),
        ('tiros_puerta', 'tiros_puerta_e1', 'tiros_puerta_e2'),
        ('posesion',     'posesion_e1',     'posesion_e2'),
    ]:
        if ml_key_e1 in mkt_regs:
            try:
                stats_equipos['e1'][se_key] = round(float(mkt_regs[ml_key_e1].predict(X_pred)[0]), 1)
            except Exception:
                pass
        if ml_key_e2 in mkt_regs:
            try:
                stats_equipos['e2'][se_key] = round(float(mkt_regs[ml_key_e2].predict(X_pred)[0]), 1)
            except Exception:
                pass

    mercados = simular_mercados(
        g1_exp=float(g1_exp), g2_exp=float(g2_exp),
        prob_win=w_c, prob_draw=d_c, prob_loss=l_c,
        equipo1=equipo1, equipo2=equipo2,
        lam_corners_e1=lam_corners_e1, lam_corners_e2=lam_corners_e2,
        lam_amarillas_e1=lam_amarillas_e1, lam_amarillas_e2=lam_amarillas_e2,
    )

    return {
        'equipo1': equipo1,
        'equipo2': equipo2,
        'fase': fase,
        'es_final': es_final,
        'n_runs': n_runs,
        'elo_e1': float(elo_e1),
        'elo_e2': float(elo_e2),
        'diff_elo': float(elo_e1 - elo_e2),
        'forma_e1': forma_e1,
        'forma_e2': forma_e2,
        'h2h': h2h,
        'modelos': modelos_out,
        'consenso': {'pred': pred_global, 'win': w_c, 'draw': d_c, 'loss': l_c},
        'goles': goles_out,
        'mercados': mercados,
        'stats_equipos': stats_equipos,
        'ultimos_e1': ultimos_partidos(df, equipo1, n=3),
        'ultimos_e2': ultimos_partidos(df, equipo2, n=3),
        'equipos_desconocidos': desconocidos,
    }


# ============================================================================
# EJECUTAR
# ============================================================================
if __name__ == "__main__":
    FILEPATH = r"C:\Users\fehgb\OneDrive\Desktop\prediccion futybol\creando_dataset_modificado.xlsx"

    try:
        results = main(FILEPATH)

        # ── Equipos disponibles en el historial ──────────────────────────
        equipos = sorted(set(
            results['df']['Equipo1'].tolist() + results['df']['Equipo2'].tolist()
        ))
        print("⚽ Equipos en el historial:")
        print("   " + " | ".join(equipos))

        # ── Predicciones de partidos futuros ─────────────────────────────
        predecir_partido("Paris",  "Juventus",       results)
        predecir_partido("Inter",    "Liverpool", results)
        predecir_partido("Monaco",    "Galatasaray",          results)

    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {FILEPATH}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


"""El comando es:

  python agregar_partido.py --fecha YYYY-MM-DD --fase Liga --si

  Ejemplos para las jornadas que vienen:

  # Jornada 5 UCL 2025-26 (25-26 nov)  python agregar_partido.py --fecha 2025-11-25 --fase Liga --si
  python agregar_partido.py --fecha 2025-11-26 --fase Liga --si

  Flags útiles:
  - --si → auto-confirma cada partido (sin esto te pregunta uno por uno)
  - --no-headless → muestra el navegador (para depurar si algo falla)
  - --debug → guarda debug_uefa.png y .txt por cada scrape
  - --fase → omítelo y te lo pregunta (default Liga)

  Recuerda que modo_fecha lanza un subprocess por partido para evitar que Chromium se cuelgue, así que tarda un rato — cada partido toma
  ~15-30s."""