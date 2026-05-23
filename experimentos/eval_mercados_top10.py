"""
Evaluación ML para los mercados del Top 10 de apuestas
Dataset: PL 2024-25 + 2025-26 (~750 partidos)
Walk-forward CV para evitar leakage temporal
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── 1. CARGAR DATOS ────���─────────────────────────────────────────────
df1 = pd.read_csv('data/premier_2024-25_enriquecido_lineups.csv', low_memory=False)
df2 = pd.read_csv('data/premier_2025-26_enriquecido_lineups.csv', low_memory=False)
df = pd.concat([df1, df2], ignore_index=True)
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
df = df.sort_values('Fecha').reset_index(drop=True)
print(f"Partidos cargados: {len(df)}")

# ���─ 2. TARGETS ──────────────────��────────────────────────────────────
def safe(col): return pd.to_numeric(df[col], errors='coerce').fillna(0)

df['corners_E1']    = safe('Saques_de_esquina_sacados_E1')
df['corners_E2']    = safe('Saques_de_esquina_sacados_E2')
df['corners_total'] = df['corners_E1'] + df['corners_E2']

df['goles_E1']      = safe('EQUIPO1_GOLES')
df['goles_E2']      = safe('EQUIPO2_GOLES')
df['goles_total']   = df['goles_E1'] + df['goles_E2']

df['amarillas_E1']  = safe('Tarjetas_amarillas_E1')
df['amarillas_E2']  = safe('Tarjetas_amarillas_E2')
df['amarillas_tot'] = df['amarillas_E1'] + df['amarillas_E2']

df['faltas_E1']     = safe('Faltas_cometidas_E1')
df['faltas_E2']     = safe('Faltas_cometidas_E2')

TARGETS = {
    'Resultado_1X2'     : (df['goles_E1'] > df['goles_E2']).astype(int),   # simplif: local gana
    'Over_2.5_Goles'    : (df['goles_total']   > 2.5).astype(int),
    'BTTS'              : ((df['goles_E1'] > 0) & (df['goles_E2'] > 0)).astype(int),
    'Corners_Over_9.5'  : (df['corners_total'] > 9.5).astype(int),
    'Corners_Over_10.5' : (df['corners_total'] > 10.5).astype(int),
    'Amarillas_Over_3.5': (df['amarillas_tot'] > 3.5).astype(int),
    'Amarillas_Over_4.5': (df['amarillas_tot'] > 4.5).astype(int),
}

# ── 3. FEATURES PRE-PARTIDO (expanding mean, sin leakage) ───────────
stat_cols = {
    'E1': ['corners_E1', 'goles_E1', 'goles_E2',   # goles_E2 = encajados de E1
           'amarillas_E1', 'faltas_E1'],
    'E2': ['corners_E2', 'goles_E2', 'goles_E1',
           'amarillas_E2', 'faltas_E2'],
}

def expanding_team_means(df, equipo_col, stat_cols_local):
    """Para cada partido, calcula la media histórica del equipo hasta ESE partido."""
    records = []
    for idx, row in df.iterrows():
        team = row[equipo_col]
        past = df.loc[:idx-1]
        as_e1 = past[past['Equipo1'] == team]
        as_e2 = past[past['Equipo2'] == team]
        # Corners sacados por el equipo
        c_e1 = as_e1['corners_E1'].mean() if len(as_e1) else np.nan
        c_e2 = as_e2['corners_E2'].mean() if len(as_e2) else np.nan
        corners_avg = np.nanmean([c_e1, c_e2]) if not (np.isnan(c_e1) and np.isnan(c_e2)) else np.nan
        # Goles marcados
        g_e1 = as_e1['goles_E1'].mean() if len(as_e1) else np.nan
        g_e2 = as_e2['goles_E2'].mean() if len(as_e2) else np.nan
        goles_avg = np.nanmean([g_e1, g_e2]) if not (np.isnan(g_e1) and np.isnan(g_e2)) else np.nan
        # Goles encajados
        ge_e1 = as_e1['goles_E2'].mean() if len(as_e1) else np.nan
        ge_e2 = as_e2['goles_E1'].mean() if len(as_e2) else np.nan
        goles_enc_avg = np.nanmean([ge_e1, ge_e2]) if not (np.isnan(ge_e1) and np.isnan(ge_e2)) else np.nan
        # Amarillas
        a_e1 = as_e1['amarillas_E1'].mean() if len(as_e1) else np.nan
        a_e2 = as_e2['amarillas_E2'].mean() if len(as_e2) else np.nan
        amarillas_avg = np.nanmean([a_e1, a_e2]) if not (np.isnan(a_e1) and np.isnan(a_e2)) else np.nan
        # Partidos jugados
        n = len(as_e1) + len(as_e2)
        records.append({
            'corners_avg'   : corners_avg,
            'goles_avg'     : goles_avg,
            'goles_enc_avg' : goles_enc_avg,
            'amarillas_avg' : amarillas_avg,
            'n_partidos'    : n,
        })
    return pd.DataFrame(records, index=df.index)

print("Calculando features pre-partido (puede tardar ~1 min)...")
feats_e1 = expanding_team_means(df, 'Equipo1', stat_cols['E1'])
feats_e1.columns = [f'E1_{c}' for c in feats_e1.columns]
feats_e2 = expanding_team_means(df, 'Equipo2', stat_cols['E2'])
feats_e2.columns = [f'E2_{c}' for c in feats_e2.columns]

X_base = pd.concat([feats_e1, feats_e2], axis=1)
X_base['es_local']          = df['Es_Local_E1'].fillna(1).astype(int)
X_base['sum_corners_avg']   = X_base['E1_corners_avg'] + X_base['E2_corners_avg']
X_base['sum_goles_avg']     = X_base['E1_goles_avg']   + X_base['E2_goles_avg']
X_base['sum_amarillas_avg'] = X_base['E1_amarillas_avg'] + X_base['E2_amarillas_avg']
X_base['diff_goles_avg']    = X_base['E1_goles_avg']   - X_base['E2_goles_avg']

# Eliminar filas con muy pocos partidos históricos (<5)
mask_min = (X_base['E1_n_partidos'] >= 5) & (X_base['E2_n_partidos'] >= 5)
X_base = X_base[mask_min].copy()

print(f"Partidos usables (≥5 hist. por equipo): {mask_min.sum()}")

# ── 4. EVALUACIÓN WALK-FORWARD ───────────────────���───────────────────
MODELS = {
    'RF'  : RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
    'GB'  : GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    'LR'  : LogisticRegression(max_iter=500, C=0.5),
}
TSS = TimeSeriesSplit(n_splits=5)

results = []

for tgt_name, y_full in TARGETS.items():
    y = y_full[mask_min]
    X = X_base.copy()

    # Seleccionar features relevantes para cada target
    if 'Corners' in tgt_name:
        feat_cols = [c for c in X.columns if 'corner' in c or 'es_local' in c or 'n_partidos' in c]
    elif 'Goles' in tgt_name or 'BTTS' in tgt_name:
        feat_cols = [c for c in X.columns if 'goles' in c or 'es_local' in c or 'n_partidos' in c]
    elif 'Amarillas' in tgt_name:
        feat_cols = [c for c in X.columns if 'amarilla' in c or 'es_local' in c or 'n_partidos' in c]
    else:
        feat_cols = X.columns.tolist()

    X_feat = X[feat_cols].fillna(X[feat_cols].median())
    baseline = max(y.mean(), 1 - y.mean())  # mayoría clase

    for model_name, model in MODELS.items():
        accs, aucs = [], []
        for train_idx, test_idx in TSS.split(X_feat):
            X_tr, X_te = X_feat.iloc[train_idx], X_feat.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if y_tr.nunique() < 2: continue

            if model_name == 'LR':
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

            m = model.__class__(**model.get_params())
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            proba = m.predict_proba(X_te)[:,1] if hasattr(m,'predict_proba') else preds
            accs.append(accuracy_score(y_te, preds))
            try: aucs.append(roc_auc_score(y_te, proba))
            except: aucs.append(0.5)

        results.append({
            'Mercado'  : tgt_name,
            'Modelo'   : model_name,
            'Accuracy' : round(np.mean(accs)*100, 1),
            'AUC'      : round(np.mean(aucs), 3),
            'Baseline' : round(baseline*100, 1),
            'Ganancia' : round((np.mean(accs) - baseline)*100, 1),
        })

# ── 5. RESULTADOS ──────────────────���─────────────────────────────────
res_df = pd.DataFrame(results)
best   = res_df.loc[res_df.groupby('Mercado')['Accuracy'].idxmax()]
best   = best.sort_values('Ganancia', ascending=False)

print("\n" + "="*70)
print("MEJOR MODELO POR MERCADO (walk-forward CV 5 splits)")
print("="*70)
print(f"{'Mercado':<22} {'Modelo':<5} {'Acc%':>6} {'Baseline%':>10} {'Ganancia':>9} {'AUC':>6}")
print("-"*70)
for _, r in best.iterrows():
    ganancia_str = f"+{r['Ganancia']:.1f}%" if r['Ganancia'] > 0 else f"{r['Ganancia']:.1f}%"
    flag = " ✓" if r['Ganancia'] > 2 else ("  " if r['Ganancia'] > 0 else " ✗")
    print(f"{r['Mercado']:<22} {r['Modelo']:<5} {r['Accuracy']:>6}%  {r['Baseline']:>8}%  {ganancia_str:>8}{flag}  AUC={r['AUC']}")

print("="*70)
print("\nTodos los modelos por mercado:")
print(res_df.to_string(index=False))

# Guardar CSV
res_df.to_csv('experimentos/resultados_mercados_top10.csv', index=False)
print("\nGuardado en experimentos/resultados_mercados_top10.csv")
