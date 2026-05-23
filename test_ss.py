"""Test del enriquecedor de lineups en 5 partidos UCL + 5 PL."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from scraper.enriquecer_lineups import (
    SSSession, construir_indice_ucl,
    enriquecer_dataset, LINEUP_COLS
)

sess = SSSession()
try:
    # ── UCL: 5 primeros partidos ──────────────────────────────────────────────
    print('\n=== UCL (5 partidos) ===')
    ucl = pd.read_excel('data/creando_dataset_modificado.xlsx').head(5).copy()
    indice = construir_indice_ucl(sess)

    # Mostrar qué partidos hay y si se encuentran en el índice
    for _, row in ucl.iterrows():
        key = (str(row['Equipo1']), str(row['Equipo2']), str(row['Fecha'])[:10])
        eid = indice.get(key)
        print(f"  {key[0]} vs {key[1]} ({key[2]}) → eid={eid}")

    ucl_enr = enriquecer_dataset(ucl, sess, indice_ucl=indice, es_pl=False)

    # Mostrar resultados
    cols_muestra = ['Equipo1','Equipo2','Formacion_E1','GK_Rating_E1',
                    'Def_Avg_Rating_E1','Mid_Avg_Rating_E1','Fwd_Avg_Rating_E1',
                    'Top1_Rating_E1','Rating_Std_E1']
    print('\nResultados UCL:')
    print(ucl_enr[cols_muestra].to_string())

    # ── PL: 5 primeros partidos ───────────────────────────────────────────────
    print('\n\n=== PL 24-25 (5 partidos) ===')
    pl = pd.read_csv('data/premier_2024-25_enriquecido.csv').head(5).copy()
    pl_enr = enriquecer_dataset(pl, sess, es_pl=True)
    print('\nResultados PL:')
    print(pl_enr[cols_muestra].to_string())

finally:
    sess.close()
