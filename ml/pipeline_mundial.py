"""
Pipeline ML para selecciones nacionales — Mundial 2026 y torneos FIFA/UEFA.

Reutiliza el pipeline UCL completo con las siguientes adaptaciones:
- Dataset: selecciones_combinado.csv (2337 partidos × 143 cols)
- Rankings FIFA en lugar de coeficiente UEFA de clubes
- Sede neutral (todos los partidos de torneos internacionales)
- ELO/forma/H2H se calculan sobre todo el historial de selecciones
"""

from pathlib import Path
import pandas as pd

# Reutiliza todo el pipeline UCL; solo se inyecta el CSV de selecciones
from ml.knime_workflow_converter import main as _main_ucl, predecir_partido as _predecir_ucl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MUNDIAL_DATASET = _PROJECT_ROOT / "data" / "selecciones_combinado.csv"

# ─── Rankings FIFA aproximados (puntos / 10 → escala similar a UEFA coef) ────
# Fuente: ranking FIFA 2024-2025, convertido a escala 0-200 para compatibilidad
# con la feature Coef_UEFA_E1/E2 (que el pipeline ya conoce).
FIFA_RANKING = {
    # TOP 10
    'Argentina':     195, 'Francia':       188, 'España':        185,
    'Inglaterra':    183, 'Brasil':         180, 'Portugal':      177,
    'Países Bajos':  172, 'Bélgica':        168, 'Italia':        165,
    'Alemania':      163,
    # 11-20
    'Colombia':      160, 'Marruecos':      158, 'Uruguay':       156,
    'Croacia':       154, 'Japón':          152, 'USA':           150,
    'México':        148, 'Senegal':        146, 'Ecuador':       144,
    'Dinamarca':     143,
    # 21-30
    'Suiza':         142, 'Austria':        140, 'Corea del Sur': 139,
    'Australia':     138, 'Turquía':        137, 'Polonia':       136,
    'Ucrania':       135, 'Serbia':         134, 'Suecia':        133,
    'Chile':         132,
    # 31-40
    'Iran':          131, 'Rumania':        130, 'Eslovaquia':    129,
    'Hungría':       128, 'Catar':          127, 'Ghana':         126,
    'Costa Rica':    125, 'Venezuela':      124, 'Canadá':        123,
    'Nigeria':       122,
    # 41-50
    'Egipto':        121, 'Costa de Marfil':120, 'Argelia':       119,
    'Camerún':       118, 'Túnez':          117, 'Arabia Saudita':116,
    'Perú':          115, 'Bolivia':        114, 'Paraguay':      113,
    'Jamaica':       112,
    # Resto / OCF
    'Irán':          131, 'Rep. Checa':     129, 'Grecia':        128,
    'Escocia':       127, 'Noruega':        126, 'Israel':        120,
    'Albania':       118, 'Gales':          130, 'Irlanda':       115,
    # WC 2018 / 2022
    'Rusia':         130, 'Arabia Saudita': 116, 'Egipto':        121,
    'Arabia':        116, 'Polonia':        136, 'Senegal':       146,
    'Dinamarca':     143, 'Bélgica':        168, 'Panamá':        108,
    'Tunisia':       117, 'Colombia':       160, 'Japón':         152,
    'Suiza':         142,
}
FIFA_RANKING_DEFAULT = 100


def _inject_fifa_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Inyecta ranking FIFA en las columnas Coef_UEFA_E1/E2/Diff_Coef_UEFA.

    Reutiliza los mismos nombres de columna que el pipeline UCL para que
    select_columns() las recoja sin modificación.
    """
    df = df.copy()
    df['Coef_UEFA_E1']   = df['Equipo1'].map(FIFA_RANKING).fillna(FIFA_RANKING_DEFAULT)
    df['Coef_UEFA_E2']   = df['Equipo2'].map(FIFA_RANKING).fillna(FIFA_RANKING_DEFAULT)
    df['Diff_Coef_UEFA'] = df['Coef_UEFA_E1'] - df['Coef_UEFA_E2']
    return df


def main_mundial(filepath: str | None = None) -> dict:
    """Entrena el pipeline completo con datos de selecciones nacionales.

    - Carga selecciones_combinado.csv (2337 partidos, todos los torneos)
    - Inyecta ranking FIFA antes de que el pipeline corra
    - Devuelve el mismo dict que main() de knime_workflow_converter
    """
    fp = filepath or str(DEFAULT_MUNDIAL_DATASET)

    # Monkey-patch temporal: el pipeline UCL llama compute_uefa_coef_features()
    # que sobreescribirá nuestras columnas. Substituimos esa función solo para
    # esta llamada para inyectar rankings FIFA en su lugar.
    import ml.knime_workflow_converter as _kw

    _original_coef = _kw.compute_uefa_coef_features

    def _fifa_coef(df):
        """Wrapper que usa ranking FIFA en lugar de coeficiente UEFA."""
        return _inject_fifa_ranking(df)

    _kw.compute_uefa_coef_features = _fifa_coef
    try:
        results = _main_ucl(fp, context_filepaths=None)
    finally:
        _kw.compute_uefa_coef_features = _original_coef

    return results


def predecir_partido_mundial(equipo1: str, equipo2: str, results: dict,
                              n_runs: int = 20, fase: str = 'Ronda 1') -> dict | None:
    """Predice un partido de selecciones usando el pipeline mundial.

    Idéntico a predecir_partido() pero:
    - Usa FIFA_RANKING en vez de UEFA_COEF para las features Coef_UEFA_*
    - Sede neutral por defecto (es_local=0) salvo si el país anfitrión juega
    """
    import ml.knime_workflow_converter as _kw

    _original_coef = _kw.UEFA_COEF
    _original_default = _kw.UEFA_COEF_DEFAULT

    _kw.UEFA_COEF = FIFA_RANKING
    _kw.UEFA_COEF_DEFAULT = FIFA_RANKING_DEFAULT
    try:
        salida = _predecir_ucl(equipo1, equipo2, results, n_runs=n_runs, fase=fase)
    finally:
        _kw.UEFA_COEF = _original_coef
        _kw.UEFA_COEF_DEFAULT = _original_default

    return salida
