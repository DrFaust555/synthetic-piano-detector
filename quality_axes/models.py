"""Where the quality tool reads its fitted models and writes retrained ones.

MODELS    folder of the fitted models used for scoring: data/quality/models (shipped with the repository: the models
          behind the printed quality column of Table 6 and the liveness critic), or the folder named by the
          environment variable SPD_QUALITY_MODELS (e.g. results/quality_models after retraining with --save).
SAVE_DIR  where --save and the extraction scripts write: results/quality_models.
TRENDS    register trends of the corpus features (data/corpus/register_trends_cal.json).
"""
import os

from detector.paths import DATA, RESULTS

MODELS = os.environ.get('SPD_QUALITY_MODELS', os.path.join(DATA, 'quality', 'models'))
SAVE_DIR = os.path.join(RESULTS, 'quality_models')
TRENDS = os.path.join(DATA, 'corpus', 'register_trends_cal.json')
FEATURES_CAL = os.path.join(DATA, 'corpus', 'features_cal.csv')
LIVENESS_CAL = os.path.join(DATA, 'quality', 'liveness_cal.csv')


def save_path(name: str) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    return os.path.join(SAVE_DIR, name)
