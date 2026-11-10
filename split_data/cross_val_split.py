from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from imblearn.pipeline import Pipeline
from imblearn.combine  import SMOTETomek
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from preprocessing import get_preprocessor

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

SCORING = {
    'accuracy'    : 'accuracy',
    'f1_weighted' : 'f1_weighted',
    'f1_macro'    : make_scorer(f1_score, average='macro', zero_division=0),
}

def build_pipeline(estimator):
    """Preprocessor -> SMOTETomek -> Classifier."""
    return Pipeline([
        ('preprocessor', get_preprocessor()),
        ('smote_tomek',  SMOTETomek(random_state=42)),
        ('classifier',   estimator),
    ])
