from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_COLS   = ['Age', 'Sleep_Hours', 'SocialMedia_Hours', 'Depression_Score']
CATEGORICAL_COLS = ['Employment_Status', 'Coping_Methods']

def get_preprocessor():
    return ColumnTransformer(transformers=[
        ('ohe',    OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_COLS),
        ('scaler', StandardScaler(), NUMERICAL_COLS)
    ], remainder='passthrough')
