import sys
from pathlib import Path

# Permet de lancer le script de n'importe quelle façon
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from data import load_raw, clean_data, TARGET,FILE

# 1. Données
df = clean_data(load_raw(FILE))
X, y = df.drop(columns=[TARGET]), df[TARGET]

# 2. Split stratifié AVANT tout prétraitement (évite la fuite de données)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Types de colonnes
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# 4. Prétraitement
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 5. Pipeline : prétraitement + modèle
model = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

# 6. Entraînement
model.fit(X_train, y_train)

# 7. Évaluation sur le test
proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 3))
print("PR-AUC  :", round(average_precision_score(y_test, proba), 3))
print(classification_report(y_test, model.predict(X_test)))

# 8. Validation croisée (mesure plus stable)
scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print("ROC-AUC (CV 5 plis) :", round(scores.mean(), 3), "±", round(scores.std(), 3))

# 9. Facteurs qui augmentent le plus le risque de churn
names = model.named_steps["pre"].get_feature_names_out()
coefs = model.named_steps["clf"].coef_[0]
top = pd.Series(coefs, index=names).sort_values(ascending=False).head(10)
print("\nTop 10 facteurs de churn :\n", top.round(2))

# 10. Sauvegarde
Path("data").mkdir(exist_ok=True)
joblib.dump(model, "data/baseline.joblib")