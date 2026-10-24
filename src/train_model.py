import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import joblib
import mlflow
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,recall_score,precision_score
from xgboost import XGBClassifier
from data import FILE,load_raw,clean_data,TARGET


# 1. Données + features supplémentaires

df = clean_data(load_raw(FILE))
df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
df["Nbservices"] = (df[service_cols]=="Yes").sum(axis=1)

X , Y = df.drop(columns=[TARGET]),df[TARGET]

# 2. Même split que la baseline (comparaison équitable)
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    train_size=0.2,
    stratify=Y,
    random_state=42
    )

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

def make_pre():
    return ColumnTransformer([
        ("num",StandardScaler(),num_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
    ])
ratio = (Y_train==0).sum()/ (Y_train==1).sum()

models = {
    "logreg": LogisticRegression(max_iter=1000,class_weight="balanced"),
    "xgboost": XGBClassifier(
         n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=ratio, eval_metric="logloss", random_state=42,
    ),
}

mlflow.set_experiment("churn radar")
Path("data").mkdir(exist_ok=True)
for name,clf in models.items():
    pipe = Pipeline([("pre",make_pre()),("clf", clf)])
    with mlflow.start_run(run_name=name):
        pipe.fit(X_train,Y_train)
        proba = pipe.predict_proba(X_test)[:,1]
        pred = pipe.predict(X_test)

        cv = cross_val_score(pipe,X,Y,cv=5,scoring="roc_auc")
        metrics = {
            "roc_auc_test": roc_auc_score(Y_test,proba),
            "roc_auc_cv": cv.mean(),
            "pr_auc": average_precision_score(Y_test, proba),
            "precision": precision_score(Y_test, pred),
            "recall": recall_score(Y_test, pred),
        }
        mlflow.log_params({k: str(v) for k,v in clf.get_params().items()})
        mlflow.log_metrics(metrics)
        print(name, {k: round(v, 3) for k, v in metrics.items()})
        joblib.dump(pipe, f"data/{name}.joblib")


