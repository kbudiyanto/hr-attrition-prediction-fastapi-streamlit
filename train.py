from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = Path("attrition_model_v3.joblib")

FEATURE_COLS = [
    "Age",
    "JobRole",
    "MaritalStatus",
    "MonthlyIncome",
    "YearsAtCompany",
    "EnvironmentSatisfaction",
    "OverTime",
]


def train_and_save(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    df = pd.read_csv(data_path)
    missing_cols = [c for c in FEATURE_COLS + ["Attrition"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan pada CSV: {missing_cols}")

    X = df[FEATURE_COLS].copy()
    y = (df["Attrition"].astype(str).str.strip() == "Yes").astype(int)

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "cv_5fold_roc_auc_mean": round(float(cv_auc.mean()), 4),
    }

    print("=== Evaluasi Model (7 Fitur) ===")
    print(f"Metrics : {metrics}")
    print(classification_report(y_test, y_pred, target_names=["No (Stay)", "Yes (Resign)"]))

    pipeline.fit(X, y)
    artifact = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLS,
        "metrics": metrics,
    }
    joblib.dump(artifact, model_path)
    print(f"Model tersimpan di '{model_path}'.")
    return metrics


if __name__ == "__main__":
    train_and_save()