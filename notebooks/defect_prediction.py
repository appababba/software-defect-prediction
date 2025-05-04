# defect_prediction.py – streamlined, non-blocking plots
# ------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for CLI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             classification_report, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import randint

# ------------------------------------------------------------
# Helper definitions
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def build_pipe(model):
    return Pipeline([
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("select", SelectKBest(f_classif, k=15)),
        ("model", model)
    ])

# Saves three plots per model and returns metrics
def report(model, X_test, y_test, title=""):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n== {title} ==")
    print(classification_report(y_test, pred, digits=3))
    # confusion
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(f"{title.replace(' ','_')}_confusion.png")
    plt.clf()
    # ROC curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(f"{title.replace(' ','_')}_roc.png")
    plt.clf()
    # PR curve
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(f"{title.replace(' ','_')}_pr.png")
    plt.clf()
    return {
        "acc": accuracy_score(y_test, pred),
        "prec": precision_score(y_test, pred),
        "rec": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba)
    }

# ------------------------------------------------------------
# Load and prepare data
cm1 = pd.read_csv("../data/cm1.csv")
jm1 = pd.read_csv("../data/jm1.csv")
cm1["defects"] = cm1["defects"].astype(int)
jm1["defects"] = jm1["defects"].astype(int)

# Split helper
def split(df):
    X = df.drop("defects", axis=1)
    y = df["defects"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

Xtr_c, Xte_c, ytr_c, yte_c = split(cm1)
Xtr_j, Xte_j, ytr_j, yte_j = split(jm1)

# ------------------------------------------------------------
# Baseline models
models = {
    "LogReg": build_pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    "RF": build_pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    "GNB": build_pipe(GaussianNB())
}
results = []
for name, pipe in models.items():
    pipe.fit(Xtr_c, ytr_c)
    r = report(pipe, Xte_c, yte_c, f"CM1_{name}")
    r.update({"set":"CM1","model":name}); results.append(r)
for name, pipe in models.items():
    pipe.fit(Xtr_j, ytr_j)
    r = report(pipe, Xte_j, yte_j, f"JM1_{name}")
    r.update({"set":"JM1","model":name}); results.append(r)

# ------------------------------------------------------------
# Tune RF on JM1 with reduced search
param_dist = {
    "smote__k_neighbors": [3],
    "model__n_estimators": randint(100,300),
    "model__max_depth": [10, None],
    "model__min_samples_leaf": [1]
}
search = RandomizedSearchCV(
    build_pipe(RandomForestClassifier(random_state=42)),
    param_distributions=param_dist,
    n_iter=10, scoring="f1", cv=CV, n_jobs=-1, random_state=42
)
search.fit(Xtr_j, ytr_j)
print("best RF params:", search.best_params_)
rf_best = search.best_estimator_
res_best = report(rf_best, Xte_j, yte_j, "JM1_RF_tuned")
res_best.update({"set":"JM1","model":"RF_tuned"}); results.append(res_best)

# ------------------------------------------------------------
# Summary
summary = pd.DataFrame(results)
summary.to_csv("summary_metrics.csv", index=False)
print("\n=== Summary Metrics ===")
print(summary)

# bar plot of F1
sns.barplot(data=summary, x="model", y="f1", hue="set")
plt.title("F1 – defect class")
plt.savefig("f1_comparison.png")
plt.clf()
