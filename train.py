"""Huấn luyện & đánh giá baseline: Logistic Regression và Linear SVM.

- Chia train/test theo `group` (GroupShuffleSplit): payload trong tập test chưa
  từng xuất hiện khi train.
- Chọn siêu tham số C bằng GridSearchCV + GroupKFold trên tập train.
- Lưu: models/<model>.joblib, reports/metrics.json, ma trận nhầm lẫn (PNG),
  top đặc trưng có trọng số lớn nhất (giải thích mô hình).

Chạy:  python -m ai.train --data data/requests.csv --model both
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

# numpy + Accelerate trên macOS in cảnh báo matmul giả (vô hại) khi fit LogisticRegression
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")  # cho các tiến trình con của GridSearchCV

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ai.features import build_features

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "svm": lambda: LinearSVC(class_weight="balanced", dual="auto", max_iter=5000),
}
PARAM_GRID = {"clf__C": [0.1, 1.0, 10.0]}


def build_pipeline(model: str) -> Pipeline:
    return Pipeline([("features", build_features()), ("clf", MODELS[model]())])


def scores_of(pipe: Pipeline, X) -> np.ndarray:
    """Điểm liên tục để tính ROC-AUC (xác suất với LR, khoảng cách lề với SVM)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    return pipe.decision_function(X)


def top_features(pipe: Pipeline, k: int = 20) -> dict:
    names = pipe.named_steps["features"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_
    if coef.shape[0] != 1:  # multiclass: chỉ lấy lớp đầu ra không phải benign
        return {}
    w = coef[0]
    order = np.argsort(w)
    fmt = lambda idx: [{"feature": str(names[i]), "weight": round(float(w[i]), 4)} for i in idx]
    return {"malicious": fmt(order[::-1][:k]), "benign": fmt(order[:k])}


def save_confusion_png(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(pipe: Pipeline, X_test, y_test, task: str) -> dict:
    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    latency_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    s = scores_of(pipe, X_test)

    if task == "binary":
        s1 = s[:, 1] if s.ndim == 2 else s
        m = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, s1),
        }
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        m["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
        labels = ["benign", "malicious"]
    else:
        labels = sorted(set(y_test) | set(y_pred))
        m = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }
        cm = confusion_matrix(y_test, y_pred, labels=labels)

    m = {k: round(float(v), 4) for k, v in m.items()}
    m["latency_ms_per_request"] = round(latency_ms, 4)
    m["confusion_matrix"] = {"labels": [str(l) for l in labels], "matrix": cm.tolist()}
    m["report"] = classification_report(y_test, y_pred, target_names=None if task != "binary" else labels,
                                        output_dict=True, zero_division=0)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/requests.csv")
    ap.add_argument("--model", choices=["logreg", "svm", "both"], default="both")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary",
                    help="binary: benign vs malicious; multiclass: benign/sqli/xss/lfi/rce")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "group" not in df.columns:  # dữ liệu ngoài (vd. log Cloud Armor) không có group
        df["group"] = np.arange(len(df))
    X = df["request"].astype(str).to_numpy()
    y = df["label"].to_numpy() if args.task == "binary" else df["category"].to_numpy()
    groups = df["group"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    print(f"Train: {len(train_idx)} dòng | Test: {len(test_idx)} dòng (payload test chưa thấy khi train)")

    models_dir, reports_dir = Path(args.models_dir), Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    names = ["logreg", "svm"] if args.model == "both" else [args.model]
    all_metrics = {}
    for name in names:
        print(f"\n=== {name} ({args.task}) ===")
        search = GridSearchCV(
            build_pipeline(name), PARAM_GRID, cv=GroupKFold(n_splits=3),
            scoring="f1" if args.task == "binary" else "f1_macro", n_jobs=-1,
        )
        t0 = time.perf_counter()
        search.fit(X_train, y_train, groups=groups[train_idx])
        train_s = time.perf_counter() - t0
        best = search.best_estimator_

        m = evaluate(best, X_test, y_test, args.task)
        m["best_params"] = search.best_params_
        m["cv_best_score"] = round(float(search.best_score_), 4)
        m["train_seconds"] = round(train_s, 2)
        if args.task == "binary":
            m["top_features"] = top_features(best)
        all_metrics[name] = m

        joblib.dump({"pipeline": best, "task": args.task}, models_dir / f"{name}_{args.task}.joblib")
        cm = np.array(m["confusion_matrix"]["matrix"])
        save_confusion_png(cm, m["confusion_matrix"]["labels"], f"{name} - {args.task}",
                           reports_dir / f"confusion_{name}_{args.task}.png")

        summary = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        print(f"best C={search.best_params_['clf__C']}  " + "  ".join(f"{k}={v}" for k, v in summary.items()))

    out = reports_dir / f"metrics_{args.task}.json"
    out.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu mô hình vào {models_dir}/ và báo cáo vào {out}")


if __name__ == "__main__":
    main()
