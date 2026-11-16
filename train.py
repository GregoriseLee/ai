"""Huấn luyện Logistic Regression phát hiện malware PE trên bộ BODMAS.

Chuẩn bị: đặt bodmas.npz và bodmas_metadata.csv (tải từ https://github.com/whyisyoung/BODMAS)
          vào thư mục data/ cạnh file này.
Chạy:     python train.py
Kết quả:  models/logreg.joblib, reports/metrics.json, reports/errors.csv
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)  # cảnh báo vô hại của numpy trên macOS

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
MODEL_PATH = HERE / "models" / "logreg.joblib"
REPORT_DIR = HERE / "reports"


def load_bodmas():
    """X: ma trận đặc trưng (mỗi dòng 1 file PE, 2381 cột), y: 0 = benign, 1 = malware."""
    npz = np.load(DATA_DIR / "bodmas.npz")
    X, y = npz["X"], npz["y"].astype(int)
    meta = pd.read_csv(DATA_DIR / "bodmas_metadata.csv")  # sha, timestamp, family
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True, errors="coerce")
    return X, y, meta


def time_split(y, timestamps, cutoff):
    """Malware trước `cutoff` -> train, từ `cutoff` trở đi -> test (malware "mới").
    File sạch (benign) chia ngẫu nhiên theo cùng tỉ lệ."""
    malware = np.where(y == 1)[0]
    benign = np.where(y == 0)[0]
    is_new = (timestamps.iloc[malware] >= cutoff).to_numpy()
    benign_train, benign_test = train_test_split(benign, test_size=is_new.mean(), random_state=42)
    return np.concatenate([malware[~is_new], benign_train]), np.concatenate([malware[is_new], benign_test])


def evaluate(y_true, prob, threshold):
    """Tính các chỉ số khi coi 'xác suất >= threshold' là malware."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(tp / (tp + fp), 4),   # trong số bị báo malware, bao nhiêu là thật
        "recall": round(tp / (tp + fn), 4),      # trong số malware thật, bắt được bao nhiêu
        "f1": round(2 * tp / (2 * tp + fp + fn), 4),
        "fpr": round(fp / (fp + tn), 5),         # tỉ lệ file sạch bị báo nhầm
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def main():
    # ---- 1. Nạp dữ liệu, lấy tập con 30 000 mẫu cho máy chạy nhanh ----
    X, y, meta = load_bodmas()
    if len(y) > 30_000:
        keep, _ = train_test_split(np.arange(len(y)), train_size=30_000, stratify=y, random_state=42)
        X, y, meta = X[keep], y[keep], meta.iloc[keep].reset_index(drop=True)

    # ---- 2. Chia dữ liệu theo thời gian ----
    # Test = malware từ 01/2020 trở đi (mô phỏng gặp malware mới chưa từng thấy).
    train, test = time_split(y, meta.timestamp, pd.Timestamp("2020-01-01", tz="UTC"))
    # Validation = 20% malware mới nhất trong phần train, dùng để chọn ngưỡng.
    ts_train = meta.timestamp.iloc[train].reset_index(drop=True)
    tr, va = time_split(y[train], ts_train, ts_train[y[train] == 1].quantile(0.8))
    tr, va = train[tr], train[va]
    print(f"Train {len(tr)} | Validation {len(va)} | Test {len(test)} mẫu")

    # ---- 3. Mô hình: các bước xử lý nối tiếp nhau ----
    model = Pipeline([
        ("log", FunctionTransformer(np.arcsinh)),    # nén giá trị rất lớn (vd. kích thước file) giống log
        ("drop_constant", VarianceThreshold(0.0)),   # bỏ cột có giá trị không đổi
        ("scale", StandardScaler()),                 # đưa các cột về cùng thang đo
        ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
    ])
    model.fit(X[tr], y[tr])  # chỉ học từ train -> không rò rỉ thông tin từ test

    # ---- 4. Chọn ngưỡng trên validation: bắt nhiều malware nhất mà FPR vẫn <= 0.5% ----
    fpr, tpr, thresholds = roc_curve(y[va], model.predict_proba(X[va])[:, 1], drop_intermediate=False)
    threshold = float(min(thresholds[fpr <= 0.005][-1], 1.0))

    # ---- 5. Đánh giá trên test ----
    prob = model.predict_proba(X[test])[:, 1]  # xác suất là malware
    y_test = y[test]
    results = {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "pr_auc": round(average_precision_score(y_test, prob), 4),
        "threshold_0.5": evaluate(y_test, prob, 0.5),
        "threshold_selected": evaluate(y_test, prob, threshold),
    }
    print(json.dumps(results, indent=2))

    # ---- 6. Phân tích lỗi: 10 file sạch điểm cao nhất + 10 malware điểm thấp nhất ----
    meta_test = meta.iloc[test].reset_index(drop=True)
    benign, malware = np.where(y_test == 0)[0], np.where(y_test == 1)[0]
    worst_fp = benign[np.argsort(-prob[benign])][:10]
    worst_fn = malware[np.argsort(prob[malware])][:10]
    rows = np.concatenate([worst_fp, worst_fn])
    errors = meta_test.iloc[rows][["sha", "family"]].copy()
    errors["type"] = ["false_positive"] * len(worst_fp) + ["false_negative"] * len(worst_fn)
    errors["probability"] = prob[rows].round(4)
    errors["wrong"] = np.concatenate([prob[worst_fp] >= threshold, prob[worst_fn] < threshold])

    # ---- 7. Lưu mô hình và kết quả ----
    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold}, MODEL_PATH)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    errors.to_csv(REPORT_DIR / "errors.csv", index=False)
    print(f"Đã lưu {MODEL_PATH} và {REPORT_DIR}/")


if __name__ == "__main__":
    main()
