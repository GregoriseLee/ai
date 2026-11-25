"""Huấn luyện Logistic Regression phát hiện malware PE trên bộ BODMAS.

Chạy (chỉ cần đưa file, không phải sắp xếp thư mục):
    python train.py bodmas.npz bodmas_metadata.csv   # đưa 2 file
    python train.py ~/Downloads/bodmas                # đưa thư mục chứa 2 file
    python train.py                                   # tự tìm trong data/ cạnh file này
Kết quả:  models/logreg.joblib, reports/metrics.json, reports/errors.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Cảnh báo "divide by zero/overflow in matmul" vô hại của numpy (Accelerate) trên macOS.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

HERE = Path(__file__).parent
DEFAULT_DATA_DIR = HERE / "data"
MODEL_PATH = HERE / "models" / "logreg.joblib"
REPORT_DIR = HERE / "reports"


def find_data_files(paths):
    """Từ 0, 1 hoặc 2 đường dẫn người dùng đưa vào, tìm ra (file .npz, file metadata .csv)."""
    paths = [Path(p).expanduser() for p in paths] or [DEFAULT_DATA_DIR]
    npz = next((p for p in paths if p.suffix == ".npz"), None)
    csv = next((p for p in paths if p.suffix == ".csv"), None)

    # Đưa thư mục, hoặc chỉ đưa 1 trong 2 file -> tìm file còn lại cùng chỗ.
    folders = [p for p in paths if p.is_dir()] + [p.parent for p in (npz, csv) if p is not None]
    for folder in folders:
        if npz is None:
            npz = next(iter(sorted(folder.glob("*.npz"))), None)
        if csv is None:
            csv = next(iter(sorted(folder.glob("*metadata*.csv")) or sorted(folder.glob("*.csv"))), None)

    for name, path in (("file đặc trưng .npz", npz), ("file metadata .csv", csv)):
        if path is None or not path.is_file():
            sys.exit(f"Không tìm thấy {name} (đã tìm trong: {', '.join(map(str, paths))}).\n"
                     "Cách chạy: python train.py bodmas.npz bodmas_metadata.csv")
    return npz.resolve(), csv.resolve()


def load_bodmas(npz_path, meta_path):
    """X: ma trận đặc trưng (mỗi dòng 1 file PE, 2381 cột), y: 0 = benign, 1 = malware."""
    npz = np.load(npz_path)
    if not {"X", "y"} <= set(npz.files):
        sys.exit(f"{npz_path} không có mảng 'X' và 'y' (đang có: {npz.files}).")
    X, y = npz["X"], npz["y"].astype(int)
    meta = pd.read_csv(meta_path)  # sha, timestamp, family
    missing = {"sha", "timestamp", "family"} - set(meta.columns)
    if missing:
        sys.exit(f"{meta_path} thiếu cột {sorted(missing)}.")
    if len(meta) != len(y):
        sys.exit(f"Số dòng không khớp: {npz_path.name} có {len(y)} mẫu, {meta_path.name} có {len(meta)} dòng.")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True, errors="coerce")
    return X, y, meta


def time_split(y, timestamps, cutoff):
    """Malware trước `cutoff` -> train, từ `cutoff` trở đi -> test (malware "mới").
    File sạch (benign) chia ngẫu nhiên theo cùng tỉ lệ."""
    malware = np.where(y == 1)[0]
    benign = np.where(y == 0)[0]
    is_new = (timestamps.iloc[malware] >= cutoff).to_numpy()
    if is_new.all() or not is_new.any():
        sys.exit(f"Không chia được theo mốc {cutoff:%Y-%m-%d}: cần có malware cả trước và sau mốc này.")
    benign_train, benign_test = train_test_split(benign, test_size=is_new.mean(), random_state=42)
    return np.concatenate([malware[~is_new], benign_train]), np.concatenate([malware[is_new], benign_test])


def safe_div(a, b):
    return round(float(a / b), 5) if b else 0.0


def evaluate(y_true, prob, threshold):
    """Tính các chỉ số khi coi 'xác suất >= threshold' là malware."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": safe_div(tp, tp + fp),   # trong số bị báo malware, bao nhiêu là thật
        "recall": safe_div(tp, tp + fn),      # trong số malware thật, bắt được bao nhiêu
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "fpr": safe_div(fp, fp + tn),         # tỉ lệ file sạch bị báo nhầm
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Logistic Regression phát hiện malware trên BODMAS.")
    parser.add_argument("files", nargs="*", help="bodmas.npz và/hoặc bodmas_metadata.csv, hoặc thư mục chứa chúng")
    parser.add_argument("--samples", type=int, default=30_000, help="số mẫu dùng để train (0 = dùng hết)")
    parser.add_argument("--cutoff", default="2020-01-01", help="malware từ ngày này trở đi làm tập test")
    args = parser.parse_args()

    # ---- 1. Nạp dữ liệu, lấy tập con cho máy chạy nhanh ----
    npz_path, meta_path = find_data_files(args.files)
    print(f"Dữ liệu: {npz_path}\n          {meta_path}")
    X, y, meta = load_bodmas(npz_path, meta_path)
    keep = np.arange(len(y))  # chỉ số gốc trong file dữ liệu
    if 0 < args.samples < len(y):
        keep, _ = train_test_split(keep, train_size=args.samples, stratify=y, random_state=42)
        X, y, meta = X[keep], y[keep], meta.iloc[keep].reset_index(drop=True)

    # ---- 2. Chia dữ liệu theo thời gian ----
    # Test = malware từ mốc cutoff trở đi (mô phỏng gặp malware mới chưa từng thấy).
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    train, test = time_split(y, meta.timestamp, cutoff)
    # Validation = 20% malware mới nhất trong phần train, dùng để chọn ngưỡng.
    ts_train = meta.timestamp.iloc[train].reset_index(drop=True)
    tr, va = time_split(y[train], ts_train, ts_train[y[train] == 1].quantile(0.8))
    tr, va = train[tr], train[va]
    sizes = {"train": len(tr), "validation": len(va), "test": len(test)}
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
    prob_va = model.predict_proba(X[va])[:, 1]
    fpr, tpr, thresholds = roc_curve(y[va], prob_va, drop_intermediate=False)
    threshold = float(min(thresholds[fpr <= 0.005][-1], 1.0))
    if threshold >= 1.0:
        print("CẢNH BÁO: không có ngưỡng nào đạt FPR <= 0.5% trên validation, dùng 0.5 thay thế.")
        threshold = 0.5

    # ---- 5. Đánh giá trên test ----
    prob = model.predict_proba(X[test])[:, 1]  # xác suất là malware
    y_test = y[test]
    results = {
        "data": {"npz": str(npz_path), "metadata": str(meta_path), "cutoff": args.cutoff, "sizes": sizes},
        "validation_selected": evaluate(y[va], prob_va, threshold),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "pr_auc": round(average_precision_score(y_test, prob), 4),
        "threshold_0.5": evaluate(y_test, prob, 0.5),
        "threshold_selected": evaluate(y_test, prob, threshold),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))

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
    joblib.dump({
        "model": model,
        "threshold": threshold,
        "npz": str(npz_path), "metadata": str(meta_path),
        "test_index": keep[test],  # chỉ số gốc của tập test -> predict.py chỉ thử trên mẫu chưa học
    }, MODEL_PATH)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    errors.to_csv(REPORT_DIR / "errors.csv", index=False)
    print(f"Đã lưu {MODEL_PATH} và {REPORT_DIR}/")


if __name__ == "__main__":
    main()
