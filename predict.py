"""Dùng mô hình đã train để dự đoán vài mẫu ngẫu nhiên trong BODMAS.

Chạy:  python predict.py 5
"""
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import joblib
import numpy as np

from train import MODEL_PATH, load_bodmas

saved = joblib.load(MODEL_PATH)              # {"model": ..., "threshold": ...}
X, y, meta = load_bodmas()

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
idx = np.random.choice(len(y), n, replace=False)
prob = saved["model"].predict_proba(X[idx])[:, 1]  # xác suất là malware

for i, p in zip(idx, prob):
    guess = "malware" if p >= saved["threshold"] else "benign"
    actual = "malware" if y[i] == 1 else "benign"
    print(f"mẫu {i:>6}: dự đoán {guess:<7} (xác suất {p:.3f}) | thực tế {actual}")
