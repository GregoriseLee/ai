"""Dùng mô hình đã train để dự đoán vài mẫu ngẫu nhiên trong tập test (mẫu mô hình chưa học).

Chạy:  python predict.py            # 5 mẫu, dùng lại file dữ liệu lúc train
       python predict.py 10
       python predict.py 10 --data bodmas.npz bodmas_metadata.csv   # nếu đã chuyển file đi chỗ khác
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

from train import MODEL_PATH, find_data_files, load_bodmas

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def main():
    parser = argparse.ArgumentParser(description="Dự đoán thử vài mẫu bằng mô hình đã train.")
    parser.add_argument("n", nargs="?", type=int, default=5, help="số mẫu muốn thử (mặc định 5)")
    parser.add_argument("--data", nargs="+", metavar="FILE", help="file/thư mục dữ liệu nếu khác lúc train")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        sys.exit(f"Chưa có mô hình {MODEL_PATH}. Chạy trước: python train.py bodmas.npz bodmas_metadata.csv")
    saved = joblib.load(MODEL_PATH)  # {"model", "threshold", "npz", "metadata", "test_index"}

    files = args.data or [saved["npz"], saved["metadata"]]
    if not args.data and not all(Path(f).is_file() for f in files):
        sys.exit(f"Không còn thấy file dữ liệu lúc train ({saved['npz']}).\n"
                 "Chỉ lại đường dẫn: python predict.py 5 --data bodmas.npz bodmas_metadata.csv")
    X, y, meta = load_bodmas(*find_data_files(files))

    pool = saved["test_index"]
    if pool.max() >= len(y):
        sys.exit("File dữ liệu không khớp với file đã dùng để train.")
    n = max(1, min(args.n, len(pool)))
    idx = np.random.default_rng().choice(pool, n, replace=False)
    prob = saved["model"].predict_proba(X[idx])[:, 1]  # xác suất là malware

    for i, p in zip(idx, prob):
        guess = "malware" if p >= saved["threshold"] else "benign"
        actual = "malware" if y[i] == 1 else "benign"
        mark = "✓" if guess == actual else "✗"
        print(f"{mark} mẫu {i:>6}: dự đoán {guess:<7} (xác suất {p:.3f}) | thực tế {actual}")


if __name__ == "__main__":
    main()
