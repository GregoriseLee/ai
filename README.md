# Đề 04 — Phát hiện malware PE trên BODMAS (Logistic Regression)

1. Tải BODMAS (https://github.com/whyisyoung/BODMAS), đặt `bodmas.npz` và `bodmas_metadata.csv` vào `data/`.
2. Train: `python train.py` → tạo `models/logreg.joblib`, `reports/metrics.json`, `reports/errors.csv`
3. Dự đoán thử: `python predict.py 5`

Cần: numpy, pandas, scikit-learn, joblib.
