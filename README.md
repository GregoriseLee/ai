# Đề 04 — Phát hiện malware PE trên BODMAS (Logistic Regression)

1. Cài thư viện: `pip install -r requirements.txt`
2. Tải BODMAS (https://github.com/whyisyoung/BODMAS) về đâu cũng được (`bodmas.npz` + `bodmas_metadata.csv`).
3. Train, chỉ cần đưa file vào:
   ```
   python train.py đường/dẫn/bodmas.npz đường/dẫn/bodmas_metadata.csv
   ```
   hoặc đưa thư mục chứa 2 file: `python train.py đường/dẫn/thư_mục`
   (không đưa gì thì tự tìm trong `data/` cạnh file code).
   → tạo `models/logreg.joblib`, `reports/metrics.json`, `reports/errors.csv`
4. Dự đoán thử 5 mẫu trong tập test: `python predict.py 5`

Tuỳ chọn: `--samples 0` để train trên toàn bộ dữ liệu (mặc định 30 000 mẫu), `--cutoff 2020-01-01` để đổi mốc chia train/test.

Lưu ý: malware được chia theo thời gian, còn file sạch chia ngẫu nhiên, nên FPR trên test có thể lạc quan hơn thực tế.
