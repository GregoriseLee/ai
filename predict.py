"""Dùng mô hình đã huấn luyện để phân loại request.

Ví dụ:
  python -m ai.predict "GET /search?q=' OR 1=1--" "GET /products?id=12"
  python -m ai.predict --model models/svm_binary.joblib --file requests.txt
"""
from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np


class Detector:
    def __init__(self, path: str = "models/logreg_binary.joblib"):
        bundle = joblib.load(path)
        self.pipe = bundle["pipeline"]
        self.task = bundle["task"]

    def predict(self, requests: list[str]) -> list[dict]:
        preds = self.pipe.predict(requests)
        if hasattr(self.pipe, "predict_proba"):
            scores = self.pipe.predict_proba(requests).max(axis=1)
            score_name = "probability"
        else:
            d = self.pipe.decision_function(requests)
            scores = np.abs(d) if d.ndim == 1 else d.max(axis=1)
            score_name = "margin"
        out = []
        for r, p, s in zip(requests, preds, scores):
            label = ("malicious" if p == 1 else "benign") if self.task == "binary" else str(p)
            out.append({"request": r, "label": label, score_name: round(float(s), 4)})
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("requests", nargs="*", help="request dạng 'METHOD /path?query'")
    ap.add_argument("--model", default="models/logreg_binary.joblib")
    ap.add_argument("--file", help="file mỗi dòng một request ('-' = stdin)")
    args = ap.parse_args()

    reqs = list(args.requests)
    if args.file:
        f = sys.stdin if args.file == "-" else open(args.file, encoding="utf-8")
        reqs += [line.rstrip("\n") for line in f if line.strip()]
    if not reqs:
        ap.error("cần ít nhất một request")

    for r in Detector(args.model).predict(reqs):
        score = {k: v for k, v in r.items() if k not in ("request", "label")}
        print(f"[{r['label']:>9}] {score}  {r['request']}")


if __name__ == "__main__":
    main()
