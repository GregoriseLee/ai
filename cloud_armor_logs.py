"""Kết nối mô hình AI với Cloud Armor qua google-cloud-logging.

Hai chức năng:
  export   Lấy log của Load Balancer (có Cloud Armor), gán nhãn theo kết quả
           Cloud Armor (DENY -> 1, ACCEPT -> 0), ghi ra CSV để huấn luyện/đánh giá.
  compare  Chạy mô hình trên CSV đó và so sánh với quyết định của Cloud Armor:
           tỉ lệ đồng thuận, các request AI bắt được mà Cloud Armor bỏ sót và ngược lại.

Lưu ý: log Cloud Armor chỉ có URL (method + path + query), KHÔNG có body.

Ví dụ:
  python -m ai.cloud_armor_logs export --project my-proj --policy armor-policy --hours 2
  python -m ai.cloud_armor_logs compare --data data/armor_logs.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


def build_filter(policy: str | None, hours: float) -> str:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    parts = [
        'resource.type="http_load_balancer"',
        "jsonPayload.enforcedSecurityPolicy.outcome:*",
        f'timestamp>="{since}"',
    ]
    if policy:
        parts.append(f'jsonPayload.enforcedSecurityPolicy.name="{policy}"')
    return " AND ".join(parts)


def entry_to_row(entry) -> dict | None:
    http = entry.http_request or {}
    payload = entry.payload if isinstance(entry.payload, dict) else {}
    esp = payload.get("enforcedSecurityPolicy", {})
    url = http.get("requestUrl")
    if not url or "outcome" not in esp:
        return None
    parts = urlsplit(url)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    method = http.get("requestMethod", "GET")
    return {
        "request": f"{method} {path}",
        "label": 1 if esp["outcome"] == "DENY" else 0,
        "category": "armor_deny" if esp["outcome"] == "DENY" else "armor_accept",
        "armor_rule_priority": esp.get("priority"),
        "armor_preconfigured": ",".join(esp.get("preconfiguredExprIds", []) or []),
        "remote_ip": http.get("remoteIp", ""),
        "timestamp": entry.timestamp.isoformat() if entry.timestamp else "",
    }


def cmd_export(args) -> None:
    from google.cloud import logging as gcl  # pip install google-cloud-logging

    client = gcl.Client(project=args.project)
    flt = build_filter(args.policy, args.hours)
    print(f"Filter: {flt}")
    rows = []
    for entry in client.list_entries(filter_=flt, order_by=gcl.DESCENDING, max_results=args.limit):
        row = entry_to_row(entry)
        if row:
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["request", "label", "category", "armor_rule_priority", "armor_preconfigured", "remote_ip", "timestamp"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    deny = sum(r["label"] for r in rows)
    print(f"Đã ghi {len(rows)} log ({deny} DENY, {len(rows) - deny} ACCEPT) vào {out}")


def cmd_compare(args) -> None:
    from ai.predict import Detector

    df = pd.read_csv(args.data)
    det = Detector(args.model)
    preds = det.predict(df["request"].astype(str).tolist())
    df["ai_malicious"] = [1 if p["label"] != "benign" else 0 for p in preds]
    df["armor_deny"] = df["label"]

    agree = (df["ai_malicious"] == df["armor_deny"]).mean()
    ai_only = df[(df.ai_malicious == 1) & (df.armor_deny == 0)]
    armor_only = df[(df.ai_malicious == 0) & (df.armor_deny == 1)]
    result = {
        "total": len(df),
        "agreement": round(float(agree), 4),
        "both_block": int(((df.ai_malicious == 1) & (df.armor_deny == 1)).sum()),
        "both_allow": int(((df.ai_malicious == 0) & (df.armor_deny == 0)).sum()),
        "ai_only_block": len(ai_only),
        "armor_only_block": len(armor_only),
        "ai_only_examples": ai_only["request"].head(10).tolist(),
        "armor_only_examples": armor_only["request"].head(10).tolist(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="lấy log Cloud Armor -> CSV có nhãn")
    e.add_argument("--project", required=True)
    e.add_argument("--policy", help="tên security policy (lọc)")
    e.add_argument("--hours", type=float, default=1.0)
    e.add_argument("--limit", type=int, default=5000)
    e.add_argument("--out", default="data/armor_logs.csv")
    e.set_defaults(func=cmd_export)

    c = sub.add_parser("compare", help="so sánh AI với quyết định Cloud Armor")
    c.add_argument("--data", default="data/armor_logs.csv")
    c.add_argument("--model", default="models/logreg_binary.joblib")
    c.add_argument("--out", default="reports/ai_vs_armor.json")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
