"""Tiền xử lý và trích xuất đặc trưng cho HTTP request.

Đặc trưng = TF-IDF n-gram ký tự (bắt được `' or`, `<scr`, `../`, `$(`...) +
một nhóm đặc trưng thủ công (độ dài, số ký tự đặc biệt...) để giải thích dễ hơn.
"""
from __future__ import annotations

import html
import math
import re
from urllib.parse import unquote_plus

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Giải mã URL lặp (chống double-encode), giải mã HTML entity, chữ thường."""
    prev = None
    s = text
    for _ in range(3):
        if s == prev:
            break
        prev, s = s, unquote_plus(s)
    s = html.unescape(s).lower()
    return _WS.sub(" ", s).strip()


# (tên đặc trưng, regex) - đếm số lần xuất hiện sau khi normalize
HANDCRAFTED = [
    ("quote", r"['\"`]"),
    ("angle", r"[<>]"),
    ("semicolon", r";"),
    ("pipe_amp", r"[|&]"),
    ("dollar_brace", r"\$\{|\$\(|\{\{"),
    ("sql_comment", r"--|#|/\*"),
    ("dot_dot_slash", r"\.\.[/\\]"),
    ("eq_compare", r"\d\s*=\s*\d|'\s*=\s*'"),
    ("sql_kw", r"\b(select|union|sleep|benchmark|waitfor|information_schema|xp_cmdshell)\b"),
    ("js_kw", r"\b(script|onerror|onload|alert|javascript|eval|document\.cookie)\b"),
    ("shell_kw", r"\b(cat|whoami|wget|curl|bash|nc|sh|powershell|cmd\.exe|uname)\b"),
    ("file_kw", r"etc/passwd|win\.ini|php://|file://|/proc/self"),
    ("pct_encoded", r"%[0-9a-f]{2}"),  # đếm trên chuỗi gốc
]
_HC_RE = [(n, re.compile(p)) for n, p in HANDCRAFTED]


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Đếm các mẫu đáng ngờ + log(độ dài). Trả về ma trận sparse."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for raw in X:
            norm = normalize(raw)
            feats = [math.log1p(len(raw))]
            for name, rx in _HC_RE:
                target = raw.lower() if name == "pct_encoded" else norm
                feats.append(len(rx.findall(target)))
            rows.append(feats)
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):
        return np.array(["log_len"] + [f"hc_{n}" for n, _ in HANDCRAFTED])


def build_features(max_features: int = 50000) -> FeatureUnion:
    char_tfidf = TfidfVectorizer(
        preprocessor=normalize,
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    handcrafted = Pipeline([("hc", HandcraftedFeatures()), ("scale", MaxAbsScaler())])
    return FeatureUnion([("char", char_tfidf), ("hc", handcrafted)])
