"""端到端验证：normalize_row 接入 tag_pipeline 后能正常处理真实数据。"""
import csv
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_video_db import normalize_row, OUTPUT_COLUMNS  # noqa: E402

CSV = Path(__file__).parent / "result_with_challenges.csv"

rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
print(f"读取 {len(rows)} 行，开始端到端验证 normalize_row ...\n")
print("输出列：", OUTPUT_COLUMNS, "\n")

for i, row in enumerate(rows, 1):
    out = normalize_row(row)
    pt = out["preference_tags"][:70]
    fi = out["fields"]
    print(f"行 {i:>2} | preference_tags={pt}...")
    print(f"       | fields={fi}")
    print()

print("端到端验证完成 — normalize_row 含 tag_pipeline 接入正常")
