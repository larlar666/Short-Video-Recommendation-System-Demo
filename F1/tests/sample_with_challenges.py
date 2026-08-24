"""
从 TikTok-10M 流中扫描前 1000 条，随机抽取 10 条。
在原有列基础上增加 challenges 和 poi_category，用于测试 tag_pipeline。
"""

import csv
import io
import json
import random
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from datasets import load_dataset

DATASET_NAME = "shrx240/TikTok-10M"
LIMIT        = 10
POOL         = 1_000
SEED         = 99          # 换一个种子，得到与之前不同的样本
OUT_FILE     = Path(__file__).parent / "result_with_challenges.csv"

# 新增 challenges 和 poi 相关字段
COLUMNS = [
    "id", "desc", "challenges",
    "duration", "digg_count", "play_count",
    "city", "poi_category",
]


def normalize_row(row: dict) -> dict:
    result = {}
    for col in COLUMNS:
        value = row.get(col)
        result[col] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
    return result


def reservoir_sample(dataset, k, max_scan, rng):
    reservoir = []
    for i, row in enumerate(dataset):
        if i >= max_scan:
            break
        norm = normalize_row(row)
        if len(reservoir) < k:
            reservoir.append(norm)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = norm
    return reservoir


def main():
    print(f"连接数据集：{DATASET_NAME}")
    print(f"扫描前 {POOL} 条，随机抽取 {LIMIT} 条，种子={SEED}\n")

    t0 = time.time()
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    rng = random.Random(SEED)
    rows = reservoir_sample(dataset, k=LIMIT, max_scan=POOL, rng=rng)
    print(f"抽样完成，耗时 {time.time() - t0:.1f} 秒")

    with OUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"已保存：{OUT_FILE}\n")

    # 简要预览 challenges 字段是否有数据
    print("═" * 70)
    for idx, r in enumerate(rows, 1):
        raw_ch = r.get("challenges", "") or ""
        try:
            ch_list = json.loads(raw_ch) if raw_ch else []
            ch_titles = [c.get("title", "") if isinstance(c, dict) else str(c)
                         for c in ch_list]
        except Exception:
            ch_titles = []
        print(f"{idx:>2}. id={r.get('id','')}  city={r.get('city','')!r}")
        print(f"    desc前60字: {(r.get('desc') or '')[:60]}")
        print(f"    challenges: {ch_titles}")
        print()
    print("═" * 70)


if __name__ == "__main__":
    main()
