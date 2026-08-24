"""
小范例测试：从 TikTok-10M 流中扫描前 1000 条，随机抽取 10 条，保存为 CSV。
用于验证运行速度和抽样结果是否正常。
"""

import csv
import json
import random
import time
from pathlib import Path

from datasets import load_dataset

# ── 测试用配置（小批量）──────────────────────
DATASET_NAME = "shrx240/TikTok-10M"
LIMIT        = 10       # 最终抽取条数
POOL         = 1_000    # 最多扫描条数
SEED         = 42
OUT_FILE     = Path(__file__).parent / "result_10_from_1000.csv"
# ────────────────────────────────────────────

COLUMNS = [
    "id", "desc", "duration",
    "digg_count", "comment_count", "play_count",
    "city", "poi_category",
]


def normalize_row(row: dict) -> dict:
    result = {}
    for col in COLUMNS:
        value = row.get(col)
        if isinstance(value, (list, dict)):
            result[col] = json.dumps(value, ensure_ascii=False)
        else:
            result[col] = value
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
    print(f"参数：扫描前 {POOL} 条，随机抽取 {LIMIT} 条，种子={SEED}\n")

    t0 = time.time()

    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    rng = random.Random(SEED)
    rows = reservoir_sample(dataset, k=LIMIT, max_scan=POOL, rng=rng)

    elapsed = time.time() - t0
    print(f"\n抽样完成，耗时 {elapsed:.1f} 秒")

    # 写入 CSV
    with OUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已保存至：{OUT_FILE}\n")

    # 在终端打印结果预览
    print("═" * 60)
    print(f"{'序号':<4}  {'id':<20}  {'城市':<10}  {'播放数':<10}  {'时长(s)'}")
    print("─" * 60)
    for idx, r in enumerate(rows, 1):
        print(f"{idx:<4}  {str(r.get('id','')):<20}  "
              f"{str(r.get('city','')):<10}  "
              f"{str(r.get('play_count','')):<10}  "
              f"{r.get('duration','')}")
    print("═" * 60)


if __name__ == "__main__":
    main()
