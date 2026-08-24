"""
从 Hugging Face 的 TikTok-10M 数据集流式读取，随机抽取 N 条，
对每条运行标签流水线（tag_pipeline），最终保存为 CSV。

用法（在 F1/ 目录下执行）：
  python scripts/build_video_db.py

输出文件：F1/data/tiktok_meta_100k.csv
  列：id · desc · challenges · duration · digg_count · comment_count
      play_count · share_count · collect_count · music_id · user_id
      poi_name · city · poi_category · preference_tags · fields
"""

import csv
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from langdetect import detect, LangDetectException


# 把 scripts/ 加入搜索路径，使 import tag_pipeline 生效
sys.path.insert(0, str(Path(__file__).parent))
from tag_pipeline import run_pipeline  # noqa: E402

# ── 配置区，按需修改 ──────────────────────────────────────────────────────────
DATASET_NAME = "shrx240/TikTok-10M"
LIMIT = 100_000               # 最终保存的条数
POOL = 1_000_000             # 最多扫描多少条（越大越随机，但越慢）
SEED = 42                    # 随机种子
OUT_FILE = Path(__file__).parent.parent / "data" / "tiktok_meta_100k.csv"
# ─────────────────────────────────────────────────────────────────────────────

# 原始数据列
RAW_COLUMNS = [
    "id",
    "desc",
    "challenges",
    "duration",
    "digg_count",
    "comment_count",
    "play_count",
    "share_count",
    "collect_count",
    "music_id",
    "user_id",
    "poi_name",
    "city",
    "poi_category",
]

# 最终写入 CSV 的列（原始列 + 标签流水线输出的两列）
OUTPUT_COLUMNS = RAW_COLUMNS + ["preference_tags", "fields"]


# 允许通过的语言：英语和西班牙语
# 其他语言（日语、韩语、俄语、阿拉伯语等）的视频会被跳过，
# 以避免 YAKE/TextRank 对非英语正文提取出无意义标签。
ALLOWED_LANGS = {"en", "es"}


def is_acceptable_language(desc: str) -> bool:
    """
    检测 desc 的主体语言是否在 ALLOWED_LANGS 中。

    策略：先剥去所有 #话题 和 @提及，只对正文部分做语言检测。
    这样能避免把"英语正文 + 日语 hashtag"的混合视频误判为非英语。
    正文不足 10 个字符时无法可靠判断，默认保留（返回 True）。
    """
    clean = re.sub(r"[#@]\S+", "", desc or "").strip()
    if len(clean) < 10:
        return True   # 正文太短，无法判断，保守地保留
    try:
        return detect(clean) in ALLOWED_LANGS
    except LangDetectException:
        return True   # 检测异常时保留，不轻易丢弃


def normalize_row(row: dict) -> dict:
    """
    将原始数据行转换为可写入 CSV 的字典：
      1. 列表/字典类型 → JSON 字符串
      2. 调用标签流水线，追加 preference_tags 和 fields 两列
    """
    result = {}
    for col in RAW_COLUMNS:
        value = row.get(col)

        # ⭐关键：保证 id 永远是字符串
        if col == "id":
            result[col] = str(value)
        else:
            result[col] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value

    tags = run_pipeline(row)
    result["preference_tags"] = json.dumps(
        tags["preference_tags"], ensure_ascii=False)
    result["fields"] = json.dumps(tags["fields"],          ensure_ascii=False)
    return result


def reservoir_sample(dataset, k: int, max_scan: int, rng: random.Random) -> list:
    """
    蓄水池抽样：从流式数据中均匀随机抽取 k 条，不需要提前知道总数据量。

    算法保证：每一条有效数据（通过语言过滤）被选中的概率完全相等。
      - 前 k 条有效行：直接放入水池
      - 第 n 条有效行（n > k）：以 k/n 的概率决定是否替换水池中的随机一条

    注意：概率计算基于 valid_count（有效行序号），而非 i（原始行序号），
    这样被过滤掉的非英语行不会干扰抽样均匀性。
    """
    reservoir = []
    valid_count = 0   # 只计通过语言过滤的行数，用于蓄水池概率计算

    for i, row in enumerate(dataset):
        if i >= max_scan:
            break

        # 语言过滤：在运行标签流水线之前完成，避免对非目标语言做无效计算
        if not is_acceptable_language(row.get("desc") or ""):
            continue

        norm = normalize_row(row)
        valid_count += 1

        if len(reservoir) < k:
            reservoir.append(norm)
        else:
            j = rng.randint(0, valid_count - 1)   # 基于有效行数，保证均匀性
            if j < k:
                reservoir[j] = norm

        if (i + 1) % 10_000 == 0:
            print(
                f"  已扫描 {i + 1:,} 条（有效 {valid_count:,} 条），水池当前 {len(reservoir):,} 条...")

    return reservoir


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"正在连接数据集：{DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    rng = random.Random(SEED)

    print(f"开始蓄水池抽样：目标 {LIMIT:,} 条，最多扫描 {POOL:,} 条，随机种子 {SEED}")
    print("（每行均会运行标签流水线，速度略慢于纯抽样，属正常现象）\n")

    rows = reservoir_sample(dataset, k=LIMIT, max_scan=POOL, rng=rng)
    rng.shuffle(rows)

    with OUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n完成！共写入 {len(rows):,} 条")
    print(f"文件保存至：{OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
