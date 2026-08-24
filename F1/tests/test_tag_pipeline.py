"""
用 result_10_from_1000.csv 测试 tag_pipeline 是否能正常运行，并打印每行的标签结果。
同时包含一组手工构造的内联用例，专门覆盖 challenges 字段的处理路径。
"""

import csv
import io
import sys
from pathlib import Path

# 强制 stdout 使用 UTF-8，避免 Windows 终端 GBK 编码报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 把 scripts/ 加入模块搜索路径，使 import tag_pipeline 生效
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tag_pipeline import run_pipeline  # noqa: E402

CSV_FILE = Path(__file__).parent / "result_10_from_1000.csv"
CSV_WITH_CHALLENGES = Path(__file__).parent / "result_with_challenges.csv"

# ── 内联用例：专门覆盖 challenges 字段路径 ────────────────────────────────────
# result_10_from_1000.csv 中没有 challenges 列，需要用这组数据验证：
#   - fyp / fypppppp 等变体是否被黑名单过滤
#   - 有效 hashtag 是否正确进入偏好标签
#   - 偏好标签是否能触发对应兴趣领域
INLINE_ROWS = [
    {
        "id": "inline_001_fitness",
        "desc": "My morning workout routine 💪 #gym #fitness #workout #fyp #morningroutine",
        "challenges": '[{"title": "gym"}, {"title": "fitness"}, {"title": "workout"}, '
                      '{"title": "fyp"}, {"title": "morningroutine"}]',
        "poi_category": "",
        "city": "",
        # 预期：fyp 被过滤；偏好标签含 gym/fitness/workout/morningroutine
        #        兴趣领域含 Physical Wellness & Fitness
    },
    {
        "id": "inline_002_food_poi",
        "desc": "Easy pasta recipe 🍝 #cooking #recipe #pasta #foodtok #fypppppp",
        "challenges": '[{"title": "cooking"}, {"title": "recipe"}, {"title": "pasta"}, '
                      '{"title": "foodtok"}, {"title": "fypppppp"}]',
        "poi_category": "Food and Drink",
        "city": "New York",
        # 预期：fypppppp 被正则过滤；poi 映射到 Food & Culinary Arts
        #        偏好标签触发 Food & Culinary Arts（与 POI 来源相同，不重复添加）
    },
    {
        "id": "inline_003_substring",
        "desc": "cats are amazing #catsoftiktok #cat #cats #catlovers #fyp",
        "challenges": '[{"title": "catsoftiktok"}, {"title": "cat"}, '
                      '{"title": "cats"}, {"title": "catlovers"}, {"title": "fyp"}]',
        "poi_category": "",
        "city": "",
        # 专门验证子串归约：cat 是 cats/catsoftiktok/catlovers 的前缀
        # 预期：只保留 "cat"（最短前缀），其余三个被归约删除
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def _print_row(i: int, row: dict, label: str = "") -> None:
    result = run_pipeline(row)
    pref = result["preference_tags"]
    fields = result["fields"]
    header = f"【{label or f'第 {i} 行'}】id={row.get('id', 'N/A')}"
    print(header)
    print(f"  城市/POI : {row.get('city', '')!r}  |  poi_category={row.get('poi_category', '')!r}")
    desc_preview = (row.get("desc") or "")[:80]
    print(f"  desc     : {desc_preview}...")
    if row.get("challenges"):
        print(f"  challenges 字段 : 已提供")
    print(f"  偏好标签 ({len(pref)}) : {pref}")
    print(f"  兴趣领域 ({len(fields)}) : {fields}")
    print("-" * 72)


def main():
    # ── 一、旧 CSV（无 challenges，走 YAKE/TextRank 路径）──
    with CSV_FILE.open(encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))

    print(f"▶ CSV 数据（{len(csv_rows)} 行，无 challenges，走 YAKE/TextRank 路径）")
    print("=" * 72)
    for i, row in enumerate(csv_rows, 1):
        _print_row(i, row)

    # ── 二、含 challenges 的新 CSV（真实 hashtag 提取路径）──
    if CSV_WITH_CHALLENGES.exists():
        with CSV_WITH_CHALLENGES.open(encoding="utf-8-sig") as f:
            ch_rows = list(csv.DictReader(f))
        print(f"\n▶ 含 challenges 的新 CSV（{len(ch_rows)} 行，验证真实 hashtag 提取）")
        print("=" * 72)
        for i, row in enumerate(ch_rows, 1):
            _print_row(i, row, label=f"新样本 {i}")
    else:
        print(f"\n⚠ 未找到 {CSV_WITH_CHALLENGES.name}，跳过该部分")

    # ── 三、内联用例（含 challenges，验证 hashtag 提取路径）──
    print(f"\n▶ 内联用例（{len(INLINE_ROWS)} 行，含 challenges，验证 hashtag 提取与子串归约）")
    print("=" * 72)
    for i, row in enumerate(INLINE_ROWS, 1):
        _print_row(i, row, label=f"内联 {i}")


if __name__ == "__main__":
    main()
