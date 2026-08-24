"""
F1 初始视频热度量化脚本
=======================
读取 F1/data/tiktok_meta_100k.csv，按以下规则计算每条视频的初始热度：

    ① H_behavior = 0.35·log(1+play) + 0.20·log(1+like)
                 + 0.15·log(1+comment) + 0.15·log(1+share)
                 + 0.15·log(1+collect)

    ② Q（内容质量，desc 与 challenges 均有效时才参与合并）
         duration_score  = exp(-(duration - d0)² / (2σ²))   d0=30, σ=20
         desc_score      = len(desc) / max_desc_len
         challenge_score = n_challenges / max_challenges
         poi_score       = 1 if poi_name 非空 else 0
         Q = 0.5·duration_score + 0.3·desc_score
           + 0.2·challenge_score + 0.1·poi_score

    ③ R（随机扰动，由 video_id hash 决定，可复现）
         R ∈ [0, 1)

    ④ H0（合并，Q 有效 = desc 去空白后非空 且 challenges 列表非空）
         H0 = 0.85·H_behavior + 0.10·Q + 0.05·R   (Q 有效)
         H0 = 0.95·H_behavior + 0.05·R             (Q 缺失)

    ⑤ 归一化
         H0_norm = (H0 - min) / (max - min + ε)

输出文件:
    F1/data/video_popularity_init.csv
    列: id, H_behavior, Q, R, H0, H0_norm

运行方式:
    python scripts/compute_initial_popularity.py
    python scripts/compute_initial_popularity.py --input custom.csv --output out.csv
"""

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────────────────────────────
D0     = 30.0    # 最佳视频时长（秒）
SIGMA  = 20.0    # 时长评分的标准差
EPS    = 1e-9    # 避免 min-max 归一化除零

DEFAULT_INPUT  = Path(__file__).parent.parent / "data" / "tiktok_meta_100k.csv"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "video_popularity_init.csv"
# ─────────────────────────────────────────────────────────────────────────────


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _safe_float(val: str, default: float = 0.0) -> float:
    """安全转 float，失败时返回 default。"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_json_list(val: str) -> list:
    """安全解析 JSON 数组字符串，失败时返回空列表。"""
    if not val:
        return []
    try:
        result = json.loads(val)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _deterministic_rand(video_id: str) -> float:
    """
    由 video_id 推导确定性随机扰动 R ∈ [0, 1)。
    同一 video_id 每次运行结果一致；不依赖全局随机种子。
    用 hash 折叠到 [0, 1) 区间。
    """
    h = hash(str(video_id)) & 0xFFFFFFFF   # 取低 32 位，恒为正
    return h / 0x100000000                 # 除以 2^32，映射到 [0, 1)


# --------------------------------------------------------------------------- #
# 第一遍：计算全库统计量（max_desc_len, max_challenges）
# --------------------------------------------------------------------------- #
def compute_global_stats(path: Path) -> tuple[float, float]:
    """
    遍历全库，得到：
        max_desc_len   - 所有 desc 的最大字符长度
        max_challenges - 所有视频 challenges 列表长度的最大值
    用于 desc_score 和 challenge_score 的归一化分母。
    """
    max_desc_len   = 1.0   # 至少 1，避免分母为 0
    max_challenges = 1.0

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc_len = len((row.get("desc") or "").strip())
            n_ch     = len(_parse_json_list(row.get("challenges", "")))
            if desc_len   > max_desc_len:   max_desc_len   = desc_len
            if n_ch       > max_challenges: max_challenges = n_ch

    return float(max_desc_len), float(max_challenges)


# --------------------------------------------------------------------------- #
# 逐行计算
# --------------------------------------------------------------------------- #
def compute_row(row: dict, max_desc_len: float, max_challenges: float) -> dict:
    """
    对单行计算 H_behavior, Q（可能 None）, R, H0, 但 H0_norm 在第二遍算。
    返回中间结果 dict（id, H_behavior, Q_valid, Q, R, H0_raw）。
    """
    vid = row["id"]

    # ── ① H_behavior ────────────────────────────────────────────────────── #
    play    = _safe_float(row.get("play_count"))
    like    = _safe_float(row.get("digg_count"))
    comment = _safe_float(row.get("comment_count"))
    share   = _safe_float(row.get("share_count"))
    collect = _safe_float(row.get("collect_count"))

    H_beh = (
        0.35 * math.log1p(play)
      + 0.20 * math.log1p(like)
      + 0.15 * math.log1p(comment)
      + 0.15 * math.log1p(share)
      + 0.15 * math.log1p(collect)
    )

    # ── ② Q（内容质量）─────────────────────────────────────────────────── #
    desc_raw     = (row.get("desc") or "").strip()
    challenges   = _parse_json_list(row.get("challenges", ""))
    q_valid      = bool(desc_raw) and len(challenges) > 0

    duration     = _safe_float(row.get("duration"), default=0.0)
    duration_score  = math.exp(-((duration - D0) ** 2) / (2 * SIGMA ** 2))

    desc_score      = len(desc_raw) / max_desc_len
    challenge_score = len(challenges) / max_challenges
    poi_score       = 1.0 if (row.get("poi_name") or "").strip() else 0.0

    Q = (
        0.5 * duration_score
      + 0.3 * desc_score
      + 0.2 * challenge_score
      + 0.1 * poi_score
    )

    # ── ③ R ─────────────────────────────────────────────────────────────── #
    R = _deterministic_rand(vid)

    # ── ④ H0（合并）────────────────────────────────────────────────────── #
    if q_valid:
        H0_raw = 0.85 * H_beh + 0.10 * Q + 0.05 * R
    else:
        H0_raw = 0.95 * H_beh + 0.05 * R

    return {
        "id":         vid,
        "H_behavior": H_beh,
        "Q_valid":    q_valid,
        "Q":          Q if q_valid else None,
        "R":          R,
        "H0_raw":     H0_raw,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_popularity(input_path: Path, output_path: Path) -> None:
    print(f"[1/3] 扫描全库统计量: {input_path}")
    max_desc_len, max_challenges = compute_global_stats(input_path)
    print(f"      max_desc_len={max_desc_len:.0f}, max_challenges={max_challenges:.0f}")

    print("[2/3] 逐行计算热度分 ...")
    rows_out: list[dict] = []

    with open(input_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            result = compute_row(row, max_desc_len, max_challenges)
            rows_out.append(result)
            if i % 10_000 == 0:
                print(f"      已处理 {i:,} 行 ...")

    print(f"      共处理 {len(rows_out):,} 行")

    # ── ⑤ 归一化 ────────────────────────────────────────────────────────── #
    print("[3/3] 归一化并写入输出文件 ...")
    h0_values = [r["H0_raw"] for r in rows_out]
    h0_min    = min(h0_values)
    h0_max    = max(h0_values)
    h0_range  = h0_max - h0_min + EPS

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "H_behavior", "Q_valid", "Q", "R", "H0", "H0_norm"],
        )
        writer.writeheader()
        for r in rows_out:
            writer.writerow({
                "id":         r["id"],
                "H_behavior": round(r["H_behavior"], 6),
                "Q_valid":    r["Q_valid"],
                "Q":          round(r["Q"], 6) if r["Q"] is not None else "",
                "R":          round(r["R"], 6),
                "H0":         round(r["H0_raw"], 6),
                "H0_norm":    round((r["H0_raw"] - h0_min) / h0_range, 6),
            })

    print(f"\n完成。输出: {output_path}")
    print(f"  H0 范围: [{h0_min:.4f}, {h0_max:.4f}]  →  H0_norm ∈ [0, 1]")
    q_count = sum(1 for r in rows_out if r["Q_valid"])
    print(f"  Q 有效比例: {q_count:,} / {len(rows_out):,} ({q_count/len(rows_out)*100:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="初始视频热度量化")
    parser.add_argument(
        "--input",  type=Path, default=DEFAULT_INPUT,
        help="F1 视频库 CSV（默认: F1/data/tiktok_meta_100k.csv）",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="输出文件路径（默认: F1/data/video_popularity_init.csv）",
    )
    return parser.parse_args()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()

    if not args.input.exists():
        print(f"[错误] 输入文件不存在: {args.input}")
        print("       请先运行 build_video_db.py 生成视频库。")
        sys.exit(1)

    build_popularity(args.input, args.output)


if __name__ == "__main__":
    main()
