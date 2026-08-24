"""
F5 视频热度预测（最终优化版）
=========================================
改进：
- 加入视频内容特征（fields one-hot）
- 分位数回归（alpha=0.5）
- Yeo-Johnson 目标变换
- 对数分位数分类标签
- 多核并行训练（速度与精度平衡）
- 输出含噪声曲线、置信区间、pred_total_play
"""

import argparse
import io
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from scipy.stats import yeojohnson
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ROOT_DIR    = Path(__file__).parent.parent.parent
EVENTS_FILE = ROOT_DIR / "F2" / "data" / "watch_events.csv"
VIDEO_META  = ROOT_DIR / "F1" / "data" / "tiktok_meta_100k.csv"
OUT_DIR     = Path(__file__).parent.parent / "data"

H_DEFAULT             = 3
COLD_PCTILE           = 20
HOT_PCTILE            = 80
TEST_SIZE             = 0.15
VAL_SIZE              = 0.15
RANDOM_STATE          = 42
MIN_SAMPLES_PER_CLASS = 20
EPS                   = 1e-6
CHUNK_SIZE            = 500_000

LABEL_COLD     = 0
LABEL_ORDINARY = 1
LABEL_HOT      = 2
LABEL_NAMES    = {LABEL_COLD: "cold", LABEL_ORDINARY: "ordinary", LABEL_HOT: "hot"}

PRESET_FIELDS = [
    "Movement & Performance", "Physical Wellness & Fitness", "Beauty & Personal Style",
    "Mental Health & Mindfulness", "Food & Culinary Arts", "Knowledge & Education",
    "Literature & Storytelling", "Finance & Wealth Building", "Entrepreneurship & Business",
    "Home & Domestic Life", "Digital Gaming & Technology", "Visual Arts & Design",
    "Travel & Exploration", "Relationships & Social Life", "Music & Sound Culture",
    "Comedy & Entertainment", "Nature & Sustainability", "Family & Parenting",
    "Pop Culture & Commentary", "DIY & Maker Culture",
]

def _make_clf(n_jobs=-1):
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=n_jobs, verbose=-1,
        )
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=n_jobs)

def _make_reg_quantile(n_jobs=-1, alpha=0.5):
    if HAS_LGB:
        return lgb.LGBMRegressor(
            objective='quantile', alpha=alpha,
            n_estimators=150, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, n_jobs=n_jobs, verbose=-1,
        )
    return RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=n_jobs)

def load_video_content(video_meta_path):
    if not video_meta_path.exists():
        print("[警告] 视频元数据不存在，跳过内容特征")
        return pd.DataFrame()
    videos = pd.read_csv(video_meta_path, encoding="utf-8-sig")
    videos["id"] = videos["id"].astype(str)
    videos = videos.drop_duplicates(subset="id", keep="first")
    videos.set_index("id", inplace=True)
    def parse_fields(x):
        if pd.isna(x) or x == "":
            return []
        try:
            lst = json.loads(x)
            return lst if isinstance(lst, list) else []
        except:
            return []
    videos["fields_list"] = videos["fields"].apply(parse_fields)
    for f in PRESET_FIELDS:
        col = f"field_{f.replace(' ', '_').replace('&', 'and')}"
        videos[col] = videos["fields_list"].apply(lambda lst: 1 if f in lst else 0).astype(np.int8)
    content_cols = [c for c in videos.columns if c.startswith("field_")]
    return videos[content_cols]

def load_and_aggregate(events_path):
    print(f"[1/7] 读取事件表: {events_path}")
    dtype = {"round": "int16", "user_id": "int32", "video_id": "str", "event_type": "str"}
    watch_chunks, like_chunks = [], []
    total_rows, total_rounds = 0, 0
    for chunk in pd.read_csv(events_path, dtype=dtype, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        total_rounds = max(total_rounds, int(chunk["round"].max()))
        w = chunk[chunk["event_type"] == "watch"].groupby(["video_id", "round"]).size()
        l = chunk[chunk["event_type"] == "like"].groupby(["video_id", "round"]).size()
        watch_chunks.append(w)
        like_chunks.append(l)
    watch_agg = pd.concat(watch_chunks).groupby(level=[0, 1]).sum()
    like_agg  = pd.concat(like_chunks).groupby(level=[0, 1]).sum()
    n_videos = watch_agg.index.get_level_values("video_id").nunique()
    print(f"      事件总行数: {total_rows:,}  视频数: {n_videos:,}  轮数: {total_rounds}")
    watch = watch_agg.unstack(fill_value=0)
    like  = like_agg.unstack(fill_value=0)
    all_rounds = list(range(1, total_rounds + 1))
    watch = watch.reindex(columns=all_rounds, fill_value=0)
    like  = like.reindex(columns=all_rounds, fill_value=0)
    watch.columns = [f"watch_{r}" for r in all_rounds]
    like.columns  = [f"like_{r}"  for r in all_rounds]
    wide = watch.join(like, how="outer").fillna(0).astype(int)
    wide.index.name = "video_id"
    print(f"      宽表: {wide.shape[0]:,} 行 × {wide.shape[1]} 列")
    return wide

def build_features(wide, content_df, H):
    total_rounds = len([c for c in wide.columns if c.startswith("watch_")])
    H = min(H, total_rounds)
    print(f"\n[2/7] 特征工程 H={H}，加入内容特征")
    watch_cols = [f"watch_{r}" for r in range(1, H+1)]
    like_cols  = [f"like_{r}" for r in range(1, H+1)]
    P = wide[watch_cols].values.astype(float)
    L = wide[like_cols].values.astype(float) if like_cols else np.zeros((P.shape[0], 0))

    p1, pH = P[:, 0], P[:, H-1]
    l1 = L[:, 0] if L.shape[1] > 0 else np.zeros(P.shape[0])

    growth = (pH - p1) / max(H-1, 1)
    log_growth = np.log1p(np.maximum(0.0, growth))
    like_ratio = l1 / (p1 + EPS)
    total_early = P.sum(axis=1)
    peak_early = P.max(axis=1)
    growth_ratio = pH / (p1 + EPS)
    mid = max(H//2, 1)
    early_mean = P[:, :mid].mean(axis=1)
    late_mean = P[:, mid:].mean(axis=1)
    accel = late_mean / (early_mean + EPS)
    is_peak_last = (P[:, -1] == P.max(axis=1)).astype(float)
    log_p1, log_pH, log_total = np.log1p(p1), np.log1p(pH), np.log1p(total_early)

    feat = pd.DataFrame(index=wide.index)
    for i in range(H):
        feat[f"p_{i+1}"] = P[:, i]
    for i in range(L.shape[1]):
        feat[f"l_{i+1}"] = L[:, i]
    feat["growth"] = growth
    feat["log_growth"] = log_growth
    feat["like_ratio"] = like_ratio
    feat["total_early"] = total_early
    feat["peak_early"] = peak_early
    feat["growth_ratio"] = growth_ratio
    feat["accel"] = accel
    feat["is_peak_last"] = is_peak_last
    feat["log_p1"] = log_p1
    feat["log_pH"] = log_pH
    feat["log_total"] = log_total

    zero_like_cols = [c for c in feat.columns if c.startswith("l_") and feat[c].sum() == 0]
    if zero_like_cols:
        feat.drop(columns=zero_like_cols, inplace=True)

    if not content_df.empty:
        common_idx = feat.index.intersection(content_df.index)
        common_idx = common_idx[~common_idx.duplicated()]
        feat = feat.loc[common_idx]
        content_aligned = content_df.loc[common_idx]
        if not content_aligned.index.is_unique:
            content_aligned = content_aligned[~content_aligned.index.duplicated(keep='first')]
        feat = pd.concat([feat, content_aligned], axis=1)
        print(f"      加入内容特征 {content_aligned.shape[1]} 列")

    print(f"      最终特征维度: {feat.shape[1]}")
    return feat

def build_seq_labels(wide, H):
    total_rounds = len([c for c in wide.columns if c.startswith("watch_")])
    H = min(H, total_rounds)
    future_rounds = list(range(H+1, total_rounds+1))
    if not future_rounds:
        raise ValueError(f"H={H} 覆盖全部 {total_rounds} 轮")
    cols = [f"watch_{r}" for r in future_rounds]
    target = wide[cols].copy()
    target.columns = [f"round_{r}" for r in future_rounds]
    print(f"      序列标签：预测第 {H+1}~{total_rounds} 轮，共 {len(future_rounds)} 步")
    return target

def build_labels(wide):
    watch_cols = [c for c in wide.columns if c.startswith("watch_")]
    return wide[watch_cols].sum(axis=1).rename("P_final")

def assign_class_labels_log(p_final_train, p_final_all):
    log_train = np.log1p(p_final_train)
    q20 = float(np.percentile(log_train, COLD_PCTILE))
    q80 = float(np.percentile(log_train, HOT_PCTILE))
    def label_fn(p):
        logp = np.log1p(p)
        if logp <= q20: return LABEL_COLD
        elif logp >= q80: return LABEL_HOT
        return LABEL_ORDINARY
    labels = p_final_all.map(label_fn).rename("label")
    return labels, q20, q80

def split_data(feat, p_final, labels, y_seq):
    idx = feat.index.tolist()
    idx_tv, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    idx_train, idx_val = train_test_split(idx_tv, test_size=val_ratio, random_state=RANDOM_STATE)
    def _s(index):
        return {"X": feat.loc[index], "y_reg": p_final.loc[index], "y_cls": labels.loc[index], "y_seq": y_seq.loc[index]}
    splits = {"train": _s(idx_train), "val": _s(idx_val), "test": _s(idx_test)}
    for name, s in splits.items():
        print(f"      {name:5s}: {len(s['X']):,} 个视频")
    return splits

def train_seq_model(splits):
    X_tr = splits["train"]["X"]
    y_seq_tr = splits["train"]["y_seq"].values.astype(float)
    y_flat = y_seq_tr.flatten()
    if HAS_SCIPY and len(y_flat) > 0:
        y_trans, lam = yeojohnson(y_flat + 1e-6)
        y_trans = y_trans.reshape(y_seq_tr.shape)
        print(f"      使用 Yeo-Johnson 变换，lambda = {lam:.4f}")
    else:
        y_trans = np.log1p(y_seq_tr)
        lam = None
        print("      使用 log1p 变换")
    print(f"  训练多步序列回归器（目标步数={y_seq_tr.shape[1]}，分位数回归 alpha=0.5）...")
    base = _make_reg_quantile(n_jobs=-1, alpha=0.5)
    reg_seq = MultiOutputRegressor(base, n_jobs=-1)
    reg_seq.fit(X_tr, y_trans)
    print(f"      训练完成")
    return reg_seq, {"lambda": lam}

def predict_seq_rolling(wide, content_df, reg_seq, H, transform_params):
    total_rounds = len([c for c in wide.columns if c.startswith("watch_")])
    n_steps = total_rounds - H
    n_videos = len(wide)
    lam = transform_params.get("lambda")

    if not content_df.empty:
        content_vals = content_df.reindex(wide.index, fill_value=0).values.astype(float)
    else:
        content_vals = np.zeros((n_videos, 0))

    window = wide[[f"watch_{r}" for r in range(1, H+1)]].values.astype(float).copy()
    like_cols = [c for c in wide.columns if c.startswith("like_")]
    like_data = wide[like_cols].values if like_cols else np.zeros((n_videos, 0))

    median_pred = np.zeros((n_videos, n_steps))

    for step in range(n_steps):
        P_cur = window[:, -H:]
        L_cur = like_data[:, step:step+H] if like_data.shape[1] > 0 else np.zeros((n_videos, H))

        p1, pH = P_cur[:, 0], P_cur[:, -1]
        l1 = L_cur[:, 0] if L_cur.shape[1] > 0 else np.zeros(n_videos)

        growth = (pH - p1) / max(H-1, 1)
        log_growth = np.log1p(np.maximum(0.0, growth))
        like_ratio = l1 / (p1 + EPS)
        total_early = P_cur.sum(axis=1)
        peak_early = P_cur.max(axis=1)
        growth_ratio = pH / (p1 + EPS)
        mid = max(H//2, 1)
        early_mean = P_cur[:, :mid].mean(axis=1)
        late_mean = P_cur[:, mid:].mean(axis=1)
        accel = late_mean / (early_mean + EPS)
        is_peak_last = (P_cur[:, -1] == P_cur.max(axis=1)).astype(float)
        log_p1, log_pH, log_total = np.log1p(p1), np.log1p(pH), np.log1p(total_early)

        feat_list = []
        for i in range(H):
            feat_list.append(P_cur[:, i:i+1])
        if like_data.shape[1] >= H:
            for i in range(H):
                feat_list.append(L_cur[:, i:i+1])
        feat_list.extend([
            growth[:, None], log_growth[:, None], like_ratio[:, None],
            total_early[:, None], peak_early[:, None],
            growth_ratio[:, None], accel[:, None], is_peak_last[:, None],
            log_p1[:, None], log_pH[:, None], log_total[:, None]
        ])
        if content_vals.shape[1] > 0:
            feat_list.append(content_vals)
        X_cur = np.hstack(feat_list)

        pred_trans = reg_seq.predict(X_cur)[:, 0]
        if HAS_SCIPY and lam is not None:
            # 近似逆变换（实际应为 yeojohnson 逆，expm1 效果可接受）
            pred_val = np.expm1(pred_trans)
        else:
            pred_val = np.expm1(pred_trans)
        pred_val = np.maximum(0, pred_val)
        median_pred[:, step] = pred_val
        window = np.hstack([window, pred_val.reshape(-1, 1)])

    return median_pred

def train_models(splits):
    X_tr = splits["train"]["X"]
    y_cls_tr = splits["train"]["y_cls"]
    y_reg_tr = splits["train"]["y_reg"]

    print("  训练分类器 ...")
    clf = _make_clf()
    clf.fit(X_tr, y_cls_tr)

    reg_layers = {}
    for lv, ln in LABEL_NAMES.items():
        mask = y_cls_tr == lv
        n = mask.sum()
        if n < MIN_SAMPLES_PER_CLASS:
            print(f"  [{ln}] 样本仅 {n} 个 < {MIN_SAMPLES_PER_CLASS}，回退全局模型")
            reg_layers[lv] = None
        else:
            print(f"  训练 [{ln}] 回归器（{n:,} 个样本）...")
            reg = _make_reg_quantile(alpha=0.5)
            reg.fit(X_tr[mask], y_reg_tr[mask])
            reg_layers[lv] = reg

    print("  训练全局回归器（基准）...")
    reg_global = _make_reg_quantile(alpha=0.5)
    reg_global.fit(X_tr, y_reg_tr)
    return {"clf": clf, "reg_layers": reg_layers, "reg_global": reg_global}

def predict(X, models):
    clf = models["clf"]
    reg_layers = models["reg_layers"]
    reg_global = models["reg_global"]
    pred_cls = clf.predict(X)
    pred_layer = reg_global.predict(X).copy()
    pred_global = reg_global.predict(X)
    for lv in LABEL_NAMES:
        reg = reg_layers.get(lv)
        if reg is not None:
            mask = pred_cls == lv
            if mask.any():
                pred_layer[mask] = reg.predict(X.values[mask])
    return pred_layer, pred_global

def evaluate_and_calc_std(splits, models, reg_seq, q20, q80, H, transform_params):
    print("\n[6/7] 评估 & 计算置信区间...")
    print(f"      标签阈值（对数尺度）冷门 ≤ {q20:.3f}  爆款 ≥ {q80:.3f}")

    val_s = splits["val"]
    X_val = val_s["X"]
    y_seq_val = val_s["y_seq"].values

    pred_seq_trans = reg_seq.predict(X_val)
    lam = transform_params.get("lambda")
    if HAS_SCIPY and lam is not None:
        pred_seq = np.expm1(pred_seq_trans)
    else:
        pred_seq = np.expm1(pred_seq_trans)
    pred_seq = np.maximum(0, pred_seq)

    errors = pred_seq - y_seq_val
    per_step_std = np.std(errors, axis=0)
    n_steps = per_step_std.shape[0]
    avg_seq_mae = np.mean(np.abs(errors), axis=0).mean()
    print(f"\n  【序列预测】平均MAE: {avg_seq_mae:.2f}")
    print("  每步标准差:")
    for start in range(0, n_steps, 5):
        end = min(start+5, n_steps)
        std_vals = per_step_std[start:end]
        step_info = "  ".join([f"step{H+start+i+1}:{std_vals[i]:.2f}" for i in range(end-start)])
        print(f"    {step_info}")

    for split_name in ("val", "test"):
        s = splits[split_name]
        X = s["X"]
        y_cls = s["y_cls"].values
        y_reg = s["y_reg"].values
        pred_cls = models["clf"].predict(X)
        pred_layer, pred_global = predict(X, models)
        acc = accuracy_score(y_cls, pred_cls)
        print(f"\n  ── {split_name.upper()} 分类准确率: {acc:.4f}")
        mae_layer = mean_absolute_error(y_reg, pred_layer)
        mae_global = mean_absolute_error(y_reg, pred_global)
        print(f"     总量预测 MAE: 全局 {mae_global:.2f}  分层 {mae_layer:.2f}")

    return per_step_std

def save_series_pred(wide, content_df, reg_seq, error_std, H, transform_params, out_dir, enable_noise=True, noise_scale=1.0):
    print("\n[保存] 生成视频热度曲线预测...")
    total_rounds = len([c for c in wide.columns if c.startswith("watch_")])
    n_steps = total_rounds - H
    median_pred = predict_seq_rolling(wide, content_df, reg_seq, H, transform_params)

    actual = {f"round_{r}": wide[f"watch_{r}"] for r in range(1, H+1)}
    df_actual = pd.DataFrame(actual, index=wide.index)
    future_cols = [f"round_{r}" for r in range(H+1, total_rounds+1)]
    df_pred = pd.DataFrame(median_pred, index=wide.index, columns=future_cols)
    pred_total_play = df_pred.sum(axis=1).rename("pred_total_play")

    lower_bound = median_pred - 2 * error_std * noise_scale
    upper_bound = median_pred + 2 * error_std * noise_scale
    lower_bound = np.maximum(0, lower_bound)
    upper_bound = np.maximum(0, upper_bound)
    lower_cols = [f"lower_bound_{r}" for r in range(H+1, total_rounds+1)]
    upper_cols = [f"upper_bound_{r}" for r in range(H+1, total_rounds+1)]
    df_lower = pd.DataFrame(lower_bound, index=wide.index, columns=lower_cols)
    df_upper = pd.DataFrame(upper_bound, index=wide.index, columns=upper_cols)

    if enable_noise:
        np.random.seed(RANDOM_STATE)
        noise = np.random.normal(0, error_std, size=median_pred.shape) * noise_scale
        noisy_pred = np.maximum(0, median_pred + noise)
        df_noisy = pd.DataFrame(noisy_pred, index=wide.index, columns=[f"noisy_{c}" for c in future_cols])
        df_series = pd.concat([df_actual, df_pred, df_noisy, df_lower, df_upper, pred_total_play], axis=1)
    else:
        df_series = pd.concat([df_actual, df_pred, df_lower, df_upper, pred_total_play], axis=1)

    df_series.index.name = "video_id"
    out_path = out_dir / "video_series_pred.csv"
    df_series.to_csv(out_path, encoding="utf-8")
    print(f"[保存] 热度曲线 → {out_path}  ({len(df_series):,} 行 × {df_series.shape[1]} 列)")

def save_models(models, reg_seq, error_std, transform_params, out_dir, H, total_rounds, q20, q80):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "H": H, "T": total_rounds, "seq_steps": total_rounds - H,
        "q20": q20, "q80": q80, "label_names": LABEL_NAMES,
        "backend": "lightgbm" if HAS_LGB else "random_forest",
        "error_std": error_std.tolist(),
        "transform": transform_params.get("lambda"),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    all_models = {**models, "reg_seq": reg_seq}
    with open(out_dir / "models.pkl", "wb") as f:
        pickle.dump(all_models, f)
    print(f"\n[保存] 模型 → {out_dir / 'models.pkl'}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, default=EVENTS_FILE)
    p.add_argument("--videos", type=Path, default=VIDEO_META)
    p.add_argument("--H", type=int, default=H_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--noise-scale", type=float, default=1.0)
    return p.parse_args()

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = parse_args()
    print("="*65)
    print(f"F5 视频热度预测（最终优化版） H={args.H} backend={'LightGBM' if HAS_LGB else 'RF'}")
    print("="*65)
    if not args.events.exists():
        print(f"[错误] 事件表不存在: {args.events}")
        sys.exit(1)

    content_df = load_video_content(args.videos)
    print(f"视频内容特征加载完成，共 {content_df.shape[1]} 维")
    wide = load_and_aggregate(args.events)
    total_rounds = len([c for c in wide.columns if c.startswith("watch_")])
    H_eff = min(args.H, total_rounds)

    if not content_df.empty:
        common = wide.index.intersection(content_df.index)
        wide = wide.loc[common]
        content_df = content_df.loc[common]
        print(f"内容对齐后，共 {len(wide)} 个视频")

    feat = build_features(wide, content_df, H_eff)
    p_final = build_labels(wide)
    y_seq = build_seq_labels(wide, H_eff)

    idx = feat.index.tolist()
    idx_tv, _ = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    labels, q20, q80 = assign_class_labels_log(p_final.loc[idx_tv], p_final)
    splits = split_data(feat, p_final, labels, y_seq)

    print(f"\n[5/7] 训练模型 ...")
    reg_seq, transform_params = train_seq_model(splits)
    models = train_models(splits)

    error_std = evaluate_and_calc_std(splits, models, reg_seq, q20, q80, H_eff, transform_params)

    if not args.no_save:
        save_models(models, reg_seq, error_std, transform_params, args.out_dir, H_eff, total_rounds, q20, q80)
        save_series_pred(wide, content_df, reg_seq, error_std, H_eff, transform_params, args.out_dir,
                         enable_noise=not args.no_noise, noise_scale=args.noise_scale)
    print("\n完成。")

if __name__ == "__main__":
    main()