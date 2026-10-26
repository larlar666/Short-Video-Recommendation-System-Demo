import ast
import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from database.config import backend as data_backend
from database.repository import load_catalog, user_watch_events, video_watch_counts
from database.errors import error_summary, error_code

logger = logging.getLogger(__name__)


def show_data_error(stage, error):
    logger.error("%s (%s)", stage, type(error).__name__, exc_info=True)
    st.error(f"{stage}：{error_summary(error)}")
    with st.expander("诊断信息"):
        st.code(error_code(error))


BASE_DIR = Path(__file__).resolve().parent

F1_VIDEO = BASE_DIR / "F1" / "data" / "tiktok_meta_100k.csv"
F2_USERS = BASE_DIR / "F2" / "data" / "users_sim.csv"
F2_EVENTS = BASE_DIR / "F2" / "data" / "watch_events.csv"
F5_SERIES = BASE_DIR / "F5" / "data" / "video_series_pred.csv"
F6_CLUSTERS = BASE_DIR / "F6" / "data" / "video_clusters.csv"
F7_CLUSTERS = BASE_DIR / "F7" / "data" / "user_clusters.csv"

# 与 F2/scripts/simulate_users.py 中 ALL_FIELDS 一致（共 20 个内容领域）
PRESET_FIELDS = [
    "Movement & Performance",
    "Physical Wellness & Fitness",
    "Beauty & Personal Style",
    "Mental Health & Mindfulness",
    "Food & Culinary Arts",
    "Knowledge & Education",
    "Literature & Storytelling",
    "Finance & Wealth Building",
    "Entrepreneurship & Business",
    "Home & Domestic Life",
    "Digital Gaming & Technology",
    "Visual Arts & Design",
    "Travel & Exploration",
    "Relationships & Social Life",
    "Music & Sound Culture",
    "Comedy & Entertainment",
    "Nature & Sustainability",
    "Family & Parenting",
    "Pop Culture & Commentary",
    "DIY & Maker Culture",
]


def render_qdrant_setup_help(*, need_video_index: bool) -> None:
    """Qdrant 连接失败（如 502）时给出与本仓库 README 一致的启动与建库步骤。"""
    with st.expander("如何启动 Qdrant 并导入向量数据", expanded=False):
        st.code(
            "docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant",
            language="bash",
        )
        lines = [
            "**Windows**：先启动 **Docker Desktop**，等待托盘图标显示 “Engine running”，再执行上方命令；若报错 `dockerDesktopLinuxEngine` / `cannot find the file specified`，即 Docker 引擎未启动或未安装。",
            "浏览器访问 http://localhost:6333/dashboard 应能打开控制台；若仍为 502，说明该端口上不是健康的 Qdrant 或被代理拦截。",
            "可在上方 URL 中尝试改为 `http://127.0.0.1:6333`，并确认本机 Docker/防火墙未占用或拦截 6333。",
            "导入 users 集合：在项目根目录执行 `python F3/scripts/build_user_index.py`",
        ]
        if need_video_index:
            lines.append("导入 videos 集合：执行 `python F4/scripts/build_video_index.py`")
        st.markdown("\n\n".join(f"- {t}" for t in lines))


def safe_import_qdrant():
    try:
        from qdrant_client import QdrantClient  # noqa: PLC0415
        from qdrant_client.models import FieldCondition, Filter, HasIdCondition, MatchValue  # noqa: PLC0415

        return QdrantClient, FieldCondition, Filter, HasIdCondition, MatchValue
    except Exception:
        return None, None, None, None, None


def safe_import_sklearn():
    try:
        from sklearn.decomposition import PCA  # noqa: PLC0415
        from sklearn.manifold import TSNE  # noqa: PLC0415

        return PCA, TSNE
    except Exception:
        return None, None


import re as _re

_TAG_CLEAN_RE = _re.compile(r"^[A-Za-z0-9 ]+$")


def _is_valid_tag(tag: str) -> bool:
    """只保留由字母、数字、空格构成的 tag；含标点（如 n't, it's, ...）视为无效。"""
    return bool(tag) and _TAG_CLEAN_RE.match(tag.strip())


def _filter_tags(tags: list) -> list:
    """对 tag 列表做清洗：去标点、去空串。"""
    return [t.strip() for t in tags if _is_valid_tag(t)]


def parse_list_cell(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


@st.cache_data(ttl=300)
def load_data(source):
    if source == "supabase":
        data = load_catalog()
        if "videos" in data:
            videos = data["videos"]
            videos["preference_tags_list"] = videos["preference_tags"].apply(parse_list_cell).apply(_filter_tags)
            videos["fields_list"] = videos["fields"].apply(parse_list_cell)
        if "users" in data:
            users = data["users"]
            users["interest_fields_list"] = users["interest_fields"].apply(parse_list_cell)
            users["interest_tags_list"] = users["interest_tags"].apply(parse_list_cell).apply(_filter_tags)
        return data
    data = {}

    if F1_VIDEO.exists():
        videos = pd.read_csv(F1_VIDEO)
        videos["preference_tags_list"] = videos["preference_tags"].apply(parse_list_cell).apply(_filter_tags)
        videos["fields_list"] = videos["fields"].apply(parse_list_cell)
        videos["id"] = videos["id"].astype(str)
        data["videos"] = videos

    if F2_USERS.exists():
        users = pd.read_csv(F2_USERS)
        users["interest_fields_list"] = users["interest_fields"].apply(parse_list_cell)
        users["interest_tags_list"] = users["interest_tags"].apply(parse_list_cell).apply(_filter_tags)
        users["user_id"] = users["user_id"].astype(int)
        data["users"] = users

    if F2_EVENTS.exists():
        _EVT_COLS = ["round", "user_id", "video_id", "event_type"]
        _EVT_DTYPE = {"round": "int32", "user_id": "int32", "video_id": "str", "event_type": "category"}
        _parquet = F2_EVENTS.with_suffix(".parquet")
        if _parquet.exists():
            events = pd.read_parquet(_parquet, columns=_EVT_COLS)
        else:
            _chunks = [
                chunk[_EVT_COLS]
                for chunk in pd.read_csv(
                    F2_EVENTS, dtype=_EVT_DTYPE, usecols=_EVT_COLS, chunksize=500_000
                )
            ]
            events = pd.concat(_chunks, ignore_index=True)
            del _chunks
            events.to_parquet(_parquet, index=False)  # 转换并保存，下次直接读
        data["events"] = events

    if F5_SERIES.exists():
        series = pd.read_csv(F5_SERIES)
        series["video_id"] = series["video_id"].astype(str)
        data["series"] = series

    if F6_CLUSTERS.exists():
        vc = pd.read_csv(F6_CLUSTERS)
        vc["video_id"] = vc["video_id"].astype(str)
        data["video_clusters"] = vc

    if F7_CLUSTERS.exists():
        uc = pd.read_csv(F7_CLUSTERS)
        uc["user_id"] = uc["user_id"].astype(int)
        data["user_clusters"] = uc

    return data


def ensure_data(data, keys):
    miss = [k for k in keys if k not in data]
    if miss:
        st.warning(f"缺少数据：{', '.join(miss)}。数据库模式请运行导入脚本，CSV 模式请生成对应文件。")
        return False
    return True


def top_counter(series_like, top_n=10):
    s = pd.Series(series_like).dropna()
    if s.empty:
        return pd.DataFrame(columns=["name", "count"])
    out = s.value_counts().head(top_n).rename_axis("name").reset_index(name="count")
    return out


def bar_chart_all_preset_fields(counts: pd.Series, title: str = "field"):
    """在 20 个预设 field 上对齐统计，缺失记为 0，保证柱状图横轴完整。"""
    full = pd.Series(0, index=PRESET_FIELDS, dtype=float)
    for f in PRESET_FIELDS:
        if f in counts.index:
            full[f] = float(counts[f])
    st.bar_chart(full.rename(title))


def user_row_field_set(row) -> set[str]:
    """单用户：primary_field 与 interest_fields 并集（用于 F3 多 field 统计）。"""
    s: set[str] = set()
    pf = row.get("primary_field")
    if pd.notna(pf) and str(pf).strip():
        s.add(str(pf).strip())
    ifis = row.get("interest_fields")
    if isinstance(ifis, str):
        ifis = parse_list_cell(ifis)
    elif not isinstance(ifis, list):
        ifis = []
    for f in ifis:
        if f and str(f).strip():
            s.add(str(f).strip())
    return s


def f1_panel(videos: pd.DataFrame):
    st.subheader("F1 视频库查看与筛选")
    st.caption("支持按视频 ID、Field 筛选（兴趣 tag 数量过多，已关闭 tag 筛选）。")

    c1, c2 = st.columns(2)
    with c1:
        keyword_id = st.text_input("视频 ID（支持包含匹配）", "")
    with c2:
        all_fields = sorted({f for lst in videos["fields_list"] for f in lst})
        selected_fields = st.multiselect("按 field 筛选", all_fields)

    mask = pd.Series(True, index=videos.index)
    if keyword_id:
        mask &= videos["id"].str.contains(keyword_id.strip(), na=False)
    if selected_fields:
        mask &= videos["fields_list"].apply(lambda x: any(f in x for f in selected_fields))

    filtered = videos[mask].copy()
    st.metric("筛选后视频数", f"{len(filtered):,}")
    st.dataframe(
        filtered[["id", "desc", "play_count", "digg_count", "preference_tags", "fields"]].head(200),
        use_container_width=True,
    )

    if not filtered.empty:
        exploded = filtered["fields_list"].explode().dropna()
        vc = exploded[exploded.isin(PRESET_FIELDS)].value_counts()
        bar_chart_all_preset_fields(vc, title="count")


def f2_panel(users: pd.DataFrame):
    st.subheader("F2 用户库查看与筛选")
    st.caption("支持按用户 ID、主兴趣领域筛选（兴趣 tag 已关闭）。")

    c1, c2 = st.columns(2)
    with c1:
        uid_text = st.text_input("用户 ID（支持精确）", "")
    with c2:
        fields = sorted(users["primary_field"].dropna().unique().tolist())
        sf = st.multiselect("主兴趣领域", fields)

    mask = pd.Series(True, index=users.index)
    if uid_text.strip().isdigit():
        mask &= users["user_id"] == int(uid_text.strip())
    if sf:
        mask &= users["primary_field"].isin(sf)

    filtered = users[mask].copy()
    st.metric("筛选后用户数", f"{len(filtered):,}")
    st.dataframe(
        filtered[
            ["user_id", "primary_field", "gamma", "interest_fields", "interest_tags", "total_watched", "total_liked"]
        ].head(200),
        use_container_width=True,
    )

    if not filtered.empty:
        vc = filtered["primary_field"][filtered["primary_field"].isin(PRESET_FIELDS)].value_counts()
        bar_chart_all_preset_fields(vc, title="count")


def build_user_vectors(users: pd.DataFrame):
    vocab = sorted(
        set(users["primary_field"].dropna().tolist())
        | {x for row in users["interest_fields_list"] for x in row}
        | {x for row in users["interest_tags_list"] for x in row}
    )
    idx = {k: i for i, k in enumerate(vocab)}
    mat = np.zeros((len(users), len(vocab)), dtype=float)
    for i, row in users.iterrows():
        if pd.notna(row["primary_field"]) and row["primary_field"] in idx:
            mat[i, idx[row["primary_field"]]] += 3.0
        for f in row["interest_fields_list"]:
            if f in idx:
                mat[i, idx[f]] += 2.0
        for t in row["interest_tags_list"]:
            if t in idx:
                mat[i, idx[t]] += 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms


def get_qdrant_client(qdrant_url: str):
    QdrantClient, _, _, _, _ = safe_import_qdrant()
    if QdrantClient is None:
        return None
    client = QdrantClient(url=qdrant_url, timeout=5)
    try:
        client.get_collections()
    except Exception as e:
        raise RuntimeError(
            f"无法连接 Qdrant（{qdrant_url}）：{e}\n"
            "请确认 Qdrant 服务已启动并监听在该地址。"
        ) from e
    return client


def qdrant_find_similar_users(qdrant_url: str, user_id: int, top_k: int):
    QdrantClient, _, Filter, HasIdCondition, _ = safe_import_qdrant()
    if QdrantClient is None:
        raise RuntimeError("未安装 qdrant-client，请安装后再使用向量检索。")

    client = get_qdrant_client(qdrant_url)
    if client is None or (not client.collection_exists("users")):
        raise RuntimeError("Qdrant 的 users 集合不存在，请先运行 F3 构建索引脚本。")

    points = client.retrieve(
        collection_name="users",
        ids=[int(user_id)],
        with_vectors=True,
        with_payload=True,
    )
    if not points:
        raise RuntimeError(f"user_id={user_id} 不在 Qdrant users 集合中。")

    target = points[0]
    result = client.query_points(
        collection_name="users",
        query=target.vector,
        limit=top_k + 1,
        query_filter=Filter(must_not=[HasIdCondition(has_id=[int(user_id)])]),
        with_payload=True,
        score_threshold=0.0,
    ).points

    rows = []
    for hit in result[:top_k]:
        rows.append(
            {
                "user_id": hit.payload.get("user_id", hit.id),
                "primary_field": hit.payload.get("primary_field", ""),
                "interest_fields": str(hit.payload.get("interest_fields", [])),
                "interest_tags": str(hit.payload.get("interest_tags", [])),
                "similarity": float(hit.score),
            }
        )
    return pd.DataFrame(rows)


def qdrant_recommend_videos(qdrant_url: str, user_id: int, top_k: int, watched_set: set[str]):
    QdrantClient, FieldCondition, Filter, _, MatchValue = safe_import_qdrant()
    if QdrantClient is None:
        raise RuntimeError("未安装 qdrant-client，请安装后再使用向量检索。")

    client = get_qdrant_client(qdrant_url)
    if client is None or (not client.collection_exists("users")) or (not client.collection_exists("videos")):
        raise RuntimeError("Qdrant 的 users/videos 集合不存在，请先运行 F3/F4 建索引脚本。")

    points = client.retrieve(
        collection_name="users",
        ids=[int(user_id)],
        with_vectors=True,
        with_payload=True,
    )
    if not points:
        raise RuntimeError(f"user_id={user_id} 不在 Qdrant users 集合中。")
    user_vec = points[0].vector

    # 先取较大候选池，再在 Python 端过滤已观看视频
    hits = client.query_points(
        collection_name="videos",
        query=user_vec,
        limit=max(200, top_k * 8),
        with_payload=True,
        score_threshold=0.0,
    ).points

    out = []
    for h in hits:
        vid = str(h.payload.get("video_id", ""))
        if vid in watched_set:
            continue
        out.append(
            {
                "id": vid,
                "desc": h.payload.get("desc", ""),
                "similarity": float(h.score),
                "play_count": int(h.payload.get("play_count", 0)),
                "digg_count": int(h.payload.get("digg_count", 0)),
                "fields": str(h.payload.get("fields", [])),
                "preference_tags": str(h.payload.get("preference_tags", [])),
            }
        )
        if len(out) >= top_k:
            break
    return pd.DataFrame(out)


def f3_panel(users: pd.DataFrame):
    st.subheader("F3 相似兴趣用户群体")
    st.caption("输入 user_id，支持本地向量检索或 Qdrant 向量检索。")

    uid = st.number_input("目标用户 ID", min_value=int(users["user_id"].min()), max_value=int(users["user_id"].max()), value=int(users["user_id"].min()), step=1)
    top_k = st.slider("返回相似用户数量", 5, 100, 20, 1)
    backend = st.radio("检索后端", ["本地数据", "Qdrant 向量"], horizontal=True)

    uidx = users.index[users["user_id"] == int(uid)]
    if len(uidx) == 0:
        st.error("用户不存在")
        return

    if backend == "本地数据":
        local_users = users.reset_index(drop=True)
        q_i = int(local_users.index[local_users["user_id"] == int(uid)][0])
        vectors = build_user_vectors(local_users)
        sims = vectors @ vectors[q_i]
        order = np.argsort(-sims)
        order = [i for i in order if i != q_i][:top_k]
        result = local_users.iloc[order][["user_id", "primary_field", "interest_fields", "interest_tags"]].copy()
        result["similarity"] = [float(sims[i]) for i in order]
    else:
        qdrant_url = st.text_input("Qdrant URL", "http://localhost:6333", key="f3_qdrant")
        try:
            result = qdrant_find_similar_users(qdrant_url, int(uid), top_k)
        except Exception as e:
            st.error(f"Qdrant 检索失败：{e}")
            st.info(
                "提示：请确认 Qdrant 已在该地址运行，且已执行 F3 的 `build_user_index.py` 创建 users 集合。"
                "无需向量库时可切换为「本地数据」。"
            )
            render_qdrant_setup_help(need_video_index=False)
            return

    st.dataframe(result, use_container_width=True)
    if not result.empty:
        # 每个相似用户对其 primary + interest_fields 并集中的领域各计 1 次，再在 20 个预设 field 上对齐展示
        acc = pd.Series(0, index=PRESET_FIELDS, dtype=int)
        for _, row in result.iterrows():
            for f in user_row_field_set(row):
                if f in acc.index:
                    acc[f] += 1
        st.caption("相似用户群体的领域分布（横轴为全部 20 个预设 field；某用户若在多个领域中则多列各 +1）")
        bar_chart_all_preset_fields(acc.astype(float), title="用户数")

        st.markdown("---")
        st.markdown("#### 相似用户群体画像分析")

        # --- 统计标签频率（已在数据加载阶段过滤了含标点的无意义标签） ---
        tag_freq: dict[str, int] = {}
        for _, row in result.iterrows():
            tags_raw = row.get("interest_tags", "")
            if isinstance(tags_raw, str):
                tags_list = _filter_tags(parse_list_cell(tags_raw))
            else:
                tags_list = _filter_tags(tags_raw) if isinstance(tags_raw, list) else []
            for tag in tags_list:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1

        col_left, col_right = st.columns(2)

        # --- 标签柱状图 (Tag Bar) ---
        with col_left:
            st.markdown("**Top-10 兴趣标签**")
            if tag_freq:
                top_tags_sorted = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                tag_df = pd.DataFrame(top_tags_sorted, columns=["tag", "count"])
                tag_chart = (
                    alt.Chart(tag_df)
                    .mark_bar(color="#4C9BE8")
                    .encode(
                        x=alt.X("count:Q", title="出现次数"),
                        y=alt.Y("tag:N", sort="-x", title=""),
                    )
                    .properties(height=300)
                )
                st.altair_chart(tag_chart, use_container_width=True)
            else:
                st.info("无标签数据")

        # --- 领域饼图：与上方「20 预设 field」柱状图同一统计口径（非仅 primary_field） ---
        with col_right:
            st.markdown("**兴趣领域分布（饼图）**")
            st.caption(
                "与柱状图一致：每名相似用户的 primary_field 与 interest_fields 并集内，"
                "每个领域各计 1 次，故总人数少时也可出现多领域扇区。"
            )
            pie_src = acc[acc > 0].sort_values(ascending=False).rename_axis("field").reset_index(name="count")
            if not pie_src.empty:
                total_c = float(pie_src["count"].sum())
                pie_src["pct_label"] = (pie_src["count"] / total_c * 100).round(1).astype(str) + "%"

                SOFT_COLORS = [
                    "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3",
                    "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD",
                    "#CCEBC5", "#FFED6F", "#A8D8B9", "#F4B8C1", "#B8D4E8",
                    "#E8C9A0", "#C9BBD9", "#F9D6A0", "#B5CBBB", "#DFD6B0",
                ]
                all_fields = sorted(pie_src["field"].unique())
                color_map = {f: SOFT_COLORS[i % len(SOFT_COLORS)] for i, f in enumerate(all_fields)}
                base = alt.Chart(pie_src).encode(
                    theta=alt.Theta("count:Q", stack=True),
                    color=alt.Color("field:N", title="兴趣领域", scale=alt.Scale(
    domain=list(color_map.keys()),
    range=list(color_map.values()),
)),
                    tooltip=["field", "count", alt.Tooltip("pct_label:N", title="占比")],
                )
                pie_chart = base.mark_arc(outerRadius=120).properties(height=320)
                labels = base.mark_text(radius=145, size=10).encode(text="pct_label:N")
                st.altair_chart(pie_chart + labels, use_container_width=True)
            else:
                st.info("无领域数据")

## 更改：        # --- 词云 (Wordcloud) ---
        if tag_freq:
            try:
                from wordcloud import WordCloud
                import io as _io
                from PIL import Image

                st.markdown("**兴趣标签词云**")

                wc = WordCloud(
                    width=1600,
                    height=900,
                    background_color="white",
                    colormap="RdYlBu",
                    max_words=120,
                    min_font_size=12,
                    prefer_horizontal=0.9,
                    scale=2,
                    random_state=42
                ).generate_from_frequencies(tag_freq)

                img = wc.to_image()

                # ✅ 直接用原图，不缩略
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)

                # 改后
                col, _ = st.columns([2, 1])
                with col:
                    st.image(buf.getvalue(), use_container_width=True)

                # 高清下载（复用同一个 buf）
                buf.seek(0)
                st.download_button(
                    label="📥下载高清词云 PNG",
                    data=buf,
                    file_name="wordcloud.png",
                    mime="image/png",
                )

            except ImportError:
                st.caption("提示：安装 `wordcloud` 库后可显示词云（`pip install wordcloud`）")
## 更改👆

def f4_panel(users: pd.DataFrame, videos: pd.DataFrame, events):
    st.subheader("F4 用户历史与相关视频推荐")
    st.caption("输入 user_id，展示历史记录并给出相关视频推荐（支持 Qdrant 向量版）。")

    uid = st.number_input(
        "目标用户 ID（F4）",
        min_value=int(users["user_id"].min()),
        max_value=int(users["user_id"].max()),
        value=int(users["user_id"].min()),
        step=1,
        key="f4_uid",
    )
    top_k = st.slider("推荐数量", 5, 50, 15, 1, key="f4_k")
    backend = st.radio("推荐后端", ["本地规则", "Qdrant 向量"], horizontal=True, key="f4_backend")

    urow = users[users["user_id"] == int(uid)]
    if urow.empty:
        st.error("用户不存在")
        return
    urow = urow.iloc[0]

## 更改：
    if events is None:
        events = cached_user_watch_events(int(uid))
    uevents = events[
        (events["user_id"] == int(uid)) &
        (events["event_type"] == "watch")
        ].merge(
        videos[["id", "desc", "fields", "preference_tags", "play_count", "digg_count"]],
        left_on="video_id",
        right_on="id",
        how="left"
    )

    watch_ids = set(uevents["video_id"].astype(str))

    st.write(f"历史观看条数：{len(uevents):,}")

    # ====== 更清晰的展示 ======
    st.dataframe(
        uevents[
            [
                "video_id",
                "desc",
                "fields",
                "preference_tags",
                "play_count",
                "digg_count",
            ]
        ].head(300),
        use_container_width=True,
    )

    st.markdown("#### 历史兴趣分布（Fields）")

    hist_fields = uevents["fields"].dropna()
    if len(hist_fields) > 0:
        exploded = hist_fields.apply(parse_list_cell).explode()
        vc = exploded.value_counts().head(15)
        st.bar_chart(vc)

## 更改👆

    if backend == "本地规则":
        fields = set(urow["interest_fields_list"])
        tags = set(urow["interest_tags_list"])
        cands = videos[~videos["id"].isin(watch_ids)].copy()
        cands["match_score"] = cands.apply(
            lambda r: 2 * len(set(r["fields_list"]) & fields) + len(set(r["preference_tags_list"]) & tags), axis=1
        )
        rec = cands.sort_values(["match_score", "play_count"], ascending=[False, False]).head(top_k)
        show_cols = ["id", "desc", "match_score", "play_count", "digg_count", "fields", "preference_tags"]
    else:
        qdrant_url = st.text_input("Qdrant URL", "http://localhost:6333", key="f4_qdrant")
        try:
            rec = qdrant_recommend_videos(qdrant_url, int(uid), top_k, watch_ids)
        except Exception as e:
            st.error(f"Qdrant 推荐失败：{e}")
            st.info(
                "提示：请确认 Qdrant 可用，且已分别创建 users（F3）与 videos（F4）向量集合。"
                "无需向量库时可切换为「本地规则」。"
            )
            render_qdrant_setup_help(need_video_index=True)
            return
        show_cols = ["id", "desc", "similarity", "play_count", "digg_count", "fields", "preference_tags"]

    st.write("推荐结果：")
    st.dataframe(rec[show_cols], use_container_width=True)

## 更改：
def f5_panel(videos: pd.DataFrame, series: pd.DataFrame, events):
    st.subheader("F5 视频热度预测")

    c1, c2 = st.columns(2)
    with c1:
        vid_kw = st.text_input("视频 ID 关键词（F5）", "")
    with c2:
        fields = sorted({f for lst in videos["fields_list"] for f in lst})
        sfields = st.multiselect("field（F5）", fields)

    # 筛选候选视频：必须同时满足用户输入的条件 + 存在于 series 中（即有预测结果）
    mask = pd.Series(True, index=videos.index)
    if vid_kw:
        mask &= videos["id"].str.contains(vid_kw.strip(), na=False)
    if sfields:
        mask &= videos["fields_list"].apply(lambda x: any(f in x for f in sfields))

    # 关键改动：只保留有预测结果的视频
    valid_video_ids = set(series["video_id"].astype(str))
    candidates = videos.loc[mask, ["id", "desc", "play_count"]].copy()
    candidates = candidates[candidates["id"].isin(valid_video_ids)]  # 过滤

    if candidates.empty:
        st.warning("没有找到有热度预测结果的视频，请尝试其他关键词或 field。")
        return

    # 限制显示数量，避免列表过长
    st.dataframe(candidates.head(500), use_container_width=True)

    selected_vid = st.selectbox("选择一个视频查看预测结果", candidates["id"].tolist())
    row = series[series["video_id"] == str(selected_vid)]
    if row.empty:
        # 理论上不会进入这里，因为 candidates 已经过滤过了
        st.warning("该视频在 F5 预测结果中不存在。")
        return

    row = row.iloc[0]
    H = 3

    # 获取所有轮次列（真实值、预测值、上下界、噪声）
    all_round_cols = [c for c in series.columns if c.startswith("round_")]
    rounds_sorted = sorted(all_round_cols, key=lambda x: int(x.split("_")[1]))

    # 真实前 H 轮
    actual_vals = [float(row[c]) for c in rounds_sorted[:H]]

    # 预测中位数（第 H+1 轮到末尾）
    pred_median_vals = [float(row[c]) for c in rounds_sorted[H:]]

    # 噪声预测曲线
    noisy_cols = [f"noisy_{c}" for c in rounds_sorted[H:]]
    if all(col in series.columns for col in noisy_cols):
        noisy_vals = [float(row[col]) for col in noisy_cols]
    else:
        noisy_vals = pred_median_vals

    # 上下界
    lower_cols = [f"lower_bound_{c.split('_')[1]}" for c in rounds_sorted[H:]]
    upper_cols = [f"upper_bound_{c.split('_')[1]}" for c in rounds_sorted[H:]]
    if all(col in series.columns for col in lower_cols) and all(col in series.columns for col in upper_cols):
        lower_vals = [float(row[col]) for col in lower_cols]
        upper_vals = [float(row[col]) for col in upper_cols]
    else:
        lower_vals = [v * 0.6 for v in pred_median_vals]
        upper_vals = [v * 1.4 for v in pred_median_vals]

    # 预测总播放量
    if "pred_total_play" in series.columns:
        total_pred = float(row["pred_total_play"])
    else:
        total_pred = sum(pred_median_vals)

    # 真实总播放量
    if events is None:
        try:
            real_rows = cached_video_watch_counts(str(selected_vid))
        except Exception as error:
            show_data_error("读取真实观看统计失败", error)
            st.info("真实观看统计暂不可用，以下仍显示已加载的预测结果。")
            real_rows = None
    else:
        real_rows = events[
            (events["video_id"] == str(selected_vid)) &
            (events["event_type"] == "watch")
        ].groupby("round").size()
    total_real = int(real_rows.sum()) if real_rows is not None else None

    # 趋势判断
    early_avg = sum(pred_median_vals[:5]) / 5 if len(pred_median_vals) > 5 else 1
    late_avg  = sum(pred_median_vals[-5:]) / 5
    if late_avg > early_avg * 1.2:
        trend = "📈 上升"
        trend_color = "green"
    elif late_avg < early_avg * 0.8:
        trend = "📉 衰减"
        trend_color = "red"
    else:
        trend = "➡️ 平稳"
        trend_color = "orange"

    # 热度分类
    if total_pred <= 22 * 30 / 127:
        label = "🥶 冷门"
        label_color = "blue"
    elif total_pred >= 127:
        label = "🔥 爆款"
        label_color = "red"
    else:
        label = "😐 普通"
        label_color = "orange"

    st.markdown("### 预测结果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("热度分类", label)
    with col2:
        st.metric("预测总播放量（第4~30轮）", f"{int(total_pred)}")
    with col3:
        st.metric("真实总播放量", str(total_real) if total_real is not None else "暂不可用")

    st.markdown(f"**热度趋势预测：** :{trend_color}[{trend}]")
    st.markdown("---")

    # 构造绘图数据
    pred_df = pd.DataFrame({
        "round": rounds_sorted[H:],
        "播放量": noisy_vals,
        "类型": ["模型预测（噪声曲线）"] * len(rounds_sorted[H:]),
    })
    combined = pred_df
    if real_rows is not None:
        full_rounds = list(range(1, 31))
        real_df = pd.DataFrame({
            "round": [f"round_{r}" for r in full_rounds],
            "播放量": [real_rows.get(r, 0) for r in full_rounds],
            "类型": ["真实完整数据"] * len(full_rounds),
        })
        combined = pd.concat([pred_df, real_df], ignore_index=True)

    # 置信区间阴影
    band_df = pd.DataFrame({
        "round": rounds_sorted[H:],
        "下界": lower_vals,
        "上界": upper_vals,
    })
    band = alt.Chart(band_df).mark_area(opacity=0.2, color="#FF9800").encode(
        x=alt.X("round:N", sort=rounds_sorted),
        y=alt.Y("下界:Q"),
        y2=alt.Y2("上界:Q"),
    )
    line_chart = alt.Chart(combined).mark_line(point=True).encode(
        x=alt.X("round:N", sort=rounds_sorted, title="轮次（天）"),
        y=alt.Y("播放量:Q", title="播放量"),
        color=alt.Color(
            "类型:N",
            scale=alt.Scale(
                domain=["模型预测（噪声曲线）", "真实完整数据"] if real_rows is not None else ["模型预测（噪声曲线）"],
                range=["#FF9800", "#4CAF50"] if real_rows is not None else ["#FF9800"]
            )
        ),
        tooltip=["round", "播放量", "类型"],
    ).properties(height=350)

    st.altair_chart(band + line_chart, use_container_width=True)
    st.caption(
        "橙色线条为模型预测曲线，橙色阴影为预测结果置信区间。"
        + ("绿色线条为全量真实播放数据。" if real_rows is not None else "真实观看统计尚未加载，图中仅显示预测结果。")
        + "本数据集为模拟数据，单轮数据存在随机波动，模型重点研判整体变化趋势。"
    )
    

    # ── 详细数据表 ──
    with st.expander("查看详细数据"):
        st.dataframe(combined, use_container_width=True)
  ## 更改👆


def build_cluster_embedding(df: pd.DataFrame, list_cols, scalar_cols=None, weights=None):
    if scalar_cols is None:
        scalar_cols = []
    if weights is None:
        weights = {}

    vocab = set()
    for col in list_cols:
        for arr in df[col]:
            vocab.update(arr)
    for col in scalar_cols:
        vocab.update(df[col].dropna().astype(str).tolist())

    vocab = sorted(vocab)
    idx = {k: i for i, k in enumerate(vocab)}
    mat = np.zeros((len(df), len(vocab)), dtype=float)

    for i, row in df.iterrows():
        for col in list_cols:
            w = float(weights.get(col, 1.0))
            for token in row[col]:
                if token in idx:
                    mat[i, idx[token]] += w
        for col in scalar_cols:
            token = str(row[col]) if pd.notna(row[col]) else ""
            if token in idx:
                mat[i, idx[token]] += float(weights.get(col, 1.0))

    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return mat / norms

## 更改
def reduce_to_2d(matrix_bytes: bytes, shape: tuple, method: str):
    # 去掉 @st.cache_data，改用 session_state 缓存
    PCA, TSNE = safe_import_sklearn()
    if PCA is None:
        raise RuntimeError("未安装 scikit-learn，无法做降维可视化。")
    matrix = np.frombuffer(matrix_bytes, dtype=np.float32).reshape(shape)
    if method == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(matrix)
    else:
        if TSNE is None:
            raise RuntimeError("当前环境不支持 t-SNE。")
        return TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto").fit_transform(matrix)
## 更改

## 更改
def scatter_cluster(df_plot: pd.DataFrame, title: str, color_by_cluster: bool = True):
    COLORS_20 = [
        "#FF9838", "#FF613D", "#FF2F00", "#FFC833", "#FFEC1F",
        "#FFFC57", "#87FC43", "#56FA00", "#00CA45", "#00A01D",
        "#21FBB8","#29FEFF", "#58BCFA", "#2C89F9", "#5317C0",
        "#6644FE", "#BA59FF", "#DE82FF", "#C2C2C2", "#848DAE",
    ]

    if color_by_cluster:
        cluster_ids = sorted(df_plot["cluster_id"].unique(), key=lambda x: int(x))
        color_enc = alt.Color(
            "cluster_id:N",
            title="Cluster",
            scale=alt.Scale(
                domain=cluster_ids,
                range=COLORS_20[: len(cluster_ids)],
            ),
        )
    else:
        color_enc = alt.value("#1f77b4")

    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=45, opacity=0.65)
        .encode(
            x=alt.X("x:Q", title="Dim-1"),
            y=alt.Y("y:Q", title="Dim-2"),
            color=color_enc,
            tooltip=["cluster_id", "id"],
        )
        .properties(height=450, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    ## 更改


def _f6_f7_all_clusters_scatter(
    merged: pd.DataFrame,
    key_prefix: str,
    list_cols: list,
    scalar_cols: list,
    weights: dict,
    title_prefix: str,
    method: str,
    max_points: int,
):
    # ✅ 用稳定参数拼 cache key
    cache_key = f"scatter_all_{key_prefix}_{method}_{max_points}_{len(merged)}"

    if cache_key not in st.session_state:
        plot_all = merged.sample(min(len(merged), max_points), random_state=42).reset_index(drop=True)
        try:
            emb = build_cluster_embedding(
                plot_all, list_cols=list_cols, scalar_cols=scalar_cols, weights=weights
            )
            emb = emb.astype(np.float32)
            xy = reduce_to_2d(emb.tobytes(), emb.shape, method)
            st.session_state[cache_key] = {
                "xy": xy,
                "cluster_id": plot_all["cluster_id"].astype(str).tolist(),
                "id": (
                    plot_all["video_id"]
                    if "video_id" in plot_all.columns
                    else plot_all["user_id"].astype(str)
                ).tolist(),
            }
        except Exception as e:
            st.warning(f"无法生成全部簇散点图：{e}")
            return

    cached = st.session_state[cache_key]
    xy = cached["xy"]
    scatter_df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "cluster_id": cached["cluster_id"],
        "id": cached["id"],
    })
    st.markdown(f"**{title_prefix}：全部 cluster（采样）**")
    scatter_cluster(scatter_df, f"{title_prefix} 二维可视化 — 全部簇（{method}）", color_by_cluster=True)


def _f6_f7_single_cluster_scatter(
    merged: pd.DataFrame,
    key_prefix: str,
    list_cols: list,
    scalar_cols: list,
    weights: dict,
    title_prefix: str,
    method: str,
    single_cid: int,
    single_max: int,
):
    sub = merged[merged["cluster_id"] == single_cid]
    if len(sub) < 3:
        st.info(f"cluster {single_cid} 内点数过少（{len(sub)}），无法稳定做二维降维。")
        return

    # ✅ 用稳定参数拼 cache key
    cache_key = f"scatter_single_{key_prefix}_{single_cid}_{method}_{single_max}_{len(sub)}"

    if cache_key not in st.session_state:
        plot_one = sub.sample(min(len(sub), single_max), random_state=42).reset_index(drop=True)
        try:
            emb1 = build_cluster_embedding(
                plot_one, list_cols=list_cols, scalar_cols=scalar_cols, weights=weights
            )
            emb1 = emb1.astype(np.float32)
            xy1 = reduce_to_2d(emb1.tobytes(), emb1.shape, method)
            st.session_state[cache_key] = {
                "xy": xy1,
                "cluster_id": plot_one["cluster_id"].astype(str).tolist(),
                "id": (
                    plot_one["video_id"]
                    if "video_id" in plot_one.columns
                    else plot_one["user_id"].astype(str)
                ).tolist(),
            }
        except Exception as e:
            st.warning(f"无法生成单簇散点图：{e}")
            return

    cached = st.session_state[cache_key]
    xy1 = cached["xy"]
    scatter_one = pd.DataFrame({
        "x": xy1[:, 0],
        "y": xy1[:, 1],
        "cluster_id": cached["cluster_id"],
        "id": cached["id"],
    })
    scatter_cluster(
        scatter_one,
        f"{title_prefix} 二维可视化 — cluster {single_cid}（{method}）",
        color_by_cluster=False,
    )


def f6_panel(video_clusters: pd.DataFrame, videos: pd.DataFrame):
    st.subheader("F6 视频聚类分析")
    st.caption("展示聚类分布、二维散点图，并给出簇内代表标签解释。")

    dist = video_clusters["cluster_id"].value_counts().sort_index()
    st.bar_chart(dist)

    merged = video_clusters.merge(
        videos[["id", "fields_list", "preference_tags_list", "play_count"]],
        left_on="video_id",
        right_on="id",
        how="left",
    )

    f6_list_cols = ["fields_list", "preference_tags_list"]
    f6_scalar_cols: list = []
    f6_weights = {"fields_list": 2.0, "preference_tags_list": 1.0}

    st.markdown("---")
    st.markdown("#### 选择 Cluster 进行可视化")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        max_cid = video_clusters["cluster_id"].max()
        f6_options = list(range(max_cid + 1))
        f6_cid = st.selectbox(
            "选择 cluster_id",
            f6_options,
            index=0,
            key="f6_vis_cid",
        )
    with col_b:
        f6_method = st.radio("降维算法", ["PCA", "t-SNE"], horizontal=True, key="f6_dr")
    with col_c:
        f6_single_max = st.slider("簇内最多采样点数", 200, 5000, 1500, 100, key="f6_single_n")

    _f6_f7_single_cluster_scatter(
        merged,
        key_prefix="f6",
        list_cols=f6_list_cols,
        scalar_cols=f6_scalar_cols,
        weights=f6_weights,
        title_prefix="F6 视频聚类",
        method=f6_method,
        single_cid=f6_cid,
        single_max=f6_single_max,
    )

    with st.expander("查看全部簇散点图（全局总览）", expanded=False):
        f6_all_max = st.slider("全局散点采样数", 500, 8000, 2500, 100, key="f6_all_n")
        if st.button("生成全局散点图", key="f6_all_btn"):
            st.session_state["f6_all_generated"] = True
        if st.session_state.get("f6_all_generated", False):
            _f6_f7_all_clusters_scatter(
                merged,
                key_prefix="f6",
                list_cols=f6_list_cols,
                scalar_cols=f6_scalar_cols,
                weights=f6_weights,
                title_prefix="F6 视频聚类",
                method=f6_method,
                max_points=f6_all_max,
            )

    st.markdown("---")
    st.markdown("#### 簇内样本与解释")
    max_cid = video_clusters["cluster_id"].max()
    cid_options = list(range(max_cid + 1))
    cid = st.selectbox("选择 cluster_id 查看样本视频", cid_options, key="f6_sample_cid")
    vids = video_clusters[video_clusters["cluster_id"] == cid]["video_id"].astype(str)
    sample = videos[videos["id"].isin(vids)].head(200)
    st.dataframe(sample[["id", "desc", "play_count", "fields", "preference_tags"]], use_container_width=True)

    cluster_rows = merged[merged["cluster_id"] == cid]
    top_fields = top_counter(cluster_rows["fields_list"].explode(), top_n=8)
    top_tags = top_counter(cluster_rows["preference_tags_list"].explode(), top_n=10)
    st.markdown("**簇解释（代表内容）**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Top Fields")
        st.dataframe(top_fields, use_container_width=True)
    with c2:
        st.write("Top Tags")
        st.dataframe(top_tags, use_container_width=True)

    # 簇解释文字放在表格后面
    F6_CLUSTER_DESC = {
        0: ("音乐文化与数字娱乐视频",
            "以音乐、歌曲和表演内容为主体，同时融合游戏科技、心理成长和商业知识等内容。该簇体现出用户对数字娱乐、兴趣探索和情感共鸣的综合需求。"),
        1: ("年轻时尚与女性审美表达视频",
            "以年轻女性的穿搭、美妆、日常vlog与审美展示为主，强调视觉风格、个人形象与时尚表达的内容类型。"),
        2: ("拉美与美国混合文化短视频内容",
            "以拉美文化元素与美国互联网流行文化融合为特征，涵盖音乐、舞蹈、日常生活与社交表达的跨文化短视频内容。"),
        3: ("都市旅行与生活方式记录视频",
            "以城市旅行、日常生活记录与生活方式展示为核心，强调在路上与都市体验感的vlog类内容。"),
        4: ("美食探店与平台化生活方式创作内容",
            "以美食展示、城市探店、生活方式分享及小型商业/创作内容为主，具有明显平台创作者经济特征的视频类型。"),
        5: ("叙事型故事与真实经历表达视频",
            "以真实故事讲述、个人经历分享与社区讨论文化为核心，包括故事类叙事、Reddit式文本/视频转述与BookTok风格内容。"),
        6: ("学习与知识获取导向短视频内容", "以学习技巧、知识讲解、教育信息与学生日常为核心的知识传播型短视频内容。"),
        7: ("地域文化与社交音乐生活方式视频",
            "以美国南部及拉美地区文化为背景，结合旅行、社交活动与音乐氛围展示的地域生活方式类内容。"),
        8: ("泛娱乐热点与剪辑文化视频",
            "以明星、影视作品、新闻热点、游戏内容及二次剪辑（edit/montage）为核心的泛娱乐信息流内容。"),
        9: ("居家审美与自然氛围生活视频", "以家庭空间、美学布置、自然氛围与温馨生活记录为核心的居家风格内容。"),
        10: ("流行音乐与粉丝文化沉浸式视频",
             "以演唱会片段、流行音乐、K-pop及粉丝文化内容为核心的音乐情绪沉浸型视频内容。"),
        11: ("效率生活与自我管理主题视频",
             "以预算管理、生活规划、自我提升与aesthetic productivity为核心的生活优化与效率型内容。"),
        12: ("美食娱乐与感官刺激视频", "以吃播、ASMR、美食探店与感官刺激体验为核心的娱乐型美食短视频内容。"),
        13: ("情感关系与人际叙事视频", "以恋爱关系、情感经历、亲密互动与真实人际故事为核心的情绪叙事类视频内容。"),
        14: ("视觉剪辑与电影化表达视频", "以电影剪辑、视觉艺术表达、都市片段与短视频edit文化为核心的高视觉密度内容。"),
        15: ("健身运动与身体表现内容视频", "以健身训练、跑步、运动表现及身体塑形记录为核心的运动型短视频内容。"),
        16: ("家庭生活与育儿记录视频", "以家庭结构、育儿日常、亲子互动与生命周期叙事为核心的家庭生活类内容。"),
        17: ("DIY制作与动手创作类视频", "以手工制作、DIY改造、建造过程与生活技巧展示为核心的创作型过程内容。"),
        18: ("视觉审美与都市音乐氛围视频", "以摄影、艺术表达、音乐氛围与都市生活场景为核心的审美导向型短视频内容。"),
        19: ("幽默娱乐与meme文化视频", "以搞笑片段、meme文化、轻喜剧与日常共鸣为核心的短时情绪娱乐型视频内容。"),
    }

    st.markdown("---")
    st.markdown("#### 簇语义解读")
    title, desc = F6_CLUSTER_DESC.get(cid, ("未知", "暂无描述"))
    st.info(f"**簇{cid}：{title}**\n\n{desc}")
    st.caption("F6 不是「视频按内容分组」，而是「根据用户行为结构推导出的内容语义结构」。")


def f7_panel(user_clusters: pd.DataFrame, users: pd.DataFrame):
    st.subheader("F7 用户聚类分析")
    st.caption("展示聚类分布、二维散点图，并给出簇内兴趣解释。")

    dist = user_clusters["cluster_id"].value_counts().sort_index()
    st.bar_chart(dist)

    uc_keys = user_clusters[["user_id", "cluster_id"]].copy()
    merged = uc_keys.merge(
        users[
            [
                "user_id",
                "primary_field",
                "interest_fields_list",
                "interest_tags_list",
                "total_watched",
                "total_liked",
            ]
        ],
        on="user_id",
        how="left",
    )

    f7_list_cols = ["interest_fields_list", "interest_tags_list"]
    f7_scalar_cols = ["primary_field"]
    f7_weights = {"primary_field": 3.0, "interest_fields_list": 2.0, "interest_tags_list": 1.0}

    st.markdown("---")
    st.markdown("#### 选择 Cluster 进行可视化")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        max_cid = user_clusters["cluster_id"].max()
        f7_options = list(range(max_cid + 1))
        f7_cid = st.selectbox(
            "选择 cluster_id",
            f7_options,
            index=0,
            key="f7_vis_cid",
        )
    with col_b:
        f7_method = st.radio("降维算法", ["PCA", "t-SNE"], horizontal=True, key="f7_dr")
    with col_c:
        f7_single_max = st.slider("簇内最多采样点数", 200, 5000, 1500, 100, key="f7_single_n")

    _f6_f7_single_cluster_scatter(
        merged,
        key_prefix="f7",
        list_cols=f7_list_cols,
        scalar_cols=f7_scalar_cols,
        weights=f7_weights,
        title_prefix="F7 用户聚类",
        method=f7_method,
        single_cid=f7_cid,
        single_max=f7_single_max,
    )

    with st.expander("查看全部簇散点图（全局总览）", expanded=False):
        f7_all_max = st.slider("全局散点采样数", 500, 8000, 2500, 100, key="f7_all_n")
        if st.button("生成全局散点图", key="f7_all_btn"):
            st.session_state["f7_all_generated"] = True
        if st.session_state.get("f7_all_generated", False):
            _f6_f7_all_clusters_scatter(
                merged,
                key_prefix="f7",
                list_cols=f7_list_cols,
                scalar_cols=f7_scalar_cols,
                weights=f7_weights,
                title_prefix="F7 用户聚类",
                method=f7_method,
                max_points=f7_all_max,
            )

    st.markdown("---")
    st.markdown("#### 簇内样本与解释")
    max_cid = user_clusters["cluster_id"].max()
    cid_options = list(range(max_cid + 1))
    cid = st.selectbox("选择 cluster_id 查看样本用户", cid_options, key="f7_sample_cid")
    uids = user_clusters[user_clusters["cluster_id"] == cid]["user_id"].astype(int)
    sample = users[users["user_id"].isin(uids)].head(200)
    st.dataframe(
        sample[["user_id", "primary_field", "interest_fields", "interest_tags", "total_watched", "total_liked"]],
        use_container_width=True,
    )

    cluster_rows = merged[merged["cluster_id"] == cid]
    top_primary = top_counter(cluster_rows["primary_field"], top_n=8)
    top_tags = top_counter(cluster_rows["interest_tags_list"].explode(), top_n=10)
    st.markdown("**簇解释（代表兴趣）**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Top Primary Fields")
        st.dataframe(top_primary, use_container_width=True)
    with c2:
        st.write("Top Interest Tags")
        st.dataframe(top_tags, use_container_width=True)

    F7_CLUSTER_DESC = {
        0: ("美食向", "核心标签：美食吃货。喜欢看吃播、美食探店和做饭教程的用户，是平台的吃货群体。"),
        1: ("商业/生活向", "核心标签：美国生活/商业关注者。关注美国商业资讯、本地生活和社会话题的用户。"),
        2: ("金融/搞钱向",
            "核心标签：理财/搞钱爱好者。对搞钱、理财、副业、商业动机类内容感兴趣，以金融/财富导向为核心，同时融合轻度生活方式与情绪放松内容消费的复合型兴趣群体。"),
        3: ("家庭育儿向", "核心标签：宝妈/家庭向用户。宝妈、新手父母，关注带娃、家庭日常和育儿经验分享。"),
        4: ("游戏/技术向", "核心标签：游戏玩家/青少年。喜欢游戏、科技和相关话题的用户，对技术类内容有较高关注。"),
        5: ("舞蹈/娱乐向", "核心标签：舞蹈/娱乐爱好者。喜欢看舞蹈、迪士尼、演唱会、体育赛事等娱乐内容的用户。"),
        6: ("故事/知识向", """核心标签：故事/知识学习者。爱听故事、看历史、学习知识的用户，是平台的"知识型观众"。"""),
        7: ("健身/运动向", "核心标签：健身/运动爱好者。关注健身、撸铁、跑步、健康生活的用户，是平台的运动党。"),
        8: ("搞笑/娱乐向", """核心标签：搞笑/娱乐观众。喜欢刷搞笑视频、段子和表情包的用户，是平台的"快乐源泉"群体。"""),
        9: ("旅行/地域向", "核心标签：美国旅行/地域爱好者。关注美国旅游、不同城市和地域生活内容的用户。"),
        10: ("情感/关系向", "核心标签：情感/恋爱关注者。关注感情、恋爱、情感语录和治愈向内容的用户。"),
        11: ("心理健康向", "核心标签：心理健康关注者。关注心理健康、情绪管理和自我疗愈内容的用户。"),
        12: ("自然/生活向", "核心标签：自然/生活爱好者。喜欢看自然风景、户外徒步、旅行和家居生活内容的用户。"),
        13: ("音乐向", "核心标签：音乐爱好者。重度音乐爱好者，喜欢听歌、看演唱会、舞蹈和音乐相关内容。"),
        14: ("美妆/时尚向", "核心标签：美妆/时尚爱好者。关注发型、美甲、化妆和时尚穿搭的用户，是平台的时尚党。"),
        15: ("美食/生活向", "核心标签：美食生活爱好者。喜欢看美食、做饭教程、吃播和家庭日常的用户。"),
        16: ("故事/知识向", "核心标签：故事/知识学习者。爱听故事、看历史、读科普和Reddit趣闻的用户。"),
        17: ("DIY/手工向", "核心标签：手工/DIY爱好者。喜欢看手工教程、DIY制作和创意手作的用户。"),
        18: ("艺术/创意向", "核心标签：艺术/创意爱好者。关注艺术、创意设计、音乐和文化相关内容的用户。"),
        19: ("家居/装修向", "核心标签：家居/装修爱好者。关注家装、室内设计、租房改造和家居布置的用户。"),
    }

    st.markdown("---")
    st.markdown("#### 簇语义解读")
    title, desc = F7_CLUSTER_DESC.get(cid, ("未知", "暂无描述"))
    st.info(f"**簇{cid}：{title}**\n\n{desc}")


@st.cache_data(ttl=300)
def cached_user_watch_events(user_id):
    return user_watch_events(user_id)


@st.cache_data(ttl=300)
def cached_video_watch_counts(video_id):
    return video_watch_counts(video_id)


def main():
    st.set_page_config(page_title="短视频推荐系统 F1-F7", layout="wide")
    st.title("短视频推荐系统（F1-F7）")
    if st.sidebar.checkbox("查看算法对照实验", key="show_algorithm_benchmarks"):
        from experiments.ui import experiments_panel

        experiments_panel()
        return
    try:
        source = data_backend()
        st.caption("Supabase / PostgreSQL 数据服务" if source == "supabase" else "CSV 离线演示模式")
        st.sidebar.caption(f"数据源：{'Supabase / PostgreSQL' if source == 'supabase' else 'CSV'}")
        if st.sidebar.button("刷新数据"):
            st.cache_data.clear()
        data = load_data(source)
    except Exception as error:
        show_data_error("数据源加载失败", error)
        st.code("python -m database.migrate\npython -m database.import_data")
        st.info("接入步骤见 database/README.md；离线演示可设置 DATA_BACKEND=csv。")
        st.stop()
    MENU_MAP = {
        "F1：构建视频信息库": "F1",
        "F2：模拟用户观看行为": "F2",
        "F3：相似用户群体分析": "F3",
        "F4：基于历史视频推荐": "F4",
        "F5：视频热度预测分析": "F5",
        "F6：视频聚类分析": "F6",
        "F7：用户聚类分析": "F7",
    }

    menu_label = st.sidebar.radio(
        "选择功能模块",
        list(MENU_MAP.keys())
    )

    menu = MENU_MAP[menu_label]

    if menu == "F1":
        if ensure_data(data, ["videos"]):
            f1_panel(data["videos"])
    elif menu == "F2":
        if ensure_data(data, ["users"]):
            f2_panel(data["users"])
    elif menu == "F3":
        if ensure_data(data, ["users"]):
            f3_panel(data["users"].reset_index(drop=True))
    elif menu == "F4":
        required = ["users", "videos"] + (["events"] if source == "csv" else [])
        if ensure_data(data, required):
            try:
                f4_panel(data["users"], data["videos"], data.get("events"))
            except Exception as error:
                show_data_error("推荐数据处理失败", error)
    elif menu == "F5":
        required = ["videos", "series"] + (["events"] if source == "csv" else [])
        if ensure_data(data, required):
            try:
                f5_panel(data["videos"], data["series"], data.get("events"))
            except Exception as error:
                show_data_error("预测结果处理失败", error)
    elif menu == "F6":
        if ensure_data(data, ["video_clusters", "videos"]):
            f6_panel(data["video_clusters"], data["videos"])
    elif menu == "F7":
        if ensure_data(data, ["user_clusters", "users"]):
            f7_panel(data["user_clusters"], data["users"])


if __name__ == "__main__":
    main()
