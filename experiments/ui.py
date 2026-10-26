"""Read-only Streamlit viewer for completed benchmark runs."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def experiments_panel():
    st.subheader("算法对照实验")
    st.caption("展示实际运行生成的结果；检索质量、聚类质量和速度分别评估。")
    results_root = Path(__file__).resolve().parent / "results"
    runs = sorted(p.parent for p in results_root.glob("*/metadata.json"))
    if not runs:
        st.info("尚未生成实验结果。在项目根目录执行以下命令。")
        st.code("python -m pip install -r requirements-experiments.txt\npython experiments/benchmark.py", language="bash")
        return
    run = st.selectbox("选择实验结果", runs, format_func=lambda path: path.name)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    st.write(f"向量来源：{metadata['source']}；运行时间（UTC）：{metadata['created_at_utc']}")
    if metadata["source"] == "local":
        st.info("本组使用项目用户数据生成的加权标签向量，比较全量搜索与聚类候选搜索。Qdrant 性能请查看单独的 qdrant 实验结果。")
    st.caption("同主兴趣领域比例、ARI/NMI 是模拟标签的一致性参考，不等于真实推荐准确率。")
    labels = {
        "method": "算法", "mean_ms": "平均查询(ms)", "p50_ms": "P50(ms)", "p95_ms": "P95(ms)",
        "recall_at_k": "Recall@K", "tie_aware_recall_at_k": "允许同分替代的Recall@K",
        "same_field_precision_at_k": "同主兴趣领域比例@K", "mean_candidates": "平均候选数",
        "mean_returned": "平均返回数", "speedup": "查询加速倍数", "index_build_s": "离线建簇(s)",
        "break_even_queries": "摊平建簇代价的查询数", "task": "任务", "n": "样本数",
        "dimensions": "维数", "mean_total_s": "总耗时均值(s)", "std_total_s": "总耗时标准差(s)",
        "mean_fit_s": "聚类耗时(s)", "mean_preprocess_s": "降维等预处理(s)",
        "silhouette_original_cosine": "原空间余弦轮廓系数", "ari_primary_field": "主兴趣ARI",
        "nmi_primary_field": "主兴趣NMI", "feature_storage_mib": "特征数组大小(MiB)",
        "speedup_vs_plus_plus": "相对原空间KMeans++加速倍数",
    }
    for filename, title in [("search_summary.csv", "相似度搜索"),
                            ("user_clustering_summary.csv", "用户聚类"),
                            ("video_clustering_summary.csv", "视频聚类")]:
        path = run / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        st.markdown(f"#### {title}")
        st.dataframe(frame.rename(columns=labels), use_container_width=True, hide_index=True)
        if filename.startswith("search"):
            info = metadata["search"]
            st.caption(f"{info['n']:,} 条向量，{info['dimensions']} 维，{info['queries']} 个查询，Top-{info['top_k']}。查询计时排除向量化与离线建簇；Qdrant 行包含 HTTP 往返。")
            st.bar_chart(frame.set_index("method")[["mean_ms", "p95_ms"]].rename(columns=labels))
        else:
            st.caption("所有方法在同一原空间和同一采样集上计算轮廓系数，越大越好；总耗时包含 SVD，排除数据加载与指标评估。特征数组大小并非进程峰值内存。")
            st.bar_chart(frame.set_index("method")[["mean_total_s"]].rename(columns=labels))
        st.download_button(f"下载{title}结果", path.read_bytes(), file_name=filename, mime="text/csv")
    with st.expander("复现实验与参数"):
        st.json(metadata)
        st.code("python experiments/benchmark.py\npython experiments/benchmark.py --source qdrant --tasks search users --out experiments/results/qdrant", language="bash")
    report = run / "report.md"
    if report.exists():
        st.download_button("下载实验报告", report.read_bytes(), file_name="实验对照报告.md", mime="text/markdown")
