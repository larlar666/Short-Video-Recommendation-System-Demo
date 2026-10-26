# 算法对照实验实测结果

数据来源：local

以下数值为本次实测，具体规模、参数、文件 SHA-256 和环境见 metadata.json。

本地实验使用加权兴趣标签向量；只有 source=qdrant 且包含 qdrant 行时，才是服务器检索实测。

同主兴趣领域比例、ARI/NMI 仅作模拟标签一致性参考，不等于真实用户推荐准确率。

检索 Recall@K 对照全量精确搜索；并列得分会影响集合重合率，因此同时给出允许同分替代的召回率。

聚类轮廓系数统一在原空间、同一采样集合计算；total 包含 SVD，排除评估和加载。

检索速度不含离线建簇；index_build_s、break_even_queries 单列其代价。feature_storage_mib 不是峰值内存。

## search_summary

| method | mean_ms | p50_ms | p95_ms | recall_at_k | tie_aware_recall_at_k | same_field_precision_at_k | mean_candidates | mean_returned | speedup | index_build_s | break_even_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cluster_probe_3 | 4.5669 | 4.4004 | 6.2024 | 1.0000 | 1.0000 | 1.0000 | 1511.0300 | 10.0000 | 1.0650 | 18.6959 | 63026.0363 |
| cluster_probe_20 | 29.6083 | 28.0874 | 42.6535 | 1.0000 | 1.0000 | 1.0000 | 9999.0000 | 10.0000 | 0.1643 | 18.6959 | — |
| exhaustive | 4.8636 | 4.8053 | 5.8768 | 1.0000 | 1.0000 | 1.0000 | 9999.0000 | 10.0000 | 1.0000 | 0.0000 | — |
| cluster_probe_1 | 1.6635 | 1.6149 | 2.1712 | 1.0000 | 1.0000 | 1.0000 | 504.3500 | 10.0000 | 2.9237 | 18.6959 | 5842.3370 |
| cluster_probe_5 | 7.6225 | 7.3016 | 9.2296 | 1.0000 | 1.0000 | 1.0000 | 2512.8700 | 10.0000 | 0.6381 | 18.6959 | — |

## user_clustering_summary

| task | method | mean_total_s | std_total_s | mean_fit_s | mean_preprocess_s | silhouette_original_cosine | ari_primary_field | nmi_primary_field | feature_storage_mib | n | dimensions | speedup_vs_plus_plus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| users | kmeans_plus_plus | 16.4970 | 2.4272 | 16.4970 | 0.0000 | 0.6032 | 1.0000 | 1.0000 | 71.4493 | 10000 | 1873 | 1.0000 |
| users | svd_kmeans_plus_plus | 1.2192 | 0.3075 | 0.3963 | 0.8229 | 0.6032 | 1.0000 | 1.0000 | 2.4414 | 10000 | 64 | 13.5307 |
| users | kmeans_random | 3.4786 | 0.7181 | 3.4785 | 0.0001 | 0.5290 | 0.9201 | 0.9776 | 71.4493 | 10000 | 1873 | 4.7424 |

## video_clustering_summary

| task | method | mean_total_s | std_total_s | mean_fit_s | mean_preprocess_s | silhouette_original_cosine | ari_primary_field | nmi_primary_field | feature_storage_mib | n | dimensions | speedup_vs_plus_plus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| videos | kmeans_plus_plus | 7.7866 | 0.9835 | 7.7866 | 0.0000 | 0.0145 | — | — | 5.5032 | 10000 | 10000 | 1.0000 |
| videos | svd_kmeans_plus_plus | 1.0401 | 0.0754 | 0.6481 | 0.3920 | 0.0158 | — | — | 2.4414 | 10000 | 64 | 7.4864 |
| videos | kmeans_random | 5.3117 | 0.6989 | 5.3117 | 0.0001 | 0.0133 | — | — | 5.5032 | 10000 | 10000 | 1.4659 |
