# 算法对照实验

在项目根目录执行：

```powershell
python -m pip install -r requirements-experiments.txt
git lfs pull --include="F2/data/*"
python experiments/benchmark.py
```

结果写入 `experiments/results/local/`：汇总 CSV、逐查询/逐轮 CSV、记录环境和参数的 `metadata.json`、自动生成的 `report.md`。Streamlit 侧栏勾选“查看算法对照实验”即可展示，不会在打开页面时重新训练，也不依赖数据库连接。

默认全量用户、随机 10,000 个被观看过的视频、100 个用户查询，每个查询重复 3 次、每种聚类重复 3 次，20 簇、SVD 64 维，种子 42、单线程。答辩引用结果时必须同时注明规模。

## 相似度搜索

- `exhaustive`：普通基准，同一批归一化向量全量计算余弦相似度，取 Top-K、排除自身。
- `cluster_probe_1/3/5/20`：离线 K-Means 建簇，查询时按离簇中心的距离选若干簇，再精确打分排序。它是本次新增实验方案，不代表 F3/F4 已采用这种路径。更多候选通常减少遗漏，速度必须实测。
- 本地使用现有用户画像的加权标签向量（主领域 3、兴趣领域 2、标签 1，累计后归一化），并非 MPNet 编码。各方法向量与查询完全相同。
- `Recall@K` 对照全量精确 Top-K 集合；同时报告允许第 K 名同分结果替代的召回率，避免相同画像使集合重合率偏低。固定同分排序规则为得分降序、数据行号升序，同分允许 `1e-6` 数值误差。
- `same_field_precision_at_k` 是返回结果中与查询用户主兴趣相同的数量除以 K，只作模拟标签一致性参考，不是真实推荐准确率；返回不足 K 条仍以 K 为分母。
- 预热后，每个查询的方法执行顺序随机打乱；输出平均/P50/P95 耗时，加速比 = 全量平均耗时 / 方法平均耗时。大于 1 倍代表加速，小于 1 倍代表更慢。
- 查询计时包含候选选择、打分和 Top-K，排除编码、数据加载、评估、离线训练；建簇耗时和摊平建簇成本需要的查询数单列。

## 对接已有 Qdrant 语义向量

启动项目原有 Qdrant 并导入 F3 用户向量，再执行：

```powershell
python experiments/benchmark.py --source qdrant --tasks search users --qdrant-url http://localhost:6333 --collection users --out experiments/results/qdrant
```

脚本只读取已有集合，不创建、删除或写入集合。下载同一批向量分别用于全量搜索、聚类候选搜索和服务器检索；运行期间不要修改集合。Qdrant 查询设置 `exact=False`、`hnsw_ef=128`（可调整），计时包含 HTTP 往返。小集合或过滤条件可能触发服务器全量扫描，因此记录集合配置，不单凭 `exact=False` 宣称实测了 HNSW 加速。已有索引历史构建时间未知，表中留空。

## 聚类

用户、视频各比较三种 sklearn 普通算法，不宣称是自创算法：

1. `kmeans_random`：随机初始化 K-Means。
2. `kmeans_plus_plus`：K-Means++ 初始化，原空间基准。两者 `n_init=10`、`max_iter=300` 相同。
3. `svd_kmeans_plus_plus`：SVD 降维、归一化后 K-Means++，对应 F6 的思路；用户侧属于新增方案，F7 原实现是语义向量直接 K-Means++。

视频从 watch 事件构造“视频 × 用户”次数稀疏矩阵，采用 F6 列 IDF 权重。原空间基准归一化后聚类，SVD 分支对加权矩阵降维后归一化，保持 F6 处理顺序。默认均匀抽样 10,000 个视频，IDF 在这一子集内计算，不冒充完整视频库结果。支持 Parquet 或 CSV：

```powershell
# 完整视频库（耗时可能较长）
python experiments/benchmark.py --tasks videos --video-limit 0 --out experiments/results/full_videos
# 仅有 CSV 时
python experiments/benchmark.py --events F2/data/watch_events.csv
```

轮廓系数统一在**原始归一化特征空间**用余弦距离、同一采样集计算，不能直接比较不同表示空间各自的系数。计时包含 SVD/归一化和 K-Means 拟合，排除数据加载、共现矩阵构建和评估；报告每种子的结果及均值/标准差。特征数组大小不是进程峰值内存。用户另报主兴趣 ARI/NMI；主兴趣参与特征构造，所以这只是描述性一致性，不能当独立测试准确率。视频没有可靠单一真实簇标签，对应列留空。

项目原 README 的 F6 轮廓系数 0.1556 来自降维后空间的欧氏距离评估，采样与运行规模也不同，不能直接与本次原空间余弦轮廓系数比较。原记录约 62 秒的计时范围不详，也不能直接作为本次加速比的分母。

真实推荐效果需要按时间切分事件、仅用训练期行为重新构建画像，再用测试期点赞算 Precision/Recall/NDCG@K。当前终态画像包含全程反馈，不能直接拿后期点赞无泄漏评测，因此本实验不报告这一类准确率。

## 验证

```powershell
python -m unittest discover -s experiments/tests -v
```

测试覆盖自身排除、同分、候选遗漏、统一原空间评估及 CLI 端到端产物。元数据记录输入 SHA-256、软件版本、线程数、种子、规模；同机复跑的耗时也可能波动。
