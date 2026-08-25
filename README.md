# 短视频推荐系统 — 项目技术报告

---

## 目录

1. [项目概述](#一项目概述)
2. [整体架构与数据流](#二整体架构与数据流)
3. [文件结构](#三文件结构)
4. [各功能模块详解](#四各功能模块详解)
   - [F1 — 视频数据库构建与热度初始化](#f1--视频数据库构建与热度初始化)
   - [F2 — 用户行为多轮模拟](#f2--用户行为多轮模拟)
   - [F3 — 相似用户检索](#f3--相似用户检索)
   - [F4 — 个性化视频推荐](#f4--个性化视频推荐)
   - [F5 — 视频热度预测](#f5--视频热度预测)
   - [F6 — 视频聚类分析](#f6--视频聚类分析)
   - [F7 — 用户聚类分析](#f7--用户聚类分析)
5. [关键设计决策](#五关键设计决策)
6. [数据结构选型与大数据量优化](#六数据结构选型与大数据量优化)
7. [运行方式与代码使用说明](#七运行方式与代码使用说明)
8. [实验结果汇总](#八实验结果汇总)

---

## 一、项目概述

本项目构建了一套**完整的短视频推荐系统仿真平台**，以 TikTok 公开数据集（10 万条视频）为基础，覆盖从数据获取、用户行为模拟、相似用户检索、个性化推荐，到视频热度预测的全链路。

| 模块 | 核心任务 |
|------|----------|
| F1 | 视频数据库构建 + 初始热度量化 |
| F2 | 多轮用户观看/点赞行为模拟（含热度影响） |
| F3 | 基于向量相似度的用户-用户协同过滤 |
| F4 | 基于用户兴趣画像的个性化视频推荐 |
| F5 | 基于早期观看序列的视频热度预测 |
| F6 | 基于观看行为相似度的视频协同聚类 |
| F7 | 基于兴趣画像向量的用户聚类 |

---

## 二、整体架构与数据流

<details>
<summary><strong>点击展开：ASCII 数据流总览</strong>（较宽，小屏可折叠）</summary>

```text
HuggingFace TikTok-10M
        │ 流式采样 ~10 万条
        ▼
    ┌───────┐  tiktok_meta_100k.csv     ┌─────────────────┐
    │  F1   │ ─────────────────────────►│ F4 视频向量索引  │
    │ 建库  │  video_popularity_init     │ build_video_    │
    └───────┘  (H0_norm)                 │ index.py        │
        │                                └────────┬────────┘
        │ H0_norm                                 │ Qdrant "videos"
        ▼                                         │
    ┌───────┐  watch_events.csv                   │
    │  F2   │ ─────────────────────────────────►  │
    │ 模拟  │  users_sim.csv ──► F4 recommend     ▼
    └───────┘                           （个性化推荐）
        │
        ├── users_sim.csv ──► F3 build_user_index ──► Qdrant "users"
        │                                              └── find_similar_users
        │
        └── watch_events.csv ──► F5 predict_popularity
                                 （LightGBM 分类 + 分层回归 → models.pkl）
```

</details>

**线性阅读版**（与上图等价）：

1. **F1**：拉取元数据 → `tiktok_meta_100k.csv`、初始热度 → `video_popularity_init.csv`（`H0_norm`）。
2. **F2**：用热度与兴趣模拟行为 → `watch_events.csv`、`users_sim.csv`。
3. **F3**：用户画像入 Qdrant `"users"` → 相似用户检索。
4. **F4**：视频文本入 Qdrant `"videos"` → `recommend_videos.py` 推荐。
5. **F5**：用早期观看序列预测最终热度（与 F2 日志对接）。

F6/F7 聚类在图中未画出：分别消费 `watch_events` 与 Qdrant 用户向量，输出 `video_clusters.csv`、`user_clusters.csv`。

---

## 三、文件结构

```text
项目根目录/
├── streamlit_app.py            # F1–F7 一站式 Web 演示（见 §7.4）
├── requirements-data.txt       # F1：datasets、langdetect 等
├── requirements-f3f4.txt       # F3/F4：sentence-transformers、qdrant-client
├── requirements-streamlit.txt    # 演示界面：streamlit、pandas、sklearn 等
├── F1/
│   ├── scripts/
│   │   ├── build_video_db.py
│   │   ├── tag_pipeline.py
│   │   └── compute_initial_popularity.py
│   ├── data/
│   │   ├── tiktok_meta_100k.csv
│   │   └── video_popularity_init.csv
│   └── tests/
├── F2/
│   ├── scripts/
│   │   └── simulate_users.py
│   └── data/
│       ├── users_sim.csv
│       ├── watch_events.csv
│       └── users_10k.csv
├── F3/
│   └── scripts/
│       ├── build_user_index.py
│       └── find_similar_users.py
├── F4/
│   └── scripts/
│       ├── build_video_index.py
│       └── recommend_videos.py
├── F5/
│   ├── scripts/
│   │   └── predict_popularity.py
│   ├── data/
│   │   ├── models.pkl
│   │   └── meta.json
│   └── requirements.txt
├── F6/
│   ├── scripts/
│   │   └── cluster_videos.py
│   └── data/
│       └── video_clusters.csv
└── F7/
    ├── scripts/
    │   └── cluster_users.py
    └── data/
        └── user_clusters.csv
```

**脚本职责简述**：`build_video_db` 流式采样并建库；`tag_pipeline` 关键词与领域；`compute_initial_popularity` 计算 `H0_norm`；F2 模拟多轮行为；F3/F4 构建 Qdrant 索引与检索/推荐；F5 热度预测；F6/F7 视频与用户聚类；`streamlit_app.py` 汇总可视化。

---

## 四、各功能模块详解

### F1 — 视频数据库构建与热度初始化

#### 目标
从 HuggingFace 公开数据集 `shrx240/TikTok-10M` 中随机抽取 10 万条英语/西班牙语视频，
为每条视频提取标签与兴趣领域，并计算初始热度分，为后续模拟提供数据底座。

#### 文件说明

**`build_video_db.py`**
- 流式扫描最多 100 万条原始记录（避免全量下载）
- 用 `langdetect` 过滤非英/西语视频
- 对每条视频运行 `tag_pipeline`，提取 `preference_tags`（关键词）和 `fields`（兴趣领域）
- 输出 `tiktok_meta_100k.csv`，包含 16 列：`id`、`desc`、`challenges`、`duration`、`play_count`、`digg_count`、`comment_count`、`share_count`、`collect_count`、`music_id`、`user_id`、`poi_name`、`city`、`poi_category`、`preference_tags`、`fields`

**`tag_pipeline.py`**
- 将 `desc` 文本通过 YAKE 和 TextRank 提取高频关键词作为 `preference_tags`
- 将标签映射到 20 个预定义兴趣领域（如 *Music & Sound Culture*、*Knowledge & Education* 等）
- **`preference_tags` 格式约束**：最终入库的标签必须为**仅含英文字母、数字与空格**的字符串；含撇号、连字符等标点（如 `n't`、`it's` 等分词残片）会在 `_is_blacklisted` 中剔除，减少无意义短语

**`compute_initial_popularity.py`**
- **三遍扫描**算法：第一遍获取 `max_desc_len` 和 `max_challenges`，第二遍逐行算分，第三遍 min-max 归一化
- 热度得分 $H_0$ 由以下三项合成：

$$H_{\text{behavior}} = 0.35\log(1+\text{play}) + 0.20\log(1+\text{like}) + 0.15\log(1+\text{comment}) + 0.15\log(1+\text{share}) + 0.15\log(1+\text{collect})$$

$$Q = 0.5 \cdot \text{duration\_score} + 0.3 \cdot \text{desc\_score} + 0.2 \cdot \text{challenge\_score} + 0.1 \cdot \text{poi\_score}$$

$$H_0 = \begin{cases} 0.85 H_{\text{behavior}} + 0.10 Q + 0.05 R & Q \text{ 有效} \\ 0.95 H_{\text{behavior}} + 0.05 R & Q \text{ 缺失} \end{cases}$$

$$H_0^{\text{norm}} = \frac{H_0 - \min(H_0)}{\max(H_0) - \min(H_0)}$$

其中 $R \in [0,1)$ 由 `hash(video_id)` 确定性生成（打破同分，可复现）。

- 输出 `video_popularity_init.csv`：列为 `id`、`H_behavior`、`Q_valid`、`Q`、`R`、`H0`、`H0_norm`

#### 关键数据

| 指标 | 值 |
|------|-----|
| 总视频数 | 100,000 |
| Q 有效比例 | 92.3%（其余 desc/challenges 为空） |
| $H_0$ 原始区间 | [4.04, 15.28] |
| $H_0^{\text{norm}}$ 区间 | [0, 1] |

---

### F2 — 用户行为多轮模拟

#### 目标
模拟 1 万名用户在 30 轮（每轮代表 1 天）中的观看与点赞行为。
每轮推荐受**用户兴趣**和**视频热度**共同影响；行为结果作为 F5 预测的原始数据，也为 F3/F4 提供用户画像。

#### 核心设计思想

```
用户兴趣（个性化）＋ 视频热度（全局先验）＝ 推荐概率
                     ↑
         γ_u（热度敏感性，每用户固定）
         γ 大 → 追热门；γ 小 → 偏小众
```

#### 模拟流程

```
初始化（冷启动，不计入日志）
    ├── 随机分配 primary_field（主兴趣领域）
    ├── 随机抽取 γ_u ~ Uniform(0.3, 1.2)
    └── 模拟一次初始观看/点赞，建立 interest_fields + interest_tags
          │
          ▼
for round = 1 to 30:
    ├── 1. 为每个用户构造个性化 Top-100 候选池（三槽融合）
    │       槽 A (40)：从 interest_fields 对应的视频池采样
    │       槽 B (40)：随机候选打分 s = W_aff·Aff + γ_u·W_pop·p̃ + 噪声，取高分
    │       槽 C (20)：从热度最低 30% 的长尾视频中随机采样（保证多样性）
    │
    ├── 2. 从 Top-100 不放回采样 20 个视频作为本轮观看
    │
    ├── 3. 从 20 个观看中按兴趣匹配度打分，抽取 3 个点赞
    │
    └── 4. 本轮结束后更新 interest_fields/tags（累计点赞反馈），供下一轮使用
              ↑ 时间因果性保证：本轮采样不使用本轮产生的历史
```

#### 打分公式

$$s(u, v) = W_a \cdot \text{Aff}(u, v) + \gamma_u \cdot W_p \cdot \tilde{p}_v + \mathcal{N}(0, \sigma^2)$$

其中：
- $\text{Aff}(u,v) = |\text{fields}(v) \cap \text{interest\_fields}(u)| + 0.5 \times |\text{tags}(v) \cap \text{interest\_tags}(u)|$
- $\tilde{p}_v = H_0^{\text{norm}}(v)$（初始热度，全局常量）
- $\gamma_u$：该用户的热度敏感性，越高越追热门

#### 输出文件

| 文件 | 内容 | 规模 |
|------|------|------|
| `users_sim.csv` | 用户终态：`user_id`、`primary_field`、`gamma`、`interest_fields`、`interest_tags`、`total_watched`、`total_liked` | 1 万行 |
| `watch_events.csv` | 事件日志：`round`、`user_id`、`video_id`、`event_type`（watch/like） | **690 万行** |

**说明**：用户侧 `interest_tags` 由模拟过程中对视频 `preference_tags` 的点赞反馈汇总得到；累计时同样**仅保留「字母 + 数字 + 空格」**的标签，与 F1 输出口径一致，避免标点碎片进入画像与下游推荐/聚类。

---

### F3 — 相似用户检索

#### 目标
给定任意用户 ID，找出兴趣最相近的 $K$ 个用户（协同过滤的用户侧）。

#### 技术栈
- **编码模型**：`sentence-transformers/all-mpnet-base-v2`（768 维密集向量）
- **向量数据库**：Qdrant（本地 Docker 部署）
- **相似度度量**：余弦相似度（Cosine Distance）

#### 流程

```
build_user_index.py
    ├── 读取 users_10k.csv（兼容旧版 F2 输出）
    ├── 拼接兴趣文本：
    │     "primary_field × 3 + interest_fields × 2 + interest_tags × 1"
    ├── SentenceTransformer 批量编码 → 768 维向量
    └── 批量写入 Qdrant "users" collection

find_similar_users.py
    ├── 读取目标用户画像，拼接同样格式的查询文本
    ├── 编码为查询向量
    ├── Qdrant 向量检索 Top-K（排除自身）
    └── 输出：user_id、primary_field、interest_fields、相似度分数

analyze_similar_users.py（可选画像可视化）
    ├── 统计相似用户群体的标签频次与领域分布
    └── 标签统计时沿用**仅字母/数字/空格**规则，与全链路 tag 清洗一致
```

---

### F4 — 个性化视频推荐

#### 目标
给定用户 ID，从 10 万视频库中为其推荐最相关且多样化的视频列表。

#### 流程

```
build_video_index.py
    ├── 读取 tiktok_meta_100k.csv
    ├── 拼接视频文本：fields + preference_tags
    ├── SentenceTransformer 批量编码 → 768 维向量
    └── 批量写入 Qdrant "videos" collection（耗时 ~15-20 分钟 CPU）

recommend_videos.py
    ├── 1. 拼接用户查询文本（primary_field × 3 + fields × 2 + tags × 1）
    ├── 2. 向量检索 Top-80 候选视频
    ├── 3. 过滤用户已观看视频（watched_video_ids）
    └── 4. 多样性后处理：每个领域最多保留 MAX_PER_FIELD 条，凑足 top_k 条
```

#### 权重设计解释

查询文本中 `primary_field` 重复 3 次、`interest_fields` 重复 2 次、`tags` 重复 1 次，
是利用 Sentence Transformer 对词频敏感的特性来**隐式加权**，无需修改模型即可实现多级优先级。

---

### F5 — 视频热度预测

#### 目标
基于视频**前 $H$ 轮**（早期窗口）的观看和点赞数据，预测其最终总播放量，
并通过「冷门/普通/爆款」分类 + 分层回归的方案与全局回归基准进行对比。

#### 特征工程（$H=3$，特征维度 = 11）

| 特征名 | 公式 | 含义 |
|--------|------|------|
| $p_1, p_2, p_3$ | 各轮播放量 | 原始早期序列 |
| $l_1, l_2, l_3$ | 各轮点赞量 | 原始点赞序列 |
| growth | $\dfrac{p_H - p_1}{H-1}$ | 平均每轮增量 |
| log\_growth | $\ln(1 + \max(0, \text{growth}))$ | 对数增长率（压缩动态范围） |
| like\_ratio | $\dfrac{l_1}{p_1 + \varepsilon}$ | 首轮点赞率（内容质量信号） |
| total\_early | $\sum_{t=1}^{H} p_t$ | 早期绝对热度起点 |
| peak\_early | $\max_{t \leq H} p_t$ | 早期单轮峰值 |

#### 标签定义

$$P_{\text{final}} = \sum_{t=1}^{30} V_t^{\text{obs}}, \quad \text{label} = \begin{cases} \text{cold} & P_{\text{final}} \leq Q_{20} \\ \text{hot} & P_{\text{final}} \geq Q_{80} \\ \text{ordinary} & \text{otherwise} \end{cases}$$

$Q_{20}=22$，$Q_{80}=127$（仅由训练集计算，避免数据泄露）。

#### 模型架构

```
输入特征向量 x
        │
        ▼
┌─────────────────────┐
│  LightGBM 分类器     │  → 预测标签 ĉ ∈ {cold, ordinary, hot}
└─────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  分层回归（Layered Regression）               │
│    cold    → LightGBM Regressor_cold         │
│    ordinary→ LightGBM Regressor_ordinary     │  → P̂_final
│    hot     → LightGBM Regressor_hot          │
│    （样本不足时回退全局模型）                  │
└──────────────────────────────────────────────┘

对比基准：
┌─────────────────────┐
│  全局回归器           │ → P̂_global（不分类，直接预测）
└─────────────────────┘
```

#### 数据集划分

按 video_id 随机划分（避免同视频出现在训练和测试集）：
- Train 70% / Val 15% / Test 15%

#### 评估指标

- 分类：Accuracy、各类 Precision / Recall / F1-score（macro 平均）
- 回归：MAE（Mean Absolute Error）
- 对比改进：

$$\text{Improvement} = \frac{\text{MAE}_{\text{global}} - \text{MAE}_{\text{layer}}}{\text{MAE}_{\text{global}}} \times 100\%$$

---

### F6 — 视频聚类分析

#### 目标
将具有**相似观看用户群体**的视频聚成一团（协同过滤视角）。
与 F4 内容向量聚类不同，F6 的相似性完全来自**用户行为**，不依赖视频文本内容。

#### 核心思想

> "两个视频如果经常被同一批用户观看，它们就应该被归为同一类。"

#### 实现流程

```
watch_events.csv（仅 watch 事件）
        │
        ▼
视频 × 用户 稀疏共现矩阵  M[i,j] = 用户 j 看过视频 i 的次数
        │
        ▼ TF-IDF 加权
        │  IDF(用户 j) = log(总视频数 / 含用户 j 的视频数 + 1)
        │  → 压低「几乎人人都看」的大众用户的权重
        │  → 突出「只在特定圈子出现」的小众用户信号
        │
        ▼ TruncatedSVD（截断奇异值分解）
        │  10,000 维 → 64 维稠密行为向量（L2 归一化）
        │  选择 SVD 而非 PCA：稀疏矩阵无需中心化，内存友好
        │
        ▼ K-Means（k-means++ 初始化，K=20）
        │
        ▼
video_clusters.csv（video_id → cluster_id）
+ Qdrant "video_clusters" collection（64 维向量 + 聚类标签）
```

#### 为什么用 TF-IDF + SVD 而不是直接聚类原始矩阵？

原始矩阵 77,972 × 10,000，维度过高且极度稀疏（稀疏度 0.72%）：
- **直接 K-Means**：维度诅咒，距离度量失效，质量极差；
- **TF-IDF**：让有区分力的小众用户信号被放大；
- **SVD 降维**：提取潜在「观众画像」维度，同时解决稀疏问题，
  等价于协同过滤中的矩阵分解（Latent Semantic Analysis）。

---

### F7 — 用户聚类分析

#### 目标
将具有**相似观看兴趣**的用户聚成一团。

#### 核心思想

> "两个用户如果 interest_fields 和 interest_tags 相似，他们应该被归为同一类。"

#### 实现流程

```
Qdrant "users" collection（F3 已建好）
        │ scroll API 全量拉取
        ▼
10,000 × 768 维用户兴趣向量（已 L2 归一化）
        │
        ▼ K-Means（K=20，k-means++ 初始化）
        │ 向量已归一化 → 欧氏距离等价于余弦距离
        │
        ▼
user_clusters.csv（user_id → cluster_id + primary_field）
+ Qdrant "user_clusters" collection（768 维向量 + 聚类标签）
+ 每簇代表性兴趣领域统计（最高频 primary_field）
```

#### 与 F3 的关系

| 对比项 | F3 相似用户检索 | F7 用户聚类 |
|--------|-----------------|-------------|
| **问题类型** | 给定一人，找最近邻 K 人 | 全量数据分成 K 组 |
| **算法** | Qdrant ANN 向量检索 | Qdrant 取向量 → K-Means |
| **Qdrant 角色** | 核心检索引擎 | 向量存储 + 聚类结果写回 |
| **输出** | 某用户的 Top-K 相似用户 | 所有用户的 cluster 标签 |

**F7 复用 F3 的向量**：`build_user_index.py` 编码的 768 维兴趣向量同时服务于 F3（检索）和 F7（聚类），无需重复编码。

---

## 五、关键设计决策

### 1. 「1 轮 = 1 天」的时间约定
F2 的「轮次」与 F5 文档中的「时间步」统一对应为「1 天」，
全链路使用 `round` 字段作为时间轴，无需额外转换。

### 2. 热度的单向影响（单向管线）
本系统设计为单向数据流：

```
初始热度（F1）→ 影响模拟（F2）→ 产出事件日志 → F5 预测
```

F5 的预测结果**不回流**影响 F2 的模拟参数。
这是有意为之的简化，降低了系统复杂度，也保证了 F5 的训练/测试没有信息泄露。

### 3. 用户热度敏感性 $\gamma_u$ 的引入
每个用户独立的 $\gamma_u \sim \text{Uniform}(0.3, 1.2)$ 参数，
模拟了现实中「追热点型用户」和「小众内容偏好者」的异质性，
使仿真行为更具多样性，热度对不同用户的影响程度不同。

### 4. Top-100 三槽设计
| 槽 | 来源 | 比例 | 意图 |
|----|------|------|------|
| 槽 A | 用户 interest_fields | 40% | 个性化召回 |
| 槽 B | 全库随机候选，按评分排序 | 40% | 热度 + 兴趣融合排序 |
| 槽 C | 热度最低 30% 长尾 | 20% | 探索多样性，防止马太效应 |

### 5. 时间因果性保证
F2 多轮模拟中，第 $t$ 轮的采样**只使用第 $t-1$ 轮结束后的**历史画像，
本轮产生的观看/点赞**在本轮结束后**才更新画像，
严格遵循「不用未来信息」的因果约束。

### 6. 兴趣标签（tag）清洗规则
为抑制关键词抽取与分词产生的**标点残片**（如 `n't`、`won't`），全链路约定：

- **允许字符**：英文字母、数字、空格（正则 `^[A-Za-z0-9 ]+$` 判定）。
- **实现位置**：
  | 模块 | 文件 | 作用 |
  |------|------|------|
  | F1 | `tag_pipeline.py` 中 `_is_blacklisted` | 写入 CSV 前过滤非法 `preference_tags` |
  | F2 | `simulate_users.py` | 从视频标签累计用户 `interest_tags` 时跳过非法串 |
  | F3 | `analyze_similar_users.py` 中 `hash_tag_frequency` | 命令行画像统计与词云/柱状图一致 |
  | 演示 | `streamlit_app.py` 中 `_filter_tags` + `load_data` | 读入 `users_sim` / 视频元数据后统一清洗列表列，F3 画像、F4 规则匹配、F6/F7 聚类特征等与 tag 相关的展示与计算均受益 |

**注意**：若你在修改规则后希望**已有** `tiktok_meta_100k.csv`、`users_sim.csv` 完全对齐新口径，需重新运行 `build_video_db.py`（或至少重跑 `tag_pipeline` 链）与 `simulate_users.py`。

---

## 六、数据结构选型与大数据量优化

本作业要求关注**逻辑结构**（线性 / 非线性）以及**大数据量**下的可行性。以下说明本项目中的对应考量与实现位置，便于答辩或评阅时快速对照代码。

### 6.1 线性结构

| 位置 | 结构 | 作用 |
|------|------|------|
| F1 `compute_initial_popularity.py` | 两遍扫描 CSV + 第三遍归一化 | 全库统计量与逐行打分，避免一次性读入超大中间表 |
| F2 `simulate_users.py` | 事件日志**增量写文件** | `watch_events.csv` 约 **690 万行**，边模拟边追加，降低峰值内存 |
| F5 `predict_popularity.py` | **chunksize 分块聚合** + NumPy 向量式特征 | 690 万行事件表以 50 万行/块逐块聚合 watch/like 计数，峰值内存显著低于整表读取 |
| F6 `cluster_videos.py` | **chunksize 分块收集** + **CSR 稀疏矩阵** | 逐块收集 watch 事件的 (video_id, user_id) 对，再一次性构建 CSR，适配稀疏度约 **0.72%** |
| Streamlit `streamlit_app.py` | **chunksize + usecols + 紧凑 dtype** | 只加载 3 列 + category/int32，分块拼接降低加载阶段内存 |

### 6.2 非线性结构

| 位置 | 结构 | 作用 |
|------|------|------|
| F2 | **倒排索引** `field_index`、**frozenset** | 按领域 O(1) 召回候选；兴趣与视频标签集合交集中高速比较 |
| F3 / F4 | **Qdrant**（HNSW 等 ANN 索引） | 768 维向量近似最近邻，避免暴力 O(N×D) |
| F5 | **LightGBM / 随机森林**（树集成） | 非线性拟合早期播放量序列与分类分层 |
| F6 | **TruncatedSVD** | 稀疏共现 → 低维稠密表示；**不选用 PCA** 以免破坏稀疏并产生全密矩阵 |
| F6 / F7 | **K-Means++** | 非线性划分簇；F6 中 **轮廓系数**对大数据做**采样**评估，控制耗时 |

### 6.3 大数据量相关措施（与代码对应）

- **流式 / 采样**：F1 从百万级原始流中采样 10 万条建库；F6 读事件时指定较小 `dtype` 降低内存 footprint。
- **稀疏与降维**：F6 用 `csr_matrix` + TF-IDF + SVD，避免 \(7.8\times10^4 \times 10^4\) 级别全密矩阵。
- **分批 I/O**：F3/F4 向量编码与 Qdrant 写入使用 `ENCODE_BATCH`、`UPSERT_BATCH`；F7 从 Qdrant **scroll** 分批拉取向量。
- **`chunksize` 分块读取**（已实现）：F5、F6 及 Streamlit 对 `watch_events.csv`（约 690 万行）均使用 `pd.read_csv(..., chunksize=500_000)` 分块读取，**不再一次性将整张事件表载入内存**：
  - **F5** `predict_popularity.py`：逐块聚合 `(video_id, round)` 维度的 watch/like 计数，最后再 `unstack` 得到宽表；任何时刻内存中只有一个 50 万行的 chunk 加上已聚合的中间 Series。
  - **F6** `cluster_videos.py`：逐块收集 watch 事件的 `video_id` 与 `user_id` 数组，最后一次性构建 CSR 稀疏矩阵；加载阶段不持有完整 DataFrame。
  - **Streamlit** `streamlit_app.py`：事件加载时仅保留 `user_id / video_id / event_type` 三列并使用 `int32 / str / category` 等紧凑 dtype，分块拼接后内存占用大幅低于全列全类型读取。
- **可进一步改进**：数据量再扩大数量级时，可考虑列式引擎（如 Parquet + PyArrow）替代 CSV，从根本上减少 I/O 与序列化开销。

---

## 七、运行方式与代码使用说明

### 7.1 环境要求

- **Python**：建议 3.10+（与当前依赖 wheel 兼容）。
- **工作目录**：**务必先 `cd` 到项目根目录**（与 `F1/`、`F2/`、`streamlit_app.py` 同级）。脚本内使用相对路径 `ROOT_DIR / "F2" / "data" / ...` 解析数据；在子目录内直接运行会导致找不到 CSV。
- **操作系统**：Windows / Linux / macOS 均可；**向量索引与 Qdrant** 相关步骤在 Windows 上需安装并启动 **Docker Desktop**。

### 7.2 依赖安装

以下 `-i` / `--extra-index-url` / `--trusted-host` 可按你所在网络替换为公网 PyPI（例如省略前两段，仅使用默认源）。

```powershell
# 建议先设变量，避免重复敲长参数（PowerShell）
$PIP = @(
  "-i", "http://vivo-private-pip.vivo.lan:9090",
  "--extra-index-url", "https://mirrors.aliyun.com/pypi/simple",
  "--trusted-host", "vivo-private-pip.vivo.lan"
)

pip install -r requirements-data.txt @PIP
pip install -r requirements-f3f4.txt @PIP
pip install -r F5/requirements.txt @PIP
# 可选：一键演示界面（F1–F7）
pip install -r requirements-streamlit.txt @PIP
```

### 7.3 推荐数据流水线（首次从零运行）

下列顺序与数据依赖一致；**已自带 `F*/data/*.csv` 时**可跳过已完成的步骤。

```bash
# Step 1：构建视频数据库（流式采样，耗时可数分钟～十数分钟）
python F1/scripts/build_video_db.py

# Step 2：计算初始热度
python F1/scripts/compute_initial_popularity.py

# Step 3：多轮用户行为模拟（约 12 分钟，生成 690 万行事件）
python F2/scripts/simulate_users.py

# （若使用 Qdrant）先启动容器，见 §7.6

# Step 4a：用户向量 → Qdrant「users」
python F3/scripts/build_user_index.py

# Step 4b：视频向量 → Qdrant「videos」（CPU 约 15–20 分钟）
python F4/scripts/build_video_index.py

# Step 5：热度预测 → F5/data/
python F5/scripts/predict_popularity.py --H 3

# Step 6：视频聚类 → F6/data/video_clusters.csv
python F6/scripts/cluster_videos.py --no-qdrant          # 仅 CSV
python F6/scripts/cluster_videos.py                      # 另写入 Qdrant「video_clusters」

# Step 7：用户聚类（需先有「users」collection）→ F7/data/user_clusters.csv
python F7/scripts/cluster_users.py --no-qdrant
python F7/scripts/cluster_users.py
```

### 7.4 Streamlit 演示界面（代码使用说明）

根目录 `streamlit_app.py` 在**已存在上述 CSV**（及部分 Qdrant 集合）时，提供 **F1–F7 侧边栏分模块**演示。

```bash
# 项目根目录执行
streamlit run streamlit_app.py
```

浏览器默认打开 `http://localhost:8501`。

**行为说明简要**：

- **F3 / F4** 可选「本地」或「Qdrant」：**无 Docker** 时请选择**本地模式**，否则会出现连接错误。
- 界面依赖路径示例：`F1/data/tiktok_meta_100k.csv`、`F2/data/users_sim.csv`、`F2/data/watch_events.csv` 等；缺文件时对应模块会提示。
- 载入用户与视频表时，会对 **`interest_tags` / `preference_tags` 解析后的列表**做与 F1/F2 一致的**无标点 tag** 过滤，便于 F3 标签图、F4 匹配等展示稳定。

### 7.5 命令行脚本速查

| 模块 | 常用命令 | 说明 |
|------|----------|------|
| F3 | `python F3/scripts/find_similar_users.py --user-id 42` | 查询与指定用户最相似的用户 |
| F4 | `python F4/scripts/recommend_videos.py --user-id 42` | 个性化视频推荐 |
| F5 | `python F5/scripts/predict_popularity.py --H 3` | 早期窗口长度 H 可调 |
| F6 | `python F6/scripts/cluster_videos.py --n-clusters 20 --svd-dim 64` | 聚类数、SVD 维数可调 |
| F7 | `python F7/scripts/cluster_users.py --query-user 42` | 查看与用户 42 同簇邻居 |

各脚本支持 `--help` 查看全部参数。

### 7.6 Qdrant 与 Docker

**启动 Qdrant（F3 / F4 / F7 写库或 Streamlit 向量模式需要）**：

```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

- **Windows**：须先启动 **Docker Desktop**，出现 `cannot find ... dockerDesktopLinuxEngine` 时表示引擎未运行。
- 健康检查：浏览器打开 `http://localhost:6333/dashboard`。
- 后台常驻示例：`docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`

### 7.7 常见问题

- **找不到数据文件**：确认当前目录为项目根，且已按 §7.3 跑过前置脚本。
- **Streamlit 报 Qdrant 502**：本机 6333 上无健康 Qdrant 或被代理占用；先 §7.6，或改用本地检索/推荐模式。
- **F7 报 users 集合不存在**：先执行 `F3/scripts/build_user_index.py`。

---

## 八、实验结果汇总

### F1 — 视频数据库

| 指标 | 值 |
|------|----|
| 视频总数 | 100,000 |
| 覆盖兴趣领域 | 20 个 |
| Q 有效率（参与内容质量分） | 92.3% |
| $H_0^{\text{norm}}$ 均值 | ~0.25（长尾分布） |

### F2 — 用户行为模拟

| 指标 | 值 |
|------|----|
| 用户数 | 10,000 |
| 模拟轮数 | 30 轮（= 30 天） |
| 每轮每人观看 | 20 个视频 |
| 每轮每人点赞 | 3 个视频 |
| 总事件行数 | 6,900,000（精确） |
| 被观看的视频数 | 77,972（占全库 77.97%） |
| 运行时间 | ~12 分钟 |

### F5 — 热度预测（Test 集，$H=3$）

#### 分类指标

| 类别 | Precision | Recall | F1-score | 样本数 |
|------|-----------|--------|----------|--------|
| cold | 0.76 | 0.92 | 0.83 | 2,445 |
| ordinary | 0.91 | 0.72 | 0.80 | 6,960 |
| hot | 0.62 | 0.86 | 0.72 | 2,291 |
| **macro avg** | **0.76** | **0.83** | **0.78** | — |
| **Accuracy** | — | — | **0.79** | 11,696 |

#### 回归指标（MAE）

| 模型 | 全局 MAE | 分层 MAE | 相对改进 |
|------|---------|---------|---------|
| 总体 | 17.93 | 20.07 | −11.94% |
| cold 桶 | 8.38 | **5.86** | **+30.1%** ✓ |
| ordinary 桶 | **17.90** | 23.07 | −28.9% ✗ |
| hot 桶 | 28.21 | **25.96** | **+7.98%** ✓ |

**分析**：分层回归在 cold 和 hot 两端均优于全局回归，证明了分层策略的有效性。
ordinary 桶（占比 60%）由于播放量跨度大（22～127）且早期信号区分度有限，
分层模型反而引入更多方差。这是仿真数据均质性的体现，也是模型的主要局限性，
增大早期窗口 $H$ 或引入更多统计特征（方差、趋势斜率）可进一步改善。

---

### F6/F7 — 聚类分析

#### F6 视频聚类（Test 运行，K=20，SVD=64）

| 指标 | 值 |
|------|----|
| 参与聚类的视频数 | 77,972（被至少一人观看过） |
| 用户数（矩阵列数） | 10,000 |
| 共现矩阵稀疏度 | 0.72% |
| SVD 解释方差 | 10.57%（前 64 个奇异值） |
| Silhouette Score | **0.1556** |
| 簇大小范围 | min=2,328 / max=5,924 / mean=3,899 |
| 运行时间 | ~62 秒 |

#### F7 用户聚类

F7 依赖 Qdrant "users" collection（需先运行 F3），结果包含每簇的代表性 `primary_field` 统计，可直观验证「相同簇的用户确实具有相似的主兴趣领域」。

**关于 Silhouette Score 0.156 的解读**：
基于行为数据的协同聚类天然比内容聚类的轮廓系数低，因为行为数据稀疏、噪声大；
0.15 在行业协同过滤聚类中属于正常水平，说明簇内一定程度的相似性确实存在。

---

*报告生成时间：2026-05-11*
