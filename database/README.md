# Supabase 数据库接入

应用默认使用 Supabase PostgreSQL，连接由 Python 服务端持有。CSV/Parquet 是算法流水线的输入输出和数据库导入源；数据库模式下页面不再读取这些文件。Qdrant 继续负责用户/视频向量与 F3/F4 向量检索，向量不会导入 Supabase。

## 已完成 001、原始事件导入曾超额的项目

先确认旧 `watch_events` 已清空并回收空间（本次项目已确认 0 行、24 KB）。不要修改已经执行过的 `001_initial.sql`，直接执行新增迁移与新导入器：

```powershell
python -m database.migrate
python -m database.import_data --datasets events
python -m database.import_data --datasets predictions video_clusters user_clusters
python -m streamlit run streamlit_app.py
```

第一条应显示 `已应用 002_compact_events.sql`。新的 `events` 导入不再写入旧的 `watch_events`。它在本机临时 SQLite 中流式去重、核对视频/用户，再一次事务提交两类紧凑数据；临时文件自动清理。整理过程可能需要几百 MB 本机临时空间，期间先显示“本地处理”，随后上传约一万条用户历史和数万条视频统计。

默认数据库软上限为 **450 MB（十进制）**，在上传前估算增量、每批前后查询实际数据库大小。超过保护值会回滚本数据集，不自动提高上限。估算并非 PostgreSQL 最终占用，磁盘中的 WAL、临时空间和其他后台写入也可能触发提供方限制，因此仍应查看 Dashboard 用量；不能保证任意数据集都适配免费额度。已有数据明显超额时先处理占用，不要反复重跑。

本次本地数据完整校验结果：6,900,000 条源记录中有 13 条完全重复事件，去重后为 5,999,988 次观看与 899,999 次点赞；整理为 10,000 条用户历史、77,972 条视频统计。配对数组原始元素约 76.2 MB（不含 PostgreSQL 行、字典与索引开销），本机临时 SQLite 峰值文件约 199 MB。此结果是离线校验，不等于云端实际占用；导入完成后命令会显示实际数据库大小。

原始事件表存在非空已提交数据时，002 会拒绝执行，避免悄悄把旧历史切换成空结果。应先导出并确认数据来源，再处理旧表，不要盲目执行 TRUNCATE。

## F5 连接超时更新

页面查询现在使用带健康检查的连接池（最多 3 个连接），在 Streamlit 重跑和 F4/F5 切换时复用连接，减少每次新建 PostgreSQL 连接的握手。迁移、导入脚本仍使用独立连接。更新后执行 `python -m pip install -r requirements-db.txt` 并重启 Streamlit。此次不需要迁移或重新导入数据。

F5 分别报告“读取真实观看统计失败”和“预测结果处理失败”；诊断区显示异常类型和 SQLSTATE，不显示数据库连接串。终端记录完整异常。真实观看查询失败时，仍显示已加载的预测结果和曲线；真实总播放量显示“暂不可用”，不会错误显示为 0，也不会绘制虚假的真实曲线。连接池等待上限 20 秒；网络完全不可达时仍会失败，需要恢复可用网络，不应据此重新导入数据。

## 1. 创建项目并配置连接

1. 在 https://supabase.com/dashboard 创建项目，记录数据库密码。
2. 点击项目的 **Connect**，复制 **Session pooler** 的 PostgreSQL URI（通常端口 5432）。它支持 IPv4，适合本机运行 Python。也可使用网络支持的 Direct connection。请不要使用 REST 项目 URL、anon key 或 service_role key 代替数据库连接串。
3. 在项目根目录执行：

```powershell
Copy-Item .env.example .env
python -m pip install -r requirements.txt
```

编辑 `.env`，将 `DATABASE_URL` 替换为完整连接串，保留 `DATA_BACKEND=supabase`。密码中的 `@`、`#` 等特殊字符需要 URL 编码；保留 `sslmode=require`。`.env` 已加入 Git 忽略。连接串仅供 Python 服务端使用，不放到浏览器或提交到仓库。

## 2. 建表

```powershell
python -m database.migrate
```

迁移脚本在事务中执行，记录版本和 SHA-256；可重复执行，已经应用的 SQL 不重复建表。后续变更新增 SQL 文件，不编辑已应用的文件。执行数据库迁移与导入使用项目数据库管理员连接。

表放在 `recommender` schema，Table Editor 中选择该 schema 查看。**不要把此 schema 加到 Data API 的 exposed schemas 中。** 应用走 PostgreSQL 连接；业务表启用了 RLS，没有浏览器公开访问策略或 PUBLIC 权限。示例的模拟用户不是 Supabase Auth 登录用户。

## 3. 校验并导入已有数据

在项目根目录执行：

```powershell
# 完整扫描文件，检查列、ID、JSON 和数值；不连接数据库
python -m database.import_data --validate-only

# 按外键依赖顺序导入全部数据
python -m database.import_data

# 也可分开执行；子表需要先有对应视频/用户
python -m database.import_data --datasets videos users popularity
python -m database.import_data --datasets events
python -m database.import_data --datasets predictions video_clusters user_clusters
```

文件尚为 Git LFS 指针时先 `git lfs pull`，或重新运行对应算法脚本。事件 CSV 缺失或为 LFS 指针时，导入器会尝试同目录的真实 `watch_events.parquet`；真实 CSV 存在时优先使用 CSV。其他文件缺失会明确报错，按需用 `--datasets` 选择要导入的文件。

普通数据集通过 PostgreSQL COPY 分批进入临时表，再按主键 upsert。每个数据集一个事务；失败则该表本次导入全部回滚，已成功的其他数据集保留。原始文件 SHA-256 与处理行数记录在 `recommender.import_runs`，相同文件再次执行跳过。用户/视频重复 ID 更新。紧凑事件则在本机以 `(user_id, round, video_id, event_type)` 去重；同一视频在不同轮次被观看仍保留，同轮观看和点赞也是不同事件，不采样、不截断历史。

`event_video_keys` 用稳定的整数代号对应完整 text 视频 ID；`user_event_history` 的配对整数数组保存视频代号和轮次，观看与点赞分开保存；`video_round_stats` 的配对数组保存每个视频的轮次与观看/点赞计数。导入器校验引用的用户和视频存在，字典本身另有视频外键；历史数组元素不是逐条 SQL 外键，依赖导入器校验。用户历史和视频统计的两份总量必须一致才会上传。配对数组保留 F4 的全部观看历史、重复观看次数以及 F5 的真实播放曲线。

普通表采用**增量 upsert**：新文件不包含的旧行不会自动删除。紧凑事件采用**整份快照替换**：文件变化后，两类历史/统计在同一个事务中替换，不会叠加旧实验；失败时先前快照保留。`event_snapshot` 标记当前已完成的源文件、去重事件数、观看/点赞总量；相同当前文件跳过，审计名称为 `events_compact_v1`。导入期间请保持源文件不变，避免同时运行多份导入任务。

所有数据集默认每批 2000 行、450 MB 软上限，均有容量保护。`--batch-size` 可调小以减少临时表和检查间隔的峰值；这不会改变最终需要的总容量。`--max-db-mb` 仅在确认真实套餐容量后调整，不可用提高参数的方式绕过免费额度。F4/F5 需要完成紧凑事件导入；没有完成快照时会提示错误，不把“未导入”显示为“没有观看记录”。

## 4. 启动应用

```powershell
python -m streamlit run streamlit_app.py
```

页面显示“Supabase / PostgreSQL 数据服务”。视频、用户、预测和聚类结果从数据库加载；F4 按用户主键读取并展开历史，F5 按视频读取已核对的每轮计数，不扫描数百万条原始事件。数据缓存五分钟，导入后点击侧栏“刷新数据”。数据库错误时显示配置提示，不自动降级读取本地文件。F3/F4 选择“Qdrant 向量”即可继续使用原向量库；启动与建索引步骤沿用根 README 的 Qdrant 说明。

离线演示可显式设置 `.env` 中 `DATA_BACKEND=csv`。CSV 模式沿用原有页面逻辑，仍需真实文件。

## 表设计

| 表 | 主键 / 用途 |
|---|---|
| videos | text 视频 ID，元数据、JSONB 领域和标签 |
| users | bigint 模拟用户 ID，兴趣画像和累计行为数 |
| video_popularity | 视频 ID 外键，初始热度 |
| watch_events | 001 遗留空表，不再承接新导入和页面查询 |
| event_video_keys | 整数代号 ↔ 完整视频 ID 字典，视频 ID 外键 |
| user_event_history | 用户 ID 外键，按用户打包观看和点赞历史 |
| video_round_stats | 视频代号外键，按视频打包各轮观看/点赞计数 |
| event_snapshot | 当前已提交行为快照及去重总量 |
| video_predictions | 视频 ID 外键，可变预测窗口指标 JSONB |
| video_clusters / user_clusters | 视频/用户 ID 外键，聚类编号 |
| import_runs / schema_migrations | 数据导入审计 / 迁移版本 |

`videos.user_id` 是原数据的创作者 ID，不关联模拟的 `users.user_id`。TikTok 视频 ID 保留为 text，整数代号仅供历史数组内部引用，避免浮点转换丢精度。F4 走用户主键、F5 走视频 ID 唯一索引和统计主键，避免为每次行为分别维护索引。视频领域另有 GIN 索引，聚类编号与播放量有普通索引。

当前改造覆盖应用的数据存储和读取。F1–F7 离线算法脚本仍产出原有文件，重新计算后运行导入命令同步；尚未把这些算法改为直接写数据库，也未将 Qdrant 迁移到 pgvector。视频媒体本身没有上传到 Supabase Storage。

## 检查与测试

```powershell
python -m unittest discover -s database/tests -v
python -m database.import_data --validate-only
# 校验完整事件去重与数组还原、统计一致性，只使用本地临时文件
python -m database.compact_events
```

连接好数据库后，可以在 Supabase SQL Editor 查看：

```sql
select * from recommender.schema_migrations;
select dataset, row_count, imported_at from recommender.import_runs order by imported_at desc;
select count(*) from recommender.videos;
select count(*) from recommender.users;
select * from recommender.event_snapshot;
select count(*) from recommender.user_event_history;
select count(*) from recommender.video_round_stats;
select pg_size_pretty(pg_database_size(current_database())) as database_size;
select relname, pg_size_pretty(pg_total_relation_size(relid)) as total_size
from pg_stat_user_tables where schemaname='recommender'
order by pg_total_relation_size(relid) desc;
```

参考：[Supabase 官方连接指南](https://supabase.com/docs/guides/database/connecting-to-postgres)。
