# AI 论文评测与检索平台

> 一个能"读懂"论文的搜索引擎：上传 PDF → AI 自动评分 → 关键词+语义双引擎检索

---

## 目录

1. [这个项目是干什么的](#一这个项目是干什么的)
2. [整体架构图（宏观）](#二整体架构图宏观)
3. [核心数据流：一次搜索的完整旅程](#三核心数据流一次搜索的完整旅程)
4. [各模块详解](#四各模块详解)
5. [索引是怎么建起来的](#五索引是怎么建起来的)
6. [搜索引擎的原理](#六搜索引擎的原理)
7. [Pipeline 五大坑 & 修复方案](#七pipeline-五大坑--修复方案)
8. [目录结构](#八目录结构)
9. [快速启动](#九快速启动)

---

## 一、这个项目是干什么的

**场景**：你有一堆 AI/CS 论文（PDF），想随时用自然语言搜索、对比、发现好论文。

**这个平台做了三件事**：

```
1. 评测  —— 上传论文 PDF → GPT 自动评分（创新性、方法论、数据集、结论）
2. 检索  —— 输入任意问题 → 双引擎（关键词+向量）找最相关论文
3. 推荐  —— 看完一篇 → 自动推荐相似论文
```

类比：把它想象成**专门针对 AI 论文的 Google Scholar + 小红书**。

---

## 二、整体架构图（宏观）

```
                        ┌─────────────────────────────────────────────┐
                        │              用 户 浏 览 器                  │
                        │         http://localhost:8000                │
                        └────────────────┬────────────────────────────┘
                                         │ HTTP / WebSocket
                                         ▼
                        ┌─────────────────────────────────────────────┐
                        │         Go API 网关  (Gin 框架)              │
                        │  ·  限流：每IP每秒最多20个请求              │
                        │  ·  鉴权：验证 JWT Token                    │
                        │  ·  路由：把请求分发到正确的服务            │
                        │  ·  WebSocket Hub：实时推送通知              │
                        └────────┬────────────────────┬───────────────┘
                                 │ gRPC                │ HTTP
                                 ▼                     ▼
              ┌──────────────────────────┐   ┌─────────────────────────┐
              │   Python 搜索服务         │   │   Python FastAPI 服务    │
              │   (grpc_server.py)       │   │   (api_server.py)       │
              │                          │   │                         │
              │   BM25 关键词检索         │   │   论文详情 / 评分       │
              │   FAISS 向量检索          │   │   评论 / 点赞           │
              │   Cross-Encoder 精排      │   │   用户注册 / 登录       │
              └──────────┬───────────────┘   └──────────┬──────────────┘
                         │                              │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
          ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
          │  Redis 缓存层     │ │  Kafka 事件总线   │ │  SQLite 数据库   │
          │                  │ │                  │ │  (papers.db)     │
          │  搜索结果缓存5分钟 │ │  用户行为异步写   │ │  论文元数据       │
          │  热度榜 ZSet      │ │  不阻塞搜索请求   │ │  评分 / 评论      │
          │  论文详情缓存1小时 │ │                  │ │  用户数据         │
          └──────────────────┘ └──────────────────┘ └──────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────────┐
                        │       Prometheus + Grafana          │
                        │   监控：延迟 / 缓存命中率 / 消费积压   │
                        └─────────────────────────────────────┘
```

**一句话理解**：用户的每个请求经过 Go 网关 → 搜索服务处理 → Redis 先查缓存，没有再查数据库和索引 → 行为数据异步写 Kafka，不影响响应速度。

---

## 三、核心数据流：一次搜索的完整旅程

### 3.1 搜索请求的旅程（ASCII 序列图）

```
用户输入: "graph neural network node classification"
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1：Go 网关收到请求                                          │
│                                                                  │
│  检查频率：这个IP最近1秒发了几次？> 20 次 → 直接拒绝（限流）       │
│  检查身份：Token 对不对？→ 不对 → 401 Unauthorized               │
│  转发给搜索服务（gRPC 协议，比 HTTP 快3-5倍）                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2：Redis 缓存检查                                           │
│                                                                  │
│  key = "search:{MD5(query)}:{mode}"                             │
│  命中 → 直接返回（< 1ms，跳过所有后续步骤）                       │
│  未命中 → 继续往下走                                              │
└──────────────────────────┬───────────────────────────────────────┘
                           │ 未命中
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3：双引擎并行检索                                           │
│                                                                  │
│  ┌────────────────────┐    ┌────────────────────────────────┐   │
│  │  BM25 关键词引擎    │    │  FAISS 向量引擎                 │   │
│  │                    │    │                                │   │
│  │  把查询分词          │    │  调用 OpenAI API               │   │
│  │  ["graph","neural" │    │  把查询变成 1536 维向量          │   │
│  │   "node",...]      │    │                                │   │
│  │                    │    │  在常驻内存的 FAISS 索引中       │   │
│  │  对每篇论文计算      │    │  做余弦相似度搜索               │   │
│  │  BM25 分数          │    │  → 返回分数最高的 N 篇          │   │
│  │  → top 候选篇       │    │                                │   │
│  └────────┬───────────┘    └───────────────┬────────────────┘   │
│           │                                │                    │
│           └───────────────┬────────────────┘                    │
│                           │                                     │
│              加权融合：0.3 × BM25 + 0.7 × FAISS                  │
│              质量门控：原始分 < 0.10 → 整批丢弃（防假高分）         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4：Cross-Encoder 精排（可选）                               │
│                                                                  │
│  Bi-Encoder（FAISS）：快但粗糙，相邻排名分差可能只有 0.03          │
│  Cross-Encoder：把查询和每篇论文拼在一起重新打分，精度高           │
│                                                                  │
│  top-50 候选 → Cross-Encoder 重排 → 取 top-10                    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 5：写缓存 & 写行为日志                                      │
│                                                                  │
│  结果写入 Redis（TTL=5分钟），下次相同查询直接命中                  │
│  "搜索"行为发给 Kafka（异步），Consumer 批量更新热度分             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
                    返回搜索结果 JSON
```

### 3.2 论文从 PDF 到可搜索的旅程

```
  PDF 文件
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  fetch_papers.py  （数据采集）                       │
│                                                     │
│  · 解析 PDF，提取标题、作者、摘要（abstract）          │
│  · 调用 arXiv API 获取原始摘要（比 PDF 提取更准）      │
│  · 领域过滤：不是 CS/AI 论文 → 丢弃（防数据污染）      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  paper_rubric_eval.py  （GPT 评测）                  │
│                                                     │
│  · 用 LangGraph 7节点流水线评测                       │
│  · 评分维度：研究问题、方法、数据集、结论、创新性       │
│  · 低分论文触发"重评"环节（质量兜底）                  │
│  · 结果写入 SQLite（papers.db）                      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  build_index.py  （建索引）                          │
│                                                     │
│  BM25 索引：                                        │
│  · 拼接 标题 + 摘要 + 方法 + 数据集 字段             │
│  · rank_bm25 在内存中实时建（每次搜索重建，很轻量）    │
│                                                     │
│  FAISS 索引：                                       │
│  · 用 OpenAI text-embedding-3-small 生成 1536维向量  │
│  · 存入 faiss.IndexFlatIP（内积=余弦，因为已归一化）  │
│  · 序列化到磁盘 paper_index_api.faiss               │
│  · 启动时加载一次，常驻内存（搜索 < 5ms）             │
└─────────────────────────────────────────────────────┘
```

---

## 四、各模块详解

### 4.1 数据层：SQLite (`papers.db`)

| 表名 | 存什么 | 关键字段 |
|------|--------|---------|
| `papers` | 每篇论文的评测结果 | paper（标题/主键）、abstract、paper_quality_score、methodology、domain |
| `comments` | 用户评论 | paper、nickname、content、likes |
| `users` | 注册用户 | email、password_hash、nickname |
| `user_behaviors` | 用户行为日志 | user_id、paper、action（view/like/search） |
| `paper_images` | 论文配图 | paper、image_path、caption |

**为什么用 SQLite 而不是 PostgreSQL？**
当前数据量（~150篇）完全够用。架构已预留 PostgreSQL 迁移脚本（`migrate_to_postgres.py`），到万级用户时直接切换。

---

### 4.2 搜索层：BM25 + FAISS 混合检索

```
BM25（关键词派）                    FAISS（语义派）
─────────────────                  ────────────────────
"GraphSAGE" → 精确命中              "图上的消息传递"→ 能理解意思
专有名词效果好                       同义词/描述效果好
不理解语义                           不区分专有名词拼写

两者互补，加权融合：
  最终分 = 0.7 × FAISS分 + 0.3 × BM25分
```

**质量门控**（防止垃圾结果混入）：
- FAISS 最高余弦相似度 < 0.10 → 整批结果丢弃（意味着数据库里没有相关论文）
- BM25 最高原始分 < 1.0 → 整批结果丢弃（没有有效的关键词匹配）

---

### 4.3 缓存层：Redis 三种用法

```
┌─────────────────────────────────────────────────────────────┐
│  String（字符串）：搜索结果缓存                               │
│  key: "search:a3f2c9d1:hybrid"                             │
│  value: JSON 搜索结果                                       │
│  TTL: 5分钟（搜索结果不需要太新）                             │
├─────────────────────────────────────────────────────────────┤
│  ZSet（有序集合）：实时热度榜                                 │
│  member: 论文标题                                           │
│  score: 点赞×3 + 评论×2 + 浏览×1 + 质量分×10                │
│  查 Top-N：O(log N) 极快                                    │
├─────────────────────────────────────────────────────────────┤
│  Hash（哈希）：论文详情缓存                                  │
│  key: "paper:{paper_id}"                                   │
│  field: title, abstract, score, ...                        │
│  TTL: 1小时（更新后手动删除让它重建）                         │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.4 异步层：Kafka 事件总线

**为什么需要 Kafka？**

没有 Kafka 的情况：
```
用户搜索 → 搜索逻辑 → 写行为日志到SQLite → 更新热度分 → 返回结果
                      ↑ 这两步占了20ms，让用户等
```

有 Kafka 之后：
```
用户搜索 → 搜索逻辑 → 发消息到Kafka → 立即返回结果（快20ms）
                           ↓
                      Kafka Consumer（后台）
                           ↓
                      批量写SQLite + 更新热度分
```

三个 Topic（消息频道）：
- `user-behaviors`：浏览/点赞/搜索行为
- `hot-score-delta`：热度分增量
- `eval-complete`：评测完成通知

---

### 4.5 网关层：Go Gin

**为什么用 Go 而不是 Python 来做网关？**

Python 有 GIL（全局解释器锁），同一时刻只有一个线程在真正运行代码。
Go 的 goroutine 极其轻量（2KB 初始栈），单机可以同时处理 10 万个并发连接。

网关做的三件事：
1. **限流**：每个 IP 每秒最多 20 个请求，超过直接返回 429
2. **鉴权**：解析 JWT Token，验证用户身份
3. **路由**：把 `/search` 转发给搜索服务，把 `/paper` 转发给 FastAPI

---

### 4.6 监控层：Prometheus + Grafana

关键指标一览：

| 指标名 | 含义 | 报警阈值 |
|--------|------|---------|
| `search_latency_seconds{mode="hybrid"}` | 混合检索 p99 延迟 | > 500ms |
| `cache_hit_total{type="search"}` | Redis 搜索缓存命中率 | < 60% |
| `kafka_consumer_lag` | 行为日志消费积压 | > 1000 条 |
| `faiss_index_search_ms` | 向量检索耗时 | > 50ms |
| `active_websocket_connections` | 实时连接数 | 监控趋势 |

---

## 五、索引是怎么建起来的

### FAISS 向量索引

```
每篇论文
   │
   ▼
build_text()：拼接文本
   优先用 abstract（原始摘要，信息量最大）
   没有摘要时才用 GPT 提取的字段（备用）
   │
   ▼
OpenAI text-embedding-3-small
   把文字 → 1536 维浮点向量
   │
   ▼
L2 归一化（faiss.normalize_L2）
   向量变成单位向量（长度=1）
   这样内积 = 余弦相似度
   │
   ▼
faiss.IndexIDMap2(IndexFlatIP(1536))
   用论文的 rowid 作为 ID
   存入索引
   │
   ▼
保存到磁盘 paper_index_api.faiss
   启动时加载一次，常驻内存
   搜索时无需磁盘 I/O（< 5ms）
```

### BM25 索引

BM25 不需要提前建索引存磁盘，每次搜索时临时建（150篇，毫秒级）：

```
搜索时：
   所有论文记录 → build_corpus() → 每篇拼成一段文字
                                        │
                                        ▼
                               BM25Okapi(tokenized_corpus)
                               自动计算每个词的 IDF 值
                               │
                               ▼
                          get_scores(tokenize(query))
                          返回每篇论文的 BM25 分数
```

---

## 六、搜索引擎的原理

### FAISS 怎么算相似度？

FAISS 用**余弦相似度**（因为向量已归一化，内积 = 余弦）：

```
query: "how to handle long-tail distribution in recommendation"
   ↓ embedding
query_vec = [0.02, -0.15, 0.08, ..., 0.11]  ← 1536个数字

每篇论文也有自己的向量：
paper_vec = [0.03, -0.12, 0.09, ..., 0.13]

相似度 = query_vec · paper_vec（点积）
       = 0.02×0.03 + (-0.15)×(-0.12) + ...
       = 0 表示完全不相关，1 表示完全一样

暴力计算 151篇 × 1536维 ≈ 23万次乘法，约 5ms，可接受
```

### BM25 怎么算相关度？

BM25 核心思想：**罕见词命中奖励高，常见词命中奖励低**

```
查询："GraphSAGE node classification"
分词：["graphsage", "node", "classification"]

对每个词 t，计算论文 d 的贡献：

  score(t, d) = IDF(t) × tf_score(t, d)

  IDF(t) = log((总篇数 - 含t篇数 + 0.5) / (含t篇数 + 0.5) + 1)
         GraphSAGE：只有2篇提到 → IDF 高 → 权重大
         node：50篇提到 → IDF 低 → 权重小

  tf_score 还会对文档长度归一化（短文档里出现1次 = 长文档里出现多次）

  论文总分 = Σ score(t, d) 对所有查询词求和
```

### 为什么要两个引擎混合？

| 查询类型 | BM25 效果 | FAISS 效果 |
|---------|----------|-----------|
| `GraphSAGE` 专有名词 | 好（精确匹配） | 一般（依赖语义向量） |
| `图上的消息传递` 语义描述 | 差（没有这些词） | 好（理解语义） |
| `graph neural network` 通用术语 | 中 | 中 |

混合 = 两者优势互补，覆盖更多查询类型。

---

## 七、Pipeline 五大坑 & 修复方案

这是在 164 篇语料库上发现的系统性问题，每个都有真实的错误案例：

### 坑 1：搜 GNN 返回流体力学论文
**根因**：上传时没有领域校验，非 CS 论文混入了向量空间  
**现象**：搜 "graph neural network" 前10结果里有流体力学论文  
**修复**：`build_index.py` 加入 `is_cs_paper()` 白名单过滤

```python
# 修复前：所有论文无差别建索引
for record in records:
    vec = encode(build_text(record))
    index.add(vec)

# 修复后：只有 CS/AI 论文进索引
for record in records:
    if is_cs_paper(record):   # ← 关键一行
        vec = encode(build_text(record))
        index.add(vec)
```

### 坑 2：向量质量差，搜索不准
**根因**：Embedding 用的是 GPT 压缩后的字段（300词→60词），专有名词被泛化  
**现象**：GraphSAGE 论文的向量里没有 "GraphSAGE" 这个词，搜不到  
**修复**：优先用原始摘要 (`abstract`) 建向量

```python
# 修复前：用 GPT 提取的压缩字段
def build_text(record):
    return f"{title}. {research_question} {methodology}"

# 修复后：优先用原始摘要
def build_text(record):
    abstract = record.get("abstract", "").strip()
    if abstract:
        return f"{title}. {abstract}"   # ← 信息量最大的字段
    return f"{title}. {research_question} {methodology}"  # 降级
```

### 坑 3：无精排，召回即返回
**根因**：Bi-Encoder（FAISS）训练目标是快速召回，排名精度不高  
**现象**：rank 3 和 rank 8 的原始分差只有 0.03，但相关性差距很大  
**修复**：top-50 候选再用 Cross-Encoder 精排

```
Bi-Encoder（FAISS）→ 快速召回 top-50 → Cross-Encoder → 精排返回 top-10
    快，但不精                                  慢，但精
    毫秒级                                      几百毫秒
```

### 坑 4：相对归一化让垃圾分数看起来很高
**根因**：除以 max 之后，即使最高原始分只有 0.08 也变成 1.0  
**现象**：完全不相关的查询也能返回 "高分" 结果  
**修复**：加绝对质量门控

```python
FAISS_MIN_QUALITY = 0.10  # 余弦相似度阈值

if max_score < FAISS_MIN_QUALITY:
    return {}   # 整批丢弃，不归一化，不返回垃圾结果
```

### 坑 5：BM25 关键词匹配失效
**根因**：BM25 语料用 GPT 改写字段，GraphSAGE 变成了 "graph sampling method"  
**现象**：搜 "GraphSAGE" 时 BM25 完全找不到那篇论文  
**修复**：build_corpus() 加入 abstract 字段

```python
# 修复后：原始摘要排在最前，专有名词完整保留
parts = [
    r.get("paper", ""),      # 标题
    r.get("abstract", ""),   # ← P5修复：原始摘要，GraphSAGE等词完整
    r.get("methodology", ""),  # GPT字段（备用）
    ...
]
```

**核心教训**：
> 最大的问题不是算法，是数据。专有名词丢了、摘要被压缩了、领域外论文混进来了——
> 用最好的 Rerank 模型也救不回来。**先清洗数据 → 保留原始文本 → 加阈值兜底 → 最后才是 Cross-Encoder。**

---

## 八、目录结构

```
agenteval_test01/
│
├── scripts/                    # 核心逻辑
│   ├── api_server.py           # FastAPI 服务（端口 8000）
│   ├── hybrid_search.py        # BM25 + FAISS 混合检索引擎
│   ├── build_index.py          # 构建 FAISS 向量索引
│   ├── db.py                   # SQLite 数据库操作
│   ├── paper_rubric_eval.py    # GPT 论文评测（LangGraph 流水线）
│   ├── fetch_papers.py         # PDF 解析 + arXiv 摘要采集
│   ├── eval_metrics.py         # 评估指标（搜索质量诊断）
│   ├── hot_score.py            # Redis 热度榜
│   ├── kafka_producer.py       # 行为事件发送
│   ├── kafka_consumer.py       # 行为事件消费
│   ├── grpc_server.py          # gRPC 搜索服务
│   ├── metrics.py              # Prometheus 埋点
│   └── migrate_to_postgres.py  # PostgreSQL 迁移脚本（备用）
│
├── gateway/                    # Go API 网关
│   ├── main.go                 # Gin 路由 + 限流 + 鉴权
│   ├── hub.go                  # WebSocket 实时推送
│   └── prom.go                 # 网关侧 Prometheus 指标
│
├── tests/                      # 测试文件
│   ├── test_pipeline_fixes.py  # 五大修复的单元测试（32个）
│   ├── test_eval_metrics.py    # 评估指标测试（27个）
│   ├── test_faiss_inmemory.py  # FAISS 内存加载测试
│   ├── test_redis_cache.py     # Redis 缓存测试
│   ├── test_hot_score.py       # 热度榜测试
│   ├── test_kafka.py           # Kafka 事件总线测试
│   ├── test_grpc.py            # gRPC 接口测试
│   └── test_metrics.py         # 监控埋点测试
│
├── data/                       # 数据文件（git ignore）
│   ├── papers.db               # SQLite 数据库
│   └── paper_index_api.faiss   # FAISS 向量索引
│
├── docs/                       # 文档
│   └── 新项目.md               # 架构设计与实施进度
│
├── frontend/
│   └── index.html              # 前端页面
│
├── docker-compose.yml          # 一键启动所有服务
├── Dockerfile.api              # Python 搜索服务镜像
├── Dockerfile.gateway          # Go 网关镜像
└── requirements.txt            # Python 依赖
```

---

## 九、快速启动

### 方式一：Docker Compose（推荐，一键启动）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

# 2. 启动所有服务
docker compose up -d

# 3. 查看服务状态
docker compose ps
```

启动后访问：
- 前端界面：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- Grafana 监控：`http://localhost:3000`（admin/admin）

### 方式二：本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python scripts/db.py

# 构建 FAISS 索引（需要 OpenAI Key）
python scripts/build_index.py --mode api

# 启动搜索服务
python scripts/api_server.py

# （可选）启动 Go 网关
cd gateway && go run .
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 只测试搜索 Pipeline 修复
python -m pytest tests/test_pipeline_fixes.py -v

# 诊断搜索质量
python scripts/eval_metrics.py --verbose
```

### 常见问题

**Q：FAISS 索引和数据库不同步怎么办？**
```bash
# 重建索引（会调用 OpenAI API）
python scripts/build_index.py --mode api
```

**Q：搜索结果质量差怎么诊断？**
```bash
# 运行评估报告
python scripts/eval_metrics.py --verbose

# 清理非 CS/AI 论文
python scripts/eval_metrics.py --purge-ood
```

**Q：abstract 字段为空影响搜索质量怎么解决？**
```bash
# 重新抓取 arXiv 摘要
python scripts/fetch_papers.py --fill-abstract
```

---

*文档更新：2026-04-16 | 数据规模：151篇 CS/AI 论文 | 架构版本：v2.0（五大 Pipeline 修复后）*
