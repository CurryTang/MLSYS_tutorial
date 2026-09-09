# System Design 01 · 无状态服务

课程位置：本篇 → [[SystemDesign01B Virtualization Containers|01B 虚拟化与容器]]

先组件，后案例。组件讲接口和挡住什么问题。案例用同一套模板把组件拼起来。

```system-design-overview-visual
```

```text
案例
1. Functional requirements × 3
2. Non-functional requirements × 3
3. Basic workflow + schema + QPS
4. Deep dive × 3：每个 ≥2 方案，tradeoff，preferred
5. 一条 end-to-end

估算
1 day ≈ 1e5 seconds
avg QPS ≈ daily requests / 1e5
peak QPS ≈ avg × peak factor
concurrency ≈ QPS × latency_seconds
DB QPS ≠ API QPS
```

只算会改变形态的量：cache、replica、shard、队列。

---

Stateless 不是“系统没有状态”。它指服务进程不独占任何不可丢失的状态。实例收到输入，读取外部状态，完成计算，再把结果写回共享系统或返回客户端。

```text
replaceable instance
= code + config + request context + rebuildable cache
```

如果杀掉任意一台实例会丢 session、文件或业务进度，这一层还不够无状态。

---

## 1 · 为什么先学无状态

Load balancer 只有在请求能落到任意健康实例时，才可以真正做水平扩展：

```text
Client
  -> Load Balancer
      -> API-1
      -> API-2
      -> API-3
```

若 user A 的 session 只在 API-1 内存里，LB 就必须做 sticky routing。API-1 挂掉后 session 仍然丢失，扩缩容和滚动发布也会变麻烦。

无状态实例带来的直接收益：

- 故障实例可以直接替换；
- 扩容只需启动同版本副本；
- rolling deployment 不需要搬业务状态；
- scheduler 可以把 workload 放到任意有容量的节点。

状态没有消失，只是被放到了更适合保存它的系统里。

---

## 2 · 先给状态分类

| 状态 | 例子 | 放在哪里 |
|---|---|---|
| Authoritative | 订单、用户、任务最终状态 | Database / durable log |
| Durable blob | 图片、模型、checkpoint | Object storage |
| Coordination | lease、lock、leader epoch | DB / coordination service |
| Rebuildable | cache、index、materialized view | Redis / local cache / derived store |
| Request-local | trace context、中间变量 | 当前进程内存 |

关键不是“能不能放内存”，而是进程退出后能否接受它消失。Request-local state 和可重建 cache 留在本地没有问题。

### 一个判断问题

```text
If this process disappears now,
what information is permanently lost?
```

答案应当是“没有业务事实丢失”。最多损失正在计算但尚未确认的工作，客户端或 queue 可以安全重试。

---

## 3 · 常见的三次外置

### Session 外置

有状态做法：

```text
API-1 memory: session_42 -> user_42
```

更容易扩展的做法：

```text
opaque session_id
  -> shared session store
  -> user and permission context
```

也可以使用短期签名 token，让实例本地验证身份。不过权限即时变更、强制登出和封禁仍可能需要查 version 或 revoke state。JWT 并非自动无状态的完美方案。

### 文件外置

不要把上传文件只留在某个 pod 的 `/tmp`。API 保存 metadata，bytes 进入 object storage：

```text
DB: file_id, owner_id, object_key, checksum, status
Object store: actual bytes
```

大文件通常用 signed URL 直传，避免 API server 变成数据搬运层。存储选择见 [[SystemDesign04 Storage Systems|04 存储系统]]。

### 长任务外置

20 分钟的 report 不应占住一个 HTTP request：

```text
POST /jobs
  -> persist job(status=queued)
  -> enqueue(job_id)
  -> 202 Accepted

Worker
  -> claim job with lease
  -> execute
  -> persist result
```

API 不记任务，worker 也不永久拥有任务。Queue 的 ack、重投见 [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]。重试能安全，是因为下面的幂等设计，不是因为换了队列。

---

## 4 · 幂等与一致性

无状态让超时和故障变成重试。重试两次若变成两笔订单，系统就不一致。幂等是设计手段；一致性是要保住的不变量。

两件事先分开：

| | 问的是 | 不在这层解决 |
|---|---|---|
| 幂等 | 同一意图执行 N 次，可见结果是否仍是一次 | HTTP 状态码碰巧相同 |
| 一致性（不变量） | 提交之后哪条业务规则仍成立 | 副本迟了几毫秒，见 [[SystemDesign02 Database Paradigms|02]] |

```text
payment captured at most once
at most one order per (sale_id, user_id)
inventory never < 0
```

先写下这些句子，再选键和事务边界。不要写“系统要强一致”就停。

### 幂等怎么设计

自然幂等：`GET`、按 id 的 `PUT`/`DELETE`、`SET key value`。  
不自然：`POST` 创建、`INCR`、转账、扣库存。这类必须外加键。

```text
1. 客户端生成 request_id（同一点击、同一超时重试，同一个 id）
2. 共享存储插入 (user_id, request_id)，UNIQUE
3. 同一事务里做业务写（建单、扣款）
4. 再来一次：唯一键冲突 → 返回已存结果，不再写第二笔
```

```sql
INSERT INTO idempotency (
  user_id, request_id, status, response
) VALUES (?, ?, 'pending', NULL);

-- UNIQUE 冲突：读已有行
-- completed → 原 response
-- pending   → 等，或 409
```

进行中的第二发不能当“没有这回事”。只查 Redis 再决定插入，两个实例会一起通过。约束在 source of truth 上，和业务写同一事务。

键选业务意图，不选 body hash：

| 键 | 挡住什么 | 挡不住什么 |
|---|---|---|
| `request_id` | 超时重试、LB 重放 | 用户两个设备各点一次 |
| `UNIQUE(sale_id, user_id)` | 每用户每场一单 | 同一次点击的网络重试（应用层仍要 `request_id`） |
| payload hash | 几乎什么都挡错 | 两笔独立同金额订单被当成一次 |

Redis `SET NX` / soldout 是 fast reject，不是这张表。见 [[SystemDesign01D Redis|01D]]、[[SystemDesign10 Flash Sale|10]]。

### 一致性怎么设计

1. 写下不变量，写到哪条 API、哪次读必须看见。  
2. 能进同一行或同一事务的，放进一个 commit。跨库不要 2PC 到 HTTP。  
3. 标出允许陈旧的面：商品页、cache、Redis tokens。订单确认页读 primary。  
4. 重试路径走同一不变量：幂等键 + unique + 事务。否则 at-least-once 会写出第二笔。  
5. 条件写补并发：`UPDATE ... WHERE version = n` 或 `balance >= 100`。失败则重读，不盲重放。

副本何时可见是 [[SystemDesign02 Database Paradigms|02]] 的 read contract。队列侧 exactly-once 是效果，不是 broker 承诺，见 [[SystemDesign06 Async Messaging Systems|06]]。

---

## 5 · API 和 worker 怎样保持可替换

创建类 API 的幂等边界在上一节。这里只补进程怎么死、怎么被换。

### Worker 使用 lease

Worker 领取任务时写入：

```text
worker_id
leased_until
attempt_count
```

执行期间续租；进程崩溃后 lease 过期，任务可以由其他 worker 重新领取。业务写入仍需幂等，因为旧 worker 可能已经提交结果，却没来得及 ack。

### 本地 cache 必须可丢

适合留在进程里的东西：

```text
parsed config
tokenizer
public keys
hot metadata
compiled template
```

实例重启后可以从 source of truth 重新加载。更新传播可用短 TTL、version check 或主动 invalidation；不要让本地 cache 成为唯一真相。

---

## 6 · 部署时还有四个细节

### Readiness 和 liveness 分开

- Liveness：进程是否卡死，需要重启？
- Readiness：当前是否适合接新流量？

启动时先加载配置和必要资源，再变为 ready。下线时先撤销 readiness，停止接新请求，然后等待 in-flight work 结束。

### Graceful shutdown

```text
receive SIGTERM
-> mark unready
-> stop accepting new work
-> finish / cancel in-flight work
-> flush safe telemetry
-> exit before grace period ends
```

Worker 应停止领取新任务，并释放或缩短未完成任务的 lease。单纯 `kill -9` 会把正常发布变成故障注入。

### Autoscaling 看饱和指标

API 常看 CPU、concurrency、p99 和 request queue；worker 更适合看 queue depth、oldest message age 与消费速率。

只看平均 CPU 容易漏掉单 shard 热点或 I/O wait。扩容也有延迟，必须配合 headroom、rate limit 和 backpressure。

### 配置不能只在本机手改

镜像和配置应版本化。Secret 通过 secret manager 或受控挂载注入，避免写入 image。容器和 VM 的交付边界见下一篇 [[SystemDesign01B Virtualization Containers|01B 虚拟化与容器]]。

---

## 7 · “无状态”会把压力移到哪里

外置状态让计算层简单，却增加了共享依赖：

- Session store 故障会影响全部 API replica；
- 每次请求查远端状态会增加 latency；
- Queue、DB 和 object storage 需要各自的 replication 与 backup；
- Cache miss 可能形成同步风暴。

因此外部状态系统要有连接池、timeout、bulkhead、cache 和容量计划。不能把复杂度推到 Redis 或 DB 后就假装问题消失了。

### Sticky routing 什么时候可以用

Sticky session 不是禁用项。WebSocket、游戏房间和 LLM KV cache 都可能从 affinity 获益。但 durable state 仍要外置，并准备实例失效后的恢复路径。

```text
affinity = optimization
durable external state = correctness
```

把两者分开，系统才能在命中 affinity 时快，实例丢失时仍然正确。

---

## 8 · LLM serving 的状态分层

LLM serving 比普通 API 多了昂贵的 GPU-local state，但原则没变：

| 状态 | 处理方式 |
|---|---|
| Conversation / request | durable store 或由客户端携带 |
| KV cache | GPU-local、best effort、可重新 prefill |
| Shared prefix cache | 可重建的分布式 cache |
| Model weights | object store / model registry，可重新加载 |
| Adapter / LoRA | versioned artifact，可按需加载 |

KV cache 丢失会变慢，不应让对话语义丢失。Router 可以优先把同一 session 送回原 worker；miss 时从 durable conversation 重新 prefill。

Paged KV cache、prefix cache、disaggregated prefill/decode 和 offload 都是状态放置优化，不改变 source of truth。细节放在 MLSYS 的 serving 章节，这里不重复。

---

## 9 · 缓存与会话的专属组件

Redis 是最常用来存储这些可重建状态和共享协作状态（如 session、分布式锁）的组件。详情见专篇 [[SystemDesign01D Redis|01D Redis]]。

一句话收尾：服务实例可以有内存状态，但不能独占不可恢复的业务事实。状态应该外置到正确的系统中，这才是系统真正可替换、可扩展的基础。
