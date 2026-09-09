# System Design 02 · 数据库

课程位置：[[SystemDesign01D Redis|01D Redis]] → 本篇 → [[SystemDesign04 Storage Systems|04 存储系统]]

数据库选型先看两件事：哪些业务不变量必须原子成立，系统最重要的访问路径是什么。产品名字放到后面。

```text
transaction boundary -> correctness
access pattern       -> data layout and indexes
```

“SQL 不能扩展”或“NoSQL 没有事务”都太粗。现代产品的能力有重叠，差别在默认数据模型、事务边界和扩展代价。

| API | 单位 | 例子 |
|---|---|---|
| SQL | row / txn | Postgres, MySQL |
| KV get/put | key | Dynamo, Redis-as-DB (usually wrong) |
| Document | doc by id | Mongo |
| Wide-column | partition + clustering | Cassandra |
| Graph | vertex/edge walk | Neo4j |

---


## 1 · RDBMS vs NoSQL


## RDBMS：先表达关系和约束

关系模型把数据放进 row 和 table，通过 primary key、foreign key、unique constraint 和 transaction 表达不变量。

```sql
BEGIN;

UPDATE accounts
SET balance = balance - 100
WHERE account_id = 1 AND balance >= 100;

UPDATE accounts
SET balance = balance + 100
WHERE account_id = 2;

COMMIT;
```

这段代码的重点不是 SQL 语法，而是两个余额变化属于同一个提交边界。任意一步失败，整个转账都不能留下半成品。
代价也很直接：跨节点 transaction、join 和全局 constraint 很难随 shard 数量一起扩展。


## NoSQL：先围绕访问路径组织数据

NoSQL 不是一种数据库。KV、document、wide-column 和 graph 的数据模型不同，但很多系统共同强调 partition-local access。

```text
GetUser(user_id)
ListOrders(user_id, created_at range)
GetFeed(viewer_id, cursor)
```

建模时先为这些读取选择 partition key 和 sort key。为了让一次请求命中单 partition，数据可能被反范式化：

```json
{
  "user_id": "u42",
  "profile": {"name": "Kai"},
  "shipping_city": "Seattle"
}
```

城市改名时，多个 document 可能要更新。读变简单，写入和一致性成本上升。
它不适合拿来逃避建模。Partition key 选错后，hot key、scan 和跨 partition transaction 会一起出现。


## 场景与建议

| 场景 | 推荐选型 | 原因 |
|---|---|---|
| 订单、财务账本 | RDBMS | 跨行关系密集，需要强一致事务和约束 |
| 用户 Profile | KV / Document | 按 `user_id` 整体读取，结构经常变化 |
| Feed Timeline | Sorted KV | 按 viewer_id 分区，按时间线排序 |
| 日志与事件 | Log system | 高吞吐 append，保留时间，不是 OLTP |

---


## 2 · Transaction 与并发控制

ACID 可以这样记：

| 性质 | 实际问题 |
|---|---|
| Atomicity | 会不会只完成一半？ |
| Consistency | 提交后约束是否仍成立？ |
| Isolation | 并发操作会看到什么中间状态？ |
| Durability | 返回成功后，故障会不会让结果消失？ |

先写不变量，再决定 transaction 范围：

```text
order.total == sum(order_items)
payment may be captured at most once
username must be unique
inventory cannot fall below zero
```

如果这些条件必须跨多个 entity 原子成立，关系数据库或 distributed SQL 更省心。
保证不变量通常需要并发控制：

- **Pessimistic lock**：写时阻塞读写，防止冲突，但容易产生排队和死锁。
- **Optimistic lock**：依靠版本号 (version) 允许并发，提交时验证。冲突少时效率高。
- **Unique constraint**：通过插入唯一键阻止重复，是最实用的幂等工具。键怎么选、进行中的第二发怎么办，见 [[SystemDesign01 Stateless Service|01]]。

当业务跨越多个独立节点时：

- **2PC (Two-Phase Commit)**：依赖 coordinator 和 participants 提供跨节点事务。缺点是如果 coordinator 在 prepare 阶段后死亡，participants 会阻塞。绝不应跨越不信任的网络或将 HTTP-to-Stripe 放入 2PC。
- **Saga / Compensation**：当工作流跨越不同系统，无法共享数据库事务时，通过执行反向补偿操作回滚已提交的步骤。


## Database transaction 和 business workflow 分离

单库 transaction 通常在毫秒内结束。跨 payment、inventory 的流程可能持续数分钟。

```text
local transaction
  -> write order + outbox
  -> async payment command
  -> state transition
  -> compensation when needed
```

这类流程靠 state machine 和 outbox，指向 [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]。

---


## 3 · Consistency per API

一致性必须绑定到具体操作。[[SystemDesign01 Stateless Service|01]] 的幂等保住的是不变量在重试下仍成立。下面这轴是：写成功之后，哪一次读看得到。

```text
User updates profile to v2
User immediately reads profile
```

可能的 contract：

- **Linearizable read**：像只有一个最新副本；
- **Read-your-writes**：该用户至少能读到自己的 v2；
- **Monotonic read**：已经看到 v2 后不会退回 v1；
- **Eventual consistency**：没有新写入时，副本最终收敛。

同一个系统可以混用。订单确认页读 primary，公开商品页读 replica。

---


## 4 · Scale reads: replica

Primary-Replica（主从）是读扩展的基础。写请求进入 Primary，Replica 负责重放日志同步数据。

```text
Client write -> Primary
Client read  -> Replica 1 / Replica 2
```

所有修改数据的操作（INSERT、UPDATE）都走向 Primary。


## 核心机制

以 MySQL 为例，核心是 binlog。

```text
Client -> Primary: write request
Primary -> Primary: execute mutation
Primary -> Binlog: append change event
Replica -> Binlog: pull changes after known position
Replica -> Relay Log: write relay log
Replica -> Replica: replay relay log
```

读写分离。写请求进入 Primary，读请求分散到多个 Replica。
这可以防止慢查询、报表和备份拖垮主库。零停机备份。


## Replication Lag

Primary-Replica 通常是异步的。这意味着 replication lag，产生过期读 (stale read)。

```text
User changes name -> Write to Primary succeeds
User reads immediately -> Routed to lagging Replica
Page shows old name
```

增加读容量和可用性的代价是副本滞后。要实现 Read-your-writes，必须短时间内强制读 Primary 或等待特定 position。

常见处理策略：

| 场景 | 策略 |
| --- | --- |
| 立即读取自己的写入 | Read-your-writes：短时间内强制读 Primary |
| 允许短暂旧数据 | 读 Replica |
| 需要一致性的关键路径 | 写入后读 Primary，或使用同步复制 |
| 副本滞后严重 | 从读流量池中移除滞后的 Replica |



## QPS ≠ DB QPS
一次 API 请求可能触发多次查库、写入。
用 Little's Law 估算并发连接：
```text
concurrency ≈ QPS × latency
```
Replica 提供读扩展，不是额外的写容量。

---


## 5 · Scale writes/capacity: shard

复制是保存同一份数据的多份拷贝；分片是让不同节点保存不同的数据。

```text
Replication: Every machine has a complete copy
Sharding: Every machine only saves a portion of the data
```

如果单库数据过大，或单主写压力过高，就需要分片。


## 基本思路

```text
user_id % 4 = 0 -> Shard 0
user_id % 4 = 1 -> Shard 1
user_id % 4 = 2 -> Shard 2
user_id % 4 = 3 -> Shard 3
```

这样每个机器只负责一部分用户。


## 分片键选择

一个好的分片键必须满足三点：
1. 必须经常出现在查询中。若未提供，查询只能广播给所有分片。
2. 数据分布尽可能均匀。`hash(user_id)` 通常更均衡。（热点 key ≠ uniform key，超级用户需要额外缓存）。
3. 最小化跨分片查询。

```text
Query -> Shard 0 / Shard 1 / Shard 2 / Shard 3
All Shards -> Merge / Sort / Aggregate -> Response
```

跨分片的 Join、跨分片事务和全局排序是分片系统的最大成本。


## Multi-Primary：多写入入口
Multi-Primary 允许系统有多个写入入口（active-active），最终都要承受全量数据复制和冲突处理，所以它主要是高可用方案，而不是 2× 吞吐扩展。

---


## 6 · 结合：Shard 与 Replica

生产环境中，这两种技术几乎总是同时部署：

```text
Query Router
  -> Shard 0 Primary -> Replica A / Replica B
  -> Shard 1 Primary -> Replica C / Replica D
  -> Shard 2 Primary -> Replica E / Replica F
```

- 通过分片扩展写入吞吐和总体容量。
- 在每个分片内部，通过副本提供读扩展和高可用故障切换。

---


## 7 · Survive a failure

冗余不是“多开几台机器”这么简单。


## Failure domain
```text
process
  < machine
  < rack / power domain
  < availability zone
  < region
```
设计前回答：系统要活过哪一级故障？


## Failover 与 Fencing

真正困难的是把一个 replica 安全地提升为新 primary。
```text
1. Failure detector 怀疑 primary 不可用 (Timeout ≠ death)
2. 达到判定阈值
3. 选出数据足够新的 replica
4. 对旧 primary 做 fencing
5. 提升新 primary，更新 routing
6. 恢复流量
```

Fencing 是必须的：旧 primary 可能只是网络隔离。如果不做 fencing (通过 epoch / lease / STONITH 隔离)，两边同时写会导致 split brain。


## 部署方式

- **Active-Passive** (主备)：

| Standby | 平时做什么 | 切换速度 | 成本 |
|---|---|---|---|
| Cold | 只有备份和部署模板 | 分钟到小时 | 低 |
| Warm | 实例运行，数据持续同步，容量可能较小 | 数十秒到分钟 | 中 |
| Hot | 完整容量在线，数据接近实时同步 | 秒级 | 高 |

- **Active-Active** (多活)：同 row 并发写是难点，合并代价极高；通常偏好单写 single-writer ownership。
- **Quorum (N/W/R)**：`W + R > N` 让读写集合相交，但它本身不自动提供 linearizability。


## Replica 不是 Backup

副本会迅速复制正常写入，也会迅速复制误删和坏数据。Backup (snapshot/WAL/immutable) 用来回到过去。
- **RPO** (Recovery Point Objective)：最坏可丢最近几分钟写入。
- **RTO** (Recovery Time Objective)：故障后多快恢复服务。

Multi-AZ 第一步；Multi-region 只有在有新故障或延迟目标时才引入。

---


## 8 · 短选择表

| 需求 | 合理起点 | 要明确的代价 |
|---|---|---|
| Read-heavy | Primary + read replicas | Lag, read-your-writes |
| Write-heavy | Shard | 跨分片查询、join |
| Single-row invariant | KV / Document | 复杂关系维护 |
| Cross-row invariant | RDBMS | 扩展受限 |
| Global low-latency write | Partition ownership / Active-active | 冲突处理、fencing |

---


## 9 · 一手资料

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MySQL Replication](https://dev.mysql.com/doc/refman/8.0/en/replication.html)
- [MongoDB Sharding](https://www.mongodb.com/docs/manual/sharding/)
- [Dynamo: Amazon's Highly Available Key-value Store](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [Cassandra - A Decentralized Structured Storage System](https://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf)
- [Spanner: Google's Globally-Distributed Database](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf)
- [Vitess: Database Clustering System for MySQL](https://vitess.io/)















