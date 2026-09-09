# System Design 06 · 消息队列

课程位置：[[SystemDesign04 Storage Systems|04 存储系统]] → 本篇 → [[SystemDesign09 Consistent Hashing|09 一致性哈希]]

异步架构重新安排了责任转移的时间点。它并不保证一个函数本身执行得更快。核心问题是：如果在当前进程立刻崩溃，在哪个时刻能够确定任务已被持久接管？

```async-messaging-architecture-visual
```

## 1 · 核心 API 与两个确认

消息系统的高层 API 通常围绕投递和消费展开：
- **Produce**：将消息写入系统。
- **Consume**：从系统中获取消息。
- **Ack (Acknowledge)**：确认消息处理成功。
- **Nack / Requeue**：处理失败，要求系统退回并重试。
- **Seek / Commit Offset**：定位或提交当前的消费进度。

系统中存在两个截然不同的确认（Ack）：

```text
Client -> API: 发起请求
API -> Queue: 将任务入队
Queue -> API: Broker 确认 (Durable Ack)
API -> Client: 202 Accepted 并返回 job_id

Queue -> Worker: 派发任务
Worker -> DB: 执行业务写入
Worker -> Queue: Consumer 确认 (Worker Ack)
```

1. **Broker 确认**：Broker 接收到消息并持久化后，向 API 确认。这标志着消息系统正式接管任务。如果 API 在此之前就向客户端返回成功，那就是 Fire-and-Hope。
2. **Consumer 确认**：消费者完成业务副作用（例如写入 DB）后向 Broker 确认，表示该消息可以删除或推进位置。

## 2 · 拓扑模式与语义分类

分析异步场景时，需要分清拓扑结构和投递语义：

```text
1 to 1 (Point to Point):   Sender -> Queue -> One logical receiver
1 to N (Fan-out):          Publisher -> Topic -> Subscriber A / B / C
N to 1 (Fan-in):           Producers -> Collector -> Stream processor
N to N (Event Backbone):   Producers -> Event Bus/Log -> Consumer groups
```

即使拓扑都是 1-to-1，背后的语义也可能完全不同：可能是指定了唯一的接收者，可能是让任意 Worker 抢占执行，也可能是双向的 Request/Reply 模式。

### Queue、Pub/Sub 与 Log 语义对比

| 模型 | 特性 | 消费机制 |
|---|---|---|
| **Queue** (队列) | Competing Consumers 模式。一条消息只会被一个消费者成功处理。 | 多个 Worker 竞争抢占同一队列中的消息。 |
| **Pub/Sub** (发布/订阅) | Fan-out 模式。每一个逻辑订阅者（Subscriber）都能获得消息的完整拷贝。 | 每个订阅者各自获取副本，订阅者内部的 Worker 之间再竞争。 |
| **Partitioned Log** (日志) | 追加写入，独立于消费进度保留历史，支持多租户重放（Replay）。 | Kafka 等系统通过 Consumer Group 决定语义：同组是 Queue，不同组是 Pub/Sub。 |

Topic 只是一个名字或逻辑归属，并不直接决定底层的分发语义。

## 3 · Kafka 作为 Partitioned Log

Kafka 没有采用传统的 Queue 模式，而是将核心抽象换成了可保留、可定位、可重放的追加日志（Partitioned Log）。

- **Topic、Partition 与 Offset**：Producer 根据 Key 路由消息到特定的 Partition。Partition 内部保证追加顺序。Consumer 消费并保存自己所在的 Offset。消息的保留期（Retention）独立于消费进度。
- **并行度限制**：在传统的 Consumer Group 中，一个 Partition 同时只能被组内的一个 Consumer 实例独占消费，因此最大有效并行度 ≤ Partition 数量。
- **复制与容错**：每个 Partition 具有一个 Leader 和多个 Replicas。`acks` 参数与 `min.insync.replicas` 配置项共同决定了消息不丢失的法定人数（Quorum）。
- **优势场景**：跨系统数据管道（CDC）、历史重放（Replay）、多租户流式分析、以及需要按 Key 保证局部顺序的场景。
- **弱点与陷阱**：单条任务的细粒度确认（Per-task ack）、复杂的基于 Header 的路由规则不是其强项。如果遇到永远无法反序列化的 Poison Record，可能会彻底卡住要求严格按 Offset 推进的 Consumer。

## 4 · 投递、幂等与顺序

### 投递语义
- **At-most-once**：先推进 Offset，再处理。如果处理时崩溃，消息丢失。
- **At-least-once**：处理成功后再 Ack。若由于网络隔离造成 Ack 失败，系统会重发，导致重复。
- **Exactly-once effect**：精确一次效果。需要注意，Kafka 内部的事务保证无法涵盖外部系统。真正的 Exactly-once 需要依赖 Consumer 端对外部系统操作的幂等性。

### 幂等 (Idempotency)
幂等键应当是具备业务唯一性的标识（如 `order_id` 与 `status` 组合），绝不能使用 Payload Body 的 Hash。否则，当用户发起两笔金额、商品完全独立的订单时，Hash 查重会错误地丢弃第二笔订单。表怎么建、进行中怎么挡，见 [[SystemDesign01 Stateless Service|01]]。

### 局部有序
全局有序意味着必须将所有流量压到单一节点排队处理。实践中几乎总是妥协为**局部有序**。通过设置相同的 `partition_key`（例如 `user_id`），保证同一用户的操作进入同一 Partition 并按顺序处理。

### 死信队列 (DLQ)
无限重试会引发 Retry Storm。发生永久失败（如数据校验不过）的消息应进入死信队列 (DLQ)，以便保留原始 `event_id`、失败原因和堆栈，供后续人工审查（Inspect）和工具重放（Replay）。

## 5 · Outbox 模式与 CDC

在微服务中，不能在业务代码中执行粗糙的 Dual Write，这会产生无法消除的故障窗口：

```text
1. 数据库事务执行：UPDATE orders SET status = 'PAID'
2. API 调用：publish OrderPaid 到消息中间件
```

如果步骤 2 失败，数据库已变更而下游永远不知；若颠倒顺序，步骤 2 成功而步骤 1 回滚，下游则收到幽灵事件。简单的重试无法判断故障点在哪一步。

### Transactional Outbox
正确做法是复用本地数据库事务，将状态变更与事件记录一起提交：

```sql
BEGIN;

-- 1. 更新业务状态
UPDATE orders SET status = 'PAID' WHERE order_id = :order_id;

-- 2. 插入事件到 Outbox 表
INSERT INTO outbox_events(event_id, event_type, payload) 
VALUES (:event_id, 'order.paid', :payload);

COMMIT;
```

事务提交后，再由独立的后台进程或 CDC 架构（如 Debezium 的 Outbox Event Router）轮询表或解析 Binlog，将事件可靠投递给 Broker。

### 使用数据库作为 Queue
如果在小规模或低并发场景下，直接使用关系型数据库也可以构建可靠的队列：

```sql
BEGIN;

-- 寻找未处理的任务，使用 SKIP LOCKED 避免消费者之间的锁竞争
SELECT job_id FROM jobs 
WHERE status = 'READY' AND available_at <= now()
ORDER BY available_at, job_id
FOR UPDATE SKIP LOCKED LIMIT 10;

-- 将其标记为运行中，并设置超时租约
UPDATE jobs SET status = 'RUNNING', attempts = attempts + 1 
WHERE job_id = ANY(:claimed_ids);

COMMIT;
```
当系统吞吐量上升、表急剧膨胀并影响在线事务时，再迁移至独立 MQ。

## 6 · 怎么选

| 系统 | 适用场景 | 核心机制 |
|---|---|---|
| **RabbitMQ** | 细粒度任务处理、复杂路由分发 | Exchange 路由，Queue 消费后即删 |
| **Kafka** | 海量数据管道、多次回放与流计算 | Partitioned Log，Offset 追踪，顺序 I/O |
| **Managed Queue** | 云原生环境下的默认首选 | API 驱动，免运维自动扩容 (SQS / PubSub) |
| **DB Table** | 单体架构或小规模服务 | 本地事务保证，`SKIP LOCKED` 避免竞争 |

决策表（6 行文本块）：

```text
1. Do you need strict history replay or stream analytics? -> Kafka
2. Do you need complex routing and per-message ack for tasks? -> RabbitMQ
3. Can you just use a cloud managed API? -> SQS / PubSub
4. Is your scale small? -> Database table queue (SKIP LOCKED)
```

## 7 · Event Bus 与 Webhook

**Event Bus** 是一个多对多路由器的泛称。它包含事件接入（Ingest）、匹配过滤（Match）和可靠投递（Durable Delivery），控制面与数据面分离。通常底层构建在 DB、RabbitMQ 或 Kafka 上。

**Webhook** 则是该路由器支持的一种常见 HTTP 目标端点。为了安全与隔离，它要求使用 HMAC 进行签名校验，并对各租户设置独立限流与重试退避（Backoff）策略。

## 8 · 观测指标与一手资料

**关键指标**：
- **Oldest unacked age**（最旧未确认消息延迟）：比只看 CPU 利用率更能反映系统是否落后。
- **Consumer lag**、**DLQ rate**（死信产生率）。

一手资料：
- [RabbitMQ: Consumer Acknowledgements and Publisher Confirms](https://www.rabbitmq.com/docs/confirms)
- [RabbitMQ: Quorum Queues](https://www.rabbitmq.com/docs/quorum-queues)
- [RabbitMQ: Native AMQP 1.0 and AMQP history](https://www.rabbitmq.com/blog/2024/08/05/native-amqp)
- [OASIS AMQP 1.0 Standard](https://www.oasis-open.org/standard/amqp/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Kafka: a Distributed Messaging System for Log Processing, NetDB 2011](https://www.odbms.org/2011/01/kafka-a-distributed-messaging-system-for-log-processing/)
- [Debezium Outbox Event Router](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)
- [PostgreSQL SELECT: SKIP LOCKED](https://www.postgresql.org/docs/current/sql-select.html)
- [Amazon EventBridge: Event buses](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-event-bus.html)
- [GitHub webhook best practices](https://docs.github.com/en/webhooks/using-webhooks/best-practices-for-using-webhooks)
- [Stripe webhook best practices](https://docs.stripe.com/webhooks)
