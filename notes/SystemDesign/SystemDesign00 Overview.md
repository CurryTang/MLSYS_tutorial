# System Design 00 · 怎么读

这套笔记分两块。先过组件，再过案例。组件说明一个东西是什么、接口长什么样、能挡住哪类问题。案例用同一套模板把组件拼成一个系统。

```system-design-overview-visual
```

---

## 组件

每篇尽量短。先看 high-level API，再看适用场景。

| | 笔记 | 解决什么 |
|---|---|---|
| 01 | [[SystemDesign01 Stateless Service\|无状态服务]] | 进程不持有不可丢的状态，才能随便扩、随便换。重试靠幂等保住不变量 |
| 01B | [[SystemDesign01B Virtualization Containers\|虚拟化与容器]] | VM 虚拟机器，container 隔离进程 |
| 01C | [[SystemDesign01C Kubernetes\|Kubernetes]] | 把容器放到节点上；训练还要 gang、队列、checkpoint。职责分层 vs 拓扑分层 |
| 01D | [[SystemDesign01D Redis\|Redis]] | 快的共享可变状态：cache、session、锁、限流、fast reject/allow。不是订单表 |
| 02 | [[SystemDesign02 Database Paradigms\|数据库]] | RDB / NoSQL、事务与锁、副本、分片、failover |
| 04 | [[SystemDesign04 Storage Systems\|存储]] | block / file / object 各存什么 |
| 06 | [[SystemDesign06 Async Messaging Systems\|消息队列]] | 把活从请求路径上挪走且不丢。Kafka 是 partitioned log 的例子 |
| 09 | [[SystemDesign09 Consistent Hashing\|一致性哈希]] | 节点增减时少搬数据 |

旧的 03（扩展）和 05（复制）已经并进 02。

---

## 案例

每篇一张图，然后按固定顺序讲：

```text
1. Functional requirements × 3
2. Non-functional requirements × 3
3. Basic workflow + schema + QPS
4. Deep dive × 3
   每个至少两个方案，写清 tradeoff，再给 preferred
5. 一条完整 end-to-end workflow
```

| | 笔记 | 这条主路径 |
|---|---|---|
| 07 | [[SystemDesign07 Photo Sharing Feed\|图片分享与 Feed]] | 上传原图，生成 home feed |
| 08 | [[SystemDesign08 LLM Async RL Platform\|异步 LLM RL]] | rollout 和训练拆开，stale 有上界 |
| 10 | [[SystemDesign10 Flash Sale\|秒杀]] | 高峰准入，库存不超卖 |

术语见 [[SystemDesign99 Glossary\|99]]。

---

## 估算时用得到的几行

```text
1 day ≈ 1e5 seconds
avg QPS ≈ daily requests / 1e5
peak QPS ≈ avg × peak factor
concurrency ≈ QPS × latency_seconds
DB QPS ≠ API QPS
```

内部 fan-out、重试、每个请求打几次库，会把数字放大。只算会改变形态的量：要不要 cache、要不要 replica、要不要 shard、要不要队列。
