# System Design 00 · How to read

Two parts. Components first, then cases. A component note is the interface and the problem it blocks. A case uses one template to assemble those components.

```system-design-overview-visual
```

---

## Components

Keep each note short. API first, then when to use it.

| | Note | What it is for |
|---|---|---|
| 01 | [[SystemDesign01 Stateless Service\|Stateless service]] | A process that does not own durable facts can be replaced and scaled. Retries keep invariants via idempotency |
| 01B | [[SystemDesign01B Virtualization Containers\|VM / container]] | A VM is a machine. A container is isolated processes |
| 01C | [[SystemDesign01C Kubernetes\|Kubernetes]] | Place containers on nodes. Training still needs gang, queues, checkpoint. Responsibility vs topology layers |
| 01D | [[SystemDesign01D Redis\|Redis]] | Fast shared mutable state: cache, session, lock, rate limit, fast reject/allow. Not the order table |
| 02 | [[SystemDesign02 Database Paradigms\|Database]] | RDB / NoSQL, txn and locks, replicas, shards, failover |
| 04 | [[SystemDesign04 Storage Systems\|Storage]] | Block / file / object |
| 06 | [[SystemDesign06 Async Messaging Systems\|Message queue]] | Move work off the request path without losing it. Kafka is the partitioned-log example |
| 09 | [[SystemDesign09 Consistent Hashing\|Consistent hashing]] | Move less data when nodes change |

Notes 03 (scaling) and 05 (replication) now live in 02.

---

## Cases

One figure, then the same outline:

```text
1. Functional requirements × 3
2. Non-functional requirements × 3
3. Basic workflow + schema + QPS
4. Deep dive × 3
   each: ≥2 options, tradeoff, preferred
5. One end-to-end workflow
```

| | Note | Main path |
|---|---|---|
| 07 | [[SystemDesign07 Photo Sharing Feed\|Photo feed]] | Upload originals, build home feed |
| 08 | [[SystemDesign08 LLM Async RL Platform\|Async LLM RL]] | Split rollout from training; bound staleness |
| 10 | [[SystemDesign10 Flash Sale\|Flash sale]] | Admit the burst, no oversell |

Glossary: [[SystemDesign99 Glossary\|99]].

---

## Numbers that change the shape

```text
1 day ≈ 1e5 seconds
avg QPS ≈ daily requests / 1e5
peak QPS ≈ avg × peak factor
concurrency ≈ QPS × latency_seconds
DB QPS ≠ API QPS
```

Fan-out, retries, and DB hits per request amplify the number. Only compute quantities that change the design: cache, replica, shard, queue.
