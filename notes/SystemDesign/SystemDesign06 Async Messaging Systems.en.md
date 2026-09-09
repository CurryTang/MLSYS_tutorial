# System Design 06 · Message Queue

Course location: [[SystemDesign04 Storage Systems|04 Storage]] → this note → [[SystemDesign09 Consistent Hashing|09 Consistent Hashing]]

Asynchronous architecture rearranges the timing of responsibility transfer. It does not guarantee that the function itself runs faster. The core question is: if the current process crashes immediately, at what moment can the system be certain the work is durably accepted?

```async-messaging-architecture-visual
```

## 1 · High-Level API and Two Acknowledgments

The high-level API of a messaging system typically revolves around delivery and consumption:
- **Produce**: Write a message into the system.
- **Consume**: Retrieve a message from the system.
- **Ack (Acknowledge)**: Confirm the message was processed successfully.
- **Nack / Requeue**: Indicate processing failure and request a retry.
- **Seek / Commit Offset**: Locate or commit the current consumption progress.

There are two distinct acknowledgments in the system:

```text
Client -> API: Initiate request
API -> Queue: Enqueue task
Queue -> API: Broker confirmation (Durable Ack)
API -> Client: 202 Accepted and return job_id

Queue -> Worker: Dispatch task
Worker -> DB: Execute business write
Worker -> Queue: Consumer confirmation (Worker Ack)
```

1. **Broker Ack**: The broker confirms to the API after receiving and persisting the message. This marks the system officially taking over the task. If the API returns success to the client before this, it's Fire-and-Hope, not Fire-and-Forget.
2. **Consumer Ack**: The consumer confirms to the broker after completing the business side effects (e.g., writing to a DB), indicating the message can be deleted or the offset advanced.

## 2 · Topology Patterns and Semantics

When analyzing asynchronous scenarios, it is necessary to distinguish between connection topology and delivery semantics:

```text
1 to 1 (Point to Point):   Sender -> Queue -> One logical receiver
1 to N (Fan-out):          Publisher -> Topic -> Subscriber A / B / C
N to 1 (Fan-in):           Producers -> Collector -> Stream processor
N to N (Event Backbone):   Producers -> Event Bus/Log -> Consumer groups
```

Even if the topology is 1-to-1, the underlying semantics can be completely different: it might specify a unique receiver, let any worker compete for execution, or be a two-way Request/Reply pattern.

### Queue, Pub/Sub, and Log Semantics Comparison

| Model | Characteristic | Consumption Mechanism |
|---|---|---|
| **Queue** | Competing Consumers. A message is successfully processed by only one consumer. | Multiple workers compete for messages in the same queue. |
| **Pub/Sub** | Fan-out. Each logical subscriber gets a full copy of the message. | Subscribers fetch their own copies, and workers within a subscriber compete. |
| **Partitioned Log** | Append-only. Retains history independently of consumption progress, supporting replay. | Systems like Kafka use Consumer Groups to determine semantics: same group acts as a Queue, different groups act as Pub/Sub. |

A Topic is merely a name or logical grouping; it does not directly dictate the underlying distribution semantics.

## 3 · Kafka as a Partitioned Log

Kafka did not adopt the traditional Queue model. Instead, its core abstraction is an append-only log that is retainable, locatable, and replayable (Partitioned Log).

- **Topic, Partition, and Offset**: A producer routes a message to a specific Partition based on its Key. Order is guaranteed within a Partition. Consumers track their own Offset. Message retention is independent of the consumption progress.
- **Parallelism Limit**: In a traditional Consumer Group, a single Partition can only be exclusively consumed by one consumer instance at a time. Thus, the maximum effective parallelism ≤ the number of Partitions.
- **Replication and Fault Tolerance**: Each Partition has a Leader and multiple Replicas. The `acks` parameter combined with the `min.insync.replicas` configuration determines the quorum needed for a safe write.
- **Strengths**: Highly suitable for cross-system data pipelines (CDC), history replay, multi-tenant stream analytics, and scenarios requiring local ordering by Key.
- **Weaknesses and Traps**: Fine-grained per-task acknowledgments and complex header-based routing rules are not its strengths. A Poison Record (e.g., one that fails deserialization perpetually) can completely stall a consumer strictly adhering to offset advancement.

## 4 · Delivery, Idempotency, and Ordering

### Delivery Semantics
- **At-most-once**: Advance the offset first, then process. If a crash occurs during processing, the message is lost.
- **At-least-once**: Ack after successful processing. If the Ack fails due to network partition, the system resends, causing duplication.
- **Exactly-once effect**: True exactly-once requires encompassing the boundaries of external systems. Kafka's internal transactions cannot cover external API calls. This relies on the consumer's idempotency regarding external operations.

### Idempotency
An idempotency key should be a unique business identifier (e.g., a combination of `order_id` and `status`), never just a hash of the payload body. Otherwise, if a user initiates two independent orders for the same item and amount, deduplication by hash would erroneously discard the second order. Table shape and in-flight duplicates: [[SystemDesign01 Stateless Service|01]].

### Local Ordering
Global ordering forces all traffic through a single processing node, sacrificing concurrency. In practice, systems settle for **local ordering**. By using the same `partition_key` (e.g., `user_id`), all operations for the same user land in the same Partition and are processed sequentially.

### Dead-Letter Queue (DLQ)
Infinite retries lead to a Retry Storm. Messages encountering permanent failures (e.g., data validation errors) should be routed to a Dead-Letter Queue (DLQ) preserving the original `event_id`, failure reason, and stack trace for subsequent manual inspection and replay.

## 5 · Outbox Pattern and CDC

In microservices, executing a raw Dual Write in business code creates an uneliminatable failure window:

```text
1. Execute DB transaction: UPDATE orders SET status = 'PAID'
2. API Call: publish OrderPaid to the message broker
```

If step 2 fails, the database is updated but downstream systems are oblivious. If the order is reversed and step 2 succeeds but step 1 rolls back, phantom events are emitted. Simple retries cannot resolve this state inconsistency.

### Transactional Outbox
The correct approach is to reuse the local database transaction to commit the state change alongside the event record:

```sql
BEGIN;

-- 1. Update business state
UPDATE orders SET status = 'PAID' WHERE order_id = :order_id;

-- 2. Insert event into Outbox table
INSERT INTO outbox_events(event_id, event_type, payload) 
VALUES (:event_id, 'order.paid', :payload);

COMMIT;
```

After the transaction commits, a separate background publisher or a CDC tool (e.g., Debezium's Outbox Event Router) polls the table or parses the binlog to reliably relay the events to the broker.

### Using the Database as a Queue
At a smaller scale or with low concurrency, a relational database can be used directly to build a reliable queue:

```sql
BEGIN;

-- Find unprocessed tasks, using SKIP LOCKED to avoid lock contention
SELECT job_id FROM jobs 
WHERE status = 'READY' AND available_at <= now()
ORDER BY available_at, job_id
FOR UPDATE SKIP LOCKED LIMIT 10;

-- Mark them as running and set a lease timeout
UPDATE jobs SET status = 'RUNNING', attempts = attempts + 1 
WHERE job_id = ANY(:claimed_ids);

COMMIT;
```
When throughput scales, tables bloat, and online transactions are impacted, you can migrate to a dedicated MQ.

## 6 · How to Choose

| System | Use Case | Core Mechanism |
|---|---|---|
| **RabbitMQ** | Fine-grained task processing, complex routing | Exchange routing, consume-and-delete |
| **Kafka** | High-volume data pipelines, replay, stream processing | Partitioned Log, Offset tracking, sequential I/O |
| **Managed Queue** | Cloud-native default choice | API driven, zero-ops scaling (SQS / PubSub) |
| **DB Table** | Monoliths or small-scale services | Local transaction guarantee, `SKIP LOCKED` |

Decision matrix (6-line text block):

```text
1. Do you need strict history replay or stream analytics? -> Kafka
2. Do you need complex routing and per-message ack for tasks? -> RabbitMQ
3. Can you just use a cloud managed API? -> SQS / PubSub
4. Is your scale small? -> Database table queue (SKIP LOCKED)
```

## 7 · Event Bus and Webhook

An **Event Bus** is a general term for a many-to-many router. It encompasses ingestion, match filtering, and durable delivery, with separated control and data planes. It is usually built on top of a DB, RabbitMQ, or Kafka.

A **Webhook** is a common HTTP target endpoint supported by this router. For security and isolation, it requires HMAC signature verification and separate rate-limiting and retry backoff strategies per tenant.

## 8 · Observability and Primary Sources

**Key Metrics**:
- **Oldest unacked age**: Reflects system lag much better than simply looking at CPU utilization.
- **Consumer lag** and **DLQ rate**.

Primary Sources:
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
