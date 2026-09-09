# System Design 02 · Database

Course Location: [[SystemDesign01D Redis|01D Redis]] → This Section → [[SystemDesign04 Storage Systems|04 Storage Systems]]

When choosing a database, consider two things first: which business invariants must be atomically satisfied, and what are the system's most critical access paths. The product name comes later.

```text
transaction boundary -> correctness
access pattern       -> data layout and indexes
```

"SQL cannot scale" or "NoSQL has no transactions" are too simplistic. Modern products have overlapping capabilities; the differences lie in default models, transaction boundaries, and scaling costs.

| API | Unit | Example |
|---|---|---|
| SQL | row / txn | Postgres, MySQL |
| KV get/put | key | Dynamo, Redis-as-DB (usually wrong) |
| Document | doc by id | Mongo |
| Wide-column | partition + clustering | Cassandra |
| Graph | vertex/edge walk | Neo4j |

---


## 1 · RDBMS vs NoSQL


## RDBMS: Expressing Relationships and Constraints First

The relational model organizes data into rows and tables, expressing invariants through primary keys, foreign keys, unique constraints, and transactions.

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

If any step fails, the entire transfer must not leave behind a partial state. 
The costs are direct: cross-node transactions, joins, and global constraints are difficult to scale.


## NoSQL: Organizing Data Around Access Patterns First

NoSQL emphasizes partition-local access.

```text
GetUser(user_id)
ListOrders(user_id, created_at range)
GetFeed(viewer_id, cursor)
```

To ensure a single request hits one partition, data may be denormalized:

```json
{
  "user_id": "u42",
  "profile": {"name": "Kai"},
  "shipping_city": "Seattle"
}
```

Reads become simpler, but writes and consistency costs increase.
If the partition key is chosen incorrectly, hot keys, scans, and cross-partition transactions will emerge.


## Scenarios

| Scenario | Recommendation | Reason |
|---|---|---|
| Orders / Ledger | RDBMS | Dense relationships, strong invariants |
| Profile by `user_id` | KV / Document | Fetched by ID, flexible schema |
| Feed Timeline | Sorted KV | Partitioned by viewer, time-sorted |
| Logs and Events | Log system | High-throughput append, not OLTP |

---


## 2 · Transactions, Locks, 2PC

ACID can be remembered as follows:

| Property | Practical Question |
|---|---|
| Atomicity | Could it be only half-finished? |
| Consistency | Does the constraint still hold after commit? |
| Isolation | What intermediate states will concurrent operations see? |
| Durability | Will the result disappear after a failure once success is returned? |

Define the invariants first, then determine the transaction scope:

```text
order.total == sum(order_items)
payment may be captured at most once
username must be unique
inventory cannot fall below zero
```

Enforcing invariants requires concurrency control:
- **Pessimistic lock**: Blocks concurrent reads/writes; introduces queuing and deadlocks.
- **Optimistic lock**: Uses versioning, validates on commit. Efficient when conflicts are rare.
- **Unique constraint**: Prevents duplication via primary keys, acting as a highly practical idempotency tool. Key choice and in-flight duplicates: [[SystemDesign01 Stateless Service|01]].

When spanning independent nodes:
- **2PC (Two-Phase Commit)**: Uses coordinator and participants. If the coordinator dies after prepare, participants block. Never span untrusted networks or HTTP-to-Stripe calls across 2PC.
- **Saga / Compensation**: Reverts applied steps by executing compensating operations when databases cannot be shared.


## Local vs Workflow
A single-database transaction usually finishes within milliseconds.

```text
local transaction
  -> write order + outbox
  -> async payment command
  -> state transition
  -> compensation when needed
```

See [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]].

---


## 3 · Consistency per API

Consistency must be bound to specific operations. Idempotency in [[SystemDesign01 Stateless Service|01]] keeps invariants true under retry. This axis is: after a successful write, which read sees it.

```text
User updates profile to v2
User immediately reads profile
```

Possible contracts:
- **Linearizable read**: Acts as if there is only one latest copy.
- **Read-your-writes**: The user can read their own writes.
- **Monotonic read**: Once v2 is seen, it will not revert to v1.
- **Eventual consistency**: Replicas eventually converge.

Mix these in the same system: read primary for order confirmation, read replica for public product pages.

---


## 4 · Scale reads: replica

The primary receives writes, and replicas replicate the primary's data.

```text
Client write -> Primary
Client read  -> Replica 1 / Replica 2
```

All modifications go to the primary.


## Core Mechanism

```text
Client -> Primary: write request
Primary -> Primary: execute mutation
Primary -> Binlog: append change event
Replica -> Binlog: pull changes after known position
Replica -> Relay Log: write relay log
Replica -> Replica: replay relay log
```

Read/write splitting distributes read traffic. 
This prevents slow queries and backups from dragging down the primary.


## Replication Lag
Asynchronous replication causes stale reads:

```text
User changes name -> Write to Primary succeeds
User reads immediately -> Routed to lagging Replica
Page shows old name
```

You trade strong consistency for read capacity. Read-your-writes requires pinning reads to the primary.

Common handling strategies:

| Scenario | Strategy |
| --- | --- |
| Read own writes immediately | Read-your-writes: Force read from primary for a short time |
| Brief stale data acceptable | Read from replica |
| Critical path requiring consistency | Read from primary after write, or use synchronous replication |
| Replica lag is severe | Remove lagging replica from read pool |



## Concurrency and DB QPS
**User QPS != DB QPS** (one API request may hit the DB many times).
Estimate concurrency using Little's Law: 
```text
concurrency ≈ QPS × latency
```
Replicas scale read capacity, not write capacity.

---


## 5 · Scale writes/capacity: shard

Replication saves multiple copies; partitioning saves different data.

```text
Replication: Every machine has a complete copy
Sharding: Every machine only saves a portion of the data
```


## Basic Idea

```text
user_id % 4 = 0  ->  Shard 0
user_id % 4 = 1  ->  Shard 1
user_id % 4 = 2  ->  Shard 2
user_id % 4 = 3  ->  Shard 3
```

Each machine is responsible for a portion of users, distributing capacity and write pressure.


## Sharding Key
A good sharding key must be:
1. Frequently included in queries (else broadcast to all shards).
2. Uniformly distributed (hot keys ≠ uniform keys).
3. Minimize cross-shard joins and transactions.

```text
Query -> Shard 0 / Shard 1 / Shard 2 / Shard 3
All Shards -> Merge / Sort / Aggregate -> Response
```
Cross-shard operations are incredibly expensive.


## Multi-Primary
Multi-primary replication gives multiple write endpoints (active-active) but does not double write capacity because all nodes must replicate all writes eventually.

---


## 6 · Together: Shard + Replica

Primary-replica replication is often performed within each shard:

```text
Query Router
  -> Shard 0 Primary -> Replica A / Replica B
  -> Shard 1 Primary -> Replica C / Replica D
  -> Shard 2 Primary -> Replica E / Replica F
```

- Capacity and write scaling from sharding;
- Read scaling, backups, and HA from replication.

---


## 7 · Survive a failure


## Failure Domains
```text
process
  < machine
  < rack / power domain
  < availability zone
  < region
```
Determine which level of failure the system must survive.


## Failover and Fencing
```text
1. Failure detector suspects primary is unavailable (Timeout ≠ death)
2. Threshold reached
3. Select a fresh replica
4. Fence the old primary
5. Promote new primary, update routing
6. Restore traffic
```

The goal of fencing is to ensure the old node cannot write. You must fence the old primary (using epoch / lease / STONITH) before routing. If both write, split brain occurs.


## Modes
- **Active-Passive**:

| Standby | What it does normally | Switchover Speed | Cost |
|---|---|---|---|
| Cold | Only backups and deployment templates | Minutes to hours | Low |
| Warm | Instance running, data syncing, capacity may be smaller | Tens of seconds to minutes | Medium |
| Hot | Full capacity online, data near real-time sync | Seconds | High |

- **Active-Active**: Same-row writes are the hard part; single-writer ownership is preferred.
- **Quorum (N/W/R)**: `W + R > N` ensures intersection, but does not provide linearizability by itself.


## Replicas are not Backups
Replicas rapidly copy deletions. Backups (snapshots / WAL) are to go back in time.
- **RPO** (Recovery Point Objective): Acceptable data loss window.
- **RTO** (Recovery Time Objective): How quickly service must be restored.

Multi-AZ is the first step. Multi-region requires new failure/latency targets.

---


## 8 · Short Choice Table

| Requirement | Start With | Cost |
|---|---|---|
| Read-heavy | Primary + read replicas | Lag, read-your-writes |
| Write-heavy | Shard | Cross-shard queries |
| Single-row invariant | KV / Document | Lack of relations |
| Cross-row invariant | RDBMS | Write bottlenecks |
| Global low-latency write | Partition ownership / Active-active | Conflicts, fencing |

---


## 9 · Sources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MySQL Replication](https://dev.mysql.com/doc/refman/8.0/en/replication.html)
- [MongoDB Sharding](https://www.mongodb.com/docs/manual/sharding/)
- [Dynamo: Amazon's Highly Available Key-value Store](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [Cassandra - A Decentralized Structured Storage System](https://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf)
- [Spanner: Google's Globally-Distributed Database](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf)
- [Vitess: Database Clustering System for MySQL](https://vitess.io/)















