# System Design 07 · Photo Sharing and Home Feed

Course location: [[SystemDesign09 Consistent Hashing|09 Consistent Hashing]] → this note → [[SystemDesign08 LLM Async RL Platform|08 Async LLM RL]]

Design an Instagram-like photo sharing system with two independent main paths: a media path for uploading and distributing large files, and a feed path for generating low-latency timelines based on the social graph.

## 1. Functional requirements

1. Users can upload a photo and create a post with a caption.
2. Users can follow or unfollow other users.
3. Users can read their paginated home feed.

```text
Out of scope: Short video / Reels / Stories / Direct messages
Search and explore / Ads / Complex photo editing / Live streaming.
Likes and comments are handled later.
As separate write paths.
```

## 2. Non-functional requirements

| Goal | Metrics & Constraints |
|---|---|
| Feed latency | Feed metadata p99 < 200 ms + CDN bytes |
| Durability | READY original survives one AZ |
| Consistency | Owner read-your-writes, feed may lag 5s |

## 3. Workflow, schema, and QPS

```text
Client -> API Gateway (auth + rate limit)
  -> Upload Service (create session)
      <- returns post_id + signed URL
Client -> Object Storage (direct upload bytes)
Client -> Upload Service (commit metadata)
      -> DB: status PROCESSING
Object Storage -> Event Log -> Media Processor (create renditions)
      -> DB: status READY
      -> Transactional Outbox: PostReady event
```

| Entity | Primary Keys / Schema |
|---|---|
| User | user_id, profile |
| Follow | follower_id, followee_id, created_at, state |
| Post | post_id, author_id, caption, status, ready_at |
| Media | post_id, original_key, renditions |
| TimelineEntry| viewer_id, sort_key, post_id, author_id, source |

| Parameter | Estimate |
|---|---|
| DAU | 50M |
| Peak Feed Read API QPS | ~140K |
| CDN Peak Egress | ~1.34 Tb/s (95% hit rate) |
| Media Storage Growth | ~18 TB/day |

## 4. Architecture

```photo-sharing-architecture-visual
```

- Separation of control and data planes: use signed URLs to avoid API bandwidth bottlenecks.
- Async media processing: decouple uploads and processing for better retry idempotency.
- Hybrid fan-out: balance write amplification by checking follower counts.
- Batch metadata: timelines only store IDs to aggregate hydration queries efficiently.

## 5. Deep dives

### 1. Bytes through API vs signed URL to object store
A. Bytes through API
+ Simplifies client logic by sending metadata and binary data in a single HTTP POST request.
- Stateless API instances become high-throughput forwarding proxies, causing bandwidth saturation and increased timeout risks.
B. Signed URL to object store
+ Separates control and data planes. API servers only issue lightweight short-lived tokens.
- Introduces state discrepancies, requiring a state machine to track asynchronous uploads.
Prefer B. Change in the picture: Must maintain a state machine. READY state is not equivalent to an HTTP 200 during upload; it relies on event queues for asynchronous confirmation.

### 2. Pull vs push vs hybrid feed
A. Pull
+ Minimal write cost. Follow graph changes take effect immediately without synchronization.
- Severe read amplification causing $O(F 	imes k)$ merge cost that drastically increases tail latency during peak hours.
B. Push
+ Timelines are pre-computed, enabling extremely fast single-point paginated reads.
- Massive write amplification. Celebrity posts can overwhelm queues and generate useless data for inactive users.
C. Hybrid feed
+ Standard active users receive pushes while celebrity feeds are pulled and merged at read time.
- Increased complexity requiring a runtime scheduler to route based on dynamic constraints.
Prefer C. Change in the picture: Background workers push normal updates while the Feed service pulls and merges celebrity outboxes dynamically.

### 3. Stale feed ID vs authz on hydrate
A. Stale feed ID
+ Explicit caching of visibility state in the timeline allows for the fastest possible read paths.
- Privacy changes and block actions are delayed, causing irreversible content leaks before asynchronous cleanup completes.
B. Authz on hydrate
+ Timeline IDs can tolerate minor distribution delays, but permission filtering is strictly synchronized during metadata hydration.
- The hydration phase might incur additional pagination overhead to filter out invalid items.
Prefer B. Change in the picture: ID distribution can lag, but permission checks during reads are never eventual.

## 6. End-to-end workflow

```text
1. Client requests upload session from API.
2. API returns post_id (PENDING) and signed URL.
3. Client uploads raw bytes to Object Storage.
4. Client commits metadata via API, status changes to PROCESSING.
5. Object Storage emits event to processing queue.
6. Worker creates thumbnails, updates DB to READY.
7. Outbox triggers PostReady event for feed system.
8. Hybrid fan-out pushes post to active followers.
9. Follower reads feed, merges pull list, hydrates items.
10. If worker crashes during push, idempotency key avoids duplicate feed entries on retry.
11. Client receives final feed page and loads images via CDN.
12. (Failure) Read timeout during ranking falls back to chronological list.
```

Primary sources: Instagram Engineering Blog early architecture.
