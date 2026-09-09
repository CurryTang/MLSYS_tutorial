# System Design 07 · 设计图片分享与 Home Feed

课程位置：[[SystemDesign09 Consistent Hashing|09 一致性哈希]] → 本篇 → [[SystemDesign08 LLM Async RL Platform|08 异步 LLM RL 平台]]

设计 Instagram-like 图片分享系统，分为两条独立的主链路：Media path 处理上传与分发，Feed path 处理如何在关注图上生成低延迟时间线。

## 1. Functional requirements

1. 用户可以上传一张图片并创建带 caption 的 post。
2. 用户可以 follow / unfollow 其他用户。
3. 用户可以分页读取自己的 home feed。

```text
Out of scope: 短视频 / Reels / Stories / 私信
搜索与发现页 / 广告 / 复杂图片编辑 / 直播。
Likes and comments are handled later.
As separate write paths.
```

## 2. Non-functional requirements

| 目标 | 指标与约束 |
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

| 参数 | 估算值 |
|---|---|
| DAU | 50M |
| Peak Feed Read API QPS | ~140K |
| CDN Peak Egress | ~1.34 Tb/s (95% hit rate) |
| Media Storage Growth | ~18 TB/day |

## 4. Architecture

```photo-sharing-architecture-visual
```

- 控制面与数据面分离：利用 Signed URL 规避 API 服务器宽带瓶颈。
- 异步媒体处理：上传与处理解耦，保障重试幂等性。
- 混合推送模式：根据粉丝基数切分读写放大路径。
- 批处理元数据：Timeline 仅存储 ID，使用聚合请求读取完整信息。

## 5. Deep dives

### 1. Bytes through API vs signed URL to object store
A. Bytes through API
+ 客户端逻辑简单，一次请求同时完成元数据和二进制数据的上传。
- Stateless API 实例变成高吞吐的搬运层，耗尽带宽并大幅增加请求超时的风险。
B. Signed URL to object store
+ 控制面与数据面分离。上传大文件直接连接存储节点，API 服务器只处理轻量级短链接签发。
- 引入了状态的不一致性，上传完成不代表资源可以立刻使用。
Prefer B. Change in the picture: 必须维护状态机。READY 状态不等同于上传时 HTTP 200 返回，依赖事件队列的异步确认机制确保可用。

### 2. Pull vs push vs hybrid feed
A. Pull
+ 发帖成本极低。关系链变动可以实时生效。
- 读取阶段 $O(F \times k)$ 复杂度导致读放大，多路归并耗时严重。
B. Push
+ 提前组装时间线，读取耗时短、性能极佳。
- 严重的写放大，大 V 发帖会压垮队列，产生无用数据。
C. Hybrid feed
+ 大多数活跃低粉作者使用推模式，大 V 使用拉模式，均衡读写压力。
- 系统复杂度高，需要运行时调度器分配策略。
Prefer C. Change in the picture: 后台 worker 对常规作者推送，Feed 服务在读取时合并大 V 的独立队列数据。

### 3. Stale feed ID vs authz on hydrate
A. Stale feed ID
+ 在 Timeline 里直接缓存完整的可见性状态，读取最快。
- 取关、封禁或设为私密的变更不能即时生效，极易出现内容泄露的安全隐患。
B. Authz on hydrate
+ Timeline ID 允许短时间的滞后，但在最后填充元数据时进行同步权限过滤。
- Hydrate 阶段可能会有部分无效 ID 导致请求需要过滤多余条目。
Prefer B. Change in the picture: ID 分发可以容忍延迟，但权限检查在读取阶段不是最终一致的，必须强同步。

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

一手资料：Instagram Engineering Blog 早期架构文章。
