# DAG-Aware vLLM Scheduling — 面试准备文档

> 基于 `dag-aware-schedule` 分支，相对上游 `b1388b1fb`（v0.19.1 snapshot）的全部修改分析。
>
> **作者**: 於晨阳 (ycy) + 迟云帆 (cyf)
> **总共**: 3 个 commit，13 个文件，+484/-15 行

---

## 简历描述（建议版）

> 基于 vLLM 实现 DAG-aware Serving Framework：扩展 ChatCompletion 协议支持 Workflow 拓扑元数据传递，设计基于关键路径（Critical Path）的两级优先级调度算法（DAG 级最短剩余优先 + 节点级关键路径优先），实现动态 DAG 拓扑运行时扩展机制以支持未知轮次的 Subagent 多轮调度。

### 原版 vs 建议版对照

| 问题 | 原版 | 建议版 |
|---|---|---|
| "扩展 API"太笼统 | 扩展 API 支持 Workflow 元数据传递与 DAG 管理 | 扩展 ChatCompletion 协议 + 5 个 REST 端点 + 运行时动态拓扑修改 |
| "两级"没解释 | 设计两级优先级调度算法 | DAG 级最短剩余优先 + 节点级关键路径优先 |
| 缺核心创新点 | — | 动态 DAG 拓扑运行时扩展 + 局部缓存失效 |

---

## 一、整体架构

```
Client (Harness / Multi-Agent Framework)
    │
    │  POST   /v1/dags                                   注册 DAG 拓扑（nodes + edges）
    │  GET    /v1/dags                                   列出已注册的 DAG
    │  PATCH  /v1/dags/{dag_id}/nodes/{node_id}         在线更新节点代价（LLM 自估计）
    │  POST   /v1/dags/{dag_id}/nodes/{parent}/next-round  运行时动态追加节点（subagent 多轮）
    │  POST   /v1/dags/{dag_id}/nodes/{node_id}/complete   标记节点完成（最终轮）
    │  POST   /v1/chat/completions  {dag_id, node_id, ...} 携带拓扑元数据的推理请求
    ▼
┌─────────────────────────────────────────────────────┐
│  API Layer  (vllm/dag_api_router.py)                │
│  FastAPI APIRouter, 5 个端点, Pydantic 校验          │
└──────────────────────┬──────────────────────────────┘
                       │ 委托调用
                       ▼
┌─────────────────────────────────────────────────────┐
│  Registry Layer  (vllm/dag_registry.py)              │
│  _DAGStore 单例（sys.modules 注入，跨 import 路径唯一）│
│  _DAGEntry: nodes / successors / predecessors        │
│             cp_cache / dag_remaining_cache            │
│             completed_nodes                           │
│                                                       │
│  compute_priority(dag_id, node_id)                    │
│    → (dag_remaining_cp, -node_cp_remaining)           │
└──────────────────────┬──────────────────────────────┘
                       │ 查询 priority
                       ▼
┌─────────────────────────────────────────────────────┐
│  Serving Layer  (vllm/entrypoints/openai/.../serving.py)│
│  从 registry 获取 priority → 传入 engine.generate() │
│  请求完成后 → mark_completed()                       │
└──────────────────────┬──────────────────────────────┘
                       │ priority: tuple[int, int] | int
                       ▼
┌─────────────────────────────────────────────────────┐
│  Engine Layer  (vllm/v1/engine/)                     │
│  priority 类型从 int 扩展为 tuple[int,int] | int     │
│  Request.__init__ 内部归一化: int → (int, 0)         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  vLLM Priority Min-Heap Scheduler                     │
│  值越小越优先，二元组字典序比较                        │
│  (3, -500) < (3, -300) < (5, -500)                   │
└─────────────────────────────────────────────────────┘
```

**所有改动对非 DAG 请求完全透明**：`dag_id` / `node_id` 都是 Optional，不带这些字段的请求走原有逻辑（`request.priority` 作为单 int）。

---

## 二、两级优先级调度算法（核心）

### 2.1 公式

```
priority = (-dag_remaining_cp, -node_cp_remaining)
```

vLLM 调度器使用 **min-heap**（值越小越优先）。二元组按字典序比较：先比第一级，同分再比第二级。

### 2.2 第一级：DAG 级调度（shortest-DAG-first）

```
dag_remaining_cp = max{ cp_remaining(n) | n ∈ frontier_nodes }
```

- **frontier_nodes**: 当前未完成节点中**入度为 0** 的节点（所有前驱均已完成，或为源节点）
- **为什么用 max 而非 sum？** 并行分支取 max 避免重复计算。若 `A→B` 和 `A→C` 是并行分支，剩余代价是 `max(B, C)` 而非 `B+C`——它们可以同时跑
- **语义**：整个 DAG 剩余工作量越小的 DAG，整体越优先调度 → **让短 DAG 先跑完，减少平均 makespan**

### 2.3 第二级：节点级调度（critical-path-first）

```
node_cp_remaining = node_cost + max{ cp_remaining(s) | s ∈ successors }
```

- 递归定义，叶子节点 `cp_remaining = node_cost`
- 带 memoization：`_cp_cache` 保证每个节点只算一次，O(V+E)
- 优先级取反（`-node_cp_remaining`）→ **关键路径越长的节点越优先**（先跑能解锁更多下游工作，减少 DAG makespan）

### 2.4 实例

假设 DAG 拓扑：
```
     A (cost=100)
    / \
 B(50) C(200)
    \ /
     D (cost=80)
```

- `cp_remaining(D)` = 80
- `cp_remaining(B)` = 50 + 80 = 130
- `cp_remaining(C)` = 200 + 80 = 280
- `cp_remaining(A)` = 100 + max(130, 280) = 380

初始 frontier = {A}，dag_remaining_cp = 380

A 完成后 frontier = {B, C}，dag_remaining_cp = max(130, 280) = 280

节点优先级排序（同 DAG 内）：C（-280）> B（-130）> D（-80）——含 C 的关键路径更长，先调度 C

---

## 三、动态 DAG 拓扑扩展（关键创新）

### 3.1 问题

Multi-agent workflow 中，subagent 的 LLM 调用轮数在 DAG 注册时**未知**（取决于模型实际推理——tool call 是否继续、任务是否完成）。

静态 DAG 无法表达这种"运行时才知道有几轮"的场景。

### 3.2 解决方案：`append_round`

**注册时**只注册每个 subagent 的第一轮（如 `sub_A`）。后续轮次通过运行时 API 动态追加。

**拓扑变换**：
```
  原边:  parent → [s1, s2, ...]
  变为:  parent → new_node → [s1, s2, ...]
```

- `new_node` **继承** parent 的全部原有后继
- parent 的后继被替换为仅 `[new_node]`
- 旧后继的 predecessors 从 parent 改为 new_node

**示例** — 两个 subagent 并行处理，其中一个需要额外一轮：
```
初始 DAG:
   sub_A ──→ join
   sub_B ──→ join

sub_A 跑完后需要第二轮 → append_round(parent="sub_A", new="sub_A__r1"):
   sub_A → sub_A__r1 → join
   sub_B ──────────────→ join    ← join 仍然正确等待所有分支！
```

**生命周期**：
1. 注册时只注册第一轮节点
2. 每个中间轮完成后，调用 `POST .../next-round` 插入下一轮
3. 最终轮（不再继续），调用 `POST .../complete` 退出 frontier

### 3.3 局部缓存失效（性能优化）

**问题**：插入新节点后，哪些节点的 `cp_remaining` 需要重算？

**分析**：`cp_remaining(x)` 递归定义为 `x.cost + max(cp_remaining(x's successors))`，只依赖后继。插入新节点改变了 parent 的后继集合路径长度，因此只需要清除 parent **及其祖先**（沿 predecessors 反向 BFS 到根）的 cp_cache。

**复杂度**：O(DAG 深度) 而非 O(总节点数)。对于高 fan-out、每个 subagent 平均 3-5 轮的场景，这避免了每轮都触发全图重算。

**正确性保证**：兄弟分支的 cp_cache 值天然不依赖被修改节点的任何信息，保留不清不会读到脏值。

---

## 四、API 端点一览

| 方法 | 路径 | 功能 |
|---|---|---|
| `POST` | `/v1/dags` | 注册 DAG（nodes + edges），返回 dag_id + num_nodes |
| `GET` | `/v1/dags` | 列出所有已注册 DAG 的 ID |
| `PATCH` | `/v1/dags/{dag_id}/nodes/{node_id}` | 在线更新节点 expected_tokens（LLM 自估计后回调） |
| `POST` | `/v1/dags/{dag_id}/nodes/{parent}/next-round` | 在 parent 后动态追加一个运行时节点（subagent 多轮） |
| `POST` | `/v1/dags/{dag_id}/nodes/{node_id}/complete` | 标记节点完成（最终轮，退出 frontier） |

### 4.1 请求/响应示例

**注册 DAG**:
```json
POST /v1/dags
{
  "dag_id": "task-001",
  "nodes": [
    {"node_id": "sub_A", "expected_tokens": 500},
    {"node_id": "sub_B", "expected_tokens": 800},
    {"node_id": "join", "expected_tokens": 300}
  ],
  "edges": [["sub_A", "join"], ["sub_B", "join"]]
}
```

**在线代价更新**:
```json
PATCH /v1/dags/task-001/nodes/sub_A
{"expected_tokens": 620}
```

**动态追加轮次**:
```json
POST /v1/dags/task-001/nodes/sub_A/next-round
{"new_node_id": "sub_A__r2", "expected_tokens": 450}
```

**Chat Completion（携带 DAG 元数据）**:
```json
POST /v1/chat/completions
{
  "dag_id": "task-001",
  "node_id": "sub_A__r2",
  "model": "llama-3-8b",
  "messages": [...]
}
```

---

## 五、改动文件清单

### 新增文件（3 个）

| 文件 | 行数 | 说明 |
|---|---|---|
| `vllm/dag_registry.py` | 459 | DAG 注册表核心：单例存储、cp_remaining 计算、两级 priority、动态节点追加、缓存管理 |
| `vllm/dag_api_router.py` | 215 | FastAPI 路由：5 个 REST 端点 + attach_router 入口 |
| `deploy_to_venv.sh` | 30 | 一键部署脚本（rsync 到目标 venv） |

### 修改文件（10 个）

| 文件 | 改动 | 说明 |
|---|---|---|
| `vllm/entrypoints/openai/chat_completion/protocol.py` | +16 | `ChatCompletionRequest` 新增 `dag_id` 和 `node_id` 可选字段 |
| `vllm/entrypoints/openai/chat_completion/serving.py` | +26/-1 | 从 registry 查询 priority 替代 request.priority；请求完成后 mark_completed |
| `vllm/entrypoints/openai/api_server.py` | +4 | 注册 dag_api_router 到 FastAPI app |
| `vllm/v1/request.py` | +6/- | priority 类型 `int → tuple[int, int] \| int`，内部归一化 |
| `vllm/v1/engine/__init__.py` | +6/- | `EngineCoreRequest.priority` 类型扩展 |
| `vllm/v1/engine/async_llm.py` | +8/- | 所有 `generate()` 签名的 priority 类型扩展 |
| `vllm/v1/engine/input_processor.py` | +3/- | 同上 |
| `vllm/engine/protocol.py` | +4/- | `EngineClient` 抽象接口 priority 类型扩展 |
| `vllm/entrypoints/llm.py` | +2/- | 离线推理接口 priority 类型扩展 |
| `vllm/tool_parsers/hermes_tool_parser.py` | +14/- | 处理带 `\<think\>` 标签的推理模型输出，tool call 提取前 strip CoT 块 |

---

## 六、设计原则与工程细节

### 6.1 向后兼容

- `priority` 类型 `tuple[int, int] | int`，内部归一化：`int → (int, 0)`
- `dag_id` / `node_id` 均为 Optional，默认 None
- 只有当两者均非空 **且** registry 中存在该 DAG 时，才走 DAG-aware 路径
- 找不到 DAG 时 fallback 到 `request.priority`
- 非 DAG 请求的行为完全不变

### 6.2 单例机制

**问题**：Python 多 import 路径（`vllm.dag_registry` vs `dag_registry` vs 相对 import）可能导致同一个模块被加载多次，创建多个 `_dags` 字典。

**方案**：`_DAGStore` 实例存储在 `sys.modules["__dag_registry_store__"]`，不挂在模块的 global 变量上。任何 import 路径拿到的都是 sys.modules 中的同一个对象。

### 6.3 线程安全

所有对 `_DAGStore._dags` 的读写操作都在 `threading.Lock()` 保护下进行。vLLM 引擎是多线程处理请求的，必须保证并发注册/查询不出现脏读。

### 6.4 缓存策略

- `_cp_cache: dict[str, int]`：缓存每个节点的 cp_remaining 值，memoization 保证每个节点只计算一次
- `_dag_remaining_cache: int | None`：缓存 DAG 级剩余关键路径，completed_nodes 变化或节点 cost 更新时失效
- 局部失效而非全清：`_invalidate_ancestors(node_id)` 仅清除受影响节点的祖先链，O(深度) 而非 O(节点数)

---

## 七、面试追问清单

### Q: 为什么不把 DAG 存外部数据库？
单进程内存存储，延迟微秒级。调度决策在毫秒级完成，不适合引入 IPC/网络开销。这个场景不需要持久化——DAG 的生命周期等于一次 workflow 执行。

### Q: 时间复杂度？
- DAG 注册：O(V+E)
- compute_priority：缓存命中 O(1)，首次 O(V+E)
- append_round：O(depth)，局部缓存失效
- 所有操作均摊接近 O(1)

### Q: 和 vLLM 原生 priority 机制的关系？
vLLM 已有 int 型 priority 字段用于请求调度。你的工作将其扩展为二元组 `(dag_priority, node_priority)`，利用 min-heap 的字典序比较实现两级排序。非 DAG 请求继续用 `(priority_int, 0)`。

### Q: 做了什么实验验证？
（需要你补充具体数据 — makespan、throughput、tail latency 对比 FIFO baseline / node-only priority / 两级 priority）

### Q: hermes_tool_parser 改了什么？为什么？
处理使用推理模型（如 DeepSeek-R1）作为 subagent 时的兼容性问题：这些模型输出包含 `\<think\>...\</think\>` CoT 块，原始 parser 在 `model_output` 直接搜 tool_call_start_token，但 think 块内可能包含类似 token 导致误匹配。修改后先用正则 strip 掉 think 块，再在剩余文本中搜索。

### Q: 和现有的 workflow orchestration（如 Dify、LangGraph）有什么区别？
那些是做 workflow 编排的框架，而你是在 **serving/inference 层**做 DAG-aware 调度。两者互补——编排层定义 DAG 拓扑，调用你的 API 注册到 serving 层，serving 层据此做优先级调度。关键区别是你让推理引擎**感知** DAG 结构来做全局优化，而不是把每个节点当作独立的推理请求。

### Q: 如果 DAG 中某个节点失败怎么办？
当前设计中，失败由客户端/harness 处理。节点失败后不调用 mark_completed，该分支的 cp_remaining 继续计入 dag_remaining_cp，DAG 不会被认为是"完成"的。重启节点通过 append_round 追加新节点（替换失败节点在 DAG 中的位置）。

---

## 八、Commit 时间线

| 顺序 | Commit | 作者 | 日期 | 内容 |
|---|---|---|---|---|
| 1 | `2a92c83d2` | 迟云帆 (cyf) | 2026-05-07 | DAG Registry 注册/查询 + ChatCompletion 协议扩展 |
| 2 | `b21e42d58` | 迟云帆 (cyf) | 2026-05-07 | PATCH 端点 + LLM 在线代价估计 + 缓存失效 |
| 3 | `ac9b83ffa` | 於晨阳 (ycy) | 2026-06-11 | 两级优先级调度 + 动态节点追加 + 全链路类型扩展 |
