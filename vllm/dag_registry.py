"""
DAGRegistry — server-side DAG metadata store for DAG-aware scheduling.

Clients register their task DAGs once via POST /v1/dags. Subsequent chat
completion requests can carry dag_id + node_id fields so that the serving
layer computes scheduling priority from the DAG's critical path, rather than
requiring the client to supply a raw priority integer.

Priority encoding (二元组, vLLM min-heap, 值越小越优先):
    priority = (-dag_remaining_cp, -node_cp_remaining)

  - 第一级 dag_remaining_cp：整个 DAG 剩余的关键路径长度越小，
    该 DAG 的所有节点整体排在前面，保证"工作量最小的 DAG 先跑"。
  - 第二级 -node_cp_remaining：同一 DAG 内，节点自身关键路径越长
    （即解锁下游工作越多）越优先，最小化 DAG makespan。

dag_remaining_cp 定义：
    从所有"尚未完成且入度（在未完成子图中）为 0"的节点出发，
    取各节点 cp_remaining 的最大值，即当前剩余子图的真实关键路径长度。
    （对并行分支求 max 而非 sum，避免重复计算。）

单例机制：
    _DAGStore 通过 sys.modules 绑定到固定 key "__dag_registry_store__"。
    无论 dag_registry 以何种路径被 import（vllm.dag_registry /
    dag_registry / 相对 import 等），所有调用方共享同一个 _DAGStore
    实例，从根本上避免重复 import 导致的多份 _dags 字典问题。

动态节点扩展（subagent 多轮对话）：
    Register 时只需注册每个 subagent 的第一轮节点（node_id，例如 "sub_A"）。
    后续轮次未知 —— harness 在每轮 LLM 调用完成、决定还要继续下一轮时，
    调用 append_round(dag_id, parent_node_id="sub_A", new_node_id="sub_A__r1",
    expected_tokens=...) 把下一轮作为新节点插入 DAG。

    拓扑变化：parent → [旧后继] 变为 parent → new_node → [旧后继]，
    new_node 继承 parent 原有的全部下游依赖，因此任何 join 节点依然会正确地
    等待所有分支真正走完（无论分支走了几轮）才被触发。

    parent 节点在 append_round 后被标记为 completed（它已被新轮次"消费"，
    退出 frontier），new_node 成为该分支新的 frontier 节点。

    缓存失效是局部的：cp_remaining(x) 只依赖 x 的后继，因此插入新节点只需
    清除 parent 及其祖先链（沿 predecessors 反向回溯到根）的 cp_cache 条目，
    不影响兄弟分支，复杂度 O(DAG 深度) 而非 O(节点总数)。这对高 fan-out、
    每个 subagent 平均 3-5 轮的场景很重要 —— 避免每轮都触发全图重算。
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAG entry
# ---------------------------------------------------------------------------

@dataclass
class _DAGEntry:
    nodes: dict[str, int]              # node_id -> expected_tokens
    successors: dict[str, list[str]]   # node_id -> [successor node_ids]
    predecessors: dict[str, list[str]] # node_id -> [predecessor node_ids]

    _cp_cache: dict[str, int] = field(default_factory=dict, init=False)
    # None 表示缓存已失效，需要重新计算
    _dag_remaining_cache: int | None = field(default=None, init=False)
    completed_nodes: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # cp_remaining：节点自身 + 后继的最大关键路径（静态，不受 completed 影响）
    # ------------------------------------------------------------------

    def cp_remaining(self, node_id: str) -> int:
        """Total critical-path token cost from this node (inclusive) onward."""
        if node_id in self._cp_cache:
            return self._cp_cache[node_id]
        cost = self.nodes.get(node_id, 0)
        downstream = max(
            (self.cp_remaining(s) for s in self.successors.get(node_id, [])),
            default=0,
        )
        result = cost + downstream
        self._cp_cache[node_id] = result
        return result

    def downstream_cp(self, node_id: str) -> int:
        """Max cp_remaining among successors (excludes this node's own cost)."""
        return max(
            (self.cp_remaining(s) for s in self.successors.get(node_id, [])),
            default=0,
        )

    # ------------------------------------------------------------------
    # 局部缓存失效：仅清除受影响节点的祖先链，而非整图
    # ------------------------------------------------------------------

    def _invalidate_ancestors(self, node_id: str) -> None:
        """
        清除 node_id 自身及其所有祖先（沿 predecessors 反向 BFS 直到 root）的
        cp_cache 条目。

        cp_remaining(x) 的递归定义只依赖 x 的后继，因此任何节点 cost 或拓扑变化
        只会影响"以它为后继的所有祖先"的 cp_remaining 值，与它的兄弟分支或
        非祖先节点无关 —— 这就是局部失效相比 entry._cp_cache.clear() 的正确性
        基础：兄弟分支的 cp_cache 条目天然不依赖 node_id，留着不清也不会读到
        脏值。

        复杂度 O(祖先链长度)，而非 O(全图节点数)。对于 subagent 多轮场景，
        祖先链长度通常是 DAG 深度，远小于总节点数（尤其是高 fan-out 的 DAG）。
        """
        visited: set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            self._cp_cache.pop(cur, None)
            stack.extend(self.predecessors.get(cur, []))

    # ------------------------------------------------------------------
    # dag_remaining_cp：当前剩余子图的关键路径长度
    # ------------------------------------------------------------------

    def dag_remaining_cp(self) -> int:
        """剩余子图的关键路径长度（max，非 sum）。

        在未完成节点中，找出入度为 0 的 frontier 节点
        （所有前驱均已完成，或本身就是源节点），取其 cp_remaining 的最大值。
        """
        if self._dag_remaining_cache is not None:
            return self._dag_remaining_cache
        result = self._compute_dag_remaining_cp()
        self._dag_remaining_cache = result
        logger.debug(
            "dag_remaining_cp computed: %d  (completed=%s)",
            result, self.completed_nodes,
        )
        return result

    def _compute_dag_remaining_cp(self) -> int:
        if not self.nodes:
            return 0
        frontier = [
            n for n in self.nodes
            if n not in self.completed_nodes
            and all(p in self.completed_nodes
                    for p in self.predecessors.get(n, []))
        ]
        if not frontier:
            return 0
        return max(self.cp_remaining(n) for n in frontier)

    def invalidate_dag_remaining_cache(self) -> None:
        """节点完成或 cost 更新后调用，使 dag_remaining 缓存失效。"""
        self._dag_remaining_cache = None

    # ------------------------------------------------------------------
    # 动态节点追加：subagent 运行时按轮次扩展 DAG
    # ------------------------------------------------------------------

    def append_round(
        self,
        *,
        parent_node_id: str,
        new_node_id: str,
        expected_tokens: int,
        mark_parent_completed: bool = True,
    ) -> bool:
        """
        在 parent_node_id 和它原有的后继之间插入一个新节点 new_node_id，
        用于表达"subagent 的下一轮对话"这一运行时才出现的节点。

        拓扑变化：
            原边  parent → [s1, s2, ...]
            变为  parent → new_node → [s1, s2, ...]

        即 new_node 继承 parent 原有的全部后继，parent 的后继被替换为仅
        [new_node]。这保证了 join 节点（多个分支汇合的下游节点）依然能在
        所有分支各自走完自己的轮次之后才被触发 —— 因为 new_node 的后继
        集合就是 parent 之前指向的下游集合，传递性保持不变。

        parent_node_id 不存在时返回 False（调用方应当 404）。
        new_node_id 已存在时返回 False（避免重复 round 误覆盖拓扑），调用方
        应当视为幂等失败而非异常 —— 常见于客户端重试 PATCH 请求的场景。

        mark_parent_completed:
            默认 True —— 一旁的 round 完成后，旧 round 节点即被"消费"，从
            frontier 中退出，新 round 节点成为新的 frontier。
            设为 False 可用于"预先占位但尚未真正完成"的场景（例如还在
            decode 中途就想让下一轮的优先级提前生效），此时 parent 仍保留
            在 frontier 集合判定逻辑之外（因为它已有后继，不会再是 frontier）。
        """
        if parent_node_id not in self.nodes:
            return False
        if new_node_id in self.nodes:
            return False  # idempotent no-op; caller treats as failure to retry-safe

        old_children = self.successors.get(parent_node_id, [])

        # 1. 注册新节点
        self.nodes[new_node_id] = max(1, expected_tokens)
        self.successors[new_node_id] = old_children
        self.predecessors[new_node_id] = [parent_node_id]

        # 2. 重连：parent 的后继替换为 [new_node_id]
        self.successors[parent_node_id] = [new_node_id]

        # 3. 旧后继的 predecessors 指向从 parent 改为 new_node_id
        for child in old_children:
            preds = self.predecessors.setdefault(child, [])
            if parent_node_id in preds:
                preds[preds.index(parent_node_id)] = new_node_id
            elif new_node_id not in preds:
                preds.append(new_node_id)

        # 4. 局部失效：new_node 插入会改变 parent 及其所有祖先的 cp_remaining
        #    （因为 parent 的 downstream 现在多了一层）。new_node 本身和
        #    old_children 的 cp_remaining 值不变（cp_remaining 只依赖后继，
        #    new_node 的后继集合 == parent 原来的后继集合，值相同，可保留）。
        self._invalidate_ancestors(parent_node_id)

        if mark_parent_completed:
            self.completed_nodes.add(parent_node_id)
        self.invalidate_dag_remaining_cache()

        logger.debug(
            "append_round: parent=%s new_node=%s tokens=%d old_children=%s",
            parent_node_id, new_node_id, expected_tokens, old_children,
        )
        return True


# ---------------------------------------------------------------------------
# Singleton store — 绑定到 sys.modules，跨 import 路径共享同一实例
# ---------------------------------------------------------------------------

_STORE_KEY = "__dag_registry_store__"


class _DAGStore:
    """进程级单例，持有所有 DAG 的状态。

    通过 sys.modules[_STORE_KEY] 存储，确保无论 dag_registry 以哪种路径
    被 import，所有调用方都操作同一份数据。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dags: dict[str, _DAGEntry] = {}

    # ------------------------------------------------------------------
    # public API（供模块级函数委托调用）
    # ------------------------------------------------------------------

    def register(
        self,
        dag_id: str,
        nodes: dict[str, int],
        edges: list[tuple[str, str]],
    ) -> None:
        successors: dict[str, list[str]] = {n: [] for n in nodes}
        predecessors: dict[str, list[str]] = {n: [] for n in nodes}
        for src, dst in edges:
            successors.setdefault(src, []).append(dst)
            predecessors.setdefault(dst, []).append(src)
        with self._lock:
            self._dags[dag_id] = _DAGEntry(
                nodes=dict(nodes),
                successors=successors,
                predecessors=predecessors,
            )
        logger.debug("registered dag '%s' with %d nodes", dag_id, len(nodes))

    def update_node_cost(self, dag_id: str, node_id: str, new_cost: int) -> bool:
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None or node_id not in entry.nodes:
                return False
            entry.nodes[node_id] = max(1, new_cost)
            # 局部失效：cost 变化只影响 node_id 自身及其祖先的 cp_remaining，
            # 不影响兄弟分支或下游节点 —— 全图 clear() 在节点数较多、
            # 高频 Layer-2 重估场景下是不必要的开销。
            entry._invalidate_ancestors(node_id)
            entry.invalidate_dag_remaining_cache()
            return True

    def append_round(
        self,
        dag_id: str,
        parent_node_id: str,
        new_node_id: str,
        expected_tokens: int,
    ) -> bool:
        """
        在已完成（或即将完成）的 parent_node_id 之后追加一个运行时才出现的
        新节点 new_node_id，代表 subagent 的下一轮对话。

        Returns False if dag_id unknown, parent_node_id unknown, or
        new_node_id already exists (idempotent-safe for retried PATCHes).
        """
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None:
                return False
            return entry.append_round(
                parent_node_id=parent_node_id,
                new_node_id=new_node_id,
                expected_tokens=expected_tokens,
            )

    def compute_priority(
        self, dag_id: str, node_id: str
    ) -> tuple[int, int] | None:
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None or node_id not in entry.nodes:
                return None
            dag_remaining  = entry.dag_remaining_cp()
            node_remaining = entry.cp_remaining(node_id)
        return (dag_remaining, -node_remaining)

    def mark_completed(self, dag_id: str, node_id: str) -> bool:
        with self._lock:
            entry = self._dags.get(dag_id)
            if entry is None:
                logger.warning(
                    "mark_completed: dag '%s' not found (store id=%d)",
                    dag_id, id(self._dags),
                )
                return False
            if node_id not in entry.nodes:
                logger.warning(
                    "mark_completed: node '%s' not found in dag '%s'",
                    node_id, dag_id,
                )
                return False
            entry.completed_nodes.add(node_id)
            entry.invalidate_dag_remaining_cache()
            logger.debug(
                "mark_completed: dag='%s' node='%s' completed=%s",
                dag_id, node_id, entry.completed_nodes,
            )
            return True

    def list_dags(self) -> list[str]:
        with self._lock:
            return list(self._dags.keys())


def _get_store() -> _DAGStore:
    """返回进程唯一的 _DAGStore 实例。

    首次调用时创建并写入 sys.modules[_STORE_KEY]；后续调用直接返回缓存值。
    无论本模块以 'vllm.dag_registry'、'dag_registry' 还是相对路径被 import，
    所有调用方拿到的都是同一个对象。
    """
    store = sys.modules.get(_STORE_KEY)
    if store is None:
        store = _DAGStore()
        sys.modules[_STORE_KEY] = store  # type: ignore[assignment]
        logger.debug("_DAGStore created (id=%d)", id(store))
    return store  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 模块级公开函数（保持原有调用接口不变）
# ---------------------------------------------------------------------------

def register(
    dag_id: str,
    nodes: dict[str, int],
    edges: list[tuple[str, str]],
) -> None:
    """
    Register a DAG.

    Parameters
    ----------
    dag_id : str
        Unique identifier for this DAG (e.g. a task UUID).
    nodes : dict[str, int]
        Mapping of node_id to expected token output cost.
    edges : list[tuple[str, str]]
        Directed edges (src_node_id, dst_node_id) representing dependencies.
    """
    _get_store().register(dag_id, nodes, edges)


def update_node_cost(dag_id: str, node_id: str, new_cost: int) -> bool:
    """
    Update a node's expected token cost with an online LLM estimate.

    Clears only the affected node's ancestor chain from the cp_remaining
    cache (not the whole DAG) so subsequent compute_priority() calls use the
    new cost. Thread-safe; returns False if dag_id/node_id unknown.
    """
    return _get_store().update_node_cost(dag_id, node_id, new_cost)


def append_round(
    dag_id: str,
    parent_node_id: str,
    new_node_id: str,
    expected_tokens: int,
) -> bool:
    """
    Append a runtime-discovered round node after parent_node_id.

    Use this when a subagent's number of LLM rounds is not known upfront:
    register only the first round at DAG-registration time, then call this
    once per subsequent round as the conversation actually progresses.

    The new node is spliced in between parent_node_id and parent's existing
    successors, so any downstream join node still waits correctly. parent_node_id
    is marked completed (it has been "consumed" by the new round), and the
    new node becomes the new frontier node for this branch.

    Returns False if dag_id or parent_node_id is unknown, or if new_node_id
    is already registered (idempotent-safe — a retried PATCH from the client
    won't corrupt the topology a second time).
    """
    return _get_store().append_round(
        dag_id, parent_node_id, new_node_id, expected_tokens
    )


def compute_priority(dag_id: str, node_id: str) -> tuple[int, int] | None:
    """
    Return the scheduling priority for (dag_id, node_id).

    Returns
    -------
    (dag_priority, node_priority) : tuple[int, int]
        Both components are negated so that larger values sort first in
        vLLM's min-heap PriorityRequestQueue:
          - dag_priority  = -dag_remaining_cp  (DAG 剩余关键路径越长越优先)
          - node_priority = -node_cp_remaining (节点关键路径越长越优先)

    Returns None if dag_id or node_id is not registered, allowing the
    caller to fall back to request.priority.
    """
    return _get_store().compute_priority(dag_id, node_id)


def mark_completed(dag_id: str, node_id: str) -> bool:
    """
    Mark a node as completed and invalidate the dag_remaining cache.

    节点完成后，剩余子图的关键路径可能缩短，必须使 dag_remaining_cp 缓存
    失效，确保后续 compute_priority 使用最新值。
    """
    return _get_store().mark_completed(dag_id, node_id)


def list_dags() -> list[str]:
    return _get_store().list_dags()