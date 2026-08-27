# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Env-gated CUDA-event timing for the MoE forward path.

Enable with ``VLLM_MOE_TIMING_DEBUG=1``. Requires ``--enforce-eager``.

Timing is aggregated across the whole run: every
``VLLM_MOE_TIMING_AGG_EVERY`` MoE-layer forwards (default 4800) the pending
CUDA events are synchronized and folded into cumulative per-stage counters
(no output). The final per-stage summary (calls / total / avg / share) is
printed once at process exit; if ``VLLM_MOE_TIMING_OUT`` is set to a file
path, the summary is also written as JSON to ``<path>.pid<pid>.json``.

Stop the server gracefully (Ctrl+C / SIGTERM) so the atexit report runs;
``kill -9`` skips it.

Optional: ``VLLM_MOE_TIMING_PRINT_EVERY`` > 0 additionally prints the
running cumulative summary every that many MoE-layer forwards (default 0).
"""

import atexit
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("VLLM_MOE_TIMING_DEBUG", "0") == "1"
_AGG_EVERY = int(os.environ.get("VLLM_MOE_TIMING_AGG_EVERY", "4800"))
_PRINT_EVERY = int(os.environ.get("VLLM_MOE_TIMING_PRINT_EVERY", "0"))
_OUT_PATH = os.environ.get("VLLM_MOE_TIMING_OUT", "")

# Batch-size buckets (tokens per MoE-layer forward); the last bucket is open.
# Small buckets correspond to decode steps, large ones to prefill chunks.
_BUCKETS: list[tuple[int, int]] = [
    (1, 64),
    (65, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
    (1025, 2048),
    (2049, 4096),
    (4097, 8192),
    (8193, 10**9),
]

# (layer_name, stage_name) -> list of (forward_id, start_event, end_event)
_pending: dict[
    tuple[str, str], list[tuple[int, torch.cuda.Event, torch.cuda.Event]]
] = defaultdict(list)
# stage_name -> [calls, total_ms]
_stats: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
# cross-stream overlap between combine_comm (comm stream) and shared
# (main stream): [n_pairs, sum_ratio, sum_overlap_ms]
_overlap: list[float] = [0, 0.0, 0.0]
# (bucket, stage_name) -> [calls, total_ms, sumsq_ms2]
_bucket_stats: dict[tuple[tuple[int, int], str], list[float]] = defaultdict(
    lambda: [0, 0.0, 0.0]
)
# bucket -> number of forwards
_bucket_forwards: dict[tuple[int, int], int] = defaultdict(int)
_fwd_id = 0
_fwd_tokens: dict[int, int] = {}
_num_forward_calls = 0
_reported = False

# Per-layer routing histograms. These stay on the device while serving so
# route recording does not introduce a device-to-host synchronization for
# every MoE layer. They are copied to CPU only when the final report is built.
_logical_route_counts: dict[str, torch.Tensor] = {}
_physical_route_counts: dict[str, torch.Tensor] = {}
_route_calls: dict[str, int] = defaultdict(int)
_route_tokens: dict[str, int] = defaultdict(int)
_route_num_ranks: dict[str, int] = {}
_route_round_robin: dict[str, bool] = {}


@contextmanager
def stage(layer_name: str, stage_name: str) -> Iterator[None]:
    """Record a CUDA event pair around a block; a no-op when disabled."""
    if not _ENABLED:
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    _pending[(layer_name, stage_name)].append((_fwd_id, start, end))


def _accumulate_route_ids(
    counts_by_layer: dict[str, torch.Tensor],
    layer_name: str,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> None:
    counts = counts_by_layer.get(layer_name)
    if counts is None:
        counts = torch.zeros(
            num_experts,
            dtype=torch.int64,
            device=expert_ids.device,
        )
        counts_by_layer[layer_name] = counts
    elif counts.numel() != num_experts:
        raise ValueError(
            f"MoE route histogram size changed for {layer_name}: "
            f"{counts.numel()} -> {num_experts}"
        )

    flat_ids = expert_ids.detach().reshape(-1).long()
    counts.scatter_add_(
        0,
        flat_ids,
        torch.ones_like(flat_ids, dtype=counts.dtype),
    )


def record_routes(
    layer_name: str,
    logical_ids: torch.Tensor,
    physical_ids: torch.Tensor,
    num_logical_experts: int,
    num_physical_experts: int,
    num_ranks: int,
    round_robin: bool,
) -> None:
    """Accumulate logical and physical expert routing histograms."""
    if not _ENABLED:
        return
    _accumulate_route_ids(
        _logical_route_counts,
        layer_name,
        logical_ids,
        num_logical_experts,
    )
    _accumulate_route_ids(
        _physical_route_counts,
        layer_name,
        physical_ids,
        num_physical_experts,
    )
    _route_calls[layer_name] += 1
    _route_tokens[layer_name] += logical_ids.shape[0]
    _route_num_ranks[layer_name] = num_ranks
    _route_round_robin[layer_name] = round_robin


def _route_distribution(counts: torch.Tensor) -> dict:
    values = counts.cpu().tolist()
    total = int(sum(values))
    num_experts = len(values)
    mean = total / num_experts if num_experts else 0.0
    variance = (
        sum((value - mean) ** 2 for value in values) / num_experts
        if num_experts
        else 0.0
    )
    cv = variance**0.5 / mean if mean > 0 else 0.0
    sorted_values = sorted(values, reverse=True)

    def top_share(k: int) -> float:
        return sum(sorted_values[:k]) / total if total else 0.0

    return {
        "total_assignments": total,
        "active_experts": sum(value > 0 for value in values),
        "max_over_mean": max(values) / mean if mean > 0 else 0.0,
        "coefficient_of_variation": cv,
        "top1_share": top_share(1),
        "top5_share": top_share(5),
        "counts": values,
    }


def _routing_summary() -> dict:
    summary = {}
    for layer_name, logical_counts in sorted(_logical_route_counts.items()):
        physical_counts = _physical_route_counts[layer_name]
        num_ranks = _route_num_ranks[layer_name]
        physical_values = physical_counts.cpu().tolist()
        assert len(physical_values) % num_ranks == 0
        if _route_round_robin[layer_name]:
            rank_assignments = [
                sum(physical_values[rank::num_ranks]) for rank in range(num_ranks)
            ]
        else:
            experts_per_rank = len(physical_values) // num_ranks
            rank_assignments = [
                sum(
                    physical_values[
                        rank * experts_per_rank : (rank + 1) * experts_per_rank
                    ]
                )
                for rank in range(num_ranks)
            ]
        rank_mean = sum(rank_assignments) / num_ranks if num_ranks else 0.0
        summary[layer_name] = {
            "calls": _route_calls[layer_name],
            "tokens": _route_tokens[layer_name],
            "logical": _route_distribution(logical_counts),
            "physical": _route_distribution(physical_counts),
            "rank_assignments": rank_assignments,
            "rank_max_over_mean": (
                max(rank_assignments) / rank_mean if rank_mean > 0 else 0.0
            ),
            "rank_balancedness": (
                rank_mean / max(rank_assignments)
                if rank_assignments and max(rank_assignments) > 0
                else 0.0
            ),
        }
    return summary


def _bucket_for(num_tokens: int) -> tuple[int, int]:
    for lo, hi in _BUCKETS:
        if lo <= num_tokens < hi:
            return (lo, hi)
    return _BUCKETS[-1]


def _aggregate() -> None:
    """Synchronize and fold pending events into the cumulative counters."""
    torch.cuda.synchronize()
    seen_fwds: set[int] = set()
    for (_layer, stage_name), events in _pending.items():
        for fid, s, e in events:
            total = s.elapsed_time(e)
            _stats[stage_name][0] += 1
            _stats[stage_name][1] += total
            bucket = _bucket_for(_fwd_tokens.get(fid, -1))
            _bucket_stats[(bucket, stage_name)][0] += 1
            _bucket_stats[(bucket, stage_name)][1] += total
            _bucket_stats[(bucket, stage_name)][2] += total * total
            if fid not in seen_fwds:
                seen_fwds.add(fid)
                _bucket_forwards[bucket] += 1

    # Cross-stream overlap between the combine collective (comm stream) and
    # the shared experts (main stream). CUDA events live on a global device
    # timeline, so elapsed_time() across streams is valid. Per forward:
    #   union_span = earliest start -> latest end
    #               = max of the 4 pairwise elapsed values
    #   overlap    = shared_dur + comb_dur - union_span
    # Both stages use the same layer label and are appended once per forward,
    # so pairing by position is valid (requires every layer to have a shared
    # expert, which holds for the Qwen MoE layers instrumented here).
    comm_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
        defaultdict(list)
    )
    shared_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
        defaultdict(list)
    )
    for (layer, stage_name), events in _pending.items():
        if stage_name == "combine_comm":
            comm_events[layer].extend((s, e) for (_f, s, e) in events)
        elif stage_name == "shared":
            shared_events[layer].extend((s, e) for (_f, s, e) in events)
    for layer, comm in comm_events.items():
        shared = shared_events.get(layer, [])
        for (cs, ce), (ss, se) in zip(comm, shared):
            comb_dur = cs.elapsed_time(ce)
            shared_dur = ss.elapsed_time(se)
            span = max(
                cs.elapsed_time(ce),
                cs.elapsed_time(se),
                ss.elapsed_time(ce),
                ss.elapsed_time(se),
            )
            overlap = comb_dur + shared_dur - span
            shorter = min(comb_dur, shared_dur)
            ratio = overlap / shorter if shorter > 1e-3 else 0.0
            ratio = min(max(ratio, 0.0), 1.0)
            _overlap[0] += 1
            _overlap[1] += ratio
            _overlap[2] += overlap
    _pending.clear()
    _fwd_tokens.clear()


def _print_summary(running: bool = False) -> None:
    rows = []
    for stage_name, (calls, total) in sorted(_stats.items()):
        avg = total / calls if calls else 0.0
        rows.append((stage_name, calls, total, avg))
    # Sequential main-stream stages roughly add up to the layer latency;
    # use their sum as the share denominator (stages hidden by overlap
    # will show a small share, which is the point of this tool).
    total_avg = sum(r[3] for r in rows) or 1.0
    header = "cumulative" if running else "final"
    print(
        f"[MoE timing] {header} summary over {_num_forward_calls} "
        f"MoE-layer forwards ({torch.cuda.get_device_name()})",
        flush=True,
    )
    for stage_name, calls, total, avg in rows:
        print(
            f"  {stage_name:<10} calls={calls:>8}  total={total:>10.2f}ms "
            f"avg={avg:>7.3f}ms  share={100 * avg / total_avg:>5.1f}%",
            flush=True,
        )
    if _overlap[0] > 0:
        n, sum_ratio, sum_overlap = _overlap
        print(
            f"  {'OVERLAP':<10} pairs={n:>8}  avg_overlap={sum_overlap / n:>7.3f}ms"
            f"  ratio={100 * sum_ratio / n:>5.1f}%  (shared ∥ combine_comm)",
            flush=True,
        )
    if _bucket_forwards:
        total_fwds = sum(_bucket_forwards.values())
        print(
            "  --- per batch-size bucket (tokens per MoE forward; "
            "small=decode, large=prefill) ---",
            flush=True,
        )
        for bucket in sorted(_bucket_forwards):
            n = _bucket_forwards[bucket]
            lo, hi = bucket
            label = f"{lo}+" if hi >= 10**9 else f"{lo}-{hi - 1}"
            stages: dict[str, float] = {}
            for (b, s), (calls, total, _sq) in _bucket_stats.items():
                if b == bucket and calls > 0:
                    stages[s] = total / calls
            avg_layer = sum(stages.values())
            parts = "  ".join(f"{s}={v:.3f}" for s, v in sorted(stages.items()))
            print(
                f"  [{label:>9}]  fwds={n:>7} ({100 * n / total_fwds:>5.1f}%)  "
                f"avg_layer={avg_layer:.3f}ms  |  {parts}",
                flush=True,
            )
    if _logical_route_counts:
        print("  --- per-layer logical route skew ---", flush=True)
        for layer_name, route in _routing_summary().items():
            logical = route["logical"]
            print(
                f"  {layer_name}: assignments={logical['total_assignments']} "
                f"active={logical['active_experts']} "
                f"max/mean={logical['max_over_mean']:.3f} "
                f"cv={logical['coefficient_of_variation']:.3f} "
                f"top1={100 * logical['top1_share']:.1f}% "
                f"top5={100 * logical['top5_share']:.1f}% "
                f"rank_bal={route['rank_balancedness']:.3f}",
                flush=True,
            )


def flush(num_tokens: int | None = None) -> None:
    """Called once per MoE-layer forward; folds timings into cumulative stats.

    num_tokens is the batch size (tokens) of this forward, used to bucket
    per-batch-size statistics (decode vs prefill chunk sizes).
    """
    global _num_forward_calls, _fwd_id
    if not _ENABLED:
        return
    _num_forward_calls += 1
    _fwd_tokens[_fwd_id] = num_tokens if num_tokens is not None else -1
    _fwd_id += 1
    if _num_forward_calls % _AGG_EVERY != 0:
        return
    _aggregate()
    if _PRINT_EVERY > 0 and _num_forward_calls % _PRINT_EVERY == 0:
        _print_summary(running=True)


def _report_and_write() -> None:
    """atexit handler: print the final summary and optionally write JSON."""
    global _reported
    if _reported:
        return
    _reported = True
    if not _ENABLED:
        return
    try:
        if _pending:
            _aggregate()
        _print_summary()
        if _OUT_PATH:
            out = {
                "pid": os.getpid(),
                "device": torch.cuda.get_device_name(),
                "num_forward_calls": _num_forward_calls,
                "stages": {
                    name: {
                        "calls": int(calls),
                        "total_ms": round(total, 3),
                        "avg_ms": round(total / calls, 4) if calls else 0.0,
                    }
                    for name, (calls, total) in _stats.items()
                },
                "buckets": {
                    f"{lo}-{hi}": {
                        "fwds": _bucket_forwards.get((lo, hi), 0),
                        "stages": {
                            s: {
                                "calls": int(c),
                                "avg_ms": round(t / c, 4) if c else 0.0,
                            }
                            for (b, s), (c, t, _sq) in _bucket_stats.items()
                            if b == (lo, hi) and c > 0
                        },
                    }
                    for (lo, hi) in _BUCKETS
                    if (lo, hi) in _bucket_forwards
                },
                "routing": _routing_summary(),
            }
            path = f"{_OUT_PATH}.pid{os.getpid()}.json"
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[MoE timing] stats written to {path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[MoE timing] final report failed: {e}", flush=True)


atexit.register(_report_and_write)
