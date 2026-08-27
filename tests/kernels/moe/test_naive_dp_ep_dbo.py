# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import torch

import vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep as agrs


class _FakeEPGroup:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def dispatch(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        is_sequence_parallel=False,
        extra_tensors=None,
    ):
        self.calls.append("dispatch")
        assert extra_tensors is None
        return hidden_states, topk_weights, topk_ids

    def combine(self, hidden_states, is_sequence_parallel=False):
        self.calls.append("combine")
        return hidden_states + 1


class _FakeWeightAndReduce:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def apply(self, **kwargs):
        self.calls.append("weight_and_reduce")
        return kwargs["fused_expert_output"].sum(dim=1)


def _patch_dbo(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(agrs, "dbo_enabled", lambda: True)
    monkeypatch.setattr(
        agrs,
        "dbo_yield_and_switch_from_compute_to_comm",
        lambda: calls.append("yield_compute_to_comm"),
    )
    monkeypatch.setattr(
        agrs,
        "dbo_yield_and_switch_from_comm_to_compute",
        lambda: calls.append("yield_comm_to_compute"),
    )
    monkeypatch.setattr(agrs, "get_ep_group", lambda: _FakeEPGroup(calls))


def test_async_support_mode_selection(monkeypatch):
    prepare_finalize = agrs.MoEPrepareAndFinalizeNaiveDPEPModular(
        num_dispatchers=2
    )

    monkeypatch.setattr(agrs, "dbo_enabled", lambda: False)
    monkeypatch.setattr(agrs.envs, "VLLM_MOE_NAIVE_AGRS_ASYNC", False)
    assert not prepare_finalize.supports_async()

    monkeypatch.setattr(agrs.envs, "VLLM_MOE_NAIVE_AGRS_ASYNC", True)
    assert prepare_finalize.supports_async()

    monkeypatch.setattr(agrs.envs, "VLLM_MOE_NAIVE_AGRS_ASYNC", False)
    monkeypatch.setattr(agrs, "dbo_enabled", lambda: True)
    assert prepare_finalize.supports_async()


def test_async_combine_stream_is_shared_per_device(monkeypatch):
    created_streams = []
    monkeypatch.setattr(agrs.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        agrs.torch.cuda,
        "Stream",
        lambda device: created_streams.append(object()) or created_streams[-1],
    )
    agrs._ASYNC_COMBINE_STREAMS.clear()

    first = agrs._get_async_combine_stream()
    second = agrs._get_async_combine_stream()

    assert first is second
    assert len(created_streams) == 1
    agrs._ASYNC_COMBINE_STREAMS.clear()


def test_dbo_prepare_dispatches_between_stream_switches(monkeypatch):
    calls: list[str] = []
    _patch_dbo(monkeypatch, calls)
    prepare_finalize = agrs.MoEPrepareAndFinalizeNaiveDPEPModular(
        num_dispatchers=2
    )
    hidden_states = torch.randn(4, 8)
    topk_weights = torch.randn(4, 2)
    topk_ids = torch.zeros((4, 2), dtype=torch.int64)

    receiver = prepare_finalize.prepare_async(
        hidden_states,
        topk_weights,
        topk_ids,
        num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=Mock(),
        defer_input_quant=True,
    )

    assert calls == [
        "yield_compute_to_comm",
        "dispatch",
        "yield_comm_to_compute",
    ]
    dispatched, scale, metadata, dispatched_ids, dispatched_weights = receiver()
    assert dispatched is hidden_states
    assert scale is None
    assert metadata is None
    assert dispatched_ids is topk_ids
    assert dispatched_weights is topk_weights


def test_dbo_finalize_combines_between_stream_switches(monkeypatch):
    calls: list[str] = []
    _patch_dbo(monkeypatch, calls)
    prepare_finalize = agrs.MoEPrepareAndFinalizeNaiveDPEPModular(
        num_dispatchers=2
    )
    output = torch.zeros(4, 8)
    fused_expert_output = torch.randn(4, 2, 8)
    topk_weights = torch.randn(4, 2)
    topk_ids = torch.zeros((4, 2), dtype=torch.int64)

    receiver = prepare_finalize.finalize_async(
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=_FakeWeightAndReduce(calls),
    )

    assert calls == [
        "weight_and_reduce",
        "yield_compute_to_comm",
        "combine",
        "yield_comm_to_compute",
    ]
    assert torch.count_nonzero(output) == 0
    receiver()
    torch.testing.assert_close(output, fused_expert_output.sum(dim=1) + 1)
