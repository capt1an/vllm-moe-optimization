# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark for the routed grouped-GEMM pipeline (Triton fused_moe_kernel).

Decomposes ``fused_experts`` into align + kernel, measures the padding waste
of ``moe_align_block_size`` and the effective TFLOPS of the Triton kernel,
against a dense-GEMM control at the same FLOPs. Targeted at the
Qwen1.5-MoE shape under 2-rank EP: E=30 local experts, N=1408, K=2048,
topk=4, bf16.

Usage (run with the environment that has vLLM installed, e.g. the
motivation venv):

    /home/ycy/dag-aware-vllm/experiments/motivation/.venv/bin/python \
        benchmarks/kernels/benchmark_moe_pipeline.py \
        --m 32 128 512 2048 8192 --iters 100
"""

import argparse

import torch

from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    get_default_config,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

# A6000 (GA102) bf16 tensor-core dense peak, TFLOPS (for reference only).
_PEAK_TFLOPS = 154.9


def time_fn(fn, warmup: int, iters: int) -> float:
    """Return mean GPU time of fn() in ms (CUDA events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+",
                        default=[32, 128, 512, 2048, 8192])
    parser.add_argument("--e", type=int, default=30,
                        help="local experts (EP=2 for Qwen1.5-MoE)")
    parser.add_argument("--n", type=int, default=1408,
                        help="expert intermediate size")
    parser.add_argument("--k", type=int, default=2048, help="hidden size")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    # Optional tile overrides for kernel experiments (override_config path).
    parser.add_argument("--block-m", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--group-m", type=int, default=None)
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print(f"GPU: {torch.cuda.get_device_name()}  "
          f"peak bf16 dense ~= {_PEAK_TFLOPS} TFLOPS")
    print(f"shape: E={args.e} N={args.n} K={args.k} topk={args.topk} bf16")
    print(f"{'M':>6} | {'config src':<20} {'tiles M/N/K/G':<16} "
          f"{'pad%':>6} {'align':>7} {'full':>7} {'kernel':>7} "
          f"{'TFLOPS':>8} {'dense':>7} {'denseTF':>8} {'%peak':>6}")
    print("-" * 110)

    for M in args.m:
        hidden = torch.randn(M, args.k, device=device, dtype=dtype)
        w1 = torch.randn(args.e, 2 * args.n, args.k, device=device, dtype=dtype)
        w2 = torch.randn(args.e, args.k, args.n, device=device, dtype=dtype)
        topk_weights = torch.rand(M, args.topk, device=device)
        topk_ids = torch.randint(
            0, args.e, (M, args.topk), device=device
        ).to(torch.int32)

        # --- config selection (same path as production: dtype=None for bf16) ---
        config = try_get_optimal_moe_config(
            w1.size(), w2.size(), args.topk, None, M
        )
        default = get_default_config(M, args.e, args.n, args.k, args.topk, None)
        cfg_src = "default" if config == default else "tuned-json"

        overrides = {}
        if args.block_m is not None:
            overrides["BLOCK_SIZE_M"] = args.block_m
        if args.block_n is not None:
            overrides["BLOCK_SIZE_N"] = args.block_n
        if args.block_k is not None:
            overrides["BLOCK_SIZE_K"] = args.block_k
        if args.group_m is not None:
            overrides["GROUP_SIZE_M"] = args.group_m
        if args.num_warps is not None:
            overrides["num_warps"] = args.num_warps
        if args.num_stages is not None:
            overrides["num_stages"] = args.num_stages
        if overrides:
            cfg_src = "override"
            config = dict(config)
            config.update(overrides)
            config["SPLIT_K"] = 1
        tiles = f"{config['BLOCK_SIZE_M']}/{config['BLOCK_SIZE_N']}/" \
                f"{config['BLOCK_SIZE_K']}/{config.get('GROUP_SIZE_M', 1)}"

        # --- align: padding waste + per-expert imbalance ---
        block_size = config["BLOCK_SIZE_M"]
        sorted_ids, _expert_ids, num_post_pad = moe_align_block_size(
            topk_ids, block_size, args.e
        )
        padded = num_post_pad.item()
        pad_pct = 100.0 * (padded - M * args.topk) / (M * args.topk)

        def align_fn():
            moe_align_block_size(topk_ids, block_size, args.e)

        def full_fn():
            if overrides:
                with override_config(config):
                    fused_experts(
                        hidden, w1, w2, topk_weights, topk_ids, inplace=False
                    )
            else:
                fused_experts(
                    hidden, w1, w2, topk_weights, topk_ids, inplace=False
                )

        align_ms = time_fn(align_fn, args.warmup, args.iters)
        full_ms = time_fn(full_fn, args.warmup, args.iters)
        kernel_ms = max(full_ms - align_ms, 0.0)

        # --- dense control: same FLOPs + activation, two matmuls ---
        rows = M * args.topk
        a = torch.randn(rows, args.k, device=device, dtype=dtype)
        w1d = torch.randn(2 * args.n, args.k, device=device, dtype=dtype)
        w2d = torch.randn(args.k, args.n, device=device, dtype=dtype)

        def dense_fn():
            out = a @ w1d.t()                       # [rows, 2N]
            mid = out[:, : args.n] * torch.nn.functional.silu(out[:, args.n :])
            mid @ w2d.t()                           # [rows, N] x [N, K]

        dense_ms = time_fn(dense_fn, args.warmup, args.iters)

        # FLOPs per token-expert pair: gemm1 2*K*2N + gemm2 2*N*K
        flops = rows * (2 * args.k * 2 * args.n + 2 * args.n * args.k)
        kernel_tflops = flops / (kernel_ms * 1e-3) / 1e12
        dense_tflops = flops / (dense_ms * 1e-3) / 1e12

        print(
            f"{M:>6} | {cfg_src:<20} {tiles:<16} {pad_pct:>5.1f}% "
            f"{align_ms:>6.3f}ms {full_ms:>6.3f}ms {kernel_ms:>6.3f}ms "
            f"{kernel_tflops:>7.1f} {dense_ms:>6.3f}ms {dense_tflops:>7.1f} "
            f"{100 * kernel_tflops / _PEAK_TFLOPS:>5.1f}%"
        )

    # per-expert token distribution for the last (largest) M
    counts = torch.bincount(
        torch.randint(0, args.e, (args.m[-1] * args.topk,), device="cpu"),
        minlength=args.e,
    )
    print(
        f"\nper-expert token counts at M={args.m[-1]} (uniform routing): "
        f"min={counts.min().item()} mean={counts.float().mean().item():.0f} "
        f"max={counts.max().item()}"
    )


if __name__ == "__main__":
    main()
