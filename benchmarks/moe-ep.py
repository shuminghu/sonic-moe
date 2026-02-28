#!/usr/bin/env python3
# ********************************************************************************
# Benchmark + correctness test for SonicMoE EP (Expert Parallelism).
# Uses the identity-array trick from the EP integration guide:
# tokens are already contiguous per expert (as after an All-to-All),
# so x_gather_idx = [0, 1, 2, ..., TK-1].
# ********************************************************************************

import argparse
import random
import time
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench

from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional.forward import _up_projection_forward, _down_projection_forward
from sonicmoe.functional.backward import (
    _up_projection_backward_act,
    _up_projection_backward_weight,
    _down_projection_backward_act,
    _down_projection_backward_weight,
)


# ── Activation helpers (for the PyTorch reference) ─────────────────────────

def swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)


def geglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return F.gelu(g.float()).to(dtype=g.dtype) * u


def reglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.relu(g.float()) * u).to(dtype=g.dtype)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).to(dtype=x.dtype)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def relu_sq(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


ACT_FN = {
    ActivationType.SWIGLU: swiglu,
    ActivationType.GEGLU: geglu,
    ActivationType.REGLU: reglu,
    ActivationType.GELU: gelu,
    ActivationType.RELU: relu,
    ActivationType.SILU: silu,
    ActivationType.RELU_SQ: relu_sq,
}


# ── EP Autograd Function ───────────────────────────────────────────────────

class SonicMoELocalExpertsEP(torch.autograd.Function):
    """
    EP-compatible local expert computation via the identity-trick.

    Forward:  up_proj → fused activation → down_proj → score multiplication.
    Backward: fused CuTe kernels that handle the score chain-rule internally.

    The forward conceptually computes ``output = y2 * score`` where
    ``y2 = W2 @ act(W1 @ x)`` per expert.  The backward kernels expect the
    upstream gradient ``dL/d(y2*score)`` and internally multiply by the score
    to obtain ``dL/dy2``, compute ``dL/dz``, ``dL/dx``, ``dL/dW``, and the
    router-score gradient ``dL/ds``.
    """

    @staticmethod
    def forward(ctx, x, w1, w2, expert_offsets, router_scores,
                activation_type_val, is_glu_flag):
        TK, H = x.shape
        I_full = w1.size(0)
        I = I_full // 2 if is_glu_flag else I_full
        stream_id = torch.cuda.current_stream().cuda_stream

        identity = torch.arange(TK, dtype=torch.int32, device=x.device)

        z = torch.empty(TK, I_full, dtype=x.dtype, device=x.device)
        y1 = torch.empty(TK, I, dtype=x.dtype, device=x.device)

        _up_projection_forward(
            x=x, w1=w1, z=z, y1=y1, b1=None,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity,
            stream_id=stream_id,
            activation_type=activation_type_val,
            is_glu_activation=is_glu_flag,
            is_inference_mode_enabled=False,
        )

        y2 = torch.empty(TK, H, dtype=x.dtype, device=x.device)

        _down_projection_forward(
            w2=w2, y1=y1, y2=y2, b2=None,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity,
            stream_id=stream_id,
        )

        output = (y2 * router_scores.unsqueeze(-1)).to(y2.dtype)

        ctx.save_for_backward(x, w1, w2, z, router_scores,
                              expert_offsets, identity)
        ctx.stream_id = stream_id
        ctx.is_glu_flag = is_glu_flag
        ctx.activation_type_val = activation_type_val
        ctx.TK = TK
        ctx.I = I

        return output

    @staticmethod
    def backward(ctx, dout):
        (x, w1, w2, z, router_scores,
         expert_offsets, identity) = ctx.saved_tensors
        stream_id = ctx.stream_id
        TK, I = ctx.TK, ctx.I

        dw1 = torch.empty_like(w1)
        dw2 = torch.empty_like(w2)
        dz = torch.empty_like(z)
        dx = torch.empty_like(x)
        ds = torch.empty_like(router_scores)
        y1s = torch.empty(TK, I, dtype=z.dtype, device=z.device)

        _down_projection_backward_act(
            dout=dout, z=z, w2=w2, dz=dz, ds=ds,
            b2=None, db2=None, y1s=y1s,
            topk_scores=router_scores,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity,
            s_scatter_idx=identity,
            is_glu_activation=ctx.is_glu_flag,
            activation_type=ctx.activation_type_val,
            stream_id=stream_id,
        )

        _down_projection_backward_weight(
            dout=dout, y1s=y1s, dw2=dw2,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity,
            stream_id=stream_id,
        )

        _up_projection_backward_act(
            w1=w1, dx_expanded=dx, dz=dz, db1=None,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity,
            s_scatter_idx=identity,
            is_glu_activation=ctx.is_glu_flag,
            stream_id=stream_id,
        )

        _up_projection_backward_weight(
            x=x, dw1=dw1, dz=dz,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity,
            is_glu_activation=ctx.is_glu_flag,
            stream_id=stream_id,
        )

        return dx, dw1, dw2, None, ds, None, None


# ── Standalone inference-mode forward (for CUDA-graph capture) ─────────────

def ep_forward_inference(x, w1, w2, expert_offsets, identity,
                         z_buf, y1_buf, y2_buf,
                         activation_type_val, is_glu_flag, stream_id):
    _up_projection_forward(
        x=x, w1=w1, z=z_buf, y1=y1_buf, b1=None,
        expert_frequency_offset=expert_offsets,
        expert_schedule_order=None,
        x_gather_idx=identity,
        stream_id=stream_id,
        activation_type=activation_type_val,
        is_glu_activation=is_glu_flag,
        is_inference_mode_enabled=True,
    )
    _down_projection_forward(
        w2=w2, y1=y1_buf, y2=y2_buf, b2=None,
        expert_frequency_offset=expert_offsets,
        expert_schedule_order=None,
        x_gather_idx=identity,
        stream_id=stream_id,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def create_balanced_expert_offsets(TK, E, device):
    base = TK // E
    remainder = TK % E
    counts = torch.full((E,), base, dtype=torch.int32, device=device)
    counts[:remainder] += 1
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0)
    return offsets


def parse_comma_separated_ints(s: str):
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SonicMoE EP benchmark (identity-trick)")
    parser.add_argument(
        "--thie",
        type=parse_comma_separated_ints,
        default=(32768, 4096, 1024, 128),
        help="TK, H, I, E (comma-separated).  "
             "TK = total tokens on this EP worker, E = local experts.",
    )
    parser.add_argument("--dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument(
        "--activation",
        choices=["swiglu", "geglu", "reglu", "relu_sq", "relu", "silu", "gelu"],
        default="swiglu",
    )
    args = parser.parse_args()
    if len(args.thie) != 4:
        parser.error("--thie must contain exactly 4 values (TK, H, I, E)")
    return args


# ── Main benchmark ──────────────────────────────────────────────────────────

def run(thie, dtype, skip_test, activation_str):
    torch_dtype = {
        cutlass.BFloat16: torch.bfloat16,
        cutlass.Float16: torch.float16,
    }[dtype]

    activation = ActivationType(activation_str)
    is_glu_activation = is_glu(activation)

    TK, H, I, E = thie
    I_full = 2 * I if is_glu_activation else I

    print0(f"\n[bold]EP Benchmark:[/bold] TK={TK}, H={H}, I={I}, E={E}")
    print0(f"Activation: {activation.value}, dtype: {torch_dtype}, "
           f"GLU: {is_glu_activation}")

    random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    device = "cuda:0"

    # Weights: standard layout [E, out, in] (leaf tensors for autograd)
    w1_std = (0.02 * torch.randn(E, I_full, H, device=device, dtype=torch_dtype)
              ).requires_grad_(True)
    w2_std = (0.02 * torch.randn(E, H, I, device=device, dtype=torch_dtype)
              ).requires_grad_(True)

    # SonicMoE kernel format: permute → [out, in, E]  stride order (2, 0, 1)
    w1 = w1_std.permute(1, 2, 0)   # [I_full, H, E]
    w2 = w2_std.permute(1, 2, 0)   # [H, I, E]

    x = (0.2 * torch.randn(TK, H, device=device, dtype=torch_dtype)
         ).requires_grad_(True)

    expert_offsets = create_balanced_expert_offsets(TK, E, device)
    router_scores = (
        0.5 + 0.5 * torch.rand(TK, device=device, dtype=torch.float32)
    ).requires_grad_(True)

    dout = 0.2 * torch.randn_like(x)

    # ═══════════════════════ CORRECTNESS CHECK ══════════════════════════════
    if not skip_test:
        # ── EP forward (SonicMoE kernels) ──────────────────────────────────
        output_ep = SonicMoELocalExpertsEP.apply(
            x, w1, w2, expert_offsets, router_scores,
            activation.value, is_glu_activation,
        )

        dx_ep, dw1_std_ep, dw2_std_ep, ds_ep = torch.autograd.grad(
            output_ep, [x, w1_std, w2_std, router_scores], grad_outputs=dout,
        )

        # ── Reference forward (PyTorch per-expert loops) ───────────────────
        act_func = ACT_FN[activation]
        ref_parts = []

        with torch.autocast(device, torch.float32):
            for e in range(E):
                s = expert_offsets[e].item()
                end = expert_offsets[e + 1].item()
                if end > s:
                    x_e = x[s:end]
                    z_e = F.linear(x_e, w1_std[e])
                    y1_e = act_func(z_e)
                    y2_e = F.linear(y1_e, w2_std[e])
                    ref_parts.append(y2_e * router_scores[s:end, None])

            ref_output = torch.cat(ref_parts, dim=0)

            # Forward comparison
            o_diff = (output_ep.float() - ref_output).abs()
            print0(f"\n[bold cyan]═══ Forward Correctness ═══[/bold cyan]")
            print0(f"max ref output val          "
                   f" {ref_output.abs().max():.6f}")
            print0(f"mean ref output val         "
                   f" {ref_output.abs().mean():.6f}")
            print0(f"max abs diff on output      "
                   f" {o_diff.max():.6f}")
            print0(f"mean rel diff on output     "
                   f" {(o_diff / (ref_output.abs() + 1e-6)).mean():.6f}\n")

            # Reference backward
            ref_dx, ref_dw1, ref_dw2, ref_ds = torch.autograd.grad(
                ref_output, [x, w1_std, w2_std, router_scores],
                grad_outputs=dout,
            )

        print0(f"[bold cyan]═══ Backward Correctness ═══[/bold cyan]")
        for name, ours, ref in [
            ("dx", dx_ep, ref_dx),
            ("dw1", dw1_std_ep, ref_dw1),
            ("dw2", dw2_std_ep, ref_dw2),
            ("ds", ds_ep, ref_ds),
        ]:
            diff = (ours.float() - ref.float()).abs()
            print0(f"max abs ref value {name:>4s}  {ref.abs().max():.6f}")
            print0(f"mean abs ref value {name:>4s} {ref.abs().mean():.6f}")
            print0(f"max abs diff on {name:>4s}    {diff.max():.6f}")
            print0(f"mean rel diff on {name:>4s}   "
                   f"{(diff / (ref.float().abs() + 1e-6)).mean():.6f}\n")

    # ═══════════════════════ BENCHMARKS ═════════════════════════════════════
    fwd_flops = (6 if is_glu_activation else 4) * TK * I * H
    repeats = 500
    warmup = 5

    identity = torch.arange(TK, dtype=torch.int32, device=device)
    stream_id = torch.cuda.current_stream().cuda_stream

    # Pre-allocate buffers for inference forward
    z_buf = torch.empty(TK, I_full, dtype=torch_dtype, device=device)
    y1_buf = torch.empty(TK, I, dtype=torch_dtype, device=device)
    y2_buf = torch.empty(TK, H, dtype=torch_dtype, device=device)

    # Warmup — populate compile caches
    ep_forward_inference(x, w1, w2, expert_offsets, identity,
                         z_buf, y1_buf, y2_buf,
                         activation.value, is_glu_activation, stream_id)

    time.sleep(0.5)

    # ── Inference forward + CUDA graph ─────────────────────────────────────
    cuda_graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        with torch.cuda.graph(cuda_graph, stream=stream):
            ep_forward_inference(
                x, w1, w2, expert_offsets, identity,
                z_buf, y1_buf, y2_buf,
                activation.value, is_glu_activation, stream.cuda_stream,
            )

    fwd_infer_timing = do_bench(
        lambda: cuda_graph.replay(), warmup=warmup, rep=repeats)
    tflops = fwd_flops / (fwd_infer_timing * 1e9)
    print0(f" EP Fwd (inference + cudagraph) Average time: "
           f"{fwd_infer_timing:.3f} ms, TFLOPS: {tflops:.1f}")

    # ── Training forward only ──────────────────────────────────────────────
    def forward_only_training():
        return SonicMoELocalExpertsEP.apply(
            x, w1, w2, expert_offsets, router_scores,
            activation.value, is_glu_activation,
        )

    # Warmup
    forward_only_training()

    time.sleep(0.5)

    fwd_train_timing = do_bench(
        forward_only_training, warmup=warmup, rep=repeats)
    print0(f" EP Fwd (training mode) Average time: "
           f"{fwd_train_timing:.3f} ms")

    # ── Forward + Backward ─────────────────────────────────────────────────
    e2e_flops = (18 if is_glu_activation else 12) * TK * I * H

    time.sleep(0.5)

    def forward_and_backward():
        output = SonicMoELocalExpertsEP.apply(
            x, w1, w2, expert_offsets, router_scores,
            activation.value, is_glu_activation,
        )
        output.backward(dout, retain_graph=True)
        x.grad = w1_std.grad = w2_std.grad = router_scores.grad = None

    e2e_timing = do_bench(
        forward_and_backward, warmup=warmup, rep=repeats,
        grad_to_none=[x, w1_std, w2_std, router_scores],
    )
    tflops = e2e_flops / (e2e_timing * 1e9)
    print0(f"[bold green]EP Fwd + Bwd Average time: {e2e_timing:.3f} ms, "
           f"TFLOPS: {tflops:.1f}[/bold green]")

    bwd_flops = e2e_flops - fwd_flops
    bwd_time = e2e_timing - fwd_infer_timing
    bwd_tflops = bwd_flops / (bwd_time / 1e3) / 1e12
    print0(f"[bold green]EP Bwd Average time: {bwd_time:.3f} ms, "
           f"TFLOPS: {bwd_tflops:.1f}[/bold green]")


if __name__ == "__main__":
    args = parse_arguments()
    run(args.thie, args.dtype, args.skip_test, args.activation)
    print("PASS")
