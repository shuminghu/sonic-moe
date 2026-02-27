# Integrating SonicMoE with Expert Parallelism (EP)

---

**SonicMoE** introduces state-of-the-art memory and compute optimizations for fine-grained Mixture of Experts (MoE) models. While the paper primarily evaluates single-GPU execution (using Gather Fusion to handle scattered tokens), its core kernels are incredibly valuable for multi-node **Expert Parallelism (EP)** setups.

This guide explains how to adapt SonicMoE for EP, bypassing its built-in gather logic while retaining its massive memory and throughput benefits.

## 1. Why use SonicMoE with EP?

In an EP setup, tokens are routed across the network via an All-to-All communication step. By the time tokens arrive at a specific GPU, they are already packed into a **contiguous tensor**. You do not need SonicMoE's "Gather Fusion."

However, you *do* want SonicMoE's other two major innovations:

1. **Epilogue Fusion:** SonicMoE fuses the `SwiGLU` activation function natively into the GEMM's epilogue. In the backward pass, it simultaneously computes the activation gradient ($dH$) and the router gradient ($dS$) before writing to HBM.
2. **O(TKd) Memory Bypassing:** By leveraging the fused epilogue, SonicMoE completely bypasses the need to cache the massive intermediate $Y$ tensor, drastically lowering peak activation memory.
3. **Ping-Pong Scheduling:** The kernels aggressively overlap memory IO with Tensor Core math, perfectly hiding the latency of memory-bound fine-grained experts.

## 2. The Solution: The "Identity Array" Trick

Because EP provides contiguous tokens, you cannot use SonicMoE's high-level `MoE` class (which assumes scattered tokens on a single GPU). Instead, you must call the **raw C++ functional bindings** directly.

To force the SonicMoE kernels to process your contiguous EP data without rewriting the CUDA code, you pass an **Identity Array** (`[0, 1, 2, ..., N]`) to the kernel's `x_gather_idx` argument.

### Does this "No-Op Gather" hurt performance?

**No.** Passing an identity array forces the kernel to issue `cp.async` instructions for sequential indices. You will not suffer a performance penalty because:

- **Hardware Coalescing:** The GPU's memory controller sees sequential `cp.async` requests and automatically coalesces them into massive, contiguous burst reads from HBM, operating at peak memory bandwidth.
- **Instruction Hiding:** Because SonicMoE uses asynchronous `cp.async` instructions and Ping-Pong scheduling, the few clock cycles spent reading the identity array are completely hidden behind the heavy math being executed by the Tensor Cores.

## 3. Implementation Template

Below is a drop-in PyTorch `autograd.Function` template. You can insert this between your forward All-to-All (Scatter) and backward All-to-All (Combine) steps.

```python
import torch
from sonicmoe.enums import ActivationType
from sonicmoe.functional.forward import _up_projection_forward, _down_projection_forward
from sonicmoe.functional.backward import (
    _up_projection_backward_act,
    _up_projection_backward_weight,
    _down_projection_backward_act,
    _down_projection_backward_weight,
)

class SonicMoELocalExperts(torch.autograd.Function):
    """
    Executes the local expert computation for an EP setup using SonicMoE kernels.
    Assumes `x_gathered` is received contiguously from a forward All-to-All.
    """
    @staticmethod
    def forward(
        ctx,
        x_gathered: torch.Tensor,           # [Total_Tokens, Hidden_Dim] (Contiguous)
        w1: torch.Tensor,                   # [2*Intermediate_Dim, Hidden_Dim, Local_Experts]
        w2: torch.Tensor,                   # [Hidden_Dim, Intermediate_Dim, Local_Experts]
        expert_offsets: torch.Tensor,       # [Local_Experts + 1] (e.g., [0, N1, N1+N2])
        router_scores: torch.Tensor,        # [Total_Tokens]
        activation_type=ActivationType.SWIGLU
    ):
        TK, H = x_gathered.shape
        I = w1.size(0) // 2  # Assuming SwiGLU
        stream_id = torch.cuda.current_stream().cuda_stream

        # 1. The Identity Trick: Force contiguous loading
        identity_indices = torch.arange(TK, dtype=torch.int32, device=x_gathered.device)

        # 2. Allocate intermediate and output tensors
        z = torch.empty(TK, 2 * I, dtype=x_gathered.dtype, device=x_gathered.device)
        y1 = torch.empty(TK, I, dtype=x_gathered.dtype, device=x_gathered.device)
        y2 = torch.empty(TK, H, dtype=x_gathered.dtype, device=x_gathered.device)

        # 3. Up-Projection (Math + Fused SwiGLU Epilogue)
        _up_projection_forward(
            x=x_gathered, w1=w1, z=z, y1=y1, b1=None,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity_indices,  # Sequential load via identity
            stream_id=stream_id,
            activation_type=activation_type.value,
            is_glu_activation=True,
            is_inference_mode_enabled=False,
        )

        # 4. Down-Projection
        _down_projection_forward(
            w2=w2, y1=y1, y2=y2, b2=None,
            expert_frequency_offset=expert_offsets,
            expert_schedule_order=None,
            x_gather_idx=identity_indices,
            stream_id=stream_id,
        )

        ctx.save_for_backward(x_gathered, w1, w2, z, router_scores, expert_offsets, identity_indices)
        ctx.stream_id = stream_id

        return y2  # Ready for local router score multiplication and return All-to-All

    @staticmethod
    def backward(ctx, dy2: torch.Tensor):
        x_gathered, w1, w2, z, router_scores, expert_offsets, identity_indices = ctx.saved_tensors
        stream_id = ctx.stream_id

        # Allocate Gradients
        dw1 = torch.empty_like(w1)
        dw2 = torch.empty_like(w2)
        dz = torch.empty_like(z)
        dx_gathered = torch.empty_like(x_gathered)

        y1s = torch.empty(x_gathered.size(0), w2.size(1), dtype=z.dtype, device=z.device)
        ds = torch.empty_like(router_scores) # Router gradients!

        # 1. Down-Projection Backward (Fused dSwiGLU and dS calculation)
        _down_projection_backward_act(
            dout=dy2, z=z, w2=w2, dz=dz, ds=ds, b2=None, db2=None, y1s=y1s,
            topk_scores=router_scores,
            expert_frequency_offset=expert_offsets, expert_schedule_order=None,
            x_gather_idx=identity_indices, s_scatter_idx=identity_indices,
            is_glu_activation=True, activation_type=ActivationType.SWIGLU.value,
            stream_id=stream_id,
        )
        _down_projection_backward_weight(
            dout=dy2, y1s=y1s, dw2=dw2,
            expert_frequency_offset=expert_offsets, expert_schedule_order=None,
            x_gather_idx=identity_indices, stream_id=stream_id,
        )

        # 2. Up-Projection Backward
        _up_projection_backward_act(
            w1=w1, dx_expanded=dx_gathered, dz=dz, db1=None,
            expert_frequency_offset=expert_offsets, expert_schedule_order=None,
            x_gather_idx=identity_indices, s_scatter_idx=identity_indices,
            is_glu_activation=True, stream_id=stream_id,
        )
        _up_projection_backward_weight(
            x=x_gathered, dw1=dw1, dz=dz,
            expert_frequency_offset=expert_offsets, expert_schedule_order=None,
            x_gather_idx=identity_indices, is_glu_activation=True,
            stream_id=stream_id,
        )

        # dx_gathered is contiguous and ready to be sent backwards over the All-to-All network
        return dx_gathered, dw1, dw2, None, ds, None
```

## 4. Critical: Weight Formatting

SonicMoE kernels bypass standard PyTorch matrix multiplication and rely on highly specific memory layouts optimized for Hopper Tensor Cores. Your EP framework **must** format the expert weights exactly as follows before passing them into the function above:

- **`w1` (Up-Projection):**
    - Shape: `(2 * Intermediate_Dim, Hidden_Dim, Local_Experts)`
    - Stride Order: `(2, 0, 1)`
    - Note: The Gate and Up projection rows must be interleaved `[gate_row0, up_row0, gate_row1, up_row1...]`.
- **`w2` (Down-Projection):**
    - Shape: `(Hidden_Dim, Intermediate_Dim, Local_Experts)`
    - Stride Order: `(2, 0, 1)`
