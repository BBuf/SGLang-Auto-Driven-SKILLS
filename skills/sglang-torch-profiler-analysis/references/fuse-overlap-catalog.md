# Fuse And Overlap Catalog

This catalog is the source-backed lookup table that the profiler skill should
consult before labeling a fuse or overlap opportunity as novel.

Use it like this:

1. Start from the three `triage` tables.
2. Match top rows against the `Trace keywords` and `Primary code` columns below.
3. If a finding matches an existing row, report it as:
   - an existing optimization path that is missing, disabled, or regressed, or
   - an already-known pattern that should be re-applied to the current model shape.
4. Only call a finding "new" when it does not fit any row in this catalog.

The catalog is grouped by reusable optimization family, not by one specific model.

## 1. LLM / SRT fused-kernel families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| FlashInfer unified `allreduce_fusion` | `cross_device_reduce_1stage*`<br>`all_reduce`<br>`FusedAddRMSNormKernel`<br>`rmsnorm*` | `python/sglang/srt/layers/flashinfer_comm_fusion.py`<br>`python/sglang/srt/layers/layernorm.py::_forward_with_allreduce_fusion`<br>`python/sglang/srt/layers/communicator.py::apply_flashinfer_allreduce_fusion` | FlashInfer workspace creation plus `allreduce_fusion(..., pattern=AllReduceFusionPattern.kARResidualRMSNorm, ...)` | First suspect missing / disabled / unsupported FlashInfer allreduce fusion, not a brand new TP fusion idea. |
| AITER allreduce fusion | ROCm all-reduce plus RMSNorm still split | `python/sglang/srt/layers/layernorm.py::_forward_with_allreduce_fusion`<br>`python/sglang/srt/distributed/communication_op.py::tensor_model_parallel_fused_allreduce_rmsnorm`<br>`python/sglang/srt/layers/communicator.py::apply_aiter_all_reduce_fusion` | ROCm-side fused TP all-reduce + RMSNorm with fallback to plain all-reduce plus norm | On AMD, rule out existing AITER fusion before proposing a new communication fusion. |
| Generic TP fused all-reduce wrapper | fuse table points to `_forward_with_allreduce_fusion` | `python/sglang/srt/distributed/communication_op.py::tensor_model_parallel_fused_allreduce_rmsnorm`<br>`python/sglang/srt/layers/layernorm.py::_forward_with_allreduce_fusion` | Shared wrapper used by multiple TP backends | Ask which backend-specific path should have fired here, not whether any fusion exists. |
| Fused dual residual RMSNorm | residual add plus two RMSNorm-like kernels around Grok blocks | `python/sglang/srt/layers/elementwise.py::fused_dual_residual_rmsnorm`<br>`python/sglang/srt/models/grok.py` | One Triton kernel computes intermediate residual update and next RMSNorm output together | On Grok-like residual layouts, treat split residual+norm as missing existing fusion. |
| In-place QK RMSNorm | split `q_norm` / `k_norm` kernels | `python/sglang/srt/models/utils.py::apply_qk_norm`<br>`python/sglang/jit_kernel/norm.py::fused_inplace_qknorm` | In-place JIT QK norm plus optional `alt_stream` overlap for K | Check shape, dtype, deterministic mode, and in-place legality before proposing a new QK fuse. |
| Fused QK RMSNorm + RoPE | `qknorm*` + `rope*` + `rotary*` as separate steps | `python/sglang/jit_kernel/fused_qknorm_rope.py`<br>`python/sglang/srt/models/qwen3_moe.py` | One JIT kernel applies QK RMSNorm and RoPE in-place on packed QKV | For compatible LLMs, classify split QK norm + RoPE as a missing existing fusion. |
| Fused RoPE + KV cache store | RoPE followed by KV-store, DtoD, or cache-write kernels | `python/sglang/jit_kernel/rope.py` | Shared entrypoints can route to fused RoPE + KV-store | Compare against the fused cache-store path before proposing a new KV rewrite. |
| Fused MoE dispatch / permute / combine | token permutation<br>dispatch / combine<br>grouped top-k<br>many small MoE support kernels | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`<br>`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` | `FusedMoE` plus DeepEP / FlashInfer / FuseEP / standard dispatch backends and `permute_fusion=True` | First ask whether the model is missing an existing `FusedMoE`-style path or backend-specific dispatcher path. |
| Fused MoE sum + all-reduce | routed MoE followed by explicit sum-reduce kernels | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`<br>`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` | `fuse_sum_all_reduce=True` path in the second MoE GEMM | Before inventing a new MoE reduction fuse, check whether `enable_fused_moe_sum_all_reduce` is simply off or the quant path is incompatible. |
| Kimi-K2 fused MoE gate | grouped-topk<br>sigmoid<br>router kernels dominate K2-like traces | `python/sglang/srt/layers/moe/topk.py::kimi_k2_moe_fused_gate` | Dedicated fused gate for the K2 `384 experts / num_expert_group=1` case | For K2-like layouts, this is an existing fused-gate precedent. |
| FlashInfer Cutlass MoE FP4 all-gather / reduce-scatter | `all_gather`<br>`reduce_scatter`<br>quantize<br>scale-interleave around FP4 MoE | `python/sglang/srt/layers/moe/token_dispatcher/standard.py`<br>`python/sglang/srt/layers/moe/utils.py::should_use_flashinfer_cutlass_moe_fp4_allgather` | Quantize hidden states before comm, `all_gatherv(...)`, then `reduce_scatterv(...)` on combine | If FP4 MoE still uses full-precision gather / scatter, check this path before suggesting a new comm optimization. |
| KV cache store fast path and split-copy overlap | K-cache / V-cache copies<br>`Memcpy DtoD` around decode | `python/sglang/srt/mem_cache/memory_pool.py` | `store_cache(...)` fast path plus graph-capture alt-stream overlap for K/V copies | Compare against existing cache-store fast path and copy overlap before suggesting a new cache optimization. |

## 2. LLM / SRT overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| Single-batch overlap (SBO) | MoE combine, down-gemm, shared-expert work in nearby two-stream windows | `python/sglang/srt/batch_overlap/single_batch_overlap.py` | combine vs down-gemm overlap, combine vs shared-expert overlap, one-stream dispatch+shared overlap, explicit SM partitioning and events | If exposed MoE combine sits near neighboring compute, classify it against SBO before calling it new overlap. |
| Two-batch overlap (TBO) | alternating kernel groups from two logical batches | `python/sglang/srt/batch_overlap/two_batch_overlap.py::model_forward_maybe_tbo`<br>`python/sglang/srt/model_executor/cuda_graph_runner.py`<br>`python/sglang/srt/model_executor/model_runner.py` | split-input two-batch scheduling and overlapped stage execution | When a model qualifies for TBO, alternating batch windows usually mean existing TBO is not fully effective. |
| Q and K normalization on different streams | Q-side norm and K-side norm on different streams | `python/sglang/srt/models/utils.py::apply_qk_norm`<br>`python/sglang/srt/models/qwen3.py`<br>`python/sglang/srt/models/qwen3_next.py`<br>`python/sglang/srt/models/qwen3_5.py` | Q stays on current stream, K can run on `alt_stream` in capture mode | Treat split Q/K norm as an existing overlap family when `alt_stream` is already wired. |
| DeepSeek shared-expert / routed-expert overlap | shared-expert GEMMs near DeepEP dispatch / combine | `python/sglang/srt/models/deepseek_v2.py`<br>`python/sglang/srt/batch_overlap/single_batch_overlap.py` | shared experts on `alt_stream`, overlap with dispatch / combine and down-gemm, Blackwell-specific env gating | This is an established routed-vs-shared branch overlap pattern, not a novel idea. |
| Llama4 shared branch vs routed branch overlap | shared expert branch plus routed MoE branch as adjacent windows | `python/sglang/srt/models/llama4.py` | shared expert on current stream, router+topk+routed experts on `alt_stream` | Use Llama4 as the first precedent for branch-level overlap in similar sparse models. |
| ExaoneMoE shared experts vs router experts overlap | shared expert output and router-expert output form a two-branch window | `python/sglang/srt/models/exaone_moe.py::forward_normal_dual_stream` | shared experts on current stream, router+routed experts on `alt_stream`, explicit join before combine | This is an existing dual-stream MoE overlap family. |
| Grok residual-MoE branch overlap | dense MLP and block-sparse MoE branches in parallel | `python/sglang/srt/models/grok.py::moe_with_rmoe` | dense MLP on current stream, MoE on `alt_stream`, fused dual residual RMSNorm around boundaries | Treat exposed Grok branch overlap as an existing pattern. |
| NSA dual-stream overlap | Q-proj, K-proj, RoPE, cache-store, quantization in tight two-stream windows | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | Q/K projection split, RoPE split, cache-store vs quantization overlap | NSA already contains several dual-stream overlap precedents. |
| Generic `alt_stream` overlap families | `alt_stream` plus explicit `wait_stream` / `with torch.cuda.stream(...)` | `qwen2_moe.py`<br>`qwen3_moe.py`<br>`glm4_moe.py`<br>`bailing_moe.py`<br>`llada2.py`<br>`grok.py`<br>`olmo2.py`<br>`step3p5.py`<br>`longcat_flash.py`<br>`falcon_h1.py` | model-specific overlap on attention prep, MoE branches, or cache-store | Search these families before designing a new overlap scheme from scratch. |

## 3. VLM-specific fuse and overlap families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| Vision QK norm with aux stream | vision-side QK norm or norm-like kernels before attention | `python/sglang/srt/layers/attention/vision.py` | vision QK normalization can call shared `apply_qk_norm(...)`, with K-side work on `aux_stream` | If vision QK prep is split, first check this existing aux-stream path. |
| ViT CUDA graph disables vision aux stream | expected vision overlap is absent under ViT graph | `python/sglang/srt/models/internvl.py`<br>`python/sglang/srt/layers/attention/vision.py`<br>`python/sglang/srt/environ.py::SGLANG_VIT_ENABLE_CUDA_GRAPH` | vision `aux_stream` is intentionally disabled when ViT CUDA graph is on | Missing vision overlap may be intentional, not a regression. |
| TP / DP vision-embedding gather patterns | `all_gather`, DtoD, padding, reconstruction around vision tower output | `python/sglang/srt/multimodal/mm_utils.py` | TP all-gather for vision embeddings and DP-sharded vision encoding reconstruction | Treat exposed vision gather traffic as an existing multimodal communication family. |
| VLMs reusing MoE fused paths | VLM trace exposes language-side MoE plumbing | `python/sglang/srt/models/kimi_vl.py`<br>`python/sglang/srt/models/ernie45_vl.py`<br>`python/sglang/srt/models/qwen3_vl_moe.py`<br>`python/sglang/srt/models/step3_vl.py`<br>`python/sglang/srt/models/internvl.py` | Many VLMs directly reuse language-model-side `FusedMoE` patterns | Compare with existing LLM MoE fused paths before proposing a VLM-specific MoE rewrite. |

## 4. Diffusion fused-kernel families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| Fused residual + norm + scale + shift | residual add, norm, scale, shift, gate around DiT blocks | `python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py`<br>`python/sglang/multimodal_gen/runtime/layers/layernorm.py` | `fused_scale_residual_norm_scale_shift(...)` | Treat split residual+norm+modulation as a missing existing diffusion fusion first. |
| Fused norm + scale + shift | norm followed by scale / shift elementwise kernels | `python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py`<br>`python/sglang/multimodal_gen/runtime/layers/layernorm.py` | `fused_norm_scale_shift(...)` | Existing modulation fusion already covers this family. |
| Triton scale / shift and gate-select kernels | tiny scale / shift or gate-select kernels dominate modulation blocks | `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`<br>`python/sglang/multimodal_gen/runtime/layers/elementwise.py` | `fuse_scale_shift_kernel(...)` and `fuse_layernorm_scale_shift_gate_select01_kernel(...)` | Check whether the runtime is missing these existing Triton fusions. |
| Fused add-RMSNorm and one-pass RMSNorm | residual add plus RMSNorm still split on short hidden sizes | `python/sglang/multimodal_gen/runtime/layers/layernorm.py`<br>`python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py` | `fused_add_rmsnorm(...)` and `triton_one_pass_rms_norm(...)` | For short hidden-size diffusion blocks, this is already an established fusion family. |
| Fused diffusion QK norm + RoPE | split QK norm and RoPE in diffusion attention blocks | `python/sglang/jit_kernel/diffusion/qknorm_rope.py`<br>`python/sglang/multimodal_gen/runtime/layers/layernorm.py::apply_qk_norm_rope` | `fused_inplace_qknorm_rope(...)`, with fallback to QK norm plus `apply_flashinfer_rope_qk_inplace(...)` | Distinguish between missing fused qknorm+rope and the existing FlashInfer RoPE fallback. |
| Fused QKV / added-QKV projection packing | separate Q, K, V linears or separate added-Q / K / V linears on packed-checkpoint models | `python/sglang/multimodal_gen/runtime/models/dits/flux.py`<br>`python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`<br>`python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`<br>`python/sglang/multimodal_gen/runtime/models/dits/zimage.py` | `MergedColumnParallelLinear(...)` for `to_qkv` and `to_added_qkv` under compatible quant configs | If projections remain split, first check whether packed fused projections are already supported for the current checkpoint / quant mode. |

## 5. Diffusion overlap and async-communication families

| Pattern | Trace keywords | Primary code | Existing path | Skill should conclude |
| --- | --- | --- | --- | --- |
| Ulysses sequence-parallel attention | exposed `all_to_all` around attention blocks | `python/sglang/multimodal_gen/runtime/layers/attention/layer.py`<br>`python/sglang/multimodal_gen/runtime/distributed/communication_op.py` | head / sequence redistribution before and after attention | Treat sequence-parallel all-to-all as an existing distributed attention family. |
| USP attention with all-to-all and ring attention | `all_to_all`, ring-attention comm, head / sequence reshards | `python/sglang/multimodal_gen/runtime/layers/attention/layer.py` | `_usp_input_all_to_all(...)`, `_usp_output_all_to_all(...)`, `ring_attn(...)` | This is the primary existing overlap / comm family for many diffusion models. |
| Turbo-layer async all-to-all pipelining | pipelined A2A windows with explicit waits on a comm stream | `python/sglang/multimodal_gen/runtime/layers/attention/turbo_layer.py` | looped `all_to_all_single(..., async_op=True)` plus staged postprocess on a comm stream | Treat exposed turbo A2A windows as an existing pipelined overlap pattern. |
| Async tensor broadcast and staging | async broadcast / wait windows outside attention core | `python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py` | GPU and CPU broadcast handles launched with `async_op=True` | This is already an established communication overlap family. |
| Layerwise offload prefetch overlap | H2D copies or memcpy overlapping with compute | `python/sglang/multimodal_gen/runtime/utils/layerwise_offload.py` | dedicated `copy_stream`, prefetch events, and residency tracking | If H2D prefetch is visible, compare against existing layerwise offload overlap before inventing a new offload schedule. |
| TorchInductor compute / communication reorder | compiled traces with compute and comm partially interleaved | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`<br>`python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/mova.py` | `torch._inductor.config.reorder_for_compute_comm_overlap = True` | Existing compile-time reordering may already explain partial overlap in diffusion traces. |
| Dual-stream diffusion models | two nearby compute branches inside one DiT / UNet block | `python/sglang/multimodal_gen/runtime/models/dits/hunyuan3d.py` | `use_dual_stream = True` | Treat dual-branch diffusion execution as an existing overlap family. |

## 6. Important toggles and caveats

| Toggle / env | Location | Effect on trace interpretation |
| --- | --- | --- |
| `enable_flashinfer_allreduce_fusion` | `python/sglang/srt/server_args.py` | Enables the FlashInfer TP allreduce fusion family. |
| `enable_aiter_allreduce_fusion` | `python/sglang/srt/server_args.py` | Enables ROCm AITER TP allreduce fusion. |
| `disable_flashinfer_cutlass_moe_fp4_allgather` | `python/sglang/srt/server_args.py` | Disables the FP4 quantize-before-comm MoE path. |
| `enable_two_batch_overlap` | `python/sglang/srt/server_args.py` | Enables the TBO family. |
| `enable_single_batch_overlap` | `python/sglang/srt/server_args.py` | Enables the SBO family. |
| `enable_fused_moe_sum_all_reduce` | `python/sglang/srt/server_args.py` | Enables fused MoE sum-reduce in the down path. |
| `enable_pdmux` | `python/sglang/srt/server_args.py` | Can suppress some `alt_stream` cache-copy overlap paths. |
| `SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO` | `python/sglang/srt/environ.py` | Alters how DeepSeek-style shared-expert overlap behaves on Blackwell. |
| `SGLANG_VIT_ENABLE_CUDA_GRAPH` | `python/sglang/srt/environ.py` | Can intentionally disable vision `aux_stream` overlap. |
| `SGLANG_ENABLE_FUSED_QKNORM_ROPE` | `python/sglang/multimodal_gen/runtime/layers/layernorm.py` | Gates the diffusion fused qknorm+rope path. |
| `enable_alt_stream=not self.server_args.enable_pdmux` | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | Explains why some cache-store overlap disappears under PDMUX. |

## 7. Suggested refresh commands

```bash
rg -n "allreduce_fusion|fused_allreduce_rmsnorm|enable_flashinfer_allreduce_fusion|enable_aiter_allreduce_fusion" python/sglang/srt
rg -n "fused_moe_sum_all_reduce|permute_fusion|FusedMoE|should_use_flashinfer_cutlass_moe_fp4_allgather" python/sglang/srt
rg -n "enable_two_batch_overlap|enable_single_batch_overlap|alt_stream|dual_stream|model_forward_maybe_tbo" python/sglang/srt
rg -n "apply_qk_norm_rope|apply_qk_norm_with_optional_rope|use_fused_qkv|use_fused_added_qkv" python/sglang/multimodal_gen
rg -n "async_op=True|reorder_for_compute_comm_overlap|layerwise_offload|copy_stream" python/sglang/multimodal_gen
```
