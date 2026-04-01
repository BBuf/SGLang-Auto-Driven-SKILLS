# Kimi K2/K2.5 Optimization Playbook



## Fast Mapping

| Symptom | Check first | Historical precedent | Likely fix direction |
| --- | --- | --- | --- |
| Small-batch K2 decode is router-bound | `topk.py`, `kimi_k2_moe_fused_gate.cu` | [#13150](https://github.com/sgl-project/sglang/pull/13150), [#13287](https://github.com/sgl-project/sglang/pull/13287), [#13332](https://github.com/sgl-project/sglang/pull/13332), [#13374](https://github.com/sgl-project/sglang/pull/13374), [#15306](https://github.com/sgl-project/sglang/pull/15306), [#15347](https://github.com/sgl-project/sglang/pull/15347), [#17325](https://github.com/sgl-project/sglang/pull/17325) | Prefer the best maintained 384-expert fast path, which may be `fused_topk_deepseek` or `kimi_k2_moe_fused_gate`, and keep it PCG-safe |
| K2 fused MoE is slow on a specific NVIDIA GPU | `fused_moe_triton/configs/` | [#8047](https://github.com/sgl-project/sglang/pull/8047), [#8021](https://github.com/sgl-project/sglang/pull/8021), [#8176](https://github.com/sgl-project/sglang/pull/8176), [#8178](https://github.com/sgl-project/sglang/pull/8178), [#8183](https://github.com/sgl-project/sglang/pull/8183), [#9010](https://github.com/sgl-project/sglang/pull/9010) | Reuse or retune the exact per-device config file |
| Quantized K2 thinking or K2.5 MoE path wastes memory, breaks under PCG, or changes kernel backend | `fused_marlin_moe.py`, `jit_kernel/moe_wna16_marlin.py` | [#13596](https://github.com/sgl-project/sglang/pull/13596), [#13725](https://github.com/sgl-project/sglang/pull/13725), [#15100](https://github.com/sgl-project/sglang/pull/15100), [#19181](https://github.com/sgl-project/sglang/pull/19181) | Avoid unconditional zero fills, wire real EP metadata, preserve PCG behavior, and optimize the active JIT-backed Marlin path |
| K2.5 fails to load quantized checkpoints cleanly | `scheduler.py`, `modelopt_quant.py`, `kimi_k25.py` | [#17789](https://github.com/sgl-project/sglang/pull/17789), [#18064](https://github.com/sgl-project/sglang/pull/18064), [#18370](https://github.com/sgl-project/sglang/pull/18370), [#18440](https://github.com/sgl-project/sglang/pull/18440) | Look through `text_config`, preserve weight mapping, keep quant config on the wrapper |
| K2.5 multimodal DP path scales poorly or behaves incorrectly | `kimi_k25.py`, `vision.py` | [#17991](https://github.com/sgl-project/sglang/pull/17991), [#18689](https://github.com/sgl-project/sglang/pull/18689) | Enable the DP encoder path and remove DP-attention double-reduce or launch mismatches |
| K2.5 PP or PD features do not work through the wrapper | `kimi_k25.py`, `deepseek_v2.py` | [#18434](https://github.com/sgl-project/sglang/pull/18434), [#19959](https://github.com/sgl-project/sglang/pull/19959), [#20747](https://github.com/sgl-project/sglang/pull/20747), [#21004](https://github.com/sgl-project/sglang/pull/21004) | Expose wrapper properties required by PP/PD/EPLB runtime code |
| K2.5 speculative decoding breaks with multimodal or DP attention | `kimi_k25.py`, `llama_eagle3.py` | [#19689](https://github.com/sgl-project/sglang/pull/19689), [#21391](https://github.com/sgl-project/sglang/pull/21391) | Expose Eagle3 hooks and respect `mm_input_embeds` during extend |
| AMD K2.5 int4 tuning is missing or obviously wrong | `benchmark/kernels/fused_moe_triton/` | [#19228](https://github.com/sgl-project/sglang/pull/19228) | Use wrapper-aware tuning and the `int4_w4a16` config files |

## Investigation Commands

Commands to run before editing:

```bash
git -C /path/to/sglang log --first-parent --oneline main --grep='Kimi'
git -C /path/to/sglang log --first-parent --oneline main -- python/sglang/srt/models/kimi_k25.py python/sglang/srt/layers/moe/topk.py
rg -n "kimi_k2_moe_fused_gate|fused_topk_deepseek|moe_wna16_marlin|routed_experts_weights_of_layer|mm_enable_dp_encoder|text_config" python/sglang/srt python/sglang/jit_kernel
rg -n "E=384|E=385|int4_w4a16" python/sglang/srt/layers/moe/fused_moe_triton/configs
```

If the issue looks multimodal, also inspect:

```bash
rg -n "pp_proxy_tensors|self.model = self.language_model.model|set_eagle3_layers_to_capture" python/sglang/srt/models/kimi_k25.py
```

## Workflow

### 1. Classify the runtime shape

Record all of these before editing:

- model id
- K2 vs K2.5
- text-only vs multimodal
- quant format
- TP / DP / EP / PP sizes
- speculative decoding enabled or not
- exact GPU family and backend


- K2 + 384 experts + CUDA + small batch
- K2 thinking + quantized Marlin MoE + EP
- K2.5 multimodal + PP
- K2.5 multimodal + DP attention + Eagle3

### 2. Start from the narrowest Kimi-specific hotspot

Prefer this order:

1. Shape-specific dispatch logic or wrapper contract
2. Existing runtime-exposed metadata plumbing
3. Existing tuning config files
4. Kernel code

That order matches the merged PR history. Many Kimi issues were solved without inventing a brand-new kernel.

### 3. Reuse the exact config file contract

The tuned MoE config filenames encode the optimization contract:

- Triton version
- expert count
- effective `N`
- device name
- dtype
- block shape

Examples:

- `triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`
- `triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`
- `triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`
- `triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`

If one of these fields changes, do not assume the old tuning file is still correct.

Inference from the history:

- `E=384` appears to target pure routed-expert cases.
- `E=385` likely covers a configuration with one extra expert-like slot, possibly due to shared-expert fusion.


## K2-Specific Guidance

### Router and topk

- Keep the 384-expert shortcut in `topk.py`.
- For CUDA, prefer the best maintained specialized path for the shape:
  `fused_topk_deepseek` when supported, otherwise `kimi_k2_moe_fused_gate`.
- When touching the custom op, preserve:
  - top-k semantics
  - renormalization
  - routed scaling behavior
  - PCG or compile fake registration
  - invalid-selection guards

### Quantized MoE

- `fused_marlin_moe.py` exists because the old wrapper path did too much unnecessary work.
- Avoid unconditional zeroing of large scratch buffers.
- For EP, pass real `expert_map` and `global_num_experts` only when dispatcher metadata exists.
- If the hot path is Marlin, inspect `python/sglang/jit_kernel/moe_wna16_marlin.py` too; not all meaningful optimizations still live only in the wrapper.
- Keep the quantized Marlin path PCG-safe, not just fast.

## K2.5-Specific Guidance

### Wrapper plumbing is part of optimization

`KimiK25ForConditionalGeneration` is not just a thin wrapper. Many performance features rely on it exposing the right hooks.

Do not remove or bypass these without understanding the downstream callers:

- `self.quant_config`
- `hf_to_sglang_mapper`
- `self.use_data_parallel`
- `self.model = self.language_model.model`
- `start_layer`
- `end_layer`
- `routed_experts_weights_of_layer`
- `set_eagle3_layers_to_capture`
- `get_embed_and_head`
- `set_embed_and_head`
- `pp_proxy_tensors` threading

### Quantized K2.5 launch issues

If the model is NVFP4 or another quantized variant:

- make sure the wrapper remaps HF weight names
- make sure excluded-module patterns are remapped too
- make sure MoE init looks through `text_config`
- make sure the wrapper still stores `quant_config`

### Multimodal DP encoder and attention

When `mm_enable_dp_encoder` is on:

- instantiate the vision tower with `use_data_parallel=True`
- use `run_dp_sharded_mrope_vision_model(...)`
- do not assume the local auto-batched path is enough
- verify the DP-attention path is not applying an extra reduction or otherwise perturbing the VLM flow

### Eagle3 plus multimodal or DP attention

When debugging speculative decoding:

- verify the wrapper forwards Eagle3 methods to the language model
- verify the draft model path can consume multimodal embeddings
- on extend mode, prefer `forward_batch.mm_input_embeds`
- only append fresh token embeddings for the truly new tail token


## Validation Order

### K2 kernel path

```bash
pytest -q sgl-kernel/tests/test_dsv3_router_gemm.py
pytest -q sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py
pytest -q test/registered/kernels/test_fused_topk_deepseek.py
python benchmark/bench_kimi_k2_moe_fused_gate.py
```

### K2 quantized Marlin path

```bash
pytest -q test/registered/quant/test_marlin_moe.py
pytest -q python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py
```

### K2.5 wrapper or runtime path

Start with the lightest targeted launch or unit test you have for the exact combination.

For the registered CUDA path:

```bash
pytest -q test/registered/8-gpu-models/test_kimi_k25.py
```

If the issue is specifically DP attention plus spec:

- include the `TP8+DP8+MTP` style launch shape from the registered test
- do not validate only the plain TP path

### Tuning updates

For config or tuning changes:

- rerun only the relevant tuning script and dtype
- keep the same model, TP, EP, and backend
- save the new config under the exact filename contract

## Anti-Patterns

- Do not start by editing generic DeepSeek kernels if the current Kimi path already has a dedicated specialization.
- Do not collapse K2 and K2.5 into the same debugging hypothesis. K2 is mainly a 384-expert router and MoE story; K2.5 is much more wrapper and multimodal plumbing heavy.
- Do not assume a successful TP-only text launch proves PP, PD, EPLB, DP encoder, or Eagle3 correctness.
- Do not "fix" K2.5 by bypassing wrapper fields that other runtime features rely on.
- Do not copy Kimi-specific constants or tuning filenames into a different model unless the serving shape actually matches.
