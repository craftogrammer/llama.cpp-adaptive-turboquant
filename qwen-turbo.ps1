param(
#    Path to the GGUF you want to launch. Default is validated SHIP model
#    (NEO-CODE IQ3_M, 12 GB on disk). Pass any other GGUF.

     [string]$Model = "C:\Users\craftogrammer\.cache\huggingface\hub\Qwen3.6-27B-NEO-CODE-2T-OT-IQ3_M.gguf",
# Deleted 2026-04-30 (4-domain A/B verdict): IQ4_XS-attn_qkv was slowest of three
# tested 27B paths (36 t/s vs NEO-CODE 41 t/s) AND larger on disk; HauhauCS was
# weakest on prior 4-domain bench. Leaving the comments as a delete-decision log.
#    [string]$Model = "C:\Users\craftogrammer\.cache\huggingface\hub\Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf",  # deleted 2026-04-30
#    [string]$Model = "C:\Users\craftogrammer\.cache\huggingface\hub\Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf",  # deleted earlier




    # KV cache / model context size in tokens. Default 128K = SHIP.
    # VRAM math on RTX 5080: each doubling of context adds ~750 MiB to KV.
    #   131072 (128K) - SHIP, fits IQ3_M comfortably (~14.4 GB total)
    #    65536  (64K) - safe headroom for IQ4_XS-class weights (~15.4 GB on disk)
    #    32768  (32K) - aggressive cut, leaves ~1 GB headroom for any model
    [int]$Context = 90072,

    # Per-turn cap on the model's <think> block. Lower = less context burned
    # per turn, less margin for hard problems.
    #   2048 - tight, fine for routine edits
    #   4096 - default, balanced
    #   8192 - full thinking headroom for hard problems (was original SHIP)
    [int]$ReasoningBudget = 8192,

    # Default OFF. -Fit reserves a 1024 MiB free-memory target which is
    # conservative on a 16 GB card running an edge config (e.g. IQ4_XS at
    # 90K context fits with ~580 MiB headroom — -Fit will instead exile
    # ~11 layers to CPU, killing decode speed). Only pass -Fit when you
    # know the config might overflow and prefer CPU offload to a hard OOM.
    # When off (default): all layers forced to GPU via -ngl 999; OOM is
    # the intended hard signal that you need a smaller context.
    [switch]$Fit
)

$ErrorActionPreference = "Stop"
$llama = "C:\Users\craftogrammer\.turboquant\turbo3-cuda\build\bin\llama-server.exe"
$model = $Model

if (-not (Test-Path $model)) {
    Write-Host "[qwen-turbo] model not found: $model" -ForegroundColor Red
    exit 1
}

# ============================================================================
# SHIP CONFIG (validated 2026-04-25)
#
# Runtime A/B benchmark on RTX 5080 / Qwen3.6-27B IQ3_M with the discipline
# system prompt baseline (TaskScheduler coding task, 12-test grader):
#   - Decode: 33.5 t/s @ 1.5K tokens, 31.9 t/s @ 4.5K tokens
#   - Quality: 12/12 tests pass on the model's own + external hidden suite
#   - Cache hit on multi-turn: 1820 t/s prompt-eval on cached prefix
#   - VRAM: 14.4 / 16.0 GB (model 12.0 + KV 1.5 + compute peak 1.0)
#
# Configuration is memory-bandwidth bound at ~45% of theoretical 80 t/s
# ceiling. Higher decode rate is unreachable without changing model class
# (e.g. switching to a 35B-A3B MoE — evaluated and rejected for coding
# quality reasons in SESSION_PLAN.md §6).
#
# Knobs that DON'T help (verified by stacking + reverting 2026-04-25):
#   - --prio 3, --poll 50, GGML_CUDA_GRAPH_OPT, --cache-ram 16384,
#     --checkpoint-every-n-tokens 1024 + --ctx-checkpoints 256
#   These are research-recommended on different hardware/models; on this
#   exact stack they were neutral or slightly negative. Do not re-add
#   without per-knob A/B on this hardware.
#
# TCQ experiment enabled in this branch:
#   - TURBO_LAYER_ADAPTIVE=13 promotes V only on first-2 + last-2 KV layers
#     to q8_0 while K stays turbo3_tcq. This exercises the fixed
#     (turbo3_tcq, q8_0) mixed attention path.
# ============================================================================

# TurboQuant TCQ config: turbo3_tcq KV + VRAM-fit-aware layer-adaptive auto-selection.
#
# As of 2026-04-30, llama-kv-cache.cpp auto-selects the most aggressive mode
# that fits dedicated VRAM with a 1 GiB safety margin:
#   - mode 1  (K&V first4+last4 q8_0)  +35% TG @ d=65K  picked when ≥3 GiB free
#   - mode 7  (K-only last8     q8_0)  +22% TG          picked when 2-3 GiB free
#   - mode 13 (V-only first2+last2)    baseline         picked when <2 GiB free
# Empirical on RTX 5080 / Qwen3.6-27B-NEO IQ3_M:
#   d=65K  → auto picks mode 1  → 17.15 t/s (+34.7% vs old SHIP mode 13)
#   d=90K  → auto picks mode 1  → 13.56 t/s (still fits dedicated)
#   d=128K → auto picks mode 13 → 7.30 t/s  (mode 1 would PCIe-spill)
# Set TURBO_LAYER_ADAPTIVE=N explicitly to override the auto-selector.
# Expected log line: "TCQ auto-selected mode N (KV X MiB, free Y MiB, margin 1024 MiB)"
$env:TURBO_SINK_SIZE       = "4"
$env:TURBO_NORM_ALPHA_V    = "1.04"   # harmless unless switching back to non-TCQ turbo types
$env:TURBO_TCQ_ALPHA_V     = "1.04"
$env:TURBO_INNERQ          = "4096"
$env:TURBO_INNERQ_STRENGTH = "1.0"

# Prompt cache + slot management
$env:LLAMA_ARG_KV_UNIFIED        = "1"
$env:LLAMA_ARG_CACHE_IDLE_SLOTS  = "1"

# When -Fit is on, omit -ngl so llama.cpp's auto-fitter can reduce GPU
# layers to hit its 1024 MiB free-memory target (some layers move to CPU,
# decode slows ~50% per offloaded layer). When -Fit is off (default),
# force all layers to GPU; OOM is the hard signal to lower context.
$fitArgs = if ($Fit) { @("--fit", "on") } else { @("-ngl", "999") }

Write-Host "[qwen-turbo] TURBO_LAYER_ADAPTIVE=$env:TURBO_LAYER_ADAPTIVE, K/V=turbo3_tcq, Context=$Context, Fit=$Fit" -ForegroundColor Cyan

& $llama `
  -m $model `
  --alias "qwen3.6-27b" `
  --host 0.0.0.0 --port 8080 `
  @fitArgs `
  -c $Context `
  --parallel 1 `
  --flash-attn on `
  --cache-type-k turbo3_tcq --cache-type-v turbo3_tcq `
  --batch-size 2048 --ubatch-size 1024 `
  --threads 4 --threads-batch 4 `
  --cache-ram -1 `
  --checkpoint-every-n-tokens 32768 `
  --prio 2 --prio-batch 2 `
  --poll 100 `
  --no-mmap `
  --jinja `
  --reasoning on `
  --reasoning-budget $ReasoningBudget `
  --reasoning-budget-message "Time to wrap up. Let me give my answer." `
  --presence-penalty 1.5 `
  --repeat-penalty 1.00 `
  --chat-template-kwargs '{\"preserve_thinking\": true}' `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
