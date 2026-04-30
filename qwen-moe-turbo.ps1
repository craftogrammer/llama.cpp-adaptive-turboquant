param(
    # MoE daily-driver: any Qwen3.6-35B-A3B GGUF that fits the validated KV/offload
    # config below. Default points at the current SHIP file; override with -Model
    # to swap variants.
    [string]$Model = "C:\Users\craftogrammer\.cache\huggingface\hub\models--mudler--Qwen3.6-35B-A3B-APEX-GGUF\snapshots\42c47e7a396813c593fb12c9307ada5cd8090d4b\Qwen3.6-35B-A3B-APEX-I-Compact.gguf",

    # Number of expert layers (0..N-1) routed to CPU; the rest stay on GPU.
    # Choose by GGUF size on a 16 GB card:
    #   ~16 GB Q4_K  -> NCpuMoE 8
    #   ~21 GB Q4_K  -> NCpuMoE 16
    #   ~21 GB Q6_K  -> NCpuMoE 20
    # Going lower than the size's matched value cliffs at d=65K (KV pressure +
    # expert weights blow VRAM); going higher leaves throughput on the table.
    [int]$NCpuMoE = 8,

    # KV / model context size in tokens. 128K so long-context sessions don't
    # need a server restart. Mode-1 KV at 131072 fits ~620 MiB in the auto-
    # selector budget on a 16 GB card — verified 2026-04-30 (30.4 t/s decode
    # @ d=128K with current default model).
    [int]$Context = 131072,

    # Per-turn cap on the model's <think> block.
    [int]$ReasoningBudget = 8192,

    # Default OFF — same rationale as qwen-turbo.ps1 (--fit reserves 1 GiB
    # which on the MoE expert-offload edge can exile the wrong tensors).
    [switch]$Fit
)

$ErrorActionPreference = "Stop"
$llama = "C:\Users\craftogrammer\.turboquant\turbo3-cuda\build\bin\llama-server.exe"
$model = $Model

if (-not (Test-Path $model)) {
    Write-Host "[qwen-moe-turbo] model not found: $model" -ForegroundColor Red
    exit 1
}

# ============================================================================
# SHIP CONFIG (MoE on RTX 5080 16 GB)
#
# - Active params per token = 3B (35B total). One-tenth the per-token compute
#   of a 27B dense on Blackwell IMMA tensor cores.
# - Expert offload (--n-cpu-moe N) keeps the upper layers' experts on GPU and
#   pays PCIe only for the first N layers' active routed experts per token.
# - TurboQuant TCQ KV applies to the 10 full_attention layers (head_dim=256);
#   the 30 gated_delta_net layers carry SSM state.
# - VRAM-fit auto-selector probes free GPU memory after expert offload and
#   picks the most aggressive K&V boundary q8_0 promotion mode that fits.
#   Override with TURBO_LAYER_ADAPTIVE=N for a specific mode.
# ============================================================================

$env:TURBO_SINK_SIZE       = "4"      # attention sinks at fp16
$env:TURBO_NORM_ALPHA_V    = "1.04"   # TCQ V-cache norm alpha
$env:TURBO_TCQ_ALPHA_V     = "1.04"
$env:TURBO_INNERQ          = "4096"   # InnerQ calibration window
$env:TURBO_INNERQ_STRENGTH = "1.0"

# Prompt cache + slot management
$env:LLAMA_ARG_KV_UNIFIED        = "1"
$env:LLAMA_ARG_CACHE_IDLE_SLOTS  = "1"

$fitArgs = if ($Fit) { @("--fit", "on") } else { @("-ngl", "999") }

$modelName = [System.IO.Path]::GetFileNameWithoutExtension($model)
Write-Host "[qwen-moe-turbo] $modelName, ncmoe=$NCpuMoE, ctx=$Context, Fit=$Fit" -ForegroundColor Cyan

& $llama `
  -m $model `
  --alias "qwen3.6-35b-a3b" `
  --host 0.0.0.0 --port 8080 `
  @fitArgs `
  --n-cpu-moe $NCpuMoE `
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
