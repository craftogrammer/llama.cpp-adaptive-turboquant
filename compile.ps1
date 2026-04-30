param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$CudaCompiler = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
$CudaArch = "120-real"
$BuildParallel = 4

if ($Clean) {
    Remove-Item -Path build -Recurse -Force -ErrorAction SilentlyContinue
}

$cache = Join-Path $PSScriptRoot "build\CMakeCache.txt"
if (Test-Path $cache) {
    $cachedCompiler = Select-String -Path $cache -Pattern "^CMAKE_CUDA_COMPILER:FILEPATH=(.*)$" -ErrorAction SilentlyContinue
    if ($cachedCompiler -and $cachedCompiler.Matches[0].Groups[1].Value -ne $CudaCompiler) {
        Write-Host "[INFO] CUDA compiler changed; removing stale build directory" -ForegroundColor Yellow
        Remove-Item -Path build -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (Test-Path $cache) {
    $cachedArch = Select-String -Path $cache -Pattern "^CMAKE_CUDA_ARCHITECTURES:.*=(.*)$" -ErrorAction SilentlyContinue
    if ($cachedArch -and $cachedArch.Matches[0].Groups[1].Value -ne $CudaArch) {
        Write-Host "[INFO] CUDA architecture changed; removing stale build directory" -ForegroundColor Yellow
        Remove-Item -Path build -Recurse -Force -ErrorAction SilentlyContinue
    }
}

cmake -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES="$CudaArch" `
  -DCMAKE_CUDA_COMPILER="$CudaCompiler" `
  -DGGML_CUDA_FORCE_CUBLAS=OFF `
  -DGGML_CUDA_FA=ON `
  -DGGML_CUDA_F16=ON `
  -DGGML_CUDA_NO_MXFP4=ON `
  -DGGML_NATIVE=ON `
  -DLLAMA_CURL=OFF `
  -DLLAMA_BUILD_TESTS=OFF `
  -DLLAMA_BUILD_EXAMPLES=ON `
  -DLLAMA_BUILD_SERVER=ON `
  -DBUILD_SHARED_LIBS=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[FAILED] CMake configure failed" -ForegroundColor Red
    exit $LASTEXITCODE
}

$start = Get-Date

cmake --build build --config Release --parallel $BuildParallel --target llama-server llama-cli llama-bench 2>&1 | ForEach-Object {
    if ($_ -match "error|FAILED|died|ACCESS|stopped") {
        Write-Host $_ -ForegroundColor Red
    } elseif ($_ -match "^\[\d+/\d+\]" -and $_ -notmatch "warning") {
        Write-Host $_ -ForegroundColor Gray
    }
}

$buildExitCode = $LASTEXITCODE
$elapsed = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
$server = Join-Path $PSScriptRoot "build\bin\llama-server.exe"
$cli    = Join-Path $PSScriptRoot "build\bin\llama-cli.exe"
$bench  = Join-Path $PSScriptRoot "build\bin\llama-bench.exe"

if ($buildExitCode -eq 0 -and (Test-Path $server) -and (Test-Path $cli) -and (Test-Path $bench)) {
    Write-Host ""
    Write-Host "[SUCCESS] Build done in $elapsed min" -ForegroundColor Green
    Write-Host "llama-server.exe: $([math]::Round((Get-Item $server).Length / 1MB, 2)) MB" -ForegroundColor Green
    Write-Host "llama-cli.exe:    $([math]::Round((Get-Item $cli).Length    / 1MB, 2)) MB" -ForegroundColor Green
    Write-Host "llama-bench.exe:  $([math]::Round((Get-Item $bench).Length  / 1MB, 2)) MB" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "[FAILED] Build failed after $elapsed min" -ForegroundColor Red
exit $buildExitCode
