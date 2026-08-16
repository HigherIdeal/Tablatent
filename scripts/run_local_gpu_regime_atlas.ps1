param(
    [string]$EnvName = "tablatent",
    [int]$Iterations = 400,
    [string]$Devices = "0",
    [switch]$SkipGitPull,
    [switch]$SkipPrepareData,
    [switch]$NoSaveOof
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $RepoRoot "outputs\local_runs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir "gpu_regime_atlas_$stamp.log"
$bundlePath = Join-Path $logDir "gpu_regime_atlas_$stamp.zip"

function Write-RunLine([string]$Text) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $Text"
    Write-Host $line
    Add-Content -Path $logPath -Value $line -Encoding UTF8
}

Write-RunLine "repo=$RepoRoot"
Write-RunLine "env=$EnvName iterations=$Iterations devices=$Devices"

if (-not $SkipGitPull) {
    Write-RunLine "git pull --ff-only"
    git pull --ff-only 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "git pull failed with exit code $LASTEXITCODE" }
}

Write-RunLine "GPU check"
nvidia-smi 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed; NVIDIA driver/GPU is not available" }

Write-RunLine "Python/CatBoost environment check"
conda run -n $EnvName python -c "import sys, catboost, numpy, pandas; print(sys.version); print('catboost', catboost.__version__); print('numpy', numpy.__version__); print('pandas', pandas.__version__)" 2>&1 | Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { throw "conda environment check failed" }

if (-not $SkipPrepareData) {
    Write-RunLine "prepare_data.py"
    conda run -n $EnvName python scripts/prepare_data.py 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "prepare_data.py failed" }
}

$outDir = "outputs/gpu_regime_atlas_local_$stamp"
$argsList = @(
    "scripts/run_gpu_regime_atlas.py",
    "--config", "configs/default.yaml",
    "--task-type", "GPU",
    "--devices", $Devices,
    "--iterations", "$Iterations",
    "--output-dir", $outDir
)
if (-not $NoSaveOof) {
    $argsList += "--save-oof"
}

Write-RunLine "starting GPU regime atlas -> $outDir"
& conda run -n $EnvName python @argsList 2>&1 | Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
Write-RunLine "experiment exit_code=$exitCode"
if ($exitCode -ne 0) { throw "GPU regime atlas failed with exit code $exitCode" }

$resultDir = Join-Path $RepoRoot ($outDir -replace '/', '\')
if (-not (Test-Path $resultDir)) { throw "result directory not found: $resultDir" }

$bundleItems = @($resultDir, $logPath)
Compress-Archive -Path $bundleItems -DestinationPath $bundlePath -Force

Write-Host ""
Write-Host "=== DONE ==="
Write-Host "Results : $resultDir"
Write-Host "Log     : $logPath"
Write-Host "Bundle  : $bundlePath"
Write-Host ""
Write-Host "Key file: $(Join-Path $resultDir 'expert_candidate_ranking.csv')"
