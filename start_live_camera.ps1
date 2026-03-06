# =============================================================================
#  BEUMER Fillpac - YOLO Bag Counter  |  Live Camera Launcher
#  PowerShell Setup & Launch Script   |  Version 1.1
#
#  USAGE (from PowerShell terminal):
#    .\start_live_camera.ps1                          # Full setup + launch
#    .\start_live_camera.ps1 -SkipSetup               # Skip install, just run
#    .\start_live_camera.ps1 -RtspUrl "rtsp://..."    # Override camera URL
#    .\start_live_camera.ps1 -SaveOutput              # Save annotated video
#    .\start_live_camera.ps1 -NoDisplay               # Headless / no window
#    .\start_live_camera.ps1 -Confidence 0.5          # Override confidence
# =============================================================================

param(
    [switch]$SkipSetup,
    [string]$RtspUrl = "",
    [switch]$SaveOutput,
    [switch]$NoDisplay,
    [string]$Confidence = "",
    [string]$WeightsPath = "models\weights\best.pt"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Header {
    param([string]$m)
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Magenta
    Write-Host "  $m"      -ForegroundColor Magenta
    Write-Host ("=" * 65) -ForegroundColor Magenta
}
function Write-Step { param([string]$m) Write-Host "`n[STEP] $m" -ForegroundColor Cyan }
function Write-OK { param([string]$m) Write-Host "[  OK] $m"  -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host "[WARN] $m"  -ForegroundColor Yellow }
function Write-Fail {
    param([string]$m)
    Write-Host "[FAIL] $m" -ForegroundColor Red
    Read-Host  "Press ENTER to close"
    exit 1
}

# ---------------------------------------------------------------------------
# Locate project root (always run from the script's own directory)
# ---------------------------------------------------------------------------
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $ProjectRoot

Write-Header "BEUMER Fillpac  |  YOLO Bag Counter  |  Live Camera Launcher"
Write-Host   "  Project : $ProjectRoot"
Write-Host   "  Time    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ---------------------------------------------------------------------------
# Step 0  Execution policy
# ---------------------------------------------------------------------------
Write-Step "0 / 7  Checking PowerShell execution policy"
$pol = Get-ExecutionPolicy -Scope CurrentUser
if ($pol -eq "Restricted") {
    Write-Warn "Execution policy is Restricted. Changing to RemoteSigned for this user..."
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-OK "Execution policy set to RemoteSigned"
}
else {
    Write-OK "Execution policy: $pol"
}

# ---------------------------------------------------------------------------
if (-not $SkipSetup) {

    # Step 1  Python check
    Write-Step "1 / 7  Checking Python 3.9+ installation"

    $pythonCmd = $null
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $v = & $cmd --version 2>&1
            if ($v -match "Python (\d+)\.(\d+)") {
                if ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 9) {
                    $pythonCmd = $cmd
                    Write-OK "Found $v  (command: '$cmd')"
                    break
                }
            }
        }
        catch { }
    }

    if (-not $pythonCmd) {
        Write-Host ""
        Write-Host "  Python 3.9+ was NOT found in PATH." -ForegroundColor Red
        Write-Host ""
        Write-Host "  HOW TO INSTALL:" -ForegroundColor Yellow
        Write-Host "    1. Open https://www.python.org/downloads/"
        Write-Host "    2. Download Python 3.11 (Windows installer)"
        Write-Host "    3. Run the installer and TICK 'Add Python to PATH'"
        Write-Host "    4. Restart this script"
        Write-Host ""
        Write-Host "  NVIDIA Jetson (Linux):"
        Write-Host "    sudo apt update"
        Write-Host "    sudo apt install -y python3.11 python3.11-venv python3-pip"
        Write-Fail  "Python 3.9+ required. Exiting."
    }

    # Step 2  Virtual environment
    Write-Step "2 / 7  Setting up Python virtual environment (.venv)"
    $venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $venvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

    if (Test-Path $venvPy) {
        Write-OK "Virtual environment already exists — reusing"
    }
    else {
        Write-Host "      Creating new virtual environment..."
        & $pythonCmd -m venv .venv
        if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create .venv" }
        Write-OK "Virtual environment created"
    }

    # Step 3  Upgrade pip
    Write-Step "3 / 7  Upgrading pip"
    & $venvPy -m pip install --upgrade pip --quiet
    Write-OK "pip is up to date"

    # Step 4  Install packages
    Write-Step "4 / 7  Installing Python packages (first run may take several minutes)"

    $reqFile = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $reqFile)) { Write-Fail "requirements.txt not found at: $reqFile" }

    # Detect NVIDIA GPU
    $gpuFound = $false
    try {
        $nvsmi = & nvidia-smi 2>&1
        if ($nvsmi -match "NVIDIA-SMI") { $gpuFound = $true }
    }
    catch { }

    if ($gpuFound) {
        Write-OK "NVIDIA GPU detected — installing CUDA-enabled PyTorch (cu118)"
        & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "CUDA PyTorch failed — falling back to CPU version"
            & $venvPip install torch torchvision --quiet
        }
    }
    else {
        Write-Warn "No NVIDIA GPU detected — installing CPU-only PyTorch (inference will be slower)"
        & $venvPip install torch torchvision --quiet
    }

    # Core packages (torch already installed above)
    $packages = @(
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "supervision>=0.16.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "filterpy>=1.4.5",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0"
    )
    & $venvPip install @packages --quiet
    if ($LASTEXITCODE -ne 0) { Write-Fail "Package installation failed. Check your internet connection." }
    Write-OK "All packages installed"

    # Step 5  Model weights
    Write-Step "5 / 7  Verifying model weights"
    $weightsAbs = Join-Path $ProjectRoot $WeightsPath

    if (Test-Path $weightsAbs) {
        $sizeMB = [math]::Round((Get-Item $weightsAbs).Length / 1MB, 2)
        Write-OK "Weights found: $WeightsPath  ($sizeMB MB)"
    }
    else {
        Write-Host ""
        Write-Host "  Model weights NOT found at: $weightsAbs" -ForegroundColor Red
        Write-Host ""
        Write-Host "  OPTIONS:" -ForegroundColor Yellow
        Write-Host "    A) Copy your trained best.pt to:  models\weights\best.pt"
        Write-Host "    B) Override path:"
        Write-Host "       .\start_live_camera.ps1 -WeightsPath `"path\to\your.pt`""
        Write-Fail  "Model weights missing. Exiting."
    }

}
else {
    # SkipSetup path
    Write-Step "Setup skipped (-SkipSetup flag)"
    $venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { Write-Fail ".venv not found. Run without -SkipSetup first." }
    Write-OK "Using existing virtual environment"
}

# ---------------------------------------------------------------------------
# Step 6  Resolve RTSP URL
# ---------------------------------------------------------------------------
Write-Step "6 / 7  Resolving RTSP camera URL"

if ($RtspUrl -ne "") {
    Write-OK "Using CLI-supplied RTSP URL"
}
else {
    $cfgFile = Join-Path $ProjectRoot "config\video_config.yaml"
    if (Test-Path $cfgFile) {
        $cfgText = Get-Content $cfgFile -Raw
        if ($cfgText -match 'source:\s*"([^"]+)"') {
            $RtspUrl = $Matches[1]
            Write-OK "RTSP URL loaded from config\video_config.yaml"
        }
        elseif ($cfgText -match "source:\s*'([^']+)'") {
            $RtspUrl = $Matches[1]
            Write-OK "RTSP URL loaded from config\video_config.yaml"
        }
        else {
            Write-Warn "Could not parse 'source:' from video_config.yaml"
        }
    }
    else {
        Write-Warn "config\video_config.yaml not found"
    }
}

if ($RtspUrl -eq "" -or $RtspUrl -eq "null") {
    Write-Host ""
    Write-Host "  No RTSP URL found. Please enter the camera URL:" -ForegroundColor Yellow
    Write-Host "  Example: rtsp://admin:Admin@123@192.168.1.5:554/cam/realmonitor?channel=1&subtype=0"
    $RtspUrl = Read-Host "  RTSP URL"
    if ($RtspUrl -eq "") { Write-Fail "No RTSP URL provided. Exiting." }
}

$displayUrl = $RtspUrl.Substring(0, [Math]::Min(70, $RtspUrl.Length))
Write-Host "      Camera  : $displayUrl ..."

# ---------------------------------------------------------------------------
# Step 7  Launch inference
# ---------------------------------------------------------------------------
Write-Step "7 / 7  Starting live bag counting"

if (-not (Test-Path (Join-Path $ProjectRoot "outputs"))) {
    New-Item -ItemType Directory -Path (Join-Path $ProjectRoot "outputs") | Out-Null
}

$scriptPath = Join-Path $ProjectRoot "src\inference_video.py"
$cfgArg = Join-Path $ProjectRoot "config\video_config.yaml"
$wtArg = Join-Path $ProjectRoot $WeightsPath

# Build argument list as array (safe for paths with spaces)
$runArgs = @($scriptPath, "--source", $RtspUrl, "--config", $cfgArg, "--weights", $wtArg)

if ($SaveOutput) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $out = Join-Path $ProjectRoot "outputs\live_$ts.mp4"
    $runArgs += "--output", $out
    Write-Host "      Save to : $out"
}

if ($NoDisplay) {
    $runArgs += "--no-display"
    Write-Host "      Display : headless"
}
else {
    Write-Host "      Display : OpenCV window  (press Q to stop)"
}

if ($Confidence -ne "") {
    $runArgs += "--conf", $Confidence
    Write-Host "      Conf    : $Confidence"
}

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Green
Write-Host "  YOLO Bag Counter — Live RTSP Stream" -ForegroundColor Green
Write-Host ("=" * 65) -ForegroundColor Green
Write-Host ""

& $venvPy @runArgs

$code = $LASTEXITCODE
Write-Host ""
if ($code -eq 0) {
    Write-OK "Session ended cleanly."
}
else {
    Write-Warn "Script exited with code $code. See logs\inference.log for details."
}

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Magenta
Write-Host "  Done  |  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Magenta
Write-Host ("=" * 65) -ForegroundColor Magenta
Write-Host ""
Read-Host "Press ENTER to close"
