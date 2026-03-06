# ==============================================================
# BEUMER Fillpac  |  YOLO Bag Counter  |  Live Camera Launcher
# PowerShell Setup & Launch Script  v1.2
#
# USAGE:
#   .\start_live_camera.ps1                       Full setup + launch
#   .\start_live_camera.ps1 -SkipSetup            Skip install, just run
#   .\start_live_camera.ps1 -RtspUrl "rtsp://..."  Override camera URL
#   .\start_live_camera.ps1 -SaveOutput           Record annotated video
#   .\start_live_camera.ps1 -NoDisplay            Headless / no window
#   .\start_live_camera.ps1 -Confidence 0.5       Override confidence
# ==============================================================
param(
    [switch]$SkipSetup,
    [string]$RtspUrl = "",
    [switch]$SaveOutput,
    [switch]$NoDisplay,
    [string]$Confidence = "",
    [string]$WeightsPath = "models\weights\best.pt"
)

# --------------------------------------------------------------
# Colour helpers  (no $() subexpressions inside strings)
# --------------------------------------------------------------
function Write-Header {
    param([string]$msg)
    $line = "=" * 62
    Write-Host ""
    Write-Host $line            -ForegroundColor Magenta
    Write-Host "  $msg"         -ForegroundColor Magenta
    Write-Host $line            -ForegroundColor Magenta
}
function Write-Step { param([string]$msg) Write-Host "" ; Write-Host "[STEP] $msg" -ForegroundColor Cyan }
function Write-OK { param([string]$msg) Write-Host "[  OK] $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Fail {
    param([string]$msg)
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    Read-Host "Press ENTER to close"
    exit 1
}

# --------------------------------------------------------------
# Always run from the project folder
# --------------------------------------------------------------
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $ProjectRoot

$startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Header "BEUMER Fillpac  |  YOLO Bag Counter  |  Live Camera Launcher"
Write-Host "  Project : $ProjectRoot"
Write-Host "  Started : $startTime"

# --------------------------------------------------------------
# Step 0  Execution policy
# --------------------------------------------------------------
Write-Step "0 / 7  Checking PowerShell execution policy"
$pol = Get-ExecutionPolicy -Scope CurrentUser
if ($pol -eq "Restricted") {
    Write-Warn "Policy is Restricted. Setting to RemoteSigned for this user ..."
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-OK "Execution policy updated to RemoteSigned"
}
else {
    Write-OK "Execution policy: $pol"
}

# ==============================================================
if (-not $SkipSetup) {
    # ==============================================================

    # ----------------------------------------------------------
    # Step 1  Python 3.9+
    # ----------------------------------------------------------
    Write-Step "1 / 7  Checking Python 3.9+ installation"
    $pythonCmd = $null

    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $verText = & $cmd --version 2>&1
            if ($verText -match "Python (\d+)\.(\d+)") {
                $maj = [int]$Matches[1]
                $min = [int]$Matches[2]
                if ($maj -eq 3 -and $min -ge 9) {
                    $pythonCmd = $cmd
                    Write-OK "Found $verText  (command: $cmd)"
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
        Write-Host "  HOW TO INSTALL ON WINDOWS:" -ForegroundColor Yellow
        Write-Host "    Step 1 - Go to  https://www.python.org/downloads/"
        Write-Host "    Step 2 - Download Python 3.11 Windows installer"
        Write-Host "    Step 3 - Run installer and CHECK  Add Python to PATH"
        Write-Host "    Step 4 - Restart this script"
        Write-Host ""
        Write-Host "  HOW TO INSTALL ON NVIDIA JETSON / LINUX:" -ForegroundColor Yellow
        Write-Host "    sudo apt update"
        Write-Host "    sudo apt install -y python3.11 python3.11-venv python3-pip"
        Write-Host ""
        Write-Fail "Python 3.9+ is required. Please install it and re-run."
    }

    # ----------------------------------------------------------
    # Step 2  Virtual environment
    # ----------------------------------------------------------
    Write-Step "2 / 7  Setting up Python virtual environment"
    $venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $venvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

    if (Test-Path $venvPy) {
        Write-OK "Virtual environment already exists - reusing"
    }
    else {
        Write-Host "      Creating .venv ..."
        & $pythonCmd -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Failed to create virtual environment. Check Python installation."
        }
        Write-OK "Virtual environment created at .venv"
    }

    # ----------------------------------------------------------
    # Step 3  Upgrade pip
    # ----------------------------------------------------------
    Write-Step "3 / 7  Upgrading pip"
    & $venvPy -m pip install --upgrade pip --quiet
    Write-OK "pip upgraded"

    # ----------------------------------------------------------
    # Step 4  Install packages
    # ----------------------------------------------------------
    Write-Step "4 / 7  Installing Python packages"

    $reqFile = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $reqFile)) {
        Write-Fail "requirements.txt not found at: $reqFile"
    }

    # Detect NVIDIA GPU
    $gpuFound = $false
    try {
        $nvsmiOut = & nvidia-smi 2>&1
        if ($nvsmiOut -match "NVIDIA-SMI") { $gpuFound = $true }
    }
    catch { }

    if ($gpuFound) {
        Write-OK "NVIDIA GPU detected - installing CUDA-enabled PyTorch"
        & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "CUDA PyTorch failed - falling back to CPU-only build"
            & $venvPip install torch torchvision --quiet
        }
    }
    else {
        Write-Warn "No NVIDIA GPU found - installing CPU PyTorch (slower inference)"
        & $venvPip install torch torchvision --quiet
    }

    Write-Host "      Installing core packages ..."
    & $venvPip install `
        "ultralytics>=8.0.0" `
        "opencv-python>=4.8.0" `
        "supervision>=0.16.0" `
        "numpy>=1.24.0" `
        "pandas>=2.0.0" `
        "Pillow>=10.0.0" `
        "pyyaml>=6.0" `
        "tqdm>=4.65.0" `
        "filterpy>=1.4.5" `
        "scikit-image>=0.21.0" `
        "matplotlib>=3.7.0" `
        --quiet

    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Package installation failed. Check internet connection and try again."
    }
    Write-OK "All packages installed"

    # ----------------------------------------------------------
    # Step 5  Model weights
    # ----------------------------------------------------------
    Write-Step "5 / 7  Verifying model weights"
    $weightsAbs = Join-Path $ProjectRoot $WeightsPath

    if (Test-Path $weightsAbs) {
        $rawBytes = (Get-Item $weightsAbs).Length
        $sizeMB = [math]::Round($rawBytes / 1MB, 2)
        $sizeStr = [string]$sizeMB + " MB"
        Write-OK "Weights found: $WeightsPath  ($sizeStr)"
    }
    else {
        Write-Host ""
        Write-Host "  Model weights NOT found at: $weightsAbs" -ForegroundColor Red
        Write-Host ""
        Write-Host "  OPTIONS:" -ForegroundColor Yellow
        Write-Host "    Option A - Copy your trained best.pt to:  models\weights\best.pt"
        Write-Host "    Option B - Override path at launch:"
        Write-Host '               .\start_live_camera.ps1 -WeightsPath "path\to\your.pt"'
        Write-Host ""
        Write-Fail "Model weights missing. Add best.pt and re-run."
    }

    # ==============================================================
}
else {
    # ==============================================================
    Write-Step "Setup skipped (-SkipSetup flag is set)"
    $venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $venvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
    if (-not (Test-Path $venvPy)) {
        Write-Fail ".venv not found. Run without -SkipSetup at least once."
    }
    Write-OK "Using existing virtual environment"
}
# ==============================================================

# --------------------------------------------------------------
# Step 6  Resolve RTSP URL
# --------------------------------------------------------------
Write-Step "6 / 7  Resolving RTSP camera URL"

if ($RtspUrl -ne "") {
    Write-OK "Using RTSP URL supplied on command line"
}
else {
    $cfgFile = Join-Path $ProjectRoot "config\video_config.yaml"
    if (Test-Path $cfgFile) {
        $cfgText = Get-Content $cfgFile -Raw
        if ($cfgText -match 'source:\s*"([^"]+)"') {
            $RtspUrl = $Matches[1]
            Write-OK "RTSP URL read from config\video_config.yaml"
        }
        elseif ($cfgText -match "source:\s*'([^']+)'") {
            $RtspUrl = $Matches[1]
            Write-OK "RTSP URL read from config\video_config.yaml"
        }
        else {
            Write-Warn "Could not parse source: field from video_config.yaml"
        }
    }
    else {
        Write-Warn "config\video_config.yaml not found"
    }
}

if ($RtspUrl -eq "" -or $RtspUrl -eq "null") {
    Write-Host ""
    Write-Host "  No RTSP URL found. Please enter it now:" -ForegroundColor Yellow
    Write-Host "  Example:  rtsp://admin:Admin%40123@192.168.1.5:554/cam/realmonitor?channel=1"
    $RtspUrl = Read-Host "  RTSP URL"
    if ($RtspUrl -eq "") {
        Write-Fail "No RTSP URL provided. Exiting."
    }
}

$urlPreview = $RtspUrl.Substring(0, [Math]::Min(70, $RtspUrl.Length))
Write-Host "      Camera : $urlPreview"

# --------------------------------------------------------------
# Step 7  Launch inference
# --------------------------------------------------------------
Write-Step "7 / 7  Launching live bag counting"

$outputsDir = Join-Path $ProjectRoot "outputs"
if (-not (Test-Path $outputsDir)) {
    New-Item -ItemType Directory -Path $outputsDir | Out-Null
}

$inferenceScript = Join-Path $ProjectRoot "src\inference_video.py"
$cfgArg = Join-Path $ProjectRoot "config\video_config.yaml"
$wtArg = Join-Path $ProjectRoot $WeightsPath

$runArgs = @(
    $inferenceScript,
    "--source", $RtspUrl,
    "--config", $cfgArg,
    "--weights", $wtArg
)

if ($SaveOutput) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputFile = Join-Path $outputsDir ("live_" + $ts + ".mp4")
    $runArgs += "--output"
    $runArgs += $outputFile
    Write-Host "      Output : $outputFile"
}

if ($NoDisplay) {
    $runArgs += "--no-display"
    Write-Host "      Window : headless (no display)"
}
else {
    Write-Host "      Window : OpenCV window  (press Q to stop)"
}

if ($Confidence -ne "") {
    $runArgs += "--conf"
    $runArgs += $Confidence
    Write-Host "      Conf   : $Confidence"
}

$venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host ""
$line = "=" * 62
Write-Host $line -ForegroundColor Green
Write-Host "  YOLO Bag Counter  |  Live RTSP Stream" -ForegroundColor Green
Write-Host $line -ForegroundColor Green
Write-Host ""

& $venvPy @runArgs

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-OK "Session ended cleanly"
}
else {
    Write-Warn "Script exited with code $exitCode  -  see logs\inference.log for details"
}

$endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host ""
Write-Host $line          -ForegroundColor Magenta
Write-Host "  Done  |  $endTime" -ForegroundColor Magenta
Write-Host $line          -ForegroundColor Magenta
Write-Host ""
Read-Host "Press ENTER to close"
