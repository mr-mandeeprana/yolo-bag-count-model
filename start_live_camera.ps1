# =============================================================================
#  BEUMER Fillpac - YOLO Bag Counter | Live Camera Launcher
#  PowerShell Setup & Launch Script
#  Version: 1.0 | 2026-03-06
#
#  USAGE:
#    Right-click this file → "Run with PowerShell"   (first-time or any time)
#    Or from a PowerShell terminal:
#      .\start_live_camera.ps1
#      .\start_live_camera.ps1 -SkipSetup          (skip install, just run)
#      .\start_live_camera.ps1 -RtspUrl "rtsp://..." (override camera URL)
#      .\start_live_camera.ps1 -SaveOutput          (save annotated video)
#
#  WHAT THIS SCRIPT DOES:
#    1. Checks Python (3.9+) is installed
#    2. Checks/installs Git (optional, for cloning)
#    3. Creates a Python virtual environment (.venv)
#    4. Installs all required packages from requirements.txt
#    5. Verifies model weights exist
#    6. Reads RTSP URL from config/video_config.yaml (or CLI override)
#    7. Launches the live bag-counting inference
# =============================================================================

param(
    [switch]$SkipSetup,        # Skip environment setup (just run the model)
    [string]$RtspUrl = "",     # Override RTSP URL (e.g. "rtsp://admin:pass@192.168.1.5:554/...")
    [switch]$SaveOutput,       # Save annotated video to outputs\live_output.mp4
    [switch]$NoDisplay,        # Run headless (no OpenCV window)
    [string]$Confidence = "",  # Override confidence threshold (0.0-1.0)
    [string]$WeightsPath = "models\weights\best.pt"  # Model weights path
)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
$ESC = [char]27
function Write-Header  { param([string]$msg) Write-Host "`n$ESC[95m$('='*65)$ESC[0m" ; Write-Host "$ESC[95m  $msg$ESC[0m" ; Write-Host "$ESC[95m$('='*65)$ESC[0m" }
function Write-Step    { param([string]$msg) Write-Host "`n$ESC[94m[STEP]$ESC[0m $msg" }
function Write-OK      { param([string]$msg) Write-Host "$ESC[92m[  OK]$ESC[0m $msg" }
function Write-Warn    { param([string]$msg) Write-Host "$ESC[93m[WARN]$ESC[0m $msg" }
function Write-Fail    { param([string]$msg) Write-Host "$ESC[91m[FAIL]$ESC[0m $msg" ; exit 1 }
function Write-Info    { param([string]$msg) Write-Host "      $msg" }

# ─────────────────────────────────────────────────────────────────────────────
#  SCRIPT ROOT — ensure we always run from the project directory
# ─────────────────────────────────────────────────────────────────────────────
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location $ProjectRoot

Write-Header "BEUMER Fillpac | YOLO Bag Counter — Live Camera Launcher"
Write-Info "Project root : $ProjectRoot"
Write-Info "Timestamp    : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 — Execution Policy (allow this session to run scripts)
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "0/7 — Checking PowerShell execution policy"
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted") {
    Write-Warn "Execution policy is Restricted. Setting to RemoteSigned for this user..."
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-OK "Execution policy set to RemoteSigned"
} else {
    Write-OK "Execution policy: $policy"
}

if (-not $SkipSetup) {

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 1 — Check Python 3.9+
    # ─────────────────────────────────────────────────────────────────────────
    Write-Step "1/7 — Checking Python installation"

    $pythonCmd = $null
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 9) {
                    $pythonCmd = $cmd
                    Write-OK "Found $ver (using '$cmd')"
                    break
                } else {
                    Write-Warn "Found $ver but need Python 3.9+. Please upgrade."
                }
            }
        } catch { }
    }

    if (-not $pythonCmd) {
        Write-Fail @"
Python 3.9+ not found in PATH!

  INSTALL STEPS:
    1. Download Python 3.11 from https://www.python.org/downloads/
    2. Run the installer — TICK 'Add Python to PATH'
    3. Restart this script

  For NVIDIA Jetson (ARM):
    sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip
"@
    }

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 2 — Create / reuse virtual environment
    # ─────────────────────────────────────────────────────────────────────────
    Write-Step "2/7 — Setting up Python virtual environment (.venv)"

    $venvPy   = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $venvPip  = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

    if (Test-Path $venvPy) {
        Write-OK "Virtual environment already exists — reusing"
    } else {
        Write-Info "Creating new virtual environment..."
        & $pythonCmd -m venv .venv
        if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create virtual environment" }
        Write-OK "Virtual environment created"
    }

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 3 — Upgrade pip
    # ─────────────────────────────────────────────────────────────────────────
    Write-Step "3/7 — Upgrading pip"
    & $venvPy -m pip install --upgrade pip --quiet
    Write-OK "pip is up to date"

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 4 — Install requirements
    # ─────────────────────────────────────────────────────────────────────────
    Write-Step "4/7 — Installing Python packages"

    $reqFile = Join-Path $ProjectRoot "requirements.txt"
    if (-not (Test-Path $reqFile)) { Write-Fail "requirements.txt not found at $reqFile" }

    Write-Info "Installing from requirements.txt (this may take a few minutes on first run)..."

    # Check for CUDA (GPU) — install torch with CUDA if available
    $cudaAvailable = $false
    try {
        $nvsmi = & nvidia-smi 2>&1
        if ($nvsmi -match "NVIDIA-SMI") {
            $cudaAvailable = $true
            Write-OK "NVIDIA GPU detected — will install CUDA-enabled PyTorch"
        }
    } catch { }

    if ($cudaAvailable) {
        Write-Info "Installing PyTorch with CUDA 11.8 support..."
        & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
        if ($LASTEXITCODE -ne 0) { Write-Warn "CUDA PyTorch install failed — falling back to CPU version" ; & $venvPip install torch torchvision --quiet }
    } else {
        Write-Warn "No NVIDIA GPU detected — installing CPU-only PyTorch (slower inference)"
        & $venvPip install torch torchvision --quiet
    }

    # Install remaining packages (skip torch since just installed)
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

    if ($LASTEXITCODE -ne 0) { Write-Fail "Package installation failed. Check internet connection." }
    Write-OK "All packages installed successfully"

    # ─────────────────────────────────────────────────────────────────────────
    #  STEP 5 — Verify model weights
    # ─────────────────────────────────────────────────────────────────────────
    Write-Step "5/7 — Verifying model weights"

    $weightsFile = Join-Path $ProjectRoot $WeightsPath
    if (Test-Path $weightsFile) {
        $sizeMB = [math]::Round((Get-Item $weightsFile).Length / 1MB, 2)
        Write-OK "Model weights found: $WeightsPath ($sizeMB MB)"
    } else {
        Write-Fail @"
Model weights NOT found at: $weightsFile

  Options:
    A) Copy your trained 'best.pt' to:  models\weights\best.pt
    B) Override path:  .\start_live_camera.ps1 -WeightsPath "path\to\your.pt"
    C) Download a pretrained base (no bag detection until fine-tuned):
         $venvPy -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
"@
    }

} else {
    # SkipSetup — just locate .venv python
    Write-Step "Setup skipped (--SkipSetup flag set)"
    $venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        Write-Fail ".venv not found. Run without -SkipSetup first."
    }
    Write-OK "Using existing virtual environment"
}

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Resolve RTSP URL
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "6/7 — Resolving RTSP camera URL"

if ($RtspUrl -ne "") {
    Write-OK "Using CLI-supplied RTSP URL"
} else {
    # Read from config/video_config.yaml
    $configFile = Join-Path $ProjectRoot "config\video_config.yaml"
    if (Test-Path $configFile) {
        $configContent = Get-Content $configFile -Raw
        if ($configContent -match 'source:\s*"([^"]+)"') {
            $RtspUrl = $Matches[1]
            Write-OK "RTSP URL loaded from video_config.yaml"
        } elseif ($configContent -match "source:\s*'([^']+)'") {
            $RtspUrl = $Matches[1]
            Write-OK "RTSP URL loaded from video_config.yaml"
        } else {
            Write-Warn "Could not parse 'source' from video_config.yaml"
        }
    } else {
        Write-Warn "config\video_config.yaml not found"
    }
}

if ($RtspUrl -eq "" -or $RtspUrl -eq "null") {
    Write-Host ""
    Write-Host "$ESC[93m  No RTSP URL found. Please enter the camera URL:$ESC[0m"
    Write-Host "  Example: rtsp://admin:Admin@123@192.168.1.5:554/cam/realmonitor?channel=1&subtype=0"
    $RtspUrl = Read-Host "  RTSP URL"
    if ($RtspUrl -eq "") { Write-Fail "No RTSP URL provided. Exiting." }
}

Write-Info "Camera: $($RtspUrl.Substring(0, [Math]::Min(60, $RtspUrl.Length)))..."

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Launch inference
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "7/7 — Starting live bag counting"

# Create outputs directory if it doesn't exist
$outputsDir = Join-Path $ProjectRoot "outputs"
if (-not (Test-Path $outputsDir)) { New-Item -ItemType Directory -Path $outputsDir | Out-Null }

# Build argument list
$inferenceScript = Join-Path $ProjectRoot "src\inference_video.py"
$configArg       = Join-Path $ProjectRoot "config\video_config.yaml"
$weightsArg      = Join-Path $ProjectRoot $WeightsPath

$args = @(
    "`"$inferenceScript`""
    "--source", "`"$RtspUrl`""
    "--config", "`"$configArg`""
    "--weights", "`"$weightsArg`""
)

if ($SaveOutput) {
    $timestamp  = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputFile = Join-Path $outputsDir "live_output_$timestamp.mp4"
    $args += "--output", "`"$outputFile`""
    Write-Info "Output video : $outputFile"
}

if ($NoDisplay) {
    $args += "--no-display"
    Write-Info "Display      : headless (no window)"
} else {
    Write-Info "Display      : OpenCV window (press Q to stop)"
}

if ($Confidence -ne "") {
    $args += "--conf", $Confidence
    Write-Info "Confidence   : $Confidence (CLI override)"
}

Write-Host ""
Write-Host "$ESC[92m$('='*65)$ESC[0m"
Write-Host "$ESC[92m  Launching YOLO Bag Counter — Live RTSP Stream$ESC[0m"
Write-Host "$ESC[92m$('='*65)$ESC[0m`n"

# Run the inference script
& $venvPy @args

$exitCode = $LASTEXITCODE
Write-Host ""
if ($exitCode -eq 0) {
    Write-OK "Session ended cleanly."
} else {
    Write-Warn "Script exited with code $exitCode. Check logs\inference.log for details."
}

Write-Host ""
Write-Host "$ESC[95m$('='*65)$ESC[0m"
Write-Host "$ESC[95m  Session complete | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')$ESC[0m"
Write-Host "$ESC[95m$('='*65)$ESC[0m"
Write-Host ""
Read-Host "Press ENTER to close"
