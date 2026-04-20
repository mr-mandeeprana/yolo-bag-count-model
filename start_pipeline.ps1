
# ==============================================================
# START_PIPELINE.ps1
# Complete pipeline startup script for YOLO Bag Counter
# ==============================================================
# USAGE:
#   .\start_pipeline.ps1 -Full              (Docker + YOLO)
#   .\start_pipeline.ps1 -DockerOnly        (Just Docker services)
#   .\start_pipeline.ps1 -Stop              (Stop all services)
# ==============================================================

param(
    [switch]$Full,
    [switch]$DockerOnly,
    [switch]$Stop,
    [switch]$Logs,
    [string]$Source = "",
    [string]$Weights = "models/weights/best.pt"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Header {
    param([string]$msg)
    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$msg, [switch]$success)
    if ($success) {
        Write-Host "[OK] $msg" -ForegroundColor Green
    } else {
        Write-Host "[..] $msg" -ForegroundColor Yellow
    }
}

function Write-Error {
    param([string]$msg)
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

# ==============================================================
# STOP SERVICES
# ==============================================================
function Stop-Services {
    Write-Header "STOPPING SERVICES"
    
    try {
        Write-Step "Stopping Docker containers..."
        docker-compose -f docker-compose.observability.yml down
        Write-Step "Docker containers stopped" -success
    } catch {
        Write-Error "Failed to stop Docker: $_"
    }
    
    Write-Header "All services stopped"
}

# ==============================================================
# START DOCKER SERVICES
# ==============================================================
function Start-Pipeline {
    Write-Header "STARTING YOLO BAG COUNTER PIPELINE"

    # Check prerequisites
    Write-Step "Checking Docker..."
    try {
        $docker = docker --version
        Write-Step "Docker found: $docker" -success
    } catch {
        Write-Error "Docker not found. Install Docker Desktop first."
        exit 1
    }

    # Create directories
    Write-Step "Creating log directories..."
    New-Item -ItemType Directory -Path "logs/vector" -Force | Out-Null
    New-Item -ItemType Directory -Path "logs/socketio" -Force | Out-Null
    New-Item -ItemType Directory -Path "observability/vector" -Force | Out-Null
    Write-Step "Directories created" -success

    # Start services
    Write-Step "Starting Docker containers..."
    Write-Host ""
    docker-compose -f docker-compose.observability.yml up -d
    
    Write-Host ""
    Write-Step "Containers started" -success
    
    # Wait for services to be healthy
    Write-Step "Waiting for services to become healthy..."
    Start-Sleep -Seconds 15

    # Health checks
    Write-Step "Performing health checks..."
    
    $services = @(
        @{ name = "Elasticsearch"; url = "http://localhost:9200/_cluster/health"; port = 9200 },
        @{ name = "Vector.dev"; url = "http://localhost:8686/health"; port = 8686 },
        @{ name = "Socket.IO"; url = "http://localhost:3000/health"; port = 3000 },
        @{ name = "Kibana"; url = "http://localhost:5601/api/status"; port = 5601 }
    )

    foreach ($service in $services) {
        try {
            $response = Invoke-WebRequest -Uri $service.url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Step "$($service.name) is healthy" -success
            }
        } catch {
            Write-Host "[WARN] $($service.name) not yet ready (this is normal)" -ForegroundColor Yellow
        }
    }

    Write-Header "PIPELINE SERVICES STARTED"
    Write-Host ""
    Write-Host "Services Running:" -ForegroundColor Cyan
    Write-Host "  - Elasticsearch  -> http://localhost:9200" -ForegroundColor Green
    Write-Host "  - Vector.dev     -> http://localhost:8686" -ForegroundColor Green
    Write-Host "  - Socket.IO      -> http://localhost:3000" -ForegroundColor Green
    Write-Host "  - Kibana         -> http://localhost:5601" -ForegroundColor Green
    Write-Host ""
}

# ==============================================================
# START YOLO INFERENCE
# ==============================================================
function Start-YoloInference {
    Write-Header "STARTING YOLO INFERENCE WITH PIPELINE"

    # Check source provided
    if (-not $Source) {
        Write-Host ""
        Write-Host "Available sources:" -ForegroundColor Yellow
        Write-Host "   1. Live Camera:  rtsp://user:pass@camera-ip:port/stream" 
        Write-Host "   2. Video File:   path/to/video.mp4"
        Write-Host "   3. Common test:  rtsp://192.168.1.5:554/cam/realmonitor?..."
        Write-Host ""
        
        $Source = Read-Host "Enter video source"
        if (-not $Source) {
            Write-Error "No source provided. Exiting."
            exit 1
        }
    }

    # Activate Python environment
    Write-Step "Activating Python environment..."
    & ".\.venv\Scripts\Activate.ps1"
    
    Write-Step "Python activated" -success
    Write-Host ""

    # Run inference
    Write-Step "Starting YOLO inference with Socket.IO pipeline..."
    Write-Host ""
    
    & python src/inference_video.py `
        --source $Source `
        --weights $Weights `
        --config config/video_config.yaml

    Write-Host ""
    Write-Header "YOLO Inference stopped"
}

# ==============================================================
# SHOW LOGS
# ==============================================================
function Show-Logs {
    Write-Header "STREAMING LOGS"
    
    Write-Host ""
    Write-Host "Available log streams:" -ForegroundColor Cyan
    Write-Host "  1. Socket.IO Server"
    Write-Host "  2. Vector.dev"
    Write-Host "  3. Elasticsearch"
    Write-Host "  4. Kibana"
    Write-Host ""
    
    $choice = Read-Host "Select service (1-4)"
    
    $logCommands = @{
        "1" = @{ name = "Socket.IO"; cmd = "docker-compose -f docker-compose.observability.yml logs socketio-server -f" }
        "2" = @{ name = "Vector.dev"; cmd = "docker-compose -f docker-compose.observability.yml logs vector-dev -f" }
        "3" = @{ name = "Elasticsearch"; cmd = "docker-compose -f docker-compose.observability.yml logs elasticsearch -f" }
        "4" = @{ name = "Kibana"; cmd = "docker-compose -f docker-compose.observability.yml logs kibana -f" }
    }
    
    if ($logCommands.ContainsKey($choice)) {
        Write-Header "$($logCommands[$choice].name) Logs"
        Invoke-Expression $logCommands[$choice].cmd
    } else {
        Write-Error "Invalid choice"
    }
}

# ==============================================================
# MAIN LOGIC
# ==============================================================

if ($Stop) {
    Stop-Services
    exit 0
}

if ($Logs) {
    Show-Logs
    exit 0
}

if ($DockerOnly -or $Full) {
    Start-Pipeline
    
    if ($Full) {
        Write-Host ""
        Write-Host "Ready to start YOLO inference. Run:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  .\start_pipeline.ps1 -Full -Source `"rtsp://...`"" -ForegroundColor Cyan
        Write-Host ""
        Start-YoloInference
    }
} else {
    Write-Host ""
    Write-Host "YOLO Bag Counter - Pipeline Startup Script" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\start_pipeline.ps1 -Full              # Start Docker + YOLO" -ForegroundColor Green
    Write-Host "  .\start_pipeline.ps1 -DockerOnly        # Start only Docker services" -ForegroundColor Green
    Write-Host "  .\start_pipeline.ps1 -Stop              # Stop all services" -ForegroundColor Green
    Write-Host "  .\start_pipeline.ps1 -Logs              # View service logs" -ForegroundColor Green
    Write-Host ""
}
