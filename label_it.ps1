# Labeling helper script for Bag Counting project

# 1. Install labelImg if not present
if (!(Get-Command labelImg -ErrorAction SilentlyContinue)) {
    Write-Host "Installing labelImg..." -ForegroundColor Cyan
    pip install labelImg
}

# 2. Run labelImg on the training frames
Write-Host "Opening labelImg for data\training_frames..." -ForegroundColor Green
Write-Host "Instructions:"
Write-Host "1. Click 'Open Dir' and select 'data\training_frames'"
Write-Host "2. Click 'Change Save Dir' and select 'data\processed\labels\train'"
Write-Host "3. Ensure 'YOLO' format is selected in the sidebar."
Write-Host "4. Start labeling 'bag' (Class ID 0)."

labelImg "data\training_frames"
