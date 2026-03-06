# Deployment Guide: JK Aligarh Cement - Fillpac Bag Counter

This document provides a production-grade step-by-step guide for deploying and running the YOLO-based bag counting system at the **JK Aligarh Cement** site.

---

## 🏗️ System Architecture Overview

```mermaid
graph TD
    A[Camera - RTSP Stream] --> B[Jetson Edge Device<br/>YOLO Inference + Counting Logic]
    B --> C[Local Log Storage<br/>inference.json]
    C --> D[Cloud Sync<br/>AWS IoT / API Gateway / Filebeat]
    D --> E[Cloud Dashboard<br/>Monitoring & Reporting]
```

*This flow ensures local resilience and real-time cloud visibility.*

---

## 1. 🛠️ Hardware Requirements & On-Site Setup

### A. Camera Installation

- **Camera Type**: Industrial IP Camera (1080p, 30 FPS minimum).
- **Mounting**: Use a **vibration-damped mounting bracket** above the conveyor belt exit. Cement plant vibrations can shift the counting line.
- **Orientation**: Top-down view with a 90-degree rotation (configured in `video_config.yaml`).
- **Lighting**: Maintain consistent lighting (>400 lux). Avoid backlight from the conveyor exit.

### B. Power & Stability

- **UPS Required**: A dedicated UPS must power both the Camera and the Jetson Edge Device. Sudden power loss can corrupt the TensorRT engine cache.
- **Cooling**: Ensure the device is in a well-ventilated enclosure to handle industrial temperatures.

### C. Processing Unit

- **Recommendation**: NVIDIA Jetson Orin Nano or Xavier NX.
- **Connectivity**: Static IP on the same local subnet as the camera. Internet access required for Cloud Sync.

---

## 2. 🔐 Security & Software Environment

### A. Production Environment Setup

1. **Initialize Production Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Model Optimization (TensorRT)**:

   ```python
   from ultralytics import YOLO
   model = YOLO('models/weights/best.PT')
   model.export(format='engine', device=0, half=True)
   ```

### B. Security Best Practices

**Never** store credentials in plain-text config files. Use environment variables.

In `config/video_config.yaml`:

```yaml
camera:
  source: "${CAMERA_RTSP}" # Loaded from environment
```

On the Jetson device (`/etc/environment` or `.bashrc`):

```bash
export CAMERA_RTSP="rtsp://admin:SecurePassword@192.168.1.5:554/live"
```

---

## 3. ⚙️ Persistence & Fail-Safety

### A. Automatic Restart (Systemd)

Create `/etc/systemd/system/bag-counter.service`:

```ini
[Unit]
Description=YOLO Bag Counter
After=network.target

[Service]
User=jetson
WorkingDirectory=/home/jetson/Yolo_bag_count_model
ExecStart=/home/jetson/venv/bin/python src/inference_video.py --config config/video_config.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable bag-counter
sudo systemctl start bag-counter
```

---

## 4. 📈 Counting Robustness & Monitoring

### A. Counting Configuration

- **Tracker**: Ensure BYTETrack is enabled for ID persistence across occlusions.
- **Confidence**: Target detection confidence between 0.45–0.6 based on site test.
- **Performance**: Ensure processing speed remains > 15 FPS for accurate tracking.

### B. Structured JSON Logging

The system uses structured logs for reliable cloud integration:

```json
{
  "timestamp": "2025-01-14T14:32:10",
  "bag_count": 1245,
  "confidence_avg": 0.87,
  "camera_status": "online",
  "gpu_temp": 65.0
}
```

---

## 5. ☁️ Cloud Integration (AWS-Ready)

### Option A: MQTT (AWS IoT Core)

- Publish count messages to: `topic: jk/aligarh/fillpac/count`
- Benefit: High reliability, low bandwidth, certificate-based auth.

### Option B: HTTPS (API Gateway)

- REST API endpoint for real-time telemetry.
- Lambda/DynamoDB stack for count persistence and QuickSight dashboards.

---

## 6. 🌡️ Edge Health & Failure Handling

### A. Health Monitoring

- **Temperature**: Monitor GPU/CPU temp via `/usr/bin/tegrastats` (Alert if > 85°C).
- **Storage**: Monitor disk usage (`df -h`). Rotate logs weekly.

### B. Failure Mitigation Matrix

| Scenario | Action |
| :--- | :--- |
| Camera RTSP loss | Automatic retry every 5 seconds |
| Network disconnected | Log locally; sync when restored |
| TensorRT failure | Auto-fallback to original PyTorch model |
| Line misalignment | Alert for manual recalibration |

---

## 7. 📄 Compliance & Versioning

Every deployment must be documented in a `deployment_manifest.json`:

- **Model Version**: SHA-256 hash of `best.engine`.
- **Dataset Version**: Link to training set version (e.g., v1.4.2).
- **Thresholds**: Specific confidence and ROI settings used.
- **Timestamp**: Date of deployment/last calibration.

---

**Contact**: For JK Aligarh site-specific support or audit requirements, contact the BEUMER ML Engineering Team.
