# Deployment Guide: YOLO Bag Counting System

This guide covers deployment options for the YOLO bag counting system on Fillpac production lines.

## Deployment Options Overview

| Option | Hardware | Latency | Cost | Best For |
|--------|----------|---------|------|----------|
| Edge (Jetson) | NVIDIA Jetson Xavier/Orin | <50ms | Medium | Real-time, on-site |
| Cloud GPU | AWS/Azure GPU instances | 100-300ms | High | Centralized monitoring |
| Hybrid | Jetson + Cloud | <50ms local | Medium-High | Best of both worlds |

---

## Option 1: Edge Deployment (Recommended)

### Hardware Requirements

**Recommended Devices:**
- **NVIDIA Jetson Xavier NX** (Best balance)
  - 384 CUDA cores, 48 Tensor cores
  - 8GB RAM
  - ~$400
  - Performance: 30-60 FPS with YOLOv8n

- **NVIDIA Jetson Orin Nano** (Latest)
  - 1024 CUDA cores
  - 8GB RAM
  - ~$500
  - Performance: 60-80 FPS with YOLOv8n

- **NVIDIA Jetson Nano** (Budget option)
  - 128 CUDA cores
  - 4GB RAM
  - ~$100
  - Performance: 15-25 FPS with YOLOv8n

### Setup Steps

#### 1. Prepare Model for Deployment

On your development machine:

```powershell
# Export to TensorRT (optimized for NVIDIA)
python -c "from ultralytics import YOLO; YOLO('models/weights/best.pt').export(format='engine', device=0, half=True)"

# This creates best.engine file
```

#### 2. Setup Jetson Device

```bash
# Install JetPack (includes CUDA, cuDNN, TensorRT)
# Follow: https://developer.nvidia.com/embedded/jetpack

# Install Python dependencies
sudo apt-get update
sudo apt-get install python3-pip
pip3 install ultralytics supervision opencv-python

# Verify installation
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
```

#### 3. Transfer Files to Jetson

```powershell
# From development machine
scp models/weights/best.engine jetson@jetson-ip:/home/jetson/bag_counter/
scp -r src/ jetson@jetson-ip:/home/jetson/bag_counter/
```

#### 4. Run Inference on Jetson

```bash
# On Jetson
cd /home/jetson/bag_counter

# Test with video file
python3 src/inference_video.py \
  --weights best.engine \
  --source test_video.mp4 \
  --output result.mp4

# Production: RTSP camera stream
python3 src/inference_video.py \
  --weights best.engine \
  --source rtsp://fillpac-camera-ip/stream \
  --log /var/log/bag_counter/counts.txt \
  --no-display
```

#### 5. Setup as System Service

Create `/etc/systemd/system/bag-counter.service`:

```ini
[Unit]
Description=Fillpac Bag Counter
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/bag_counter
ExecStart=/usr/bin/python3 src/inference_video.py \
  --weights best.engine \
  --source rtsp://camera-ip/stream \
  --log /var/log/bag_counter/counts.txt \
  --no-display
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable bag-counter
sudo systemctl start bag-counter
sudo systemctl status bag-counter
```

### Performance Optimization

**1. Use TensorRT FP16 (Half Precision)**
```python
YOLO('best.pt').export(format='engine', half=True)  # 2x speedup
```

**2. Reduce Input Resolution**
```python
# In inference script, resize frames
frame = cv2.resize(frame, (416, 416))  # Instead of 640x640
```

**3. Process Every Nth Frame**
```python
if frame_count % 2 == 0:  # Process every 2nd frame
    results = model(frame)
```

---

## Option 2: Cloud Deployment

### AWS Deployment

#### 1. Choose Instance Type

- **g4dn.xlarge**: 1x NVIDIA T4, 16GB GPU RAM (~$0.50/hour)
- **g5.xlarge**: 1x NVIDIA A10G, 24GB GPU RAM (~$1.00/hour)

#### 2. Setup EC2 Instance

```bash
# Launch Ubuntu 20.04 Deep Learning AMI
# SSH into instance

# Install dependencies
pip install ultralytics supervision opencv-python

# Upload model
scp models/weights/best.pt ubuntu@ec2-ip:/home/ubuntu/
```

#### 3. Stream Processing

```python
# On EC2, process RTSP streams
python src/inference_video.py \
  --weights best.pt \
  --source rtsp://fillpac-camera-ip/stream \
  --log s3://bucket/counts/log.txt  # Save to S3
```

#### 4. Setup Auto-scaling (Optional)

Use AWS Lambda + ECS for on-demand processing:
- Trigger on video upload to S3
- Process with Fargate GPU containers
- Store results in DynamoDB

### Azure Deployment

Similar to AWS, use:
- **NC6s_v3**: 1x NVIDIA V100
- Azure Container Instances with GPU
- Store logs in Azure Blob Storage

---

## Option 3: Hybrid Deployment

### Architecture

```
[Fillpac Camera] → [Jetson (Edge)] → [Cloud (Analytics)]
                         ↓
                  [Local Display]
                  [Real-time Count]
```

### Implementation

**Edge (Jetson):**
- Real-time counting
- Local logging
- Immediate alerts

**Cloud:**
- Long-term storage
- Analytics dashboard
- Multi-site aggregation

```python
# On Jetson: Send counts to cloud
import requests

def send_to_cloud(count, timestamp):
    requests.post(
        'https://api.yourcompany.com/bag-counts',
        json={'count': count, 'timestamp': timestamp, 'machine_id': 'fillpac-1'}
    )

# In inference loop
if frame_count % fps == 0:  # Every second
    send_to_cloud(line_zone.out_count, datetime.now().isoformat())
```

---

## Camera Setup

### RTSP Stream Configuration

Most IP cameras support RTSP:

```python
# Generic RTSP URL format
rtsp://username:password@camera-ip:554/stream

# Examples:
# Hikvision: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
# Dahua: rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
# Axis: rtsp://root:password@192.168.1.100/axis-media/media.amp
```

### USB Camera

```python
# USB camera (simpler for testing)
video_source = 0  # /dev/video0
cap = cv2.VideoCapture(video_source)
```

### Camera Positioning

**Optimal Setup:**
- Mount camera 2-3 meters above conveyor
- Angle: 30-45 degrees from vertical
- Ensure bags are fully visible when crossing counting line
- Avoid backlighting (camera facing away from windows/lights)

---

## Monitoring and Alerts

### Logging

```python
# In inference_video.py, add structured logging
import logging
import json

logging.basicConfig(
    filename='/var/log/bag_counter/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log counts
logging.info(json.dumps({
    'event': 'count_update',
    'count': line_zone.out_count,
    'frame': frame_count,
    'fps': current_fps
}))
```

### Alerts

```python
# Alert on discrepancy
expected_rate = 43  # bags per minute (2600/hour)
actual_rate = (line_zone.out_count / (frame_count / fps)) * 60

if abs(actual_rate - expected_rate) > 5:  # >5 bags/min difference
    send_alert(f"Count anomaly: {actual_rate:.1f} bags/min (expected {expected_rate})")
```

### Dashboard (Grafana)

1. Store counts in InfluxDB/Prometheus
2. Visualize with Grafana
3. Set up alerts for thresholds

---

## Troubleshooting

### Issue: Low FPS on Jetson

**Solutions:**
- Use TensorRT export with FP16
- Reduce input resolution to 416x416
- Use YOLOv8n instead of YOLOv8m
- Disable display (`--no-display`)

### Issue: RTSP Stream Disconnects

**Solutions:**
```python
# Add reconnection logic
while True:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Reconnecting in 5 seconds...")
        time.sleep(5)
        continue
    # Process frames...
```

### Issue: High Latency

**Solutions:**
- Deploy on edge (Jetson) instead of cloud
- Use UDP instead of TCP for RTSP: `rtsp://...?tcp=0`
- Reduce frame buffer: `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`

---

## Security Considerations

1. **Network Security:**
   - Use VPN for remote access to Jetson
   - Secure RTSP streams with authentication
   - Firewall rules to restrict access

2. **Data Privacy:**
   - Encrypt logs before sending to cloud
   - Anonymize video feeds if required
   - Comply with GDPR/data regulations

3. **Physical Security:**
   - Secure Jetson device in locked enclosure
   - Tamper-proof camera mounting

---

## Cost Estimation

### Edge Deployment (One Fillpac Machine)

| Item | Cost |
|------|------|
| Jetson Xavier NX | $400 |
| IP Camera (1080p) | $150 |
| Enclosure & Mounting | $100 |
| **Total Initial** | **$650** |
| **Ongoing** | $0/month |

### Cloud Deployment (One Machine)

| Item | Cost |
|------|------|
| g4dn.xlarge (24/7) | $360/month |
| Data transfer | $50/month |
| Storage (S3) | $10/month |
| **Total Monthly** | **$420/month** |

**Recommendation**: Edge deployment has better ROI after 2 months.

---

## Next Steps

1. Choose deployment option based on requirements
2. Procure hardware (if edge deployment)
3. Export model to appropriate format
4. Test in staging environment
5. Deploy to production with monitoring
6. Set up alerts and dashboards

For questions, refer to the main README or contact your ML team.
