# 🚀 YOLO Bag Counter - Complete Pipeline Setup Guide

## 📋 Pipeline Architecture

```
YOLO Model (Python)
    ↓ (emit via Socket.IO Client)
Socket.IO Backend Server (Node.js:3000)
    ↓ (HTTP POST)
Vector.dev (Port 8686)
    ↓ (process & index)
Elasticsearch (Port 9200)
    ↓ (visualize)
Kibana (Port 5601)
```

---

## 🔧 Prerequisites

### System Requirements
- **Docker & Docker Compose** (for containerized deployment)
- **Node.js 16+** (for local Socket.IO server)
- **Python 3.9+** (already have this)
- **4GB+ RAM** (for ELK + Vector)
- **Ports available**: 3000 (Socket.IO), 8686 (Vector), 9200 (ES), 5601 (Kibana)

### Software Setup
```bash
# Verify Docker
docker --version
docker-compose --version

# Verify Node.js (if running locally)
node --version
npm --version
```

---

## 📦 Step 1: Prepare Node.js Environment

### Option A: Run Socket.IO Server in Docker (Recommended)

No additional setup needed - Docker will build and run it.

### Option B: Run Socket.IO Server Locally

```bash
# Install Node.js dependencies
npm install

# Create logs directory
mkdir -p logs/socketio

# Test the server locally
node server.js
```

Expected output:
```
info: ====================================================
info: 🎯 Socket.IO Backend Server Started
info: ====================================================
info: 📡 Listening on: http://localhost:3000
info: 📤 Forwarding to Vector.dev: http://localhost:8686/logs
info: 🏥 Health check: http://localhost:3000/health
```

---

## 🐳 Step 2: Start Docker Containers

### Create directories for volumes

```bash
mkdir -p logs/vector
mkdir -p ./observability/vector
```

### Set up environment file

```bash
# Copy and customize environment
cp .env.example .env

# (Optional) Edit .env if needed
```

### Start all services

```bash
# Start containers (detached mode)
docker-compose -f docker-compose.observability.yml up -d

# Or verbose mode (see logs)
docker-compose -f docker-compose.observability.yml up
```

### Verify containers are running

```bash
docker-compose -f docker-compose.observability.yml ps
```

Expected output:
```
NAME                     SERVICE         STATUS
bagcounter-elasticsearch elasticsearch   Up (healthy)
bagcounter-vector        vector-dev      Up (healthy)
bagcounter-socketio      socketio-server Up (healthy)
bagcounter-kibana        kibana          Up (healthy)
bagcounter-logstash      logstash        Up
```

---

## ✅ Step 3: Verify All Services

### Health Checks

```bash
# Socket.IO Server Health
curl http://localhost:3000/health

# Vector.dev Health
curl http://localhost:8686/health

# Elasticsearch Health
curl http://localhost:9200/_cluster/health

# Kibana (browser)
open http://localhost:5601
```

---

## 🎬 Step 4: Start YOLO Inference with Pipeline

### Update your Python config

Ensure your `config/video_config.yaml` has:

```yaml
pipeline:
  enabled: true
  socketio_url: "http://localhost:3000"
  socketio_event: "bag_count"
  socketio_namespace: "/"
  socketio_path: "socket.io"
```

### Run inference with pipeline enabled

```bash
# Activate Python environment
.venv\Scripts\activate

# Run with live camera (RTSP)
python src/inference_video.py \
  --source "rtsp://your-camera-ip/stream" \
  --weights models/weights/best.pt \
  --config config/video_config.yaml

# OR run with video file
python src/inference_video.py \
  --source "path/to/video.mp4" \
  --weights models/weights/best.pt \
  --config config/video_config.yaml
```

Expected output:
```
[INFO] 🎯 YOLO Bag Counter Started
[INFO] 📹 Source: rtsp://...
[INFO] 📡 Socket.IO Pipeline Enabled
[INFO] ✅ Connected to socketio at http://localhost:3000
[INFO] Starting inference...
[DEBUG] Frame 0: 12 bags detected
[DEBUG] 📤 Pushed event: bag_count=12
```

---

## 📊 Step 5: View in Kibana Dashboard

1. **Open Kibana**: `http://localhost:5601`
2. **Create Index Pattern**:
   - Go to: Menu → Stack Management → Index Patterns
   - Create new pattern: `bag-counter-*`
   - Timestamp field: `@timestamp`
3. **Create Visualization**:
   - Go to: Menu → Analytics → Discover
   - Select index pattern: `bag-counter-*`
   - View streaming bag count data in real-time

---

## 🔍 Step 6: Monitor Logs

### Socket.IO Server Logs

```bash
# Docker logs
docker-compose -f docker-compose.observability.yml logs socketio-server -f

# Or local file
tail -f logs/socketio/socketio-server.log
```

### Vector.dev Logs

```bash
docker-compose -f docker-compose.observability.yml logs vector-dev -f

# Or local file
tail -f logs/vector/vector.log
```

### Elasticsearch Logs

```bash
docker-compose -f docker-compose.observability.yml logs elasticsearch -f
```

---

## 🧪 Testing the Pipeline

### Test 1: Manual Event Emission

```bash
# In Python REPL
import socketio
import json

socket = socketio.Client()
socket.connect('http://localhost:3000')

# Send test event
socket.emit('bag_count', {
    'parentid': 'TruckLoadingCount',
    'revision': 1,
    'value': 42,
    'time': int(time.time() * 1000),
    'commission': 'E0-00068100',
    'sensorid': 'BagsCount',
    'sourceid': 'Truck01',
    'event': 'sensor',
    'data_source_id': 'FP01'
})

socket.disconnect()
```

### Test 2: Check Vector.dev Received Data

```bash
# Query Elasticsearch for events
curl -s http://localhost:9200/bag-counter-2026.03.30/_search | jq .
```

Expected response: Documents with `bag_count` field.

---

## 🛑 Stopping Services

```bash
# Stop all containers (keep data)
docker-compose -f docker-compose.observability.yml stop

# Stop and remove containers (keep volumes)
docker-compose -f docker-compose.observability.yml down

# Remove everything including volumes (CAREFUL!)
docker-compose -f docker-compose.observability.yml down -v
```

---

## 🐛 Troubleshooting

### Socket.IO Server Won't Start

**Error**: Connection refused on localhost:3000

**Solution**:
```bash
# Check if port 3000 is in use
netstat -ano | find "3000"  # Windows
lsof -i :3000               # macOS/Linux

# Kill process or use different port
```

### Vector.dev Can't Connect to Elasticsearch

**Error**: Elasticsearch connection refused

**Solution**:
```bash
# Check Elasticsearch is healthy
curl http://localhost:9200/_cluster/health

# View Elasticsearch logs
docker-compose logs elasticsearch
```

### No Data in Kibana

**Steps to debug**:
1. Check Socket.IO server received events:
   ```bash
   docker-compose logs socketio-server | grep "bag_count"
   ```

2. Check Vector.dev processed data:
   ```bash
   docker-compose logs vector-dev | grep "Elasticsearch"
   ```

3. Check Elasticsearch has indices:
   ```bash
   curl http://localhost:9200/_cat/indices
   ```

4. Query raw data:
   ```bash
   curl http://localhost:9200/bag-counter-*/_search?pretty
   ```

---

## 📈 Performance Monitoring

### Check Vector.dev Throughput

```bash
# Watch Vector metrics
watch -n 1 'curl -s http://localhost:8686/health | jq .'
```

### Monitor Elasticsearch Metrics

```bash
curl http://localhost:9200/_stats/indexing?pretty | jq '.indices | length'
```

---

## 🚀 Production Considerations

1. **Enable Security**: Set Elasticsearch password
2. **Resource Limits**: Adjust Elasticsearch memory in docker-compose
3. **Backup**: Configure Elasticsearch snapshots
4. **Monitoring**: Add Prometheus + Grafana
5. **Load Balancing**: Use Nginx if scaling Socket.IO servers

---

## 📞 Quick Reference

| Component | Port | URL | Purpose |
|-----------|------|-----|---------|
| Socket.IO Server | 3000 | http://localhost:3000 | Event receiver |
| Vector.dev | 8686 | http://localhost:8686 | Log processing |
| Elasticsearch | 9200 | http://localhost:9200 | Storage & query |
| Kibana | 5601 | http://localhost:5601 | Visualization |

---

## ✨ Next Steps

After pipeline is running:
1. Create Kibana dashboard with bag count trends
2. Set up alerts for count anomalies
3. Add camera_id/location tracking for multi-site
4. Export data for analytics & reporting
