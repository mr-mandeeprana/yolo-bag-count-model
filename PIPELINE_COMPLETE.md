# ✅ YOLO Bag Counter - Complete Pipeline Implementation

## Summary

Your YOLO bag counter pipeline is **NOW COMPLETE** with all components wired together:

```
YOLO Model → Socket.IO Client → Socket.IO Server (Node.js) → Vector.dev → Elasticsearch → Kibana
```

---

## 📦 What Was Created

### Core Files

| File | Purpose |
|------|---------|
| `server.js` | Socket.IO Backend Server (Node.js) - listens on port 3000 |
| `package.json` | Node.js dependencies (socket.io, express, axios, winston) |
| `Dockerfile.socketio` | Docker image for Socket.IO server |
| `observability/vector/vector.toml` | Vector.dev configuration for log processing |
| `.env.example` | Environment variables template |

### Configuration & Documentation

| File | Purpose |
|------|---------|
| `docker-compose.observability.yml` | **UPDATED** - added Vector.dev & Socket.IO services |
| `PIPELINE_SETUP.md` | Step-by-step setup guide (comprehensive) |
| `ARCHITECTURE.md` | Complete architecture documentation |
| `start_pipeline.ps1` | Windows PowerShell startup script |

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Node.js Dependencies
```bash
npm install
```

### Step 2: Start Docker Services
```bash
docker-compose -f docker-compose.observability.yml up -d
```

### Step 3: Run YOLO Inference
```bash
# Activate Python environment
.venv\Scripts\activate

# Run with your camera or video
python src/inference_video.py \
  --source "rtsp://your-camera-ip/stream" \
  --weights models/weights/best.pt \
  --config config/video_config.yaml
```

**Done!** 🎉 Data flows to Kibana: `http://localhost:5601`

---

## 🔄 Complete Pipeline Flow

```
1. YOLO Detection (Python)
   ↓
   Detects bags, counts them
   Creates JSON: { value: 42, timestamp: ..., ... }
   
2. Socket.IO Client (Python socketio_pipeline_client.py)
   ↓
   Emits event to http://localhost:3000
   
3. Socket.IO Backend Server (Node.js server.js)
   ↓
   Receives event
   Transforms to Vector format
   HTTP POST to http://vector-dev:8686/logs
   
4. Vector.dev (observability/vector/vector.toml)
   ↓
   Normalizes data
   Adds metadata
   Creates index: bag-counter-YYYY.MM.DD
   Bulk inserts to Elasticsearch
   
5. Elasticsearch (http://localhost:9200)
   ↓
   Stores indexed documents
   Ready for search/aggregation
   
6. Kibana (http://localhost:5601)
   ↓
   User sees real-time bag counts
   Can create dashboards, alerts, etc.
```

---

## 🏗️ Architecture Components

### Socket.IO Server (Node.js)

**What it does**:
- Listens for Socket.IO events from Python YOLO script
- Receives: `{ value: 42, timestamp: ..., ... }`
- Converts to Vector.dev compatible JSON
- POSTs to Vector.dev HTTP endpoint

**Key Features**:
- Automatic reconnection handling
- Health check endpoint `/health`
- Comprehensive logging to `logs/socketio/`
- Docker-containerized

**Config**:
- Controlled by environment variables:
  - `SOCKETIO_PORT=3000`
  - `VECTOR_DEV_HOST=vector-dev`
  - `VECTOR_DEV_PORT=8686`

---

### Vector.dev (Log Processor)

**What it does**:
- Listens on HTTP `:8686/logs`
- Receives JSON from Socket.IO server
- Transforms/enriches the data
- Filters valid records
- Bulk inserts into Elasticsearch

**Transformations Applied**:
1. Ensure timestamp in ISO8601 format
2. Normalize bag_count to integer
3. Add metadata (batch_id, processing_pipeline)
4. Create dynamic Elasticsearch index name
5. Filter out invalid records

**Configuration**: `observability/vector/vector.toml`

---

### Docker Compose Updates

**New Services**:
- `vector-dev` - Vector.dev container
- `socketio-server` - Node.js Socket.IO server

**Networking**:
- All services on `bagcounter-network` bridge
- Can reference each other by container name
- Ports exposed to host for debugging

**Dependencies**:
- Socket.IO waits for Vector.dev
- Vector waits for Elasticsearch
- Kibana waits for Elasticsearch

---

## 🔧 Configuration Files

### Python (Already Have)
**File**: `config/video_config.yaml`
```yaml
pipeline:
  enabled: true
  socketio_url: "http://localhost:3000"  ← Points to Socket.IO server
  socketio_event: "bag_count"
  socketio_namespace: "/"
  socketio_path: "socket.io"
  socketio_transports: ["websocket", "polling"]
```

### Node.js (New)
**File**: `server.js`
- Listens on port 3000
- Forwards to Vector on port 8686
- Auto-reconnects on failure
- Logs everything

### Vector.dev (New)
**File**: `observability/vector/vector.toml`
- HTTP source on :8686
- Transformation steps
- Elasticsearch sink
- Dynamic indexing

---

## 📊 Verification Checklist

### ✅ Services Running
```bash
docker-compose -f docker-compose.observability.yml ps
```
Expected: All containers `Up` and healthy

### ✅ Health Checks
```bash
curl http://localhost:3000/health       # Socket.IO
curl http://localhost:8686/health       # Vector
curl http://localhost:9200/_cluster/health  # Elasticsearch
```

### ✅ Data Flow
1. YOLO running + emitting events (check logs)
2. Vector receives (check: `docker logs bagcounter-vector`)
3. Elasticsearch has indices (`curl http://localhost:9200/_cat/indices`)
4. Kibana shows data (http://localhost:5601 → Discover)

---

## 🧪 Test the Pipeline

### Manual Test (Python REPL)
```python
import socketio
import time

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

### Check Elasticsearch Got Data
```bash
curl -s http://localhost:9200/bag-counter-*/_search?pretty | jq '.hits.hits[0]._source'
```

### Check Kibana
1. Go to http://localhost:5601
2. Menu → Discover
3. Select `bag-counter-*` index pattern
4. Should see documents with `bag_count` field

---

## 🚀 Advanced: Using start_pipeline.ps1

**Windows PowerShell Script** for automated setup:

```bash
# Start just Docker services
.\start_pipeline.ps1 -DockerOnly

# Start Docker + YOLO inference (interactive)
.\start_pipeline.ps1 -Full

# Stop all services
.\start_pipeline.ps1 -Stop

# View logs
.\start_pipeline.ps1 -Logs
```

---

## 📚 Documentation Files

| File | Read This For |
|------|---------------|
| `PIPELINE_SETUP.md` | Detailed step-by-step setup guide |
| `ARCHITECTURE.md` | Complete architecture & data flow |
| `README.md` | Original YOLO project info |
| `start_pipeline.ps1` | Automated Windows startup |

---

## 🔍 Troubleshooting

### "Connection refused on localhost:3000"
```bash
# Check if port in use
netstat -ano | find "3000"
# Kill process or use different port
```

### "Vector.dev can't connect to Elasticsearch"
```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# Check logs
docker-compose logs elasticsearch
```

### "No data in Kibana"
```bash
# 1. Check YOLO running
ps aux | grep inference_video

# 2. Check Socket.IO events received
docker-compose logs socketio-server | grep "bag_count"

# 3. Check Vector processing
docker-compose logs vector-dev | grep "Elasticsearch"

# 4. Check Elasticsearch indices
curl http://localhost:9200/_cat/indices
```

---

## 📈 What's Next?

1. **Create Kibana Dashboard**
   - Visualize bag count trends
   - Add real-time counter
   - Show per-camera stats

2. **Set Up Alerts**
   - Alert when count drops to 0
   - Alert on high variance
   - Custom anomaly detection

3. **Multi-Site Support**
   - Add `camera_id` to data
   - Multiple YOLO instances per site
   - Centralized monitoring

4. **Export & Analytics**
   - Export hourly/daily counts
   - Generate reports
   - Integrate with ERP/MES

---

## 📞 Quick Reference

### Ports
- **3000** - Socket.IO Server (Python → Node.js)
- **8686** - Vector.dev HTTP (Node.js → Vector)
- **9200** - Elasticsearch (Vector → ES)
- **5601** - Kibana (Browser)

### Key Files
- **Python Config**: `config/video_config.yaml`
- **Node.js Server**: `server.js`
- **Vector Config**: `observability/vector/vector.toml`
- **Docker Setup**: `docker-compose.observability.yml`

### Commands
```bash
# Install dependencies
npm install

# Start services
docker-compose -f docker-compose.observability.yml up -d

# Run YOLO
python src/inference_video.py --source "..." --weights "..." --config "config/video_config.yaml"

# Check health
curl http://localhost:3000/health
curl http://localhost:9200/_cluster/health

# View Kibana
open http://localhost:5601
```

---

## ✨ You're All Set!

Your complete pipeline is ready to go:

```
✅ Socket.IO Backend Server - Receives YOLO events
✅ Vector.dev - Processes & routes data  
✅ Elasticsearch - Stores data
✅ Kibana - Visualizes results
✅ Docker Integration - Containerized deployment
✅ Comprehensive Documentation
```

**Start now with**: `docker-compose -f docker-compose.observability.yml up -d`

🚀 **Happy bag counting!**
