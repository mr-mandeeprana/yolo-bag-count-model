# 🏗️ YOLO Bag Counter - Complete Architecture

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOLO BAG COUNTER PIPELINE                           │
│                                                                               │
│  ┌──────────┐      ┌──────────────────┐      ┌────────────────┐             │
│  │   YOLO   │      │  Socket.IO       │      │  Vector.dev    │             │
│  │ Detection│─────▶│  Backend Server  │─────▶│  Log Processor │             │
│  │  (Python)│      │  (Node.js:3000)  │      │  (Port:8686)   │             │
│  └──────────┘      └──────────────────┘      └────────────────┘             │
│       │                     │                         │                       │
│       ├─ Frame              ├─ Socket emission        ├─ HTTP POST             │
│       │  Inference          │  Socket.IO event       │  JSON payload          │
│       │                     ├─ Health check          ├─ Transform data        │
│       │                     │  (http://localhost:    │  (filter, enrich)      │
│       │                     │   3000/health)         │  Log buffering         │
│       │                     │                        │                        │
│       └─ Outputs:           └─ Forwards to Vector    └─ Sends to ES           │
│          - Bag count                                                          │
│          - Timestamp                                                          │
│          - Detections                                                        │
│          - Confidence                                                        │
│                                                                               │
│                           ┌──────────────────┐                              │
│                           │  Elasticsearch   │                              │
│                           │  (Port:9200)     │                              │
│                           │  Storage & Index │                              │
│                           └──────────────────┘                              │
│                                   △                                          │
│                                   │                                          │
│                                   │ Bulk API                                │
│                                   │ Create indices                          │
│                                   │ bag-counter-YYYY.MM.DD                 │
│                                   │                                          │
│                           ┌──────────────────┐                              │
│                           │     Kibana       │                              │
│                           │  (Port:5601)     │                              │
│                           │ Visualization &  │                              │
│                           │   Dashboards     │                              │
│                           └──────────────────┘                              │
│                                   △                                          │
│                                   │                                          │
│                          User Queries & Viz                                 │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1️⃣ YOLO Detection (Python)

**File**: `src/inference_video.py`

**Role**: Detect and count bags

**Key Outputs**:
```python
{
    "parentid": "TruckLoadingCount",
    "revision": 42,
    "value": 42,
    "time": 1711836923000,
    "commission": "E0-00068100",
    "sensorid": "BagsCount",
    "sourceid": "Truck01",
    "event": "sensor",
    "data_source_id": "FP01"
}
```

**Configuration**: `config/video_config.yaml` → `pipeline` section

---

### 2️⃣ Socket.IO Backend Server (Node.js)

**File**: `server.js`

**Port**: 3000

**Role**: 
- Receive Socket.IO emissions from Python client
- Transform data to Vector.dev format
- Forward to Vector.dev HTTP endpoint
- Health check endpoint

**Health Check**:
```bash
curl http://localhost:3000/health
```

**Docker Image**: Built from `Dockerfile.socketio`

**Logs**: `logs/socketio/socketio-server.log`

---

### 3️⃣ Vector.dev (log processor)

**File**: `observability/vector/vector.toml`

**Port**: 8686

**Role**:
- HTTP source listener (receives JSON from Socket.IO server)
- Transform & enrich data
- Filter valid records
- Add Elasticsearch metadata
- Bulk ingest into Elasticsearch

**Data Transform**:
```toml
# Normalize fields
.bag_count = to_int(.value)
.timestamp = now() * 1000
.@timestamp = now()

# Add index naming
.target_index = "bag-counter-" + format_timestamp!(%Y.%m.%d, .timestamp / 1000)
```

**Outputs**:
- Elasticsearch (primary)
- Console (debug only)

---

### 4️⃣ Elasticsearch (Storage)

**Port**: 9200

**Role**: 
- Index and store bag count events
- Fast searching and aggregation
- Time-series data retention

**Index Pattern**: 
```
bag-counter-2026.03.30
bag-counter-2026.03.31
... (rotates daily)
```

**Document Structure**:
```json
{
  "@timestamp": "2026-03-30T10:15:23.000Z",
  "bag_count": 42,
  "revision": 42,
  "sensorid": "BagsCount",
  "sourceid": "Truck01",
  "commission": "E0-00068100",
  "target_index": "bag-counter-2026.03.30"
}
```

---

### 5️⃣ Kibana (Visualization)

**Port**: 5601

**Role**:
- Create index patterns
- Build dashboards
- Real-time monitoring
- Analytics and trends

**Setup**:
1. Go to `http://localhost:5601`
2. Create index pattern: `bag-counter-*`
3. Visualize in Discover or Dashboard

---

## Network & Docker Setup

### Docker Services

```yaml
Services:
├── elasticsearch      (docker.elastic.co/elasticsearch:8.13.4)
├── vector-dev         (timberio/vector:latest)
├── socketio-server    (Built from Dockerfile.socketio)
├── kibana             (docker.elastic.co/kibana:8.13.4)
└── logstash           (docker.elastic.co/logstash:8.13.4) [Optional]
```

### Network

```
All Docker services connected via: bagcounter-network (bridge)

External Access (Host → Container):
├── localhost:3000   → socketio-server:3000
├── localhost:8686   → vector-dev:8686
├── localhost:9200   → elasticsearch:9200
├── localhost:5601   → kibana:5601
└── localhost:8080   → logstash:8080 (optional)
```

### Ports

| Service | Port | Protocol | From | To |
|---------|------|----------|------|-----|
| Socket.IO | 3000 | HTTP/WS | Python YOLO | Node.js Server |
| Vector.dev | 8686 | HTTP | Node.js Server | Vector Container |
| Elasticsearch | 9200 | HTTP | Vector, Kibana | ES Container |
| Kibana | 5601 | HTTP | Browser | Kibana Container |

---

## Data Transformations

### Stage 1: YOLO → Socket.IO Client

**Input**: Frame data from camera/video
**Output**: Socket.IO event with bag count

```python
socket.emit('bag_count', {
    'parentid': 'TruckLoadingCount',
    'value': 42,
    'time': 1711836923000,
    ...
})
```

### Stage 2: Socket.IO Server Processing

**Input**: Socket.IO event from Python
**Output**: HTTP POST to Vector.dev

```javascript
// Transform in server.js
const vectorPayload = {
    timestamp: data.time,
    '@timestamp': new Date().toISOString(),
    bag_count: data.value,
    ...data
}
```

### Stage 3: Vector.dev Transform

**Input**: HTTP JSON from Socket.IO
**Output**: Elasticsearch bulk ingest

```toml
# In vector.toml
[transforms.enrich_data]
type = "remap"
source = '''
.bag_count = to_int(.value)
.@timestamp = now()
.target_index = "bag-counter-" + format_timestamp!(%Y.%m.%d, .timestamp / 1000)
'''
```

### Stage 4: Elasticsearch Indexing

**Input**: Transformed JSON from Vector
**Output**: Indexed documents in ES

```json
Index: bag-counter-2026.03.30
Doc:
{
  "@timestamp": "2026-03-30T10:15:23Z",
  "bag_count": 42,
  "sensorid": "BagsCount",
  ...
}
```

---

## Configuration Files

### Python Configuration
- `config/video_config.yaml` → `pipeline` section (Socket.IO settings)
- `config/model_config.yaml` → YOLO model parameters
- `.env` (optional) → Environment variables

### Node.js Configuration
- `server.js` → Server logic
- `package.json` → Dependencies
- Environment variables: `SOCKETIO_PORT`, `VECTOR_DEV_HOST`, `VECTOR_DEV_PORT`

### Vector.dev Configuration
- `observability/vector/vector.toml` → Processing rules

### Docker Configuration
- `docker-compose.observability.yml` → Container orchestration
- `Dockerfile.socketio` → Socket.IO server build

---

## Monitoring & Logging

### Log Locations

```
logs/
├── socketio-server.log          # Socket.IO server
├── vector/
│   └── vector.log               # Vector.dev
└── (Elasticsearch logs in Docker)
```

### Log Levels

- **Socket.IO**: Controlled by `LOG_LEVEL` env var (default: `info`)
- **Vector**: Controlled by `VECTOR_LOG_LEVEL` env var (default: `info`)

### Health Checks

```bash
# Socket.IO Server
curl http://localhost:3000/health

# Vector.dev (internal)
curl http://localhost:8686/health

# Elasticsearch
curl http://localhost:9200/_cluster/health

# Kibana
curl http://localhost:5601/api/status
```

---

## Scaling Considerations

### Single-Site (Current)
- One YOLO process
- One Socket.IO instance
- One Vector.dev instance
- Shared Elasticsearch cluster

### Multi-Site (Future)
- Multiple YOLO processes (different cameras)
- Multiple Socket.IO instances (load-balanced)
- Single Vector.dev (centralizes processing)
- Shared Elasticsearch (all data)
- Enhanced indexing by `sourceid` or `camera_id`

---

## Troubleshooting Flow

```
Issue: No data in Kibana
│
├─ Check YOLO running?
│  └─ `ps aux | grep inference_video`
│
├─ Check Socket.IO events?
│  └─ `docker logs bagcounter-socketio | grep bag_count`
│
├─ Check Vector processing?
│  └─ `docker logs bagcounter-vector | grep Elasticsearch`
│
├─ Check Elasticsearch?
│  └─ `curl http://localhost:9200/_cat/indices`
│
└─ Check Kibana index pattern?
   └─ Menu → Stack Management → Index Patterns
```

---

## Performance Characteristics

| Component | Latency | Throughput | Bottleneck |
|-----------|---------|-----------|------------|
| YOLO Detection | 30-100ms | 10-30 FPS | GPU |
| Socket.IO | 1-5ms | Unlimited | Network |
| Vector.dev | 10-50ms | 1000+ events/s | I/O |
| Elasticsearch | 100-500ms | 1000+ docs/s | Memory |
| Kibana | 500ms-2s | Dashboard refresh | Aggregation |

---

## Next Steps

1. **Deploy**: Follow `PIPELINE_SETUP.md`
2. **Monitor**: Create Kibana dashboards
3. **Scale**: Add multi-site support
4. **Automate**: Set up Elasticsearch snapshot backups
5. **Alert**: Configure alerts in Kibana for anomalies
