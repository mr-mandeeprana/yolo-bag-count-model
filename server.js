/**
 * Socket.IO Backend Server
 * Receives bag count events from YOLO model and forwards to Vector.dev
 * 
 * Pipeline:
 * YOLO (Python) → Socket.IO Client → This Server → Vector.dev → Elasticsearch → Kibana
 */

const express = require('express');
const { Server } = require('socket.io');
const http = require('http');
const axios = require('axios');
const winston = require('winston');

// Logger setup
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.simple()
    }),
    new winston.transports.File({ 
      filename: 'logs/socketio-server.log',
      maxsize: 10485760, // 10MB
      maxFiles: 5
    })
  ]
});

// Configuration
const SOCKETIO_PORT = process.env.SOCKETIO_PORT || 3000;
const VECTOR_HOST = process.env.VECTOR_HOST || 'localhost';
const VECTOR_PORT = process.env.VECTOR_PORT || 8686;
const VECTOR_PATH = process.env.VECTOR_PATH || '/logs';
const VECTOR_ENDPOINT = `http://${VECTOR_HOST}:${VECTOR_PORT}${VECTOR_PATH}`;

const app = express();
app.disable('x-powered-by');
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  },
  transports: ['websocket', 'polling']
});

// Middleware
app.use(express.json());

// Browser-friendly ingest route for manual checks
app.get('/logs', (req, res) => {
  res.status(200).json({
    status: 'ok',
    message: 'Ingest endpoint is active. Use POST with JSON body to submit events.',
    example: {
      parentid: 'TruckLoadingCount',
      revision: 1,
      value: 1,
      time: Date.now(),
      commission: 'E0-00068100',
      sensorid: 'BagsCount',
      sourceid: 'Truck01',
      event: 'sensor',
      data_source_id: 'FP01'
    }
  });
});

// Optional HTTP ingestion for manual testing (same payload contract as Socket.IO)
app.post('/logs', async (req, res) => {
  try {
    const vectorPayload = transformBagCountData(req.body || {});
    await forwardToVector(vectorPayload);
    res.status(200).json({ status: 'received', timestamp: new Date().toISOString() });
  } catch (error) {
    logger.error('Error processing HTTP /logs payload:', error.message || error);
    res.status(500).json({ status: 'error', message: error.message || 'Unknown error' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    vectorEndpoint: VECTOR_ENDPOINT
  });
});

// Socket.IO event handlers
io.on('connection', (socket) => {
  logger.info(`Client connected: ${socket.id}`);

  // Main bag_count event listener
  socket.on('bag_count', async (data) => {
    try {
      logger.debug(`Received bag_count event:`, data);

      // Transform data to Vector compatible format
      const vectorPayload = transformBagCountData(data);
      logger.debug(`Transformed payload:`, vectorPayload);

      // Forward to Vector.dev
      await forwardToVector(vectorPayload);
      socket.emit('ack', { status: 'received', timestamp: new Date().toISOString() });
    } catch (error) {
      logger.error(`Error processing bag_count:`, error);
      socket.emit('error', { message: error.message });
    }
  });

  // Handle disconnection
  socket.on('disconnect', (reason) => {
    logger.info(`Client disconnected: ${socket.id} - Reason: ${reason}`);
  });

  // Handle errors
  socket.on('error', (error) => {
    logger.error(`Socket error for ${socket.id}:`, error);
  });
});

/**
 * Transform YOLO output to Vector.dev compatible format
 */
function transformBagCountData(rawData) {
  const now = new Date();
  
  return {
    timestamp: rawData.time || rawData.timestamp || now.getTime(),
    '@timestamp': now.toISOString(),
    
    // Metadata from YOLO output
    parentid: rawData.parentid,
    revision: rawData.revision ?? rawData.bag_count ?? rawData.value,
    value: rawData.value ?? rawData.bag_count,
    bag_count: rawData.bag_count ?? rawData.value,
    
    // Sensor information
    commission: rawData.commission,
    sensorid: rawData.sensorid,
    sourceid: rawData.sourceid,
    data_source_id: rawData.data_source_id,
    event_type: rawData.event || 'sensor',
    
    // Additional metadata for analytics
    ingestion_timestamp: now.toISOString(),
    source: 'yolo_bag_counter',
    environment: process.env.ENVIRONMENT || 'production',
    
    // Extra data if provided
    ...(rawData.extra_data && rawData.extra_data)
  };
}

/**
 * Forward transformed data to Vector.dev
 */
async function forwardToVector(payload) {
  try {
    const response = await axios.post(VECTOR_ENDPOINT, payload, {
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 15000
    });

    logger.debug(`Successfully forwarded to Vector:`, {
      status: response.status,
      statusText: response.statusText
    });

    return response.data;
  } catch (error) {
    if (error.response) {
      logger.error(`Vector returned error:`, {
        status: error.response.status,
        data: error.response.data
      });
    } else if (error.code === 'ECONNREFUSED') {
      logger.error(`Cannot connect to Vector at ${VECTOR_ENDPOINT}. Is it running?`);
    } else {
      logger.error(`Error forwarding to Vector:`, error.message);
    }
    throw error;
  }
}

// Health check to Vector on startup
async function checkVectorHealth() {
  try {
    const response = await axios.get(`http://${VECTOR_HOST}:8687/health`, { timeout: 5000 });
    logger.info('Vector is healthy:', response.data);
    return true;
  } catch (error) {
    logger.warn(`Vector health check failed: ${error.message}`);
    logger.warn(`Continuing anyway - Vector may start later...`);
    return false;
  }
}

// Start server
async function startServer() {
  try {
    // Check Vector health
    await checkVectorHealth();

    server.listen(SOCKETIO_PORT, () => {
      logger.info(`====================================================`);
      logger.info(`🎯 Socket.IO Backend Server Started`);
      logger.info(`====================================================`);
      logger.info(`📡 Listening on: http://localhost:${SOCKETIO_PORT}`);
      logger.info(`📤 Forwarding to Vector: ${VECTOR_ENDPOINT}`);
      logger.info(`🏥 Health check: http://localhost:${SOCKETIO_PORT}/health`);
      logger.info(`====================================================`);
    });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

startServer();
