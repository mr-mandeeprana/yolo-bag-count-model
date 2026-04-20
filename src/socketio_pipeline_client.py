import json
import logging
import time
from typing import Optional, Dict, Any

try:
    import socketio
except Exception:
    socketio = None


class SocketIOPipelineClient:
    """
    Lightweight client for pushing bag count events to the existing Socket.IO pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("BagCounter.SocketIOPipeline")
        self.enabled = config.get('enabled', False)

        # Socket.IO settings (primary path: app -> socket.io -> vector -> elk).
        self.socketio_url = str(config.get('socketio_url', 'http://localhost:3000'))
        self.socketio_event = str(config.get('socketio_event', 'bag_count'))
        self.socketio_namespace = str(config.get('socketio_namespace', '/'))
        self.socketio_path = str(config.get('socketio_path', 'socket.io'))
        self.socket_timeout_s = float(config.get('socket_timeout_s', 3.0))
        self.socketio_transports = config.get('socketio_transports', ['websocket', 'polling'])
        self._socketio_client = None

        # Business payload metadata.
        self.parentid = str(config.get('parentid', 'TruckLoadingCount'))
        self.commission = str(config.get('commission', 'E0-00068100'))
        self.sensorid = str(config.get('sensorid', 'BagsCount'))
        self.sourceid = str(config.get('sourceid', 'Truck01'))
        self.event = str(config.get('event', 'sensor'))
        self.data_source_id = str(config.get('data_source_id', 'FP01'))

        self.base_url = self.socketio_url

        if not self.enabled:
            self.logger.info("Socket.IO pipeline integration is disabled.")
            return

        # Attempt to ping with retries to account for container startup lag
        connected = False
        max_retries = 5
        for attempt in range(max_retries):
            if self._ping():
                self.logger.info(f"Successfully connected to socketio endpoint at {self.base_url}.")
                connected = True
                break
            if attempt < max_retries - 1:
                self.logger.debug(f"Ping failed (attempt {attempt + 1}/{max_retries}), retrying in 2s...")
                time.sleep(2)
        
        if not connected:
            self.logger.warning(f"Could not ping socketio endpoint at {self.base_url}. Updates may fail. Ensure your socketio service is running.")

    def _ping(self) -> bool:
        """Verify if the endpoint is reachable."""
        return self._ensure_socketio_connected()

    def push_event(self, increment: int, total_count: int, extra_data: Optional[Dict[str, Any]] = None):
        """
        Push bag count event(s) to the configured Socket.IO pipeline.
        """
        if not self.enabled:
            return

        event_count = max(0, int(increment))
        if event_count == 0:
            return
        start_count = max(1, int(total_count) - event_count + 1)

        for offset in range(event_count):
            bag_count = start_count + offset
            document = {
                "parentid": self.parentid,
                "revision": bag_count,
                "value": 1,  # Always 1 for each increment event
                "time": int(time.time() * 1000),
                "commission": self.commission,
                "sensorid": self.sensorid,
                "sourceid": self.sourceid,
                "event": self.event,
                "data_source_id": self.data_source_id
            }

            if extra_data:
                document.update(extra_data)

            self._push_socketio(document)

    def _ensure_socketio_connected(self) -> bool:
        """Create and connect a Socket.IO client if needed."""
        if socketio is None:
            self.logger.error("python-socketio is not installed. Install dependency to use socketio mode.")
            return False

        try:
            if self._socketio_client is None:
                self._socketio_client = socketio.Client(reconnection=True, logger=False, engineio_logger=False)

            if not self._socketio_client.connected:
                # Handle common server variations for path formatting and default namespace behavior.
                namespace = (self.socketio_namespace or '/').strip() or '/'
                if not namespace.startswith('/'):
                    namespace = f"/{namespace}"

                path_candidates = [self.socketio_path]
                if self.socketio_path.startswith('/'):
                    path_candidates.append(self.socketio_path.lstrip('/'))
                else:
                    path_candidates.append(f"/{self.socketio_path}")

                # Preserve order while removing duplicates.
                seen = set()
                unique_paths = []
                for p in path_candidates:
                    if p not in seen:
                        unique_paths.append(p)
                        seen.add(p)

                last_error = None
                for path in unique_paths:
                    try:
                        connect_kwargs = {
                            'socketio_path': path,
                            'wait_timeout': self.socket_timeout_s,
                            'transports': self.socketio_transports
                        }
                        if namespace != '/':
                            connect_kwargs['namespaces'] = [namespace]

                        self._socketio_client.connect(self.socketio_url, **connect_kwargs)
                        self.socketio_path = path
                        self.socketio_namespace = namespace
                        break
                    except Exception as e:
                        last_error = e

                if not self._socketio_client.connected and last_error is not None:
                    raise last_error

            return self._socketio_client.connected
        except Exception as e:
            self.logger.error(f"Failed to connect Socket.IO client: {e}")
            return False

    def _push_socketio(self, document: Dict[str, Any]) -> None:
        """Emit one JSON payload to Socket.IO server."""
        if not self._ensure_socketio_connected():
            return

        try:
            self._socketio_client.emit(self.socketio_event, document, namespace=self.socketio_namespace)
            self.logger.debug("Pushed event to Socket.IO")
        except Exception as e:
            self.logger.error(f"Failed to push event to Socket.IO: {e}")

    def close(self):
        """Close persistent Socket.IO connection if active."""
        if self._socketio_client is not None:
            try:
                if self._socketio_client.connected:
                    self._socketio_client.disconnect()
            except Exception:
                pass
        return
