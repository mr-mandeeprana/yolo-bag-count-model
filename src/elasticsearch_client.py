import json
import logging
from datetime import datetime
import time
from typing import Optional, Dict, Any
from urllib import request, error


class ElasticsearchClient:
    """
    Lightweight client for pushing bag count events to Elasticsearch.
    Uses direct HTTP to avoid Python client/server major-version mismatch issues.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("BagCounter.Elasticsearch")
        self.enabled = config.get('enabled', False)
        self.index = config.get('index', 'bag-counts')
        self.ingestion_mode = str(config.get('ingestion_mode', 'elasticsearch')).strip().lower()

        if self.ingestion_mode not in ('elasticsearch', 'logstash'):
            self.logger.warning(
                "Unknown ingestion_mode '%s'. Falling back to 'elasticsearch'.",
                self.ingestion_mode
            )
            self.ingestion_mode = 'elasticsearch'

        host = config.get('host', 'localhost')
        port = config.get('port', 9200)
        scheme = config.get('scheme', 'http')

        if self.ingestion_mode == 'logstash':
            ls_host = config.get('logstash_host', host)
            ls_port = config.get('logstash_port', 8080)
            ls_scheme = config.get('logstash_scheme', 'http')
            ls_path = str(config.get('logstash_path', '/')).strip() or '/'
            if not ls_path.startswith('/'):
                ls_path = f"/{ls_path}"

            self.base_url = f"{ls_scheme}://{ls_host}:{ls_port}"
            self.endpoint_url = f"{self.base_url}{ls_path}"
        else:
            self.base_url = f"{scheme}://{host}:{port}"
            self.endpoint_url = f"{self.base_url}/{self.index}/_doc"

        if not self.enabled:
            self.logger.info("Elasticsearch integration is disabled.")
            return

        # Attempt to ping with retries to account for container startup lag
        connected = False
        max_retries = 5
        for attempt in range(max_retries):
            if self._ping():
                self.logger.info(f"Successfully connected to {self.ingestion_mode} endpoint at {self.base_url}.")
                connected = True
                break
            if attempt < max_retries - 1:
                self.logger.debug(f"Ping failed (attempt {attempt + 1}/{max_retries}), retrying in 2s...")
                time.sleep(2)
        
        if not connected:
            self.logger.warning(f"Could not ping {self.ingestion_mode} endpoint at {self.base_url}. Updates may fail. Ensure Docker containers are fully initialized.")

    def _ping(self) -> bool:
        """Verify if the endpoint is reachable."""
        try:
            req = request.Request(self.base_url, method='GET')
            with request.urlopen(req, timeout=3) as resp:
                # 200-299 is success
                return 200 <= resp.status < 300
        except error.HTTPError as e:
            # For Logstash, a GET might return 404 (Not Found) or 405 (Method Not Allowed),
            # but it still means the server is UP and listening.
            if self.ingestion_mode == 'logstash' and e.code in (404, 405):
                return True
            return False
        except Exception:
            return False

    def push_event(self, increment: int, total_count: int, extra_data: Optional[Dict[str, Any]] = None):
        """
        Push a bag count event to Elasticsearch.
        """
        if not self.enabled:
            return

        document = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "bag_detection",
            "bag_count": increment,
            "total_count": total_count,
            "system": "Fillpac-Bag-Counter"
        }

        # Help Logstash route into the expected Elasticsearch index.
        if self.ingestion_mode == 'logstash':
            document["target_index"] = self.index

        if extra_data:
            document.update(extra_data)

        payload = json.dumps(document).encode('utf-8')
        url = self.endpoint_url
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        try:
            req = request.Request(url, data=payload, headers=headers, method='POST')
            with request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    self.logger.debug("Pushed event to Elasticsearch")
                else:
                    self.logger.error(f"Failed to push event to Elasticsearch. HTTP {resp.status}")
        except error.HTTPError as e:
            body = ''
            try:
                body = e.read().decode('utf-8', errors='ignore')
            except Exception:
                pass
            self.logger.error(f"Failed to push event to Elasticsearch: HTTP {e.code} {body}")
        except Exception as e:
            self.logger.error(f"Failed to push event to Elasticsearch: {e}")

    def close(self):
        """No persistent connection to close for HTTP-based client."""
        return
