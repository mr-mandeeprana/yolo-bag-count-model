"""Backward compatibility shim.

Use SocketIOPipelineClient from src.socketio_pipeline_client.
Prefer the `pipeline` config section in YAML (fallback to `elasticsearch` is supported).
"""

from src.socketio_pipeline_client import SocketIOPipelineClient

# Backward compatible alias for legacy imports.
ElasticsearchClient = SocketIOPipelineClient
