"""
Service entry point for the web UI addon.

Loaded only in the headless main process. Publishes the FastAPI
application factory in the global service registry under the key
``webui``; :mod:`rayforge.serve` resolves it from there and runs it
with uvicorn.
"""

import logging

from rayforge.core.hooks import hookimpl

logger = logging.getLogger(__name__)

ADDON_NAME = "webui"
SERVICE_KEY = "webui"


@hookimpl
def register_services(service_registry):
    from .server import create_app

    service_registry.register(SERVICE_KEY, create_app, addon_name=ADDON_NAME)
