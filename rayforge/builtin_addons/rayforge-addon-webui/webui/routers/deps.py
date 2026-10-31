"""Shared request helpers for the web UI routers."""

from gettext import gettext as _

from fastapi import HTTPException, Request

from ..core_bridge import CoreBridge


def get_bridge(request: Request) -> CoreBridge:
    """The CoreBridge connecting the web loop to the core loop."""
    return request.app.state.bridge


def tree_with_revision(request: Request, tree: dict) -> dict:
    """Stamp the current document revision onto a serialized tree."""
    tree["rev"] = request.app.state.events.tree_revision
    return tree


def not_found(message: str) -> HTTPException:
    return HTTPException(status_code=404, detail=message)


def bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def conflict(message: str) -> HTTPException:
    return HTTPException(status_code=409, detail=message)


def item_not_found() -> HTTPException:
    return not_found(_("Document item not found"))


def machine_unavailable() -> HTTPException:
    return HTTPException(
        status_code=503, detail=_("No active machine configured")
    )
