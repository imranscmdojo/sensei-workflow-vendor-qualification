"""
Admin Event Logger — writes usage events to Firestore admin_events collection.

Usage:
    from admin_logger import log_admin_event, log_token_usage

    await log_admin_event("ask_message_received", user_id, {
        "sessionId": session_id,
        "input_tokens": 1200,
        "output_tokens": 800,
        "total_tokens": 2000,
        "provider": "anthropic",
    })

For sync contexts (FastAPI endpoints):
    log_admin_event_sync("ask_message_received", user_id, {...})
"""
import os
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Lazy singleton — initialized on first use
_firestore_client = None


def _get_firestore():
    global _firestore_client
    if _firestore_client is None:
        try:
            from google.cloud import firestore
            _firestore_client = firestore.Client(
                project=os.environ.get("FIREBASE_PROJECT_ID", "sensei-ask-project"),
            )
        except Exception as e:
            logger.warning(f"[admin_logger] Failed to initialize Firestore: {e}")
    return _firestore_client


def _server_timestamp() -> Any:
    """The Firestore server-time sentinel.

    This lives on the `google.cloud.firestore` module, not on the Client
    instance the singleton holds. Reading it off the client raised
    `AttributeError: 'Client' object has no attribute 'SERVER_TIMESTAMP'` on
    every single write, so the whole admin_events audit trail was failing
    silently and only visible as a warning in the log.
    """
    try:
        from google.cloud import firestore
        return firestore.SERVER_TIMESTAMP
    except Exception:
        # Better a client-assigned timestamp than a lost audit record.
        return datetime.now(timezone.utc)


def _write_admin_event(
    event_type: str,
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    db = _get_firestore()
    if db is None:
        return False
    try:
        db.collection("admin_events").add({
            "type": event_type,
            "userId": user_id,
            "metadata": metadata or {},
            "timestamp": _server_timestamp(),
        })
        return True
    except Exception as e:
        logger.warning(f"[admin_logger] Failed to log event {event_type}: {e}")
        return False


def log_admin_event_sync(
    event_type: str,
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Write an admin event to Firestore, blocking until it lands.

    For genuinely synchronous callers only. Never call this from a request
    handler or an async generator: the Firestore client is blocking, so it
    stalls the event loop, and with the audit trail actually working that stall
    became visible as the SSE response never closing after its last event.
    """
    return _write_admin_event(event_type, user_id, metadata)


def _log_admin_event_in_background(event_type: str, user_id: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> None:
    """Hand the write to a worker thread so the caller returns immediately."""
    thread = threading.Thread(
        target=_write_admin_event,
        args=(event_type, user_id, metadata),
        daemon=True,
        name="admin-logger",
    )
    thread.start()


async def log_admin_event(
    event_type: str,
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Write an admin event to Firestore without blocking the event loop."""
    _log_admin_event_in_background(event_type, user_id, metadata)
    return True


def log_token_usage(
    event_type: str,
    user_id: str,
    token_usage: dict,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """Log a token usage event. Merges token fields into metadata.

    Called after every model call, including from inside the SSE async
    generator, so it dispatches to a background thread. Audit records are not
    worth holding a qualification request open for.
    """
    metadata = {
        "input_tokens": token_usage.get("input_tokens", 0),
        "output_tokens": token_usage.get("output_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
    }
    # Include cache tokens if present (Claude)
    if "cache_read_input_tokens" in token_usage:
        metadata["cache_read_input_tokens"] = token_usage["cache_read_input_tokens"]
    if "cache_creation_input_tokens" in token_usage:
        metadata["cache_creation_input_tokens"] = token_usage["cache_creation_input_tokens"]
    if extra:
        metadata.update(extra)
    _log_admin_event_in_background(event_type, user_id, metadata)
    return True
