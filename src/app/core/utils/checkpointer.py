"""
checkpointer.py — Centralized LangGraph state checkpointer initialization.

Supports:
- Redis / Upstash Redis persistence via RedisSaver / AsyncRedisSaver using UPSTASH_REDIS_URL or REDIS_URL.
- In-memory persistence via MemorySaver as fallback for local dev / testing.
"""

import os
import logging
from typing import Optional, Any
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global cached checkpointer instance
_checkpointer_instance: Optional[Any] = None


def get_memory_checkpointer() -> MemorySaver:
    """Returns an in-memory checkpointer instance for dev/testing."""
    return MemorySaver()


def get_checkpointer() -> Any:
    """
    Returns the global configured checkpointer instance.
    Primary: Upstash Redis (UPSTASH_REDIS_URL / REDIS_URL) via langgraph.checkpoint.redis.aio.AsyncRedisSaver
    Fallback: MemorySaver
    """
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            _checkpointer_instance = AsyncRedisSaver(redis_url=redis_url)
            print("[CHECKPOINTER] AsyncRedisSaver connected to Redis/Upstash successfully.")
            return _checkpointer_instance
        except Exception as e:
            print(f"[CHECKPOINTER WARNING] Failed to initialize AsyncRedisSaver ({e}). Falling back to MemorySaver.")

    print("[CHECKPOINTER] Using MemorySaver checkpointer.")
    _checkpointer_instance = MemorySaver()
    return _checkpointer_instance


async def get_async_checkpointer() -> Any:
    """
    Returns an async Redis checkpointer (AsyncRedisSaver) using Upstash Redis if available,
    otherwise returns MemorySaver.
    """
    redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            checkpointer = AsyncRedisSaver(redis_url=redis_url)
            return checkpointer
        except Exception as e:
            logger.warning(
                f"Failed to initialize AsyncRedisSaver ({e}). Falling back to MemorySaver."
            )

    return MemorySaver()

