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
    Primary: Upstash Redis (UPSTASH_REDIS_URL / REDIS_URL) via langgraph.checkpoint.redis
    Fallback: MemorySaver
    """
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver
            from redis import Redis

            conn = Redis.from_url(redis_url)
            _checkpointer_instance = RedisSaver(conn)
            logger.info("Initialized RedisSaver checkpointer with Upstash Redis successfully.")
            return _checkpointer_instance
        except Exception as e:
            logger.warning(
                f"Failed to initialize RedisSaver ({e}). Falling back to MemorySaver."
            )

    logger.info("Using MemorySaver checkpointer.")
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
            from redis.asyncio import Redis as AsyncRedis

            conn = AsyncRedis.from_url(redis_url)
            checkpointer = AsyncRedisSaver(conn)
            logger.info("Initialized AsyncRedisSaver checkpointer with Upstash Redis successfully.")
            return checkpointer
        except Exception as e:
            logger.warning(
                f"Failed to initialize AsyncRedisSaver ({e}). Falling back to MemorySaver."
            )

    logger.info("Using MemorySaver checkpointer.")
    return MemorySaver()
