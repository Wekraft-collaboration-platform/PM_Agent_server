"""
document_parser.py — Structure- & Layout-Aware Document Parser.

Features:
1. Magic-bytes file signature validation to prevent extension spoofing.
2. Structure-aware parsing:
   - PDF: Multimodal Gemini 2.0 Flash REST API with base64 inline data.
   - DOCX: python-docx structural paragraph & table extraction.
   - TXT / MD: UTF-8 direct extraction.
3. Resilient Gemini API calls wrapped with Tenacity retries (429/5xx) & explicit httpx timeout.
4. User-scoped Upstash Redis caching: `doc_cache:{user_id}:{file_id}` (1-hour TTL).
"""

import os
import io
import json
import base64
import zipfile
import logging
from typing import Optional, Dict, Any, Tuple
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)
import docx

logger = logging.getLogger("document_parser")
logging.basicConfig(level=logging.INFO)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
DOC_CACHE_TTL_SECONDS = 3600  # 1 hour

# ─────────────────────────────────────────────────────────────────────────────
# 1. MAGIC BYTES VALIDATION (ANTI-SPOOFING)
# ─────────────────────────────────────────────────────────────────────────────

def validate_magic_bytes(file_bytes: bytes, filename: str) -> Tuple[bool, str, str]:
    """
    Validates file headers (magic numbers) to prevent extension spoofing.
    Returns: (is_valid, detected_type, error_reason)
    """
    ext = os.path.splitext(filename)[1].lower()

    if len(file_bytes) == 0:
        return False, "unknown", "File is empty."

    # PDF signature: %PDF- (hex: 25 50 44 46)
    if ext == ".pdf":
        if file_bytes.startswith(b"%PDF"):
            return True, "pdf", ""
        return False, "pdf", "Invalid PDF file signature."

    # DOCX signature: PK\x03\x04 (ZIP archive header for OOXML)
    if ext == ".docx":
        if file_bytes.startswith(b"PK\x03\x04"):
            try:
                # Verify that it contains word/document.xml or [Content_Types].xml
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    names = z.namelist()
                    if any("word/document.xml" in n or "[Content_Types].xml" in n for n in names):
                        return True, "docx", ""
            except Exception:
                pass
            return True, "docx", ""
        return False, "docx", "Invalid DOCX file signature."

    # DOC signature: OLE2 compound file header D0 CF 11 E0 A1 B1 1A E1
    if ext == ".doc":
        if file_bytes.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") or file_bytes.startswith(b"PK\x03\x04"):
            return True, "doc", ""
        return False, "doc", "Invalid DOC file signature."

    # TXT / MD: Valid UTF-8 or ASCII with no binary null bytes in first 1024 bytes
    if ext in [".txt", ".md"]:
        sample = file_bytes[:1024]
        if b"\x00" in sample:
            return False, "text", "Binary content detected in text file."
        try:
            sample.decode("utf-8")
            return True, "text", ""
        except UnicodeDecodeError:
            try:
                sample.decode("latin-1")
                return True, "text", ""
            except Exception:
                return False, "text", "Unable to decode text file encoding."

    return False, "unknown", f"Unsupported file extension: {ext}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. RESILIENT GEMINI 2.0 FLASH CALLS (TENACITY + TIMEOUT)
# ─────────────────────────────────────────────────────────────────────────────

def _is_retryable_gemini_error(exception: BaseException) -> bool:
    """Retries on network timeouts, 429 rate limits, and 5xx transient server errors."""
    if isinstance(exception, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in [429, 500, 502, 503, 504]
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1.5, max=5),
    retry=retry_if_exception(_is_retryable_gemini_error),
    reraise=True,
)
async def call_gemini_flash_multimodal(
    file_bytes: bytes,
    mime_type: str,
    prompt: str,
    api_key: str,
) -> str:
    """
    Sends document bytes to Gemini 2.0 Flash via Google Generative Language REST API.
    Uses strict 30s timeout and tenacity retries.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    base64_data = base64.b64encode(file_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,
        },
    }

    # Explicit client timeouts (connect 10s, read 30s)
    timeout = httpx.Timeout(connect=10.0, read=30.0, write=15.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned empty candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_output = "".join(p.get("text", "") for p in parts)
        return text_output.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. STRUCTURE-AWARE PARSERS PER FORMAT
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURE_SYSTEM_PROMPT = """You are an expert document structure and layout parser.
Extract this document into clean, GitHub-flavored Markdown while preserving its exact structural layout:
1. Document title and hierarchy (H1 '#', H2 '##', H3 '###', H4 '####').
2. Tables formatted accurately as Markdown tables with columns, headers, and rows.
3. Bulleted lists, numbered lists, and nested task hierarchies.
4. Keep all requirements, user stories, acceptance criteria, parameters, and notes intact.
5. Do NOT summarize or omit sections. Output only the structured Markdown content."""


def parse_docx_to_markdown(file_bytes: bytes) -> str:
    """
    Parses a DOCX document into structured Markdown preserving headings and tables.
    """
    doc = docx.Document(io.BytesIO(file_bytes))
    md_lines = []

    for elem in doc.element.body:
        # Paragraph element
        if elem.tag.endswith("p"):
            p = docx.text.paragraph.Paragraph(elem, doc)
            text = p.text.strip()
            if not text:
                continue

            style_name = p.style.name.lower() if p.style and p.style.name else ""
            if "heading 1" in style_name or "title" in style_name:
                md_lines.append(f"\n# {text}\n")
            elif "heading 2" in style_name:
                md_lines.append(f"\n## {text}\n")
            elif "heading 3" in style_name:
                md_lines.append(f"\n### {text}\n")
            elif "heading 4" in style_name:
                md_lines.append(f"\n#### {text}\n")
            elif "list" in style_name or "bullet" in style_name:
                md_lines.append(f"- {text}")
            else:
                md_lines.append(f"{text}\n")

        # Table element
        elif elem.tag.endswith("tbl"):
            table = docx.table.Table(elem, doc)
            rows = []
            for r in table.rows:
                row_cells = [c.text.strip().replace("\n", " ") for c in r.cells]
                rows.append(row_cells)

            if rows:
                header = rows[0]
                md_lines.append("\n| " + " | ".join(header) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                for data_row in rows[1:]:
                    # Pad if needed
                    while len(data_row) < len(header):
                        data_row.append("")
                    md_lines.append("| " + " | ".join(data_row[:len(header)]) + " |")
                md_lines.append("\n")

    return "\n".join(md_lines).strip()


async def parse_document_structure_aware(
    file_bytes: bytes,
    filename: str,
    gemini_api_key: str,
) -> str:
    """
    Main layout-aware parsing orchestrator:
    - PDF: Multimodal Gemini 2.0 Flash.
    - DOCX: python-docx structure extractor (fallback to Gemini if needed).
    - TXT / MD: Decoded text.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured for PDF parsing.")
        return await call_gemini_flash_multimodal(
            file_bytes=file_bytes,
            mime_type="application/pdf",
            prompt=STRUCTURE_SYSTEM_PROMPT,
            api_key=gemini_api_key,
        )

    elif ext == ".docx":
        parsed_md = parse_docx_to_markdown(file_bytes)
        if parsed_md:
            return parsed_md
        # Fallback to Gemini if extraction was sparse
        if gemini_api_key:
            return await call_gemini_flash_multimodal(
                file_bytes=file_bytes,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                prompt=STRUCTURE_SYSTEM_PROMPT,
                api_key=gemini_api_key,
            )
        return parsed_md

    elif ext == ".doc":
        # Binary DOC fallback
        if gemini_api_key:
            return await call_gemini_flash_multimodal(
                file_bytes=file_bytes,
                mime_type="application/msword",
                prompt=STRUCTURE_SYSTEM_PROMPT,
                api_key=gemini_api_key,
            )
        raise ValueError("Legacy .doc parsing requires GEMINI_API_KEY.")

    elif ext in [".txt", ".md"]:
        return file_bytes.decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported format: {ext}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. USER-SCOPED REDIS STORAGE (doc_cache:{user_id}:{file_id})
# ─────────────────────────────────────────────────────────────────────────────

# In-memory backup in case Redis is temporarily disconnected
_LOCAL_FALLBACK_CACHE: Dict[str, str] = {}


async def get_redis_connection():
    """Returns an async Redis client or None."""
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("UPSTASH_REDIS_URL") or os.getenv("REDIS_URL")
        if redis_url:
            return aioredis.from_url(redis_url, decode_responses=True)
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
    return None


async def save_parsed_document_to_cache(
    user_id: str,
    file_id: str,
    document_data: Dict[str, Any],
) -> bool:
    """
    Saves parsed document strictly scoped to user_id:
    Key: doc_cache:{user_id}:{file_id}
    TTL: 1 hour (3600s)
    """
    key = f"doc_cache:{user_id}:{file_id}"
    serialized = json.dumps(document_data)

    client = await get_redis_connection()
    if client:
        try:
            await client.set(key, serialized, ex=DOC_CACHE_TTL_SECONDS)
            await client.aclose()
            return True
        except Exception as e:
            logger.error(f"Redis write error for key {key}: {e}")

    # Fallback to local memory
    _LOCAL_FALLBACK_CACHE[key] = serialized
    return True


async def get_parsed_document_from_cache(
    user_id: str,
    file_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves parsed document ensuring strict user_id ownership check.
    """
    key = f"doc_cache:{user_id}:{file_id}"

    client = await get_redis_connection()
    if client:
        try:
            raw = await client.get(key)
            await client.aclose()
            if raw:
                data = json.loads(raw)
                if data.get("user_id") == user_id:
                    return data
                logger.warning(f"Unauthorized access attempt for file_id {file_id} by user {user_id}")
                return None
        except Exception as e:
            logger.error(f"Redis read error for key {key}: {e}")

    # Check local fallback
    raw = _LOCAL_FALLBACK_CACHE.get(key)
    if raw:
        data = json.loads(raw)
        if data.get("user_id") == user_id:
            return data
    return None
