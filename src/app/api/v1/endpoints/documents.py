"""
documents.py — FastAPI endpoint for background document upload & layout-aware parsing.
"""

import os
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status

from app.core.utils.document_parser import (
    MAX_FILE_SIZE_BYTES,
    validate_magic_bytes,
    parse_document_structure_aware,
    save_parsed_document_to_cache,
    get_parsed_document_from_cache,
)

router = APIRouter(prefix="/documents", tags=["Document Parsing"])
logger = logging.getLogger("documents_endpoint")


@router.post("/parse")
async def parse_document_endpoint(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    project_id: Optional[str] = Form(None),
):
    """
    Asynchronously parses an uploaded document (PDF, DOCX, DOC, TXT, MD) with layout awareness.
    Enforces 5MB size limit, magic-byte anti-spoofing validation, and user-scoped caching.
    """
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required for user-scoped document caching.",
        )

    filename = file.filename or "unknown_file"

    # 1. Read file bytes & enforce max file size limit (5MB)
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds 10MB size limit ({round(len(file_bytes)/(1024*1024), 2)}MB).",
        )

    # 2. Magic-bytes anti-spoofing validation
    is_valid, detected_type, error_reason = validate_magic_bytes(file_bytes, filename)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File validation failed: {error_reason}",
        )

    # 3. Retrieve Gemini API key from environment
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    # 4. Parse document with layout & structure preservation
    try:
        parsed_markdown = await parse_document_structure_aware(
            file_bytes=file_bytes,
            filename=filename,
            gemini_api_key=gemini_key,
        )
    except Exception as e:
        logger.error(f"Error parsing document '{filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse document: {str(e)}",
        )

    # 5. Generate secure unique file ID and save to user-scoped Redis cache
    file_id = f"doc_{uuid.uuid4().hex[:12]}"
    doc_payload = {
        "file_id": file_id,
        "user_id": user_id,
        "project_id": project_id,
        "file_name": filename,
        "file_size": len(file_bytes),
        "mime_type": file.content_type or detected_type,
        "parsed_markdown": parsed_markdown,
    }

    await save_parsed_document_to_cache(user_id=user_id, file_id=file_id, document_data=doc_payload)

    logger.info(f"Successfully parsed & cached {filename} as {file_id} for user {user_id}")

    return {
        "success": True,
        "file_id": file_id,
        "file_name": filename,
        "file_size": len(file_bytes),
        "status": "ready",
    }


@router.get("/{file_id}")
async def get_document_endpoint(
    file_id: str,
    user_id: str = Query(..., description="User ID for ownership verification"),
):
    """
    Retrieves cached parsed document with strict user ownership validation.
    """
    doc = await get_parsed_document_from_cache(user_id=user_id, file_id=file_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied.",
        )
    return doc
