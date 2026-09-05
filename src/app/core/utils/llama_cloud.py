import os
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


class LlamaCloudExtractor:
    """
    LlamaCloud Document Parser & Structured Task/Issue Extractor.
    Extracts actionable tasks & issues from uploaded PRDs, PDFs, Excel sheets, and images.
    """

    def __init__(self):
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY", "")

    async def parse_document_async(self, file_path_or_url: str) -> str:
        """
        Asynchronously parse document content into structured markdown using LlamaParse.
        Non-blocking execution for fast response times.
        """
        api_key = os.getenv("LLAMA_CLOUD_API_KEY", self.api_key)
        if not api_key:
            return f"Simulated parsed content from {file_path_or_url} (LLAMA_CLOUD_API_KEY not set)."

        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(api_key=api_key, result_type="markdown")
            extractions = await parser.aload_data(file_path_or_url)
            text_content = "\n\n".join([doc.text for doc in extractions])
            return text_content
        except Exception as e:
            print(f"[LlamaCloudExtractor] Error parsing file {file_path_or_url}: {e}")
            return f"Error parsing document: {e}"

    def extract_tasks_and_issues_from_markdown(self, markdown_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Structured extraction of tasks and issues from parsed document markdown.
        Returns dictionary with 4 minimal task fields and 4 minimal issue fields.
        """
        tasks = []
        issues = []

        lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]

        for i, line in enumerate(lines):
            lowered = line.lower()
            # Simple heuristic detection for demo/parser fallback
            if "issue" in lowered or "bug" in lowered or "error" in lowered:
                issues.append({
                    "title": line.lstrip("-*#1234567890. ").strip(),
                    "description": lines[i+1] if i + 1 < len(lines) else "Extracted from uploaded document",
                    "environment": "dev",
                    "severity": "critical" if "critical" in lowered or "urgent" in lowered else "medium",
                })
            elif "task" in lowered or "todo" in lowered or "feature" in lowered or "implement" in lowered:
                tasks.append({
                    "title": line.lstrip("-*#1234567890. ").strip(),
                    "description": lines[i+1] if i + 1 < len(lines) else "Extracted from uploaded PRD",
                    "priority": "high" if "high" in lowered or "urgent" in lowered else "medium",
                    "type": {"label": "PRD-Import", "color": "#3b82f6"},
                })

        return {"tasks": tasks, "issues": issues}


llama_cloud_extractor = LlamaCloudExtractor()
