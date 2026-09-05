import re
import os
import asyncio
import httpx
from typing import Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from app.core.utils import is_greeting_query

load_dotenv()

# PII Regex Patterns (EXCLUDING email address)
PII_PATTERNS = {
    "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "api_key": r"\b(?:sk-[a-zA-Z0-9]{20,}|pcsk_[a-zA-Z0-9]{20,}|llx-[a-zA-Z0-9]{20,})\b"
}

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"system override",
    r"bypass safety",
    r"you are now DAN",
    r"drop database",
]


class InputGuardrails:
    """
    3-Layer Parallel/Sequential Input Guardrails Service:
    Layer 1: PII Redaction (Phone, SSN, Credit Card, API Keys - Email Preserved)
    Layer 2: Regex & Greeting/Short Query Check (< 4 words or greeting matched)
    Layer 3: Groq LLM Guardrail (openai/gpt-oss-safeguard-20 HTTP evaluation)
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_SAFEGUARD_MODEL", "openai/gpt-oss-safeguard-20")

    def layer1_pii_redaction(self, text: str) -> str:
        """Layer 1: Redact PII patterns while leaving email addresses untouched."""
        redacted = text
        for pii_type, pattern in PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted)
        return redacted

    def layer2_regex_and_greetings(self, text: str) -> Dict[str, Any]:
        """
        Layer 2: Fast regex inspection.
        Detects injection attempts and flags greetings / short queries (< 4 words)
        for future fast-track routing.
        """
        lowered = text.lower().strip()
        words = lowered.split()

        # Check prompt injection
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                return {
                    "is_safe": False,
                    "reason": f"Prompt injection pattern detected: {pattern}",
                    "is_greeting": False,
                    "is_short_query": False,
                }

        # Check greetings using centralized is_greeting_query
        is_greeting = is_greeting_query(text)
        is_short_query = len(words) < 4

        return {
            "is_safe": True,
            "reason": None,
            "is_greeting": is_greeting,
            "is_short_query": is_short_query,
            "word_count": len(words),
        }

    async def validate_layer2_groq_llm(self, text: str) -> Tuple[bool, float, str]:
        """
        Layer 3 / Stage 2: Groq LLM Guardrail (~250ms).
        Sends HTTP request to openai/gpt-oss-safeguard-20 via Groq API.
        Returns (is_safe, risk_score, reasoning).
        If risk score > 0.8, returns False immediately.
        """
        groq_key = os.getenv("GROQ_API_KEY", self.groq_api_key)
        if not groq_key:
            return True, 0.0, "Groq API key not set; skipping LLM guardrail check."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a safety guardrail classifier. Analyze the user prompt and respond ONLY with JSON: "
                        '{"risk_score": float (0.0 to 1.0), "is_safe": boolean, "reason": string}'
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "max_tokens": 100,
        }

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    raw_content = result["choices"][0]["message"]["content"]
                    
                    import json
                    try:
                        parsed = json.loads(raw_content)
                        risk_score = float(parsed.get("risk_score", 0.0))
                        is_safe = parsed.get("is_safe", True) and (risk_score <= 0.8)
                        reason = parsed.get("reason", "Passed LLM safeguard evaluation.")
                        return is_safe, risk_score, reason
                    except Exception:
                        if "unsafe" in raw_content.lower() or "risk_score: 0.9" in raw_content or "risk_score: 1" in raw_content:
                            return False, 0.9, raw_content
                        return True, 0.0, "Passed safeguard evaluation."
                else:
                    return True, 0.0, f"Groq HTTP status {resp.status_code}, allowed."
        except Exception as e:
            return True, 0.0, f"Groq LLM check bypassed due to error: {e}"

    async def run_input_guardrails(self, text: str) -> Dict[str, Any]:
        """
        Runs Layer 1 & Layer 2 in parallel, followed by Layer 3 Groq LLM check.
        Logs: 'Passed user query from 3 steps.' on success.
        """
        redacted_text = self.layer1_pii_redaction(text)
        layer2_res = self.layer2_regex_and_greetings(redacted_text)

        if not layer2_res["is_safe"]:
            print(f"[GUARDRAIL] FAILED at Layer 2: {layer2_res['reason']}")
            return {
                "is_safe": False,
                "redacted_text": redacted_text,
                "reason": layer2_res["reason"],
                "layer2_info": layer2_res,
                "risk_score": 1.0,
            }

        is_safe_llm, risk_score, llm_reason = await self.validate_layer2_groq_llm(redacted_text)

        if not is_safe_llm or risk_score > 0.8:
            print(f"[GUARDRAIL] FAILED at Layer 3 (Groq LLM): risk_score={risk_score}, reason={llm_reason}")
            return {
                "is_safe": False,
                "redacted_text": redacted_text,
                "reason": f"Groq LLM risk score {risk_score} > 0.8 ({llm_reason})",
                "layer2_info": layer2_res,
                "risk_score": risk_score,
            }

        print("Passed user query from 3 steps.")

        return {
            "is_safe": True,
            "redacted_text": redacted_text,
            "reason": "Passed user query from 3 steps.",
            "layer2_info": layer2_res,
            "risk_score": risk_score,
            "redirect_to_router": layer2_res["is_greeting"] or layer2_res["is_short_query"],
        }


# Singleton Instance
input_guardrails = InputGuardrails()
