import os
import asyncio
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticCacheService:
    """
    Semantic Cache Service using Cohere API embeddings (embed-english-v3.0).
    Stores query embeddings and responses for fast similarity lookup (>= 0.92 threshold).
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.api_key = os.getenv("COHERE_API_KEY", "")
        self.similarity_threshold = similarity_threshold
        self.cache: List[Dict[str, Any]] = []
        self._cohere_client = None
        self._init_cohere()

    def _init_cohere(self):
        try:
            import cohere
            key = os.getenv("COHERE_API_KEY", self.api_key)
            if key:
                self._cohere_client = cohere.ClientV2(api_key=key)
                print("[SemanticCache] Cohere ClientV2 initialized successfully")
        except Exception as e:
            print(f"[SemanticCache] Failed to initialize Cohere client: {e}")

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Fetch query embedding vector from Cohere API asynchronously."""
        if not self._cohere_client:
            return None
        try:
            # Run blocking Cohere call in thread pool to prevent blocking asyncio loop
            response = await asyncio.to_thread(
                self._cohere_client.embed,
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_query",
                embedding_types=["float"],
            )
            return response.embeddings.float_[0]
        except Exception as e:
            print(f"[SemanticCache] Error generating Cohere embedding: {e}")
            return None

    async def lookup(self, query: str) -> Tuple[bool, Optional[str], float]:
        """
        Lookup similar query in semantic cache.
        Returns (is_hit, cached_response, similarity_score).
        """
        if not self.cache:
            return False, None, 0.0

        query_vector = await self.get_embedding(query)
        if query_vector is None:
            return False, None, 0.0

        best_score = 0.0
        best_response = None

        for entry in self.cache:
            score = cosine_similarity(query_vector, entry["embedding"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]

        if best_score >= self.similarity_threshold:
            print(f"[SemanticCache] HIT! Score: {best_score:.4f} >= {self.similarity_threshold}")
            return True, best_response, best_score

        return False, None, best_score

    async def store(self, query: str, response: str):
        """Store query vector and response in semantic cache."""
        query_vector = await self.get_embedding(query)
        if query_vector is not None:
            self.cache.append({
                "query": query,
                "embedding": query_vector,
                "response": response
            })
            print(f"[SemanticCache] Stored query response in cache. Cache size: {len(self.cache)}")


# Singleton instance
semantic_cache = SemanticCacheService()


async def run_guardrails_and_cache_parallel(user_text: str, is_doc_upload: bool = False) -> Dict[str, Any]:
    """
    Fires Input Guardrails and Semantic Cache lookup concurrently in PARALLEL via asyncio.gather.
    Bypasses semantic cache if query is a greeting or document upload.
    """
    from app.core.guardrails.input_guardrails import input_guardrails
    from app.core.utils import is_greeting_query

    # Define tasks for parallel execution
    guardrail_task = asyncio.create_task(input_guardrails.run_input_guardrails(user_text))

    # Central greeting check & short query check (< 4 words)
    is_greeting_prelim = is_greeting_query(user_text) or len(user_text.strip().split()) < 4

    skip_cache = is_doc_upload or is_greeting_prelim


    if skip_cache:
        print(f"[ParallelPipeline] Bypassing semantic cache (is_doc_upload={is_doc_upload}, is_greeting={is_greeting_prelim})")
        guardrail_res = await guardrail_task
        return {
            "guardrail": guardrail_res,
            "cache_hit": False,
            "cached_response": None,
            "cache_score": 0.0,
            "bypassed_cache": True,
        }

    # Run Guardrails + Cache concurrently using asyncio.gather
    cache_task = asyncio.create_task(semantic_cache.lookup(user_text))
    guardrail_res, (cache_hit, cached_response, cache_score) = await asyncio.gather(guardrail_task, cache_task)

    print(f"[ParallelPipeline] Guardrail safe: {guardrail_res.get('is_safe')}, Cache hit: {cache_hit} (score: {cache_score:.4f})")

    return {
        "guardrail": guardrail_res,
        "cache_hit": cache_hit,
        "cached_response": cached_response,
        "cache_score": cache_score,
        "bypassed_cache": False,
    }
