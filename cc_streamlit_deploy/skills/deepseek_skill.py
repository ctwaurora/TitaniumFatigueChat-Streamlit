"""DeepSeek provider adapter used by the existing scientific skills."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.api_keys import DeepSeekSettings, get_deepseek_settings
from src.deepseek_client import DeepSeekClient


DEFAULT_MODEL = "deepseek-chat"


def call_deepseek(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    stage: str = "deepseek_api",
) -> Dict[str, Any]:
    """Call DeepSeek and return an OpenAI-compatible response mapping."""
    settings = get_deepseek_settings()
    if model:
        settings = DeepSeekSettings(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=model,
        )

    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 4000
    max_tokens = max(1, min(max_tokens, 8000))

    success = False
    purpose = "deepseek_api_call"
    try:
        content = DeepSeekClient(settings).chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,
        )
        success = True
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "model": settings.model,
        }
    except Exception as exc:
        purpose = type(exc).__name__
        raise
    finally:
        try:
            from src.deepseek_usage import log_call

            log_call(
                stage=stage,
                model_name=settings.model,
                purpose=purpose,
                success=success,
            )
        except Exception:
            pass


def extract_deepseek_text(result: Dict[str, Any]) -> str:
    """Extract text from an OpenAI-compatible DeepSeek response."""
    try:
        return str(result["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        return json.dumps(result, ensure_ascii=False, indent=2)


def call_deepseek_text(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    stage: str = "deepseek_api",
) -> str:
    """Call DeepSeek with a single user prompt and return text."""
    result = call_deepseek(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stage=stage,
    )
    return extract_deepseek_text(result)


def test_api_connection() -> bool:
    """Return whether a minimal DeepSeek request succeeds."""
    try:
        result = call_deepseek(
            messages=[{"role": "user", "content": "Hello. Reply OK only."}],
            max_tokens=20,
            temperature=0.1,
            stage="connection_test",
        )
        return bool(extract_deepseek_text(result).strip())
    except Exception:
        return False
