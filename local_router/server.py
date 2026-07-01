from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .confidence import ConfidenceMetrics, choose_route, summarize_confidence


DEFAULT_MODEL_ID = "local-model"
DEFAULT_BACKEND_BASE_URL = "http://127.0.0.1:8081/v1"
MOCK_UNCERTAIN_MARKER = "[mock-router:uncertain]"
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)


@dataclass
class ServerConfig:
    model_id: str = DEFAULT_MODEL_ID
    backend_base_url: str = DEFAULT_BACKEND_BASE_URL
    backend_api_key: str = "no-key"
    host: str = "127.0.0.1"
    port: int = 8080
    max_local_tokens: int = 512
    entropy_threshold: float = 0.12
    top1_threshold: float = 0.95
    confidence_threshold: float = 0.97
    top_logprobs: int = 20
    request_timeout_s: float = 120.0
    max_concurrency: int = 1
    decision_cache_size: int = 128
    mock: bool = False


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    metrics: ConfidenceMetrics


class DecisionCache:
    def __init__(self, max_size: int):
        self.max_size = max(0, max_size)
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        if self.max_size == 0:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        self._items.move_to_end(key)
        return json.loads(json.dumps(item))

    def set(self, key: str, item: dict[str, Any]) -> None:
        if self.max_size == 0:
            return
        self._items[key] = json.loads(json.dumps(item))
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"size": len(self._items), "max_size": self.max_size}


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and item.get("type") == "image_url":
                parts.append("[image omitted]")
        return "\n".join(part for part in parts if part)
    return str(content)


def messages_to_plain_prompt(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = _message_content_to_text(message.get("content", ""))
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def strip_reasoning_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    close_matches = list(THINK_CLOSE_RE.finditer(stripped))
    if close_matches:
        after_last_think = stripped[close_matches[-1].end() :].strip()
        if after_last_think:
            return after_last_think
    if THINK_OPEN_RE.search(stripped) and not close_matches:
        return ""
    without_blocks = THINK_BLOCK_RE.sub("", stripped)
    without_tags = THINK_OPEN_RE.sub("", THINK_CLOSE_RE.sub("", without_blocks))
    return without_tags.strip()


def _logprob_entropy(logprobs: list[float]) -> tuple[float, float] | None:
    probs = [math.exp(value) for value in logprobs if math.isfinite(value)]
    probs = [min(max(value, 0.0), 1.0) for value in probs if value > 0.0]
    if not probs:
        return None
    mass = sum(probs)
    if mass > 1.0:
        probs = [value / mass for value in probs]
        residual = 0.0
    else:
        residual = 1.0 - mass
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probs)
    if residual > 1e-12:
        entropy -= residual * math.log(residual)
    return entropy, max(probs)


def _chat_logprob_values(choice: dict[str, Any]) -> tuple[list[float], list[float]]:
    entropies: list[float] = []
    top1_probs: list[float] = []
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return entropies, top1_probs

    content = logprobs.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            top_values: list[float] = []
            top_logprobs = item.get("top_logprobs")
            if isinstance(top_logprobs, list):
                for top_item in top_logprobs:
                    if isinstance(top_item, dict):
                        try:
                            top_values.append(float(top_item.get("logprob")))
                        except (TypeError, ValueError):
                            pass
            try:
                selected_logprob = float(item.get("logprob"))
                top_values.append(selected_logprob)
            except (TypeError, ValueError):
                pass
            metrics = _logprob_entropy(top_values)
            if metrics is not None:
                entropy, top1 = metrics
                entropies.append(entropy)
                top1_probs.append(top1)
        return entropies, top1_probs

    legacy_top_logprobs = logprobs.get("top_logprobs")
    legacy_token_logprobs = logprobs.get("token_logprobs")
    if isinstance(legacy_top_logprobs, list):
        for index, item in enumerate(legacy_top_logprobs):
            top_values = []
            if isinstance(item, dict):
                for value in item.values():
                    try:
                        top_values.append(float(value))
                    except (TypeError, ValueError):
                        pass
            if isinstance(legacy_token_logprobs, list) and index < len(legacy_token_logprobs):
                try:
                    top_values.append(float(legacy_token_logprobs[index]))
                except (TypeError, ValueError):
                    pass
            metrics = _logprob_entropy(top_values)
            if metrics is not None:
                entropy, top1 = metrics
                entropies.append(entropy)
                top1_probs.append(top1)
    return entropies, top1_probs


class MockRouterModel:
    def __init__(self, model_id: str):
        self.model_id = model_id

    @property
    def backend_label(self) -> str:
        return "mock"

    async def generate(self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> GenerationResult:
        prompt = messages_to_plain_prompt(messages)
        uncertain = MOCK_UNCERTAIN_MARKER in prompt
        text = (
            "This synthetic mock completion has low confidence and should delegate to the remote model."
            if uncertain
            else "This is a local answer from the small model path."
        )
        entropies = [7.0, 6.5, 6.8] if uncertain else [0.07, 0.10, 0.09]
        top1 = [0.03, 0.04, 0.03] if uncertain else [0.97, 0.96, 0.98]
        metrics = summarize_confidence(entropies, top1, vocab_size=128_000)
        return GenerationResult(
            text=text,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=max(1, len(text.split())),
            metrics=metrics,
        )


class OpenAICompatibleRouterModel:
    def __init__(self, config: ServerConfig):
        self.model_id = config.model_id
        self.base_url = config.backend_base_url.rstrip("/")
        self.api_key = config.backend_api_key
        self.top_logprobs = max(1, config.top_logprobs)
        self.timeout = config.request_timeout_s

    @property
    def backend_label(self) -> str:
        return self.base_url

    async def generate(self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, messages, max_new_tokens, temperature)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"local backend returned {exc.code}: {body}") from exc

    def _generate_sync(self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> GenerationResult:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        response = self._post_json("/chat/completions", payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("local backend returned no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("local backend returned an invalid choice")

        message = choice.get("message")
        raw_text = ""
        if isinstance(message, dict):
            raw_text = str(message.get("content") or "")
        elif "text" in choice:
            raw_text = str(choice.get("text") or "")
        text = strip_reasoning_text(raw_text)
        entropies, top1_probs = _chat_logprob_values(choice)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or max(1, len(messages_to_plain_prompt(messages).split())))
        completion_tokens = int(usage.get("completion_tokens") or max(len(entropies), len(text.split()), 1))
        metrics = summarize_confidence(entropies, top1_probs, vocab_size=128_000)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metrics=metrics,
        )


def create_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="Pi Local Router", version="0.1.0")
    model = MockRouterModel(config.model_id) if config.mock else OpenAICompatibleRouterModel(config)
    generation_semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
    decision_cache = DecisionCache(config.decision_cache_size)

    async def generate(messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> GenerationResult:
        async with generation_semaphore:
            return await model.generate(messages, max_tokens, temperature)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model": config.model_id,
            "backend": model.backend_label,
            "mock": config.mock,
            "thresholds": {
                "entropy": config.entropy_threshold,
                "top1": config.top1_threshold,
                "confidence": config.confidence_threshold,
            },
            "generation": {
                "max_concurrency": config.max_concurrency,
                "max_local_tokens": config.max_local_tokens,
                "top_logprobs": config.top_logprobs,
            },
            "decision_cache": decision_cache.stats(),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "pi-local-router",
                }
            ],
        }

    @app.post("/v1/router/decision")
    async def decision(request: Request) -> dict[str, Any]:
        started = time.perf_counter()
        payload = await request.json()
        messages = _messages(payload)
        max_tokens = _max_tokens(payload, config.max_local_tokens)
        temperature = float(payload.get("temperature", 0.2))
        entropy_threshold = float(payload.get("entropy_threshold", config.entropy_threshold))
        top1_threshold = float(payload.get("top1_threshold", config.top1_threshold))
        confidence_threshold = float(payload.get("confidence_threshold", config.confidence_threshold))
        cache_key = _decision_cache_key(
            config.model_id,
            config.backend_base_url,
            messages,
            max_tokens,
            temperature,
            entropy_threshold,
            top1_threshold,
            confidence_threshold,
        )
        cached = decision_cache.get(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return cached

        result = await generate(messages, max_tokens, temperature)
        route, reason = choose_route(result.metrics, entropy_threshold, top1_threshold, confidence_threshold)
        if route == "local" and not result.text.strip():
            route = "remote"
            reason = "local final answer empty after stripping reasoning"
        response = {
            "route": route,
            "text": result.text if route == "local" else "",
            "model": config.model_id,
            "confidence": result.metrics.confidence,
            "reason": reason,
            "route_source": "entropy",
            "metrics": asdict(result.metrics),
            "usage": _usage(result),
            "cache_hit": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        decision_cache.set(cache_key, response)
        return response

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request):
        payload = await request.json()
        messages = _messages(payload)
        max_tokens = _max_tokens(payload, config.max_local_tokens)
        temperature = float(payload.get("temperature", 0.2))
        result = await generate(messages, max_tokens, temperature)
        if payload.get("stream"):
            return StreamingResponse(
                _completion_stream(result, config.model_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return JSONResponse(_completion_payload(result, config.model_id))

    return app


def _messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    return [message for message in messages if isinstance(message, dict)]


def _decision_cache_key(
    model_id: str,
    backend_base_url: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    entropy_threshold: float,
    top1_threshold: float,
    confidence_threshold: float,
) -> str:
    body = {
        "model": model_id,
        "backend_base_url": backend_base_url,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "entropy_threshold": entropy_threshold,
        "top1_threshold": top1_threshold,
        "confidence_threshold": confidence_threshold,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _max_tokens(payload: dict[str, Any], fallback: int) -> int:
    raw = payload.get("max_tokens", payload.get("max_completion_tokens", fallback))
    try:
        return max(1, min(int(raw), fallback))
    except (TypeError, ValueError):
        return fallback


def _usage(result: GenerationResult) -> dict[str, int]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _completion_payload(result: GenerationResult, model_id: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(result),
    }


async def _completion_stream(result: GenerationResult, model_id: str) -> AsyncIterator[str]:
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"
    chunks = result.text.split(" ")
    for index, chunk in enumerate(chunks):
        delta = chunk if index == len(chunks) - 1 else f"{chunk} "
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0)
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": _usage(result),
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Route Pi requests between llama.cpp and Hugging Face providers.")
    parser.add_argument("--model-id", default=os.environ.get("LOCAL_ROUTER_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--backend-base-url", default=os.environ.get("LOCAL_ROUTER_BACKEND_BASE_URL", DEFAULT_BACKEND_BASE_URL))
    parser.add_argument("--backend-api-key", default=os.environ.get("LOCAL_ROUTER_BACKEND_API_KEY", "no-key"))
    parser.add_argument("--host", default=os.environ.get("LOCAL_ROUTER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOCAL_ROUTER_PORT", "8080")))
    parser.add_argument("--max-local-tokens", type=int, default=int(os.environ.get("LOCAL_ROUTER_MAX_TOKENS", "512")))
    parser.add_argument("--entropy-threshold", type=float, default=float(os.environ.get("LOCAL_ROUTER_ENTROPY_THRESHOLD", "0.12")))
    parser.add_argument("--top1-threshold", type=float, default=float(os.environ.get("LOCAL_ROUTER_TOP1_THRESHOLD", "0.95")))
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.environ.get("LOCAL_ROUTER_CONFIDENCE_THRESHOLD", "0.97")),
    )
    parser.add_argument("--top-logprobs", type=int, default=int(os.environ.get("LOCAL_ROUTER_TOP_LOGPROBS", "20")))
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(os.environ.get("LOCAL_ROUTER_REQUEST_TIMEOUT_S", "120")),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("LOCAL_ROUTER_MAX_CONCURRENCY", "1")),
        help="Maximum concurrent local backend requests.",
    )
    parser.add_argument(
        "--decision-cache-size",
        type=int,
        default=int(os.environ.get("LOCAL_ROUTER_DECISION_CACHE_SIZE", "128")),
        help="Number of routing decision responses to cache. Set to 0 to disable.",
    )
    parser.add_argument("--mock", action="store_true", default=os.environ.get("LOCAL_ROUTER_MOCK") == "1")
    args = parser.parse_args()
    return ServerConfig(**vars(args))


def main() -> None:
    import uvicorn

    config = parse_args()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
