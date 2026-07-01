from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_router.server import MOCK_UNCERTAIN_MARKER, ServerConfig, create_app


DEFAULT_CASES = Path(__file__).with_name("router_cases.jsonl")


@dataclass
class EvalResult:
    case_id: str
    expected_route: str
    actual_route: str
    route_source: str
    passed: bool
    confidence: float
    latency_ms: float
    cache_hit: bool
    reason: str


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
        for field in ("id", "messages", "expected_route"):
            if field not in case:
                raise ValueError(f"{path}:{line_no}: missing required field {field!r}")
        if case["expected_route"] not in {"local", "remote"}:
            raise ValueError(f"{path}:{line_no}: expected_route must be local or remote")
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no eval cases found")
    return cases


class RouterClient:
    def decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class LiveRouterClient(RouterClient):
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")

    def decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.server_url}/router/decision",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"router returned {exc.code}: {body}") from exc


class InProcessMockRouterClient(RouterClient):
    def __init__(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Using `httpx` with `starlette\.testclient` is deprecated.*",
                category=Warning,
            )
            from fastapi.testclient import TestClient

        self.client = TestClient(create_app(ServerConfig(mock=True)))

    def decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = json.loads(json.dumps(payload))
        mock_profile = payload.pop("mock_profile", None)
        if mock_profile == "uncertain":
            payload["messages"] = [
                {"role": "system", "content": MOCK_UNCERTAIN_MARKER},
                *payload["messages"],
            ]
        response = self.client.post("/v1/router/decision", json=payload)
        response.raise_for_status()
        return response.json()


def run_case(client: RouterClient, case: dict[str, Any], max_tokens: int, temperature: float) -> EvalResult:
    started = time.perf_counter()
    payload = {
        "messages": case["messages"],
        "max_tokens": case.get("max_tokens", max_tokens),
        "temperature": case.get("temperature", temperature),
    }
    for field in ("entropy_threshold", "top1_threshold", "confidence_threshold"):
        if field in case:
            payload[field] = case[field]
    if "mock_profile" in case:
        payload["mock_profile"] = case["mock_profile"]
    decision = client.decision(payload)
    latency_ms = float(decision.get("latency_ms") or ((time.perf_counter() - started) * 1000))
    actual_route = str(decision.get("route", ""))
    expected_route = str(case["expected_route"])
    return EvalResult(
        case_id=str(case["id"]),
        expected_route=expected_route,
        actual_route=actual_route,
        route_source=str(decision.get("route_source", "")),
        passed=actual_route == expected_route,
        confidence=float(decision.get("confidence", 0.0)),
        latency_ms=latency_ms,
        cache_hit=bool(decision.get("cache_hit", False)),
        reason=str(decision.get("reason", "")),
    )


def print_results(results: list[EvalResult]) -> None:
    print("case_id              expected  actual  source               pass  conf   ms      cache  reason")
    print("-" * 116)
    for result in results:
        print(
            f"{result.case_id[:20]:20} "
            f"{result.expected_route:8} "
            f"{result.actual_route:6} "
            f"{result.route_source[:20]:20} "
            f"{'yes' if result.passed else 'no ':4} "
            f"{result.confidence:0.3f} "
            f"{result.latency_ms:7.2f} "
            f"{'yes' if result.cache_hit else 'no ':5} "
            f"{result.reason}"
        )


def write_jsonl(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for result in results:
            handle.write(json.dumps(result.__dict__, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local-vs-remote router decisions.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--mock", action="store_true", help="Run against an in-process mock router server.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    client: RouterClient = InProcessMockRouterClient() if args.mock else LiveRouterClient(args.server_url)
    results: list[EvalResult] = []
    for case in cases:
        result = run_case(client, case, max_tokens=args.max_tokens, temperature=args.temperature)
        results.append(result)
        if args.fail_fast and not result.passed:
            break
    print_results(results)
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, results)
    passed = sum(result.passed for result in results)
    print(f"\npassed {passed}/{len(results)} router evals")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
