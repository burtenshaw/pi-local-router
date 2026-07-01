from fastapi.testclient import TestClient

from local_router.server import (
    MOCK_UNCERTAIN_MARKER,
    ServerConfig,
    create_app,
    parse_route_probe_label,
    prepare_probe_messages,
    strip_reasoning_text,
    strip_router_trace_text,
)


def test_mock_decision_cache_and_routes() -> None:
    client = TestClient(create_app(ServerConfig(mock=True, decision_cache_size=8)))

    easy = {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 32}
    first = client.post("/v1/router/decision", json=easy)
    second = client.post("/v1/router/decision", json=easy)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["route"] == "local"
    assert first.json()["route_source"] == "route_probe+entropy"
    assert first.json()["route_probe"]["label"] == "local"
    assert first.json()["cache_hit"] is False
    assert second.json()["route"] == "local"
    assert second.json()["cache_hit"] is True

    hard = client.post(
        "/v1/router/decision",
        json={
            "messages": [
                {"role": "system", "content": MOCK_UNCERTAIN_MARKER},
                {"role": "user", "content": "Route this from synthetic uncertainty."},
            ],
            "max_tokens": 32,
        },
    )

    assert hard.status_code == 200
    assert hard.json()["route"] == "remote"
    assert hard.json()["route_source"] == "route_probe"
    assert hard.json()["route_probe"]["label"] == "remote"


def test_health_exposes_dx_settings() -> None:
    client = TestClient(create_app(ServerConfig(mock=True, max_concurrency=2, decision_cache_size=4)))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["generation"]["max_concurrency"] == 2
    assert payload["generation"]["route_probe_enabled"] is True
    assert payload["decision_cache"]["max_size"] == 4


def test_strip_reasoning_text_removes_think_block() -> None:
    text = "<think>\nI should not show this.\n</think>\nHello!"

    assert strip_reasoning_text(text) == "Hello!"


def test_strip_reasoning_text_hides_unclosed_think_block() -> None:
    text = "<think>\nI am still reasoning and never produced a final answer."

    assert strip_reasoning_text(text) == ""


def test_strip_router_trace_text_removes_trace_line() -> None:
    text = "> router: route=remote | model=cloud | confidence=0.000\n\nFinal answer."

    assert strip_router_trace_text(text) == "Final answer."


def test_parse_route_probe_label_uses_last_label_and_marks_invalid() -> None:
    assert parse_route_probe_label("Maybe REMOTE, final answer: LOCAL") == "local"
    assert parse_route_probe_label("<think>still thinking</think>") == "invalid"


def test_prepare_probe_messages_cleans_remote_trace_and_bounds_context() -> None:
    messages = [
        {"role": "user", "content": "Give a detailed remote answer."},
        {
            "role": "assistant",
            "content": (
                "> router: route=remote | model=cloud | confidence=0.000\n\n"
                "<think>hidden reasoning</think>\n"
                + "A" * 300
                + "\nThe answer was Paris."
            ),
        },
        {"role": "user", "content": "What city was mentioned?"},
    ]

    prepared = prepare_probe_messages(messages, max_context_chars=260, max_message_chars=160)

    assert prepared[-1] == {"role": "user", "content": "What city was mentioned?"}
    assistant = next(message for message in prepared if message["role"] == "assistant")
    assert "router:" not in assistant["content"]
    assert "hidden reasoning" not in assistant["content"]
    assert len("\n".join(message["content"] for message in prepared)) <= 260


def test_natural_prompt_uses_route_probe_and_entropy_path() -> None:
    client = TestClient(create_app(ServerConfig(mock=True, decision_cache_size=0)))

    response = client.post(
        "/v1/router/decision",
        json={"messages": [{"role": "user", "content": "what is grpo in post training"}], "max_tokens": 32},
    )

    assert response.status_code == 200
    assert response.json()["route_source"] == "route_probe+entropy"
    assert response.json()["route_probe"]["enabled"] is True
    assert "task gate" not in response.json()["reason"]


def test_route_probe_can_be_disabled_for_entropy_only_mode() -> None:
    client = TestClient(create_app(ServerConfig(mock=True, decision_cache_size=0, route_probe_enabled=False)))

    response = client.post(
        "/v1/router/decision",
        json={
            "messages": [
                {"role": "system", "content": MOCK_UNCERTAIN_MARKER},
                {"role": "user", "content": "Route this from synthetic uncertainty."},
            ],
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    assert response.json()["route"] == "remote"
    assert response.json()["route_source"] == "entropy"
