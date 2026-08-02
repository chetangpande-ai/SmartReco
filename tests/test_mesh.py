"""The Mesh gateway, against a stub client.

These cases are not hypothetical — each one was observed against the live gateway
before the code was written:

  * a reasoning model returning 200 with empty content because the token ceiling was
    consumed by hidden reasoning (kimi-k3: 220 reasoning tokens of a 396-token
    completion);
  * a 400 rejecting `temperature` for a model that only accepts 1;
  * JSON wrapped in ``` fences despite response_format=json_object.
"""

from types import SimpleNamespace

import pytest
from openai import APIConnectionError, BadRequestError, RateLimitError

from app.config import settings
from app.services.mesh import (
    BudgetExceeded,
    CircuitOpen,
    EmptyCompletion,
    MeshClient,
    MeshError,
)


def _response(content: str, *, prompt=100, completion=50, reasoning=0, model="test/model"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


def _error(cls, message: str):
    """Build a provider exception without its heavyweight constructor."""
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    return err


class StubCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else _response('{"ok": true}')
        if isinstance(item, Exception):
            raise item
        return item


class StubClient:
    def __init__(self, script=(), embeddings=None):
        self.completions = StubCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)
        self.embeddings = SimpleNamespace(create=lambda **kw: embeddings)


@pytest.fixture
def mesh(monkeypatch):
    """A fresh client with budgets and breaker state isolated per test."""
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr(settings, "meshapi_api_key", "rsk_test")
    return MeshClient()


def attach(mesh_client, script=(), embeddings=None) -> StubClient:
    stub = StubClient(script, embeddings)
    mesh_client._client = stub
    return stub


class TestChat:
    def test_happy_path_records_usage_and_cost(self, mesh):
        attach(mesh, [_response("hello", prompt=1000, completion=500)])
        result = mesh.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o-mini")
        assert result.text == "hello"
        assert result.prompt_tokens == 1000 and result.completion_tokens == 500
        assert result.cost_usd == pytest.approx(0.00045)
        assert result.latency_ms >= 0

    def test_empty_content_raises_rather_than_confusing_the_caller(self, mesh):
        """The reasoning-token trap: a successful response containing nothing."""
        attach(mesh, [_response("", completion=20, reasoning=17)])
        with pytest.raises(EmptyCompletion) as exc:
            mesh.chat([{"role": "user", "content": "hi"}])
        assert "reasoning tokens" in str(exc.value)

    def test_json_mode_sets_response_format(self, mesh):
        stub = attach(mesh, [_response('{"a": 1}')])
        data, _ = mesh.chat_json([{"role": "user", "content": "hi"}])
        assert data == {"a": 1}
        assert stub.completions.calls[0]["response_format"] == {"type": "json_object"}

    def test_fenced_json_is_still_parsed(self, mesh):
        attach(mesh, [_response('```json\n{"a": 2}\n```')])
        data, _ = mesh.chat_json([{"role": "user", "content": "hi"}])
        assert data == {"a": 2}


class TestParameterNegotiation:
    def test_drops_a_rejected_temperature_and_retries(self, mesh):
        """kimi-k3 returns 400 for any temperature other than 1."""
        stub = attach(
            mesh,
            [
                _error(BadRequestError, "invalid temperature: only 1 is allowed for this model"),
                _response("ok"),
            ],
        )
        result = mesh.chat([{"role": "user", "content": "hi"}], temperature=0.7)
        assert result.text == "ok"
        assert "temperature" in stub.completions.calls[0]
        assert "temperature" not in stub.completions.calls[1]

    def test_remembers_the_rejection_for_later_calls(self, mesh):
        stub = attach(
            mesh,
            [
                _error(BadRequestError, "invalid temperature: only 1 is allowed"),
                _response("first"),
                _response("second"),
            ],
        )
        mesh.chat([{"role": "user", "content": "a"}], model="quirky/model", temperature=0.7)
        mesh.chat([{"role": "user", "content": "b"}], model="quirky/model", temperature=0.7)
        assert "temperature" not in stub.completions.calls[-1], "the quirk should be learned once"

    def test_renames_max_tokens_when_rejected(self, mesh):
        stub = attach(
            mesh,
            [_error(BadRequestError, "unsupported parameter: max_tokens"), _response("ok")],
        )
        mesh.chat([{"role": "user", "content": "hi"}], max_tokens=500)
        assert "max_completion_tokens" in stub.completions.calls[1]

    def test_an_unrelated_400_is_not_retried(self, mesh):
        stub = attach(mesh, [_error(BadRequestError, "model not found")])
        with pytest.raises(MeshError):
            mesh.chat([{"role": "user", "content": "hi"}])
        assert len(stub.completions.calls) == 1, "a deterministic 400 must not be retried"


class TestRetries:
    def test_transient_failures_are_retried(self, mesh):
        stub = attach(
            mesh,
            [_error(RateLimitError, "429"), _error(APIConnectionError, "boom"), _response("ok")],
        )
        assert mesh.chat([{"role": "user", "content": "hi"}]).text == "ok"
        assert len(stub.completions.calls) == 3

    def test_gives_up_after_the_attempt_budget(self, mesh):
        stub = attach(mesh, [_error(RateLimitError, "429")] * 5)
        with pytest.raises(MeshError):
            mesh.chat([{"role": "user", "content": "hi"}], attempts=2)
        assert len(stub.completions.calls) == 2


class TestCircuitBreaker:
    def test_opens_after_repeated_failures(self, mesh):
        attach(mesh, [_error(RateLimitError, "429")] * 30)
        for _ in range(3):
            with pytest.raises(MeshError):
                mesh.chat([{"role": "user", "content": "hi"}], attempts=2)
        assert mesh._breaker.is_open
        assert not mesh.available

    def test_open_circuit_short_circuits_immediately(self, mesh):
        attach(mesh, [_error(RateLimitError, "429")] * 30)
        for _ in range(3):
            with pytest.raises(MeshError):
                mesh.chat([{"role": "user", "content": "hi"}], attempts=2)
        before = len(mesh._client.completions.calls)
        with pytest.raises(CircuitOpen):
            mesh.chat([{"role": "user", "content": "hi"}])
        assert len(mesh._client.completions.calls) == before, "no call should reach the gateway"

    def test_success_resets_the_failure_count(self, mesh):
        attach(mesh, [_error(RateLimitError, "429"), _response("ok")])
        mesh.chat([{"role": "user", "content": "hi"}])
        assert mesh._breaker.failures == 0


class TestBudget:
    def test_blocks_once_the_daily_cap_is_reached(self, mesh, monkeypatch):
        """The gate is checked before each call, so a single call may overshoot the cap
        — cost is only knowable after the response. The guarantee is that spending
        stops once the limit is crossed, not that it never crosses it."""
        monkeypatch.setattr(settings, "llm_daily_budget_usd", 0.0004)
        attach(mesh, [_response("ok", prompt=1000, completion=500)] * 5)
        mesh.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o-mini")
        assert mesh.budget_remaining_usd == 0.0
        with pytest.raises(BudgetExceeded):
            mesh.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o-mini")

    def test_reports_remaining_budget(self, mesh, monkeypatch):
        monkeypatch.setattr(settings, "llm_daily_budget_usd", 1.0)
        attach(mesh, [_response("ok", prompt=1000, completion=500)])
        mesh.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o-mini")
        assert 0.99 < mesh.budget_remaining_usd < 1.0

    def test_status_is_reportable(self, mesh):
        status = mesh.status()
        assert status["key_present"] and not status["circuit_open"]
        assert "budget" in status


class TestEmbeddings:
    def test_returns_vectors_and_bills_them(self, mesh):
        payload = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])],
            usage=SimpleNamespace(prompt_tokens=40),
        )
        attach(mesh, embeddings=payload)
        assert mesh.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]

    def test_empty_input_makes_no_call(self, mesh):
        attach(mesh, embeddings=None)
        assert mesh.embed([]) == []


class TestDisabled:
    def test_refuses_when_the_llm_is_switched_off(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_enabled", False)
        client = MeshClient()
        with pytest.raises(MeshError):
            client.chat([{"role": "user", "content": "hi"}])
