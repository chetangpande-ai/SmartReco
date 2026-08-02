"""Pure functions: security, Mesh response handling, guardrails, ranking maths."""

import numpy as np
import pytest

from app import metrics
from app.security import (
    create_session_token,
    csrf_ok,
    decode_session_token,
    hash_password,
    new_csrf_token,
    verify_password,
)
from app.services import guardrails
from app.services.mesh import MeshError, estimate_cost, extract_json
from app.services.retrieval import (
    BM25_K1,
    Candidate,
    LexicalIndex,
    mmr_select,
    rrf_fuse,
    tokenize,
)


class TestPasswords:
    def test_round_trip(self):
        h = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", h)
        assert not verify_password("wrong", h)

    def test_salted(self):
        assert hash_password("same") != hash_password("same")

    def test_long_password_truncated_at_72_bytes(self):
        # bcrypt silently ignores bytes past 72; truncating explicitly documents it.
        h = hash_password("x" * 200)
        assert verify_password("x" * 200, h)


class TestSessionTokens:
    def test_round_trip(self):
        payload = decode_session_token(create_session_token(42, "admin"))
        assert payload["sub"] == "42"
        assert payload["role"] == "admin"

    @pytest.mark.parametrize("token", ["", "nonsense", "a.b.c"])
    def test_garbage_rejected(self, token):
        assert decode_session_token(token) is None

    def test_tampered_rejected(self):
        token = create_session_token(1, "user")
        assert decode_session_token(token[:-3] + "aaa") is None

    def test_signed_with_another_key_rejected(self):
        import jwt

        forged = jwt.encode({"sub": "1", "role": "admin"}, "other-key", algorithm="HS256")
        assert decode_session_token(forged) is None


class TestCsrf:
    def test_matching(self):
        token = new_csrf_token()
        assert csrf_ok(token, token)

    @pytest.mark.parametrize("cookie,form", [("a", "b"), (None, "b"), ("a", None), (None, None)])
    def test_rejects(self, cookie, form):
        assert not csrf_ok(cookie, form)


class TestExtractJson:
    """Observed against the live gateway: several models fence JSON even in json mode."""

    @pytest.mark.parametrize(
        "raw",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            '```\n{"a": 1}\n```',
            'Sure! Here you go:\n{"a": 1}\nHope that helps.',
            '   \n {"a": 1}   ',
        ],
    )
    def test_tolerates_wrappers(self, raw):
        assert extract_json(raw)["a"] == 1

    def test_nested_objects(self):
        data = extract_json('```json\n{"a": {"b": [1, 2]}}\n```')
        assert data["a"]["b"] == [1, 2]

    def test_raises_without_json(self):
        with pytest.raises(MeshError):
            extract_json("there is no json here")


class TestCostEstimate:
    def test_known_model(self):
        assert estimate_cost("openai/gpt-4o-mini", 1000, 500) == pytest.approx(0.00045)

    def test_unknown_model_priced_pessimistically(self):
        assert estimate_cost("who/knows", 1000, 0) > estimate_cost("openai/gpt-4o-mini", 1000, 0)

    def test_zero_tokens(self):
        assert estimate_cost("openai/gpt-4o-mini", 0, 0) == 0.0


class TestGuardrails:
    @pytest.mark.parametrize(
        "text",
        [
            "This course guarantees you a job",
            "A totally risk-free investment",
            "100% guaranteed results",
            "Become an expert overnight",
            "Double your salary with this course",
            "The secret that bootcamps don't want you to know",
        ],
    )
    def test_blocks_outcome_promises(self, text):
        assert not guardrails.check_copy(text).ok

    @pytest.mark.parametrize(
        "text",
        [
            "Only 3 seats left",
            "Limited-time offer",
            "Act now before it's gone",
            "Hurry, this ends today",
            "Last chance to join",
            "Enrollment closes Friday",
        ],
    )
    def test_blocks_fabricated_urgency(self, text):
        assert not guardrails.check_copy(text).ok

    @pytest.mark.parametrize("text", ["Get 50% off today", "You save $40", "Half-price this week"])
    def test_blocks_invented_discounts(self, text):
        assert not guardrails.check_copy(text).ok

    def test_price_must_match_catalogue(self):
        allowed = {8900, 7900}
        assert not guardrails.check_copy("Yours for $499", allowed_prices_cents=allowed).ok
        assert guardrails.check_copy("It's $89 today", allowed_prices_cents=allowed).ok

    @pytest.mark.parametrize("text", ["Email bob@example.com", "Call +1 415 555 0199"])
    def test_blocks_pii(self, text):
        assert not guardrails.check_copy(text).ok

    def test_length_cap(self):
        assert not guardrails.check_copy("x" * 1300).ok

    def test_honest_copy_passes(self):
        good = (
            "You've been digging into agentic AI and spent real time on the RAG material. "
            "These two advanced courses pick up exactly where that left off."
        )
        assert guardrails.check_copy(good).ok

    def test_scrub_keeps_honest_sentences(self):
        mixed = (
            "You've been exploring agentic AI. "
            "Only 2 seats remain so act now. "
            "The LangGraph course builds on that."
        )
        cleaned = guardrails.scrub(mixed)
        assert "seats" not in cleaned and "act now" not in cleaned
        assert "agentic AI" in cleaned and "LangGraph" in cleaned

    def test_validate_reports_every_violation(self):
        report = guardrails.validate("Guaranteed job. Only 2 seats left. 50% off.")
        assert len(report.violations) >= 3


class TestTokenizer:
    def test_drops_stopwords(self):
        assert tokenize("the best course for learning ai") == ["ai"]

    def test_keeps_technical_terms(self):
        assert set(tokenize("c++ k8s node.js")) >= {"c++", "k8s"}

    def test_drops_single_characters(self):
        assert tokenize("a b ai") == ["ai"]

    def test_empty_input(self):
        assert tokenize("") == [] and tokenize(None) == []


class TestBm25:
    @pytest.fixture
    def index(self):
        return LexicalIndex(
            [
                (1, "kubernetes containers devops"),
                (2, "kubernetes basics"),
                (3, "python programming"),
                (4, "python data analysis python"),
            ]
        )

    def test_only_matching_documents_score(self, index):
        assert {pid for pid, _ in index.search("kubernetes", 10)} == {1, 2}

    def test_length_normalisation_favours_shorter_docs(self, index):
        scores = dict(index.search("kubernetes", 10))
        assert scores[2] > scores[1]

    def test_rare_terms_outweigh_common_ones(self, index):
        rare = dict(index.search("devops", 10))[1]
        common = dict(index.search("python", 10))[3]
        assert rare > common

    @pytest.mark.parametrize("query", ["", "   ", "quantumflux"])
    def test_no_match_returns_empty(self, index, query):
        assert index.search(query, 10) == []

    def test_k1_is_the_standard_default(self):
        assert BM25_K1 == 1.5


class TestRrf:
    def test_fuses_the_union(self):
        fused = rrf_fuse([(10, 0.9), (11, 0.8), (12, 0.7)], [(12, 5.0), (13, 4.0)])
        assert set(fused) == {10, 11, 12, 13}

    def test_reciprocal_rank_arithmetic(self):
        fused = rrf_fuse([(10, 0.9), (11, 0.8), (12, 0.7)], [(12, 5.0), (13, 4.0)])
        assert fused[10]["fused"] == pytest.approx(1 / 61)
        assert fused[12]["fused"] == pytest.approx(1 / 63 + 1 / 61)

    def test_appearing_in_both_rankers_wins(self):
        fused = rrf_fuse([(10, 0.9), (11, 0.8), (12, 0.7)], [(12, 5.0), (13, 4.0)])
        assert fused[12]["fused"] > fused[10]["fused"]

    def test_raw_score_magnitude_is_irrelevant(self):
        """Cosine and BM25 are incomparable scales; fusing by rank sidesteps that."""
        fused = rrf_fuse([(10, 999999.0)], [(11, 0.0001)])
        assert fused[10]["fused"] == pytest.approx(fused[11]["fused"])

    def test_tracks_provenance(self):
        fused = rrf_fuse([(10, 0.9), (11, 0.8), (12, 0.7)], [(12, 5.0)])
        assert fused[12]["v_rank"] == 3 and fused[12]["l_rank"] == 1

    def test_empty_inputs(self):
        assert rrf_fuse([], []) == {}


class TestMmr:
    @pytest.fixture
    def setup(self):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            Candidate(product_id=i, title=f"p{i}", category="c", tier="mid",
                      price_cents=0, rating=0.0)
            for i in (1, 2, 3)
        ]
        vectors = {
            1: np.array([1.00, 0.00, 0.0], np.float32),
            2: np.array([0.99, 0.14, 0.0], np.float32),  # near-duplicate of 1
            3: np.array([0.30, 0.95, 0.0], np.float32),  # a different direction
        }
        return candidates, vectors, query

    def test_pure_relevance_takes_the_two_closest(self, setup):
        candidates, vectors, query = setup
        assert [c.product_id for c in mmr_select(candidates, vectors, query, 2, 1.0)] == [1, 2]

    def test_diversity_rejects_the_near_duplicate(self, setup):
        candidates, vectors, query = setup
        assert [c.product_id for c in mmr_select(candidates, vectors, query, 2, 0.3)] == [1, 3]

    def test_empty_candidates(self, setup):
        _, vectors, query = setup
        assert mmr_select([], vectors, query, 3, 0.5) == []

    def test_never_returns_more_than_requested(self, setup):
        candidates, vectors, query = setup
        assert len(mmr_select(candidates, vectors, query, 2, 0.5)) == 2


class TestMetrics:
    def test_counter_and_labels(self):
        metrics.inc("smartreco_test_total", 5)
        metrics.inc("smartreco_test_total", 1, kind="a")
        assert metrics.get("smartreco_test_total") == 5.0
        assert metrics.get("smartreco_test_total", kind="a") == 1.0

    def test_prometheus_exposition(self):
        metrics.gauge("smartreco_test_gauge", 3)
        text = metrics.render_prometheus()
        assert "# TYPE smartreco_test_gauge gauge" in text
        assert "smartreco_uptime_seconds" in text
