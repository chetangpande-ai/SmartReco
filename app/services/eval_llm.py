"""DeepEval's model adapter, routed through Mesh.

DeepEval metrics (FaithfulnessMetric, AnswerRelevancyMetric, GEval) need an LLM to judge
with. Handing them the OpenAI SDK directly would open a second door past Mesh — no
budget cap, no circuit breaker, no cost accounting for eval runs. This adapter is the one
integration point that keeps "every model call goes through Mesh" true here too.

Only imported by scripts/eval_generation.py and scripts/red_team.py — both already
require the optional `evals` extra to run at all, so this imports deepeval at module
scope rather than lazily like guardrails.py does for NeMo (which the main app imports on
every request and must survive not having installed).
"""

from deepeval.models import DeepEvalBaseLLM

from app.config import settings
from app.services.mesh import mesh


class MeshDeepEvalModel(DeepEvalBaseLLM):
    """Judge calls use a fixed, deliberately-chosen model rather than
    settings.meshapi_model_fast — a rubric judge being too lenient is worse than it
    being slow."""

    def __init__(self, model: str | None = None) -> None:
        self._model_name = model or settings.meshapi_model
        super().__init__(model=self._model_name)

    def load_model(self):
        return self._model_name

    def generate(self, prompt: str) -> str:
        result = mesh.chat(
            [{"role": "user", "content": prompt}],
            model=self._model_name,
            temperature=0,
        )
        return result.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name
