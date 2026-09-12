import asyncio
from types import SimpleNamespace

import pytest

from healthrag.evaluation import ragas_eval


def test_ragas_skips_abstentions_and_records_explicit_judge(monkeypatch):
    class Metric:
        async def ascore(self, **kwargs):
            return SimpleNamespace(value=0.75)

    metrics = {"faithfulness": Metric(), "context_recall": Metric()}
    report = {"rows": [
        {
            "id": "answered",
            "question": "Question?",
            "reference": "Reference.",
            "result": {
                "status": "answered",
                "claims": [{"text": "Answer."}],
                "sources": [{"passage": "Context."}],
            },
        },
        {
            "id": "abstained",
            "question": "Unknown?",
            "reference": "Unknown.",
            "result": {"status": "insufficient_evidence", "claims": [], "sources": []},
        },
    ]}
    result = asyncio.run(ragas_eval.score_report(
        report, model="judge", base_url="http://localhost:11434/v1", metrics=metrics
    ))
    assert result["judge_model"] == "judge"
    assert result["metrics"]["faithfulness"] == {"mean": 0.75, "scored_cases": 1}
    assert result["rows"][1]["status"] == "skipped"
