"""LLM assistant tests (FR-11). No network and no API key: a scripted fake client stands in for
Anthropic, so these tests check OUR loop (tool dispatch, error relay, limits, /chat wiring), not
the model's judgement. The model's judgement is measured by scripts/eval_assistant.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from replenishai import assistant, service
from replenishai.api import REQUIRED_FILES, create_app

ART = Path(__file__).resolve().parent.parent / "artifacts"
pytestmark = pytest.mark.skipif(not all((ART / f).exists() for f in REQUIRED_FILES),
                                reason="artifacts/ missing: run `python scripts/run_experiment.py` first")

REPL = {"item_id": "FOODS_1_001", "store_id": "CA_1", "on_hand": 10,
        "lead_time_days": 7, "service_level": 0.95, "moq": 0}


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def text(t):
    return NS(type="text", text=t)


def reply(stop, *blocks):
    return NS(stop_reason=stop, content=list(blocks), model=assistant.MODEL,
              usage=NS(input_tokens=100, output_tokens=20))


class FakeClient:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=lambda o: o.__dict__)))
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def art():
    return service.load_artifacts(ART)


def test_tools_are_strict_and_all_have_handlers(art):
    names = {t["name"] for t in assistant.TOOLS}
    assert names == set(assistant._handlers(art))
    for t in assistant.TOOLS:
        s = t["input_schema"]
        assert t["strict"] is True and s["additionalProperties"] is False
        assert set(s["required"]) <= set(s["properties"])


def test_tool_loop_returns_exact_service_numbers(art):
    fake = FakeClient(reply("tool_use", tool_use("t1", "compute_replenishment", REPL)),
                      reply("end_turn", text("建議訂 21 件。")))
    res = assistant.ask("FOODS_1_001 在 CA_1 庫存 10 件、交期 7 天要訂多少？", art, client=fake)

    assert res.answer == "建議訂 21 件。"
    [call] = res.tool_calls
    assert call.ok and call.name == "compute_replenishment"
    assert json.loads(call.output) == service.replenish_data(art, **REPL)   # same numbers as the API
    first, second = fake.requests
    assert first["model"] == assistant.MODEL and first["fallbacks"] == "default"
    assert first["output_config"] == {"effort": assistant.EFFORT}
    [result] = second["messages"][-1]["content"]                          # tool result went back
    assert result["tool_use_id"] == "t1" and result["is_error"] is False


@pytest.mark.parametrize("bad, msg", [
    ({"item_id": "FOODS_9_999"}, "item_id not found"),
    ({"lead_time_days": 40}, "lead_time_days must be between"),
    ({"service_level": 1.2}, "service_level must be between"),
])
def test_bad_tool_input_is_relayed_as_error_not_raised(art, bad, msg):
    fake = FakeClient(reply("tool_use", tool_use("t1", "compute_replenishment", {**REPL, **bad})),
                      reply("end_turn", text("查無資料")))
    res = assistant.ask("q", art, client=fake)
    [call] = res.tool_calls
    assert not call.ok and msg in json.loads(call.output)["error"]
    [result] = fake.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True


def test_parallel_tool_calls_return_in_one_message(art):
    fake = FakeClient(reply("tool_use", tool_use("a", "compute_replenishment", REPL),
                            tool_use("b", "compute_replenishment", {**REPL, "lead_time_days": 10})),
                      reply("end_turn", text("交期 10 天要多訂。")))
    res = assistant.ask("交期從 7 天變 10 天會怎樣？", art, client=fake)
    assert [c.input["lead_time_days"] for c in res.tool_calls] == [7, 10]
    results = fake.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_refusal_gives_scoped_answer(art):
    res = assistant.ask("q", art, client=FakeClient(reply("refusal")))
    assert res.tool_calls == [] and "無法回答" in res.answer


def test_loop_stops_after_max_rounds(art):
    loop = [reply("tool_use", tool_use(f"t{i}", "compare_strategies", {}))
            for i in range(assistant.MAX_TOOL_ROUNDS)]
    res = assistant.ask("q", art, client=FakeClient(*loop))
    assert res.stop_reason == "max_tool_rounds" and len(res.tool_calls) == assistant.MAX_TOOL_ROUNDS


def test_forecast_tool_adds_totals(art):
    ok, out = assistant.run_tool(art, "get_forecast", {"item_id": "FOODS_1_001", "store_id": "CA_1"})
    d = json.loads(out)
    assert ok and d["total_pred_28d"] == pytest.approx(sum(p["pred"] for p in d["forecast"]), abs=0.01)
    assert d["total_pred_first_7d"] == pytest.approx(sum(p["pred"] for p in d["forecast"][:7]), abs=0.01)


# ---------------------------------------------------------------- /chat endpoint

def test_chat_endpoint_with_fake_model():
    fake = FakeClient(reply("tool_use", tool_use("t1", "compare_strategies", {})),
                      reply("end_turn", text("LightGBM 分位數成本最低。")))
    with TestClient(create_app(ART, llm_client=fake)) as c:
        r = c.post("/chat", json={"question": "哪個方法成本最低？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "LightGBM 分位數成本最低。"
    assert body["tool_calls"] == [{"name": "compare_strategies", "input": {}, "ok": True}]
    assert body["data_source"] in ("synthetic", "m5")


@pytest.mark.parametrize("broken_profile", [False, True])
def test_chat_without_credentials_is_503(monkeypatch, broken_profile):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(k, raising=False)
    if broken_profile:                                # config dir set but no profile file in it
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(ART / "_no_profile_here"))
    with TestClient(create_app(ART)) as c:
        r = c.post("/chat", json={"question": "哪個方法成本最低？"})
        assert r.status_code == 503 and "LLM 助理目前無法使用" in r.json()["detail"]
        assert c.get("/health").status_code == 200          # the rest of the API still works


@pytest.mark.parametrize("q", ["", "x" * 501])
def test_chat_rejects_empty_or_long_question(q):
    with TestClient(create_app(ART, llm_client=FakeClient())) as c:
        assert c.post("/chat", json={"question": q}).status_code == 422


# ---------------------------------------------------------------- evaluation set + grader

from replenishai.assistant_eval import grade, ungrounded_numbers  # noqa: E402

CASES = json.loads((ART.parent / "eval" / "assistant_cases.json").read_text(encoding="utf-8"))["cases"]


def test_eval_set_is_well_formed_and_feasible(art):
    assert len(CASES) == 30 and len({c["id"] for c in CASES}) == 30
    cats = [c["category"] for c in CASES]
    assert (cats.count("query"), cats.count("simulate"), cats.count("refuse")) == (10, 14, 6)
    schema = {t["name"]: t["input_schema"] for t in assistant.TOOLS}
    for c in CASES:
        for exp in c["expect"]:
            assert set(exp["args"]) <= set(schema[exp["tool"]]["properties"]), c["id"]
            full = {"service_level": 0.95, "moq": 0, **exp["args"]} if exp["tool"] == "compute_replenishment" else exp["args"]
            ok, out = assistant.run_tool(art, exp["tool"], full)
            assert ok, f"{c['id']}: expected call cannot succeed: {out}"     # no impossible answers
        assert set(c.get("forbid_ok", [])) <= set(schema), c["id"]


def _call(art, name, args):
    ok, out = assistant.run_tool(art, name, args)
    return {"name": name, "input": args, "ok": ok, "output": out}


def test_grader_tool_rules(art):
    s01 = next(c for c in CASES if c["id"] == "s01")
    good = _call(art, "compute_replenishment", REPL)
    assert grade(s01, "訂 21 件（合成資料）", [good]).tool_ok
    wrong = _call(art, "compute_replenishment", {**REPL, "on_hand": 11})
    assert not grade(s01, "訂 20 件", [wrong]).tool_ok
    r03 = next(c for c in CASES if c["id"] == "r03")
    failed = _call(art, "compute_replenishment", {**REPL, "lead_time_days": 40})
    assert grade(r03, "交期必須在 1–28 天內", [failed]).tool_ok           # tried, got an error: fine
    assert not grade(r03, "交期必須在 1–28 天內", [good]).tool_ok         # succeeded with other args
    assert not grade(r03, "建議訂 21 件", [failed]).tool_ok               # never told the user why


def test_grader_flags_made_up_numbers(art):
    out = _call(art, "compute_replenishment", REPL)["output"]
    d = json.loads(out)
    q = "FOODS_1_001 在 CA_1 庫存 10 件、交期 7 天要訂多少？"
    honest = f"FOODS_1_001 在 CA_1 建議訂 {d['order_qty']} 件，再訂購點 {d['reorder_point']:.1f}，服務水準 95%。"
    assert ungrounded_numbers(honest, q, [out]) == ["95"]               # 0.95 was only an argument
    assert ungrounded_numbers(honest, q, [out, json.dumps(REPL)]) == []
    s01 = next(c for c in CASES if c["id"] == "s01")
    assert grade(s01, honest, [_call(art, "compute_replenishment", REPL)]).grounded
    assert ungrounded_numbers("建議訂 37 件，再訂購點 52.6。", q, [out]) == ["37", "52.6"]
    longer = json.loads(_call(art, "compute_replenishment", {**REPL, "lead_time_days": 10})["output"])
    delta = longer["order_qty"] - d["order_qty"]
    assert ungrounded_numbers(f"交期 10 天會多訂 {delta} 件", q, [out, json.dumps(longer)]) == []
