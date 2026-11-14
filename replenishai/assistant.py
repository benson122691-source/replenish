"""LLM replenishment assistant (project-brief differentiator 3): Claude answers questions by
CALLING TOOLS over the experiment artifacts, never by guessing numbers.

- Tools are thin wrappers over replenishai.service, the same functions the REST API uses, so the
  assistant cannot report a number the API/dashboard would not report for the same inputs.
- One question per call (no chat memory): every answer can be checked on its own, which is what
  the 30-case evaluation (scripts/eval_assistant.py) needs.
- Needs Anthropic credentials (ANTHROPIC_API_KEY). Without them `ask` raises AssistantUnavailable
  and the API answers 503; the rest of the app keeps working.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from . import service
from .service import Artifacts, UnknownSeries

MODEL = "claude-opus-5-5"
EFFORT = "medium"                 # tool routing over 4 small tools; raise only if the eval says so
MAX_TOKENS = 16000
MAX_TOOL_ROUNDS = 6               # hard stop for the agent loop (cost control)
MAX_QUESTION_CHARS = 500
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """你是 ReplenishAI 的補貨查詢助理，服務對象是零售店的採購人員。用繁體中文回答，簡潔、直接。

你只能用工具查到的資料回答：
- 每個數字都必須來自本次工具回傳的結果。不要自己估算、四捨五入成別的值，也不要補上工具沒給的數字。
- 問到補貨量或再訂購點，呼叫 compute_replenishment；問到預測銷量，呼叫 get_forecast；問到各方法的成本或誤差比較，呼叫 compare_strategies；不確定有哪些門市或品項時，呼叫 list_series。
- 「如果交期變成 X 天會怎樣」這類假設題：分別用原本與新的參數各呼叫一次 compute_replenishment，再比較兩個結果。
- 使用者沒給目前庫存時，不要假設數值，先請他提供。服務水準沒提到用 0.95，MOQ 沒提到用 0。
- 工具回傳錯誤（例如查無品項、交期超出 1–28 天）時，照實告訴使用者錯在哪裡，不要改用別的品項或參數硬算。

範圍限制：
- 你只回答這份資料裡的預測、補貨建議與方法比較。天氣、股價、其他公司、一般閒聊等，簡短說明你無法回答，並說你能做什麼。
- 你不能下單、聯絡供應商或修改任何資料；被要求執行這類動作時，說明你只能提供建議。

資料提醒：工具結果裡的 data_source 若是 synthetic，回答時要提一句「目前是合成資料，不代表真實零售表現」。成本參數是假設值。"""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": properties,
                             "required": required, "additionalProperties": False}}


_ITEM = {"type": "string", "description": "品項代碼，例如 FOODS_1_001"}
_STORE = {"type": "string", "description": "門市代碼，例如 CA_1"}

TOOLS = [
    _tool("list_series", "列出資料中所有門市，以及每個門市可查的品項代碼。", {}, []),
    _tool("get_forecast",
          "取得單一品項在單一門市的 28 天 LightGBM 日銷量預測（回測期間），含每日預測、實際值與合計。",
          {"item_id": _ITEM, "store_id": _STORE}, ["item_id", "store_id"]),
    _tool("compute_replenishment",
          "依目前庫存、交期、服務水準與 MOQ，計算再訂購點、安全庫存與建議訂購量。"
          "公式 ROP = μ·L + z·σ·√L；庫存 ≤ ROP 時訂 max(MOQ, ⌈ROP − 庫存⌉)，否則訂 0。",
          {"item_id": _ITEM, "store_id": _STORE,
           "on_hand": {"type": "integer", "description": "目前庫存件數（≥ 0）"},
           "lead_time_days": {"type": "integer", "description": "供應商交期天數（1–28）"},
           "service_level": {"type": "number", "description": "目標服務水準（0.5–0.999），未提供時用 0.95"},
           "moq": {"type": "integer", "description": "最小訂購量，未提供時用 0"}},
          ["item_id", "store_id", "on_hand", "lead_time_days", "service_level", "moq"]),
    _tool("compare_strategies",
          "取得各預測方法（季節性 naive、28 天移動平均、TSB、LightGBM、LightGBM 分位數、完美預測）"
          "的總成本、持有/缺貨成本、滿足率、預測誤差，以及相對 28 天移動平均的成本差異與 95% 信賴區間。",
          {}, []),
]


def _forecast_summary(a: Artifacts, item_id: str, store_id: str) -> dict:
    d = service.forecast_data(a, item_id, store_id)
    preds = [p["pred"] for p in d["forecast"]]
    d["total_pred_28d"] = round(sum(preds), 2)
    d["total_pred_first_7d"] = round(sum(preds[:7]), 2)
    d["mean_pred_per_day"] = round(sum(preds) / len(preds), 4)
    d["first_date"], d["last_date"] = d["forecast"][0]["date"], d["forecast"][-1]["date"]
    return d


def _handlers(a: Artifacts) -> dict[str, Callable[..., dict]]:
    return {
        "list_series": lambda: service.series_data(a),
        "get_forecast": lambda item_id, store_id: _forecast_summary(a, item_id, store_id),
        "compute_replenishment": lambda **kw: service.replenish_data(a, **kw),
        "compare_strategies": lambda: service.compare_data(a),
    }


def run_tool(a: Artifacts, name: str, args: dict) -> tuple[bool, str]:
    """Execute one tool. Returns (ok, JSON text). Bad input / unknown series become error results
    that the model must relay; they never raise into the loop."""
    fn = _handlers(a).get(name)
    if fn is None:
        return False, json.dumps({"error": f"unknown tool: {name}"})
    try:
        return True, json.dumps(fn(**args), ensure_ascii=False)
    except (UnknownSeries, ValueError, TypeError) as e:
        return False, json.dumps({"error": str(e)}, ensure_ascii=False)


class AssistantUnavailable(RuntimeError):
    """No credentials / API unreachable. The caller decides how to surface it (API: 503)."""


@dataclass
class ToolCall:
    name: str
    input: dict
    ok: bool
    output: str


@dataclass
class ChatResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    served_by: str = MODEL
    input_tokens: int = 0
    output_tokens: int = 0


def ask(question: str, a: Artifacts, client: Any | None = None, model: str = MODEL) -> ChatResult:
    """Answer one question with a manual tool loop (we need every call recorded for the eval)."""
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question longer than {MAX_QUESTION_CHARS} characters")
    if client is None:
        try:
            client = anthropic.Anthropic()
        except anthropic.CredentialsError as e:                 # broken `ant auth` profile
            raise AssistantUnavailable(f"Anthropic 憑證設定有誤：{e}") from e
    messages: list[dict] = [{"role": "user", "content": question}]
    result = ChatResult(answer="")

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = client.beta.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, tools=TOOLS,
                messages=messages, output_config={"effort": EFFORT},
                betas=[FALLBACK_BETA], fallbacks="default",   # re-run on another model if declined
            )
        except anthropic.CredentialsError as e:
            raise AssistantUnavailable(f"Anthropic 憑證設定有誤：{e}") from e
        except anthropic.AuthenticationError as e:
            raise AssistantUnavailable("Anthropic 憑證無效或未設定（ANTHROPIC_API_KEY）") from e
        except anthropic.PermissionDeniedError as e:
            raise AssistantUnavailable("這把 API key 沒有使用此模型的權限") from e
        except anthropic.RateLimitError as e:
            raise AssistantUnavailable("Anthropic API 達到速率上限，請稍後再試") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise AssistantUnavailable(f"Anthropic API 暫時錯誤（{e.status_code}）") from e
            raise                                                  # 4xx = our bug: surface it
        except anthropic.APIConnectionError as e:
            raise AssistantUnavailable("無法連線到 Anthropic API") from e
        except TypeError as e:                                     # SDK could not resolve auth
            if "auth" in str(e).lower() or "api_key" in str(e).lower():
                raise AssistantUnavailable("未設定 ANTHROPIC_API_KEY") from e
            raise

        result.stop_reason = resp.stop_reason
        result.served_by = getattr(resp, "model", model)
        usage = getattr(resp, "usage", None)
        result.input_tokens += getattr(usage, "input_tokens", 0) or 0
        result.output_tokens += getattr(usage, "output_tokens", 0) or 0
        # keep every block (thinking, fallback, tool_use) exactly as returned
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "refusal":
            result.answer = "這個問題我無法回答。我可以查詢品項預測、計算補貨建議，或比較各預測方法的成本。"
            return result
        if resp.stop_reason == "pause_turn":
            continue
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not tool_uses:
            result.answer = "".join(b.text for b in resp.content if b.type == "text").strip()
            if resp.stop_reason == "max_tokens":
                result.answer += "\n（回答過長被截斷）"
            return result

        tool_results = []
        for tu in tool_uses:                       # all results go back in ONE user message
            args = tu.input if isinstance(tu.input, dict) else json.loads(tu.input)
            ok, out = run_tool(a, tu.name, args)
            result.tool_calls.append(ToolCall(tu.name, args, ok, out))
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id,
                                 "content": out, "is_error": not ok})
        messages.append({"role": "user", "content": tool_results})

    result.answer = "查詢步驟過多，已停止。請把問題拆小一點再問。"
    result.stop_reason = "max_tool_rounds"
    return result
