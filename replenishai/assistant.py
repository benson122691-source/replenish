"""LLM replenishment assistant (project-brief differentiator 3): Gemini answers questions by
CALLING TOOLS over the experiment artifacts, never by guessing numbers.

- Tools are thin wrappers over replenishai.service, the same functions the REST API uses, so the
  assistant cannot report a number the API/dashboard would not report for the same inputs.
- One question per call (no chat memory): every answer can be checked on its own, which is what
  the 30-case evaluation (scripts/eval_assistant.py) needs.
- Needs a Gemini API key (GEMINI_API_KEY, or GOOGLE_API_KEY). Without it `ask` raises
  AssistantUnavailable and the API answers 503; the rest of the app keeps working.
- Model: GEMINI_MODEL env var, default below (has a free tier on the Gemini Developer API).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from google import genai
from google.genai import errors, types

from . import service
from .service import Artifacts, UnknownSeries

DEFAULT_MODEL = "gemini-3.8-flash"
MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
MAX_OUTPUT_TOKENS = 8192
MAX_TOOL_ROUNDS = 6               # hard stop for the agent loop (cost / quota control)
MAX_QUESTION_CHARS = 500
REQUEST_TIMEOUT_S = 90            # per model request
KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}

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
    """Provider-neutral tool spec (JSON Schema). Gemini has no strict mode, so `_coerce_args`
    re-checks types before a tool runs and the service layer still validates every value."""
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
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


class Transient(AssistantUnavailable):
    """Worth retrying later: 5xx (model overloaded) or 429. The eval script waits and retries."""


class RateLimited(Transient):
    """429 from Gemini (free-tier per-minute or per-day quota)."""


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


_SCHEMAS = {t["name"]: t["parameters"]["properties"] for t in TOOLS}


def _coerce_args(name: str, args: dict) -> dict:
    """Gemini may send 10.0 for an integer field; turn whole floats back into int per the schema."""
    props = _SCHEMAS.get(name, {})
    out = {}
    for k, v in args.items():
        if props.get(k, {}).get("type") == "integer" and isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


def _config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["parameters"]) for t in TOOLS])],
        # we run the loop ourselves: every call must be recorded for the eval
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def make_client() -> Any:
    key = next((os.environ[k] for k in KEY_ENVS if os.environ.get(k)), None)
    if not key:
        raise AssistantUnavailable("未設定 GEMINI_API_KEY")
    # Bounded wait per request; no hidden SDK retries (the eval script retries Transient itself,
    # and /chat should fail fast with 503 rather than hang the page).
    return genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=REQUEST_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)))


def _call_model(client: Any, model: str, contents: list, config) -> Any:
    try:
        return client.models.generate_content(model=model, contents=contents, config=config)
    except errors.ClientError as e:
        msg = str(e)
        if e.code == 429:
            raise RateLimited("Gemini API 達到額度上限（免費方案每分鐘／每日次數），請稍後再試") from e
        if e.code in (401, 403) or "API_KEY_INVALID" in msg or "API key not valid" in msg:
            raise AssistantUnavailable("Gemini API key 無效或沒有權限（GEMINI_API_KEY）") from e
        if e.code == 404:
            raise AssistantUnavailable(f"模型 {model} 無法使用（檢查 GEMINI_MODEL）：{e.message}") from e
        raise                                                      # other 4xx = our bug: surface it
    except errors.ServerError as e:
        raise Transient(f"Gemini API 暫時錯誤（{e.code}），請稍後再試") from e
    except httpx.TransportError as e:
        if isinstance(e, httpx.TimeoutException):
            raise Transient(f"Gemini API 超過 {REQUEST_TIMEOUT_S} 秒沒有回應，請稍後再試") from e
        raise Transient("無法連線到 Gemini API") from e


def ask(question: str, a: Artifacts, client: Any | None = None, model: str | None = None) -> ChatResult:
    """Answer one question with a manual tool loop (we need every call recorded for the eval)."""
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question longer than {MAX_QUESTION_CHARS} characters")
    model = model or MODEL
    if client is None:
        client = make_client()
    config = _config()
    contents: list = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    result = ChatResult(answer="", served_by=model)

    for _ in range(MAX_TOOL_ROUNDS):
        resp = _call_model(client, model, contents, config)
        usage = getattr(resp, "usage_metadata", None)
        result.input_tokens += getattr(usage, "prompt_token_count", 0) or 0
        result.output_tokens += ((getattr(usage, "candidates_token_count", 0) or 0)
                                 + (getattr(usage, "thoughts_token_count", 0) or 0))
        result.served_by = getattr(resp, "model_version", None) or model

        feedback = getattr(resp, "prompt_feedback", None)
        if not resp.candidates or (feedback and feedback.block_reason):
            result.stop_reason = "blocked"
            result.answer = "這個問題我無法回答。我可以查詢品項預測、計算補貨建議，或比較各預測方法的成本。"
            return result
        cand = resp.candidates[0]
        reason = getattr(cand.finish_reason, "value", str(cand.finish_reason or ""))
        result.stop_reason = reason
        if reason in BLOCKED:
            result.answer = "這個問題我無法回答。我可以查詢品項預測、計算補貨建議，或比較各預測方法的成本。"
            return result

        calls = resp.function_calls or []
        if not calls:
            parts = (cand.content.parts if cand.content else None) or []
            result.answer = "".join(p.text for p in parts if p.text and not p.thought).strip()
            if reason == "MAX_TOKENS":
                result.answer += "\n（回答過長被截斷）"
            if not result.answer:
                result.answer = "沒有產生回答，請換個方式再問一次。"
            return result

        # keep the model turn exactly as returned (thought signatures must go back unchanged)
        contents.append(cand.content)
        parts = []
        for fc in calls:                          # all results go back in ONE user turn
            args = _coerce_args(fc.name, dict(fc.args or {}))
            ok, out = run_tool(a, fc.name, args)
            result.tool_calls.append(ToolCall(fc.name, args, ok, out))
            payload = {"result": json.loads(out)} if ok else json.loads(out)   # {"error": ...}
            parts.append(types.Part(function_response=types.FunctionResponse(
                id=fc.id, name=fc.name, response=payload)))
        contents.append(types.Content(role="user", parts=parts))

    result.answer = "查詢步驟過多，已停止。請把問題拆小一點再問。"
    result.stop_reason = "max_tool_rounds"
    return result
