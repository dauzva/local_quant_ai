"""OpenRouter LLM client: free-model discovery/ranking + strategy-code and
JSON-analysis completions.

Never trusts the raw text response -- callers must run any generated Python
through src/strategy_sandbox.py before using it for anything.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import requests

from src.config import get_api_key
from src.utils import get_logger

logger = get_logger("llm")


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    """HTTP 429 -- on OpenRouter's free tier this is usually a per-model
    cooldown that won't clear in the next few seconds, so retrying the SAME
    model repeatedly just burns the retry budget. Callers should move to the
    next model immediately instead of retrying in place (see complete_text/
    complete_json below)."""


class AuthError(LLMError):
    """HTTP 401/403: the API key is missing/invalid. Every model fails the
    same way, so callers must stop instead of walking the whole fallback list
    (the old loop burned 20+ requests per run on this)."""


class DailyQuotaExhaustedError(LLMError):
    """A fresh (no-credit) OpenRouter account gets 50 free-model requests
    per DAY, account-wide -- not per-model, and not the same as an
    individual model's own cooldown. Once this hits, EVERY free model fails
    identically until the daily reset, so unlike a plain RateLimitError
    there is nothing to fail over to: raised when every model in the
    fallback chain failed specifically for this reason, so the orchestrator
    can stop the run early instead of grinding through remaining iterations
    that are all guaranteed to fail the same way."""


@dataclass
class LLMResponse:
    content: dict
    model_used: str
    raw_text: str


@dataclass
class LLMTextResponse:
    text: str
    model_used: str


# --------------------------------------------------------------------------
# Free-model ranking for strategy-code generation.
#
# OpenRouter's free-model catalog rotates constantly (models appear/disappear,
# get renamed, or stop being free) -- hardcoding a single "best" model id is
# guaranteed to go stale. Instead we rank whatever is CURRENTLY free by
# keyword heuristics, so `list-free-models`/the orchestrator self-adjust as
# the catalog changes, and fall back through the ranked list on any failure.
# --------------------------------------------------------------------------

# Positive signal: models whose id/name suggests strong instruction-following
# or code ability. Matched case-insensitively against the model id.
CODEGEN_PREFERRED_KEYWORDS = [
    "code", "coder", "qwen", "glm", "deepseek", "gemma",
    "llama", "mistral", "phi", "gpt", "claude", "grok",
]
# Soft negative signal: still usable (not excluded), but pushed down the
# fallback order -- "thinking"/reasoning-branded models are far more prone to
# spending their token budget on hidden chain-of-thought before ever writing
# code, and empirically (nvidia/nemotron-3.5-lightning:free) can still leak
# that reasoning into `content` as repetitive garbage or return prose instead
# of code even with the `reasoning: {exclude: true}` request parameter set
# (see _call_model_once) -- a plain instruct model is the safer default for
# "return exactly this fenced format" tasks.
CODEGEN_DEPRIORITIZED_KEYWORDS = ["reasoning", "thinking", "nemotron", "-r1", ":r1", "think"]
# Negative signal: models that are unlikely to reliably follow a "return
# exactly N fenced Python blocks" instruction, or aren't general text models
# at all (audio/image/video/vision/safety/domain-tuned variants).
CODEGEN_EXCLUDED_KEYWORDS = [
    "vision", "-vl", ":vl", "vl:", "audio", "lyria", "image", "video", "clip",
    "safety", "moderation", "embed", "whisper", "tts", "asr", "note-preview",
    "-fin", "-sante", "-health",
]
MIN_CONTEXT_FOR_CODEGEN = 8_000


def _keyword_score(model_id: str) -> int:
    mid = model_id.lower()
    score = sum(1 for kw in CODEGEN_PREFERRED_KEYWORDS if kw in mid)
    score -= sum(1 for kw in CODEGEN_DEPRIORITIZED_KEYWORDS if kw in mid)
    return score


def rank_models_for_codegen(free_models: list[dict]) -> list[dict]:
    """Sort free models best-first for strategy-code generation: usable text
    models only, preferring larger context and code/instruct-sounding names."""
    candidates = []
    for m in free_models:
        mid = (m.get("id") or "").lower()
        if any(kw in mid for kw in CODEGEN_EXCLUDED_KEYWORDS):
            continue
        ctx = m.get("context_length") or 0
        if ctx and ctx < MIN_CONTEXT_FOR_CODEGEN:
            continue
        candidates.append(m)

    candidates.sort(key=lambda m: (-_keyword_score(m.get("id", "")), -(m.get("context_length") or 0)))
    return candidates


class UsageTracker:
    """Accumulates token usage and cost across every LLM call in a run.

    Cost is OpenRouter's own reported figure (`usage.cost`, requested via
    `usage: {include: true}`) when present; otherwise estimated from the
    model's per-token list pricing. Free (:free) models cost 0 but their
    tokens are still counted, so you can see what a paid model WOULD cost
    by multiplying against its price."""

    def __init__(self):
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               cost: float | None, cost_is_estimate: bool, elapsed: float) -> dict:
        with self._lock:
            return self._record(model, prompt_tokens, completion_tokens, cost, cost_is_estimate, elapsed)

    def _record(self, model: str, prompt_tokens: int, completion_tokens: int,
                cost: float | None, cost_is_estimate: bool, elapsed: float) -> dict:
        entry = {
            "call": len(self.calls) + 1,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost,
            "cost_is_estimate": cost_is_estimate,
            "seconds": round(elapsed, 2),
        }
        self.calls.append(entry)
        return entry

    def summary(self) -> dict:
        by_model: dict[str, dict] = {}
        for c in self.calls:
            m = by_model.setdefault(c["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            m["calls"] += 1
            m["prompt_tokens"] += c["prompt_tokens"]
            m["completion_tokens"] += c["completion_tokens"]
            m["cost_usd"] += c["cost_usd"] or 0.0
        return {
            "num_calls": len(self.calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "total_tokens": sum(c["total_tokens"] for c in self.calls),
            "cost_usd": sum(c["cost_usd"] or 0.0 for c in self.calls),
            "any_cost_estimated": any(c["cost_is_estimate"] for c in self.calls),
            "by_model": by_model,
        }

    def log_summary(self) -> None:
        s = self.summary()
        if not s["num_calls"]:
            logger.info("LLM usage: no successful calls this run")
            return
        est = " (partly estimated from list pricing)" if s["any_cost_estimated"] else ""
        logger.info("LLM usage TOTAL: %d calls, %d prompt + %d completion = %d tokens, cost $%.6f%s",
                    s["num_calls"], s["prompt_tokens"], s["completion_tokens"], s["total_tokens"], s["cost_usd"], est)
        for model, m in s["by_model"].items():
            logger.info("  %-50s %3d calls  %8d in  %8d out  $%.6f",
                        model, m["calls"], m["prompt_tokens"], m["completion_tokens"], m["cost_usd"])

    def save(self, path) -> None:
        data = {"summary": self.summary(), "calls": self.calls}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)




class OpenRouterClient:
    """Thread-safe OpenRouter client with adaptive model failover.

    Failover is stateful across calls (and across concurrent calls):
    - a model that returns HTTP 429 is put on a cooldown and skipped by every
      later call until it expires, instead of being re-hit on each call (the
      old behavior spent hundreds of requests -- which count against the daily
      free quota -- just being told "rate limited" again);
    - models that fail/return empty are demoted by a penalty score, models that
      succeed are promoted, so the order self-tunes toward whatever is actually
      answering today; a model that fails 3x in a row is dropped for the run;
    - 401/403 is fatal (AuthError) -- the key is bad, trying other models is
      pointless.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.base_url = cfg.llm.base_url.rstrip("/")
        self.api_key = get_api_key(cfg)
        if not self.api_key:
            logger.warning("No OpenRouter API key found in environment variable %s", cfg.llm.get("api_key_env"))
        self.configured_models = list(cfg.llm.get("models", []) or [])
        self.auto_discover = bool(cfg.llm.get("auto_discover_free_models", False))
        self.max_free_models = cfg.llm.get("max_free_models")
        self.max_retries = int(cfg.llm.get("max_retries", 2))
        self.timeout = int(cfg.llm.get("timeout_seconds", 120))
        self.retry_wait = float(cfg.llm.get("retry_wait_seconds", 3))
        self.temperature = float(cfg.llm.get("temperature", 0.7))
        self.max_tokens = int(cfg.llm.get("max_tokens", 4096))
        self.cooldown_seconds = float(cfg.llm.get("rate_limit_cooldown_seconds", 60))
        self.max_cooldown_wait = float(cfg.llm.get("max_cooldown_wait_seconds", 90))
        self.reasoning_effort = cfg.llm.get("reasoning_effort", "low")
        self._resolved_models: list[str] | None = None
        self._dead_models: set[str] = set()
        self._cooldown_until: dict[str, float] = {}
        self._penalty: dict[str, float] = {}
        self._consecutive_fail: dict[str, int] = {}
        self._zero_yield: dict[str, int] = {}
        self._lock = threading.Lock()
        self.usage = UsageTracker()
        self._pricing_cache: dict[str, tuple[float, float]] | None = None

    # ---- cost accounting -------------------------------------------------

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """Fallback when the API response carries no cost: price per token
        from OpenRouter's /models list (fetched once, lazily)."""
        if model.endswith(":free"):
            return 0.0
        with self._lock:
            if self._pricing_cache is None:
                self._pricing_cache = {}
                try:
                    for m in self.list_models():
                        p = m.get("pricing") or {}
                        self._pricing_cache[m.get("id", "")] = (float(p.get("prompt") or 0), float(p.get("completion") or 0))
                except (requests.RequestException, TypeError, ValueError) as exc:
                    logger.warning("Could not fetch model pricing for cost estimate: %s", exc)
            price = self._pricing_cache.get(model)
        if price is None:
            return None
        return prompt_tokens * price[0] + completion_tokens * price[1]

    def _record_usage(self, model: str, data: dict, elapsed: float) -> None:
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        estimated = False
        if cost is None:
            cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
            estimated = cost is not None
        entry = self.usage.record(model, prompt_tokens, completion_tokens, cost, estimated, elapsed)
        cost_str = "n/a" if cost is None else f"${cost:.6f}{' (est)' if estimated else ''}"
        logger.info("LLM call #%d model=%s tokens: %d in + %d out = %d, cost %s, %.1fs",
                    entry["call"], model, prompt_tokens, completion_tokens, entry["total_tokens"], cost_str, elapsed)

    def total_cost(self) -> float:
        return self.usage.summary()["cost_usd"]

    # ---- model discovery -------------------------------------------------

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        extra = self.cfg.llm.get("request_headers", {}) or {}
        headers.update(dict(extra))
        return headers

    def list_models(self) -> list[dict]:
        url = f"{self.base_url}/models"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])

    def list_free_models(self, max_models: int | None = None) -> list[dict]:
        models = self.list_models()
        free = []
        for m in models:
            model_id = m.get("id", "")
            pricing = m.get("pricing", {}) or {}
            is_free_id = model_id.endswith(":free")
            is_zero_priced = False
            try:
                prompt_price = float(pricing.get("prompt", "1") or "1")
                completion_price = float(pricing.get("completion", "1") or "1")
                is_zero_priced = prompt_price == 0.0 and completion_price == 0.0
            except (TypeError, ValueError):
                pass
            if is_free_id or is_zero_priced:
                free.append(m)
        free.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
        if max_models:
            free = free[:max_models]
        return free

    def resolve_models(self) -> list[str]:
        """The ranked list of model ids to try, best first.

        Explicitly configured models come first, then (if auto_discover_free_models)
        whatever is currently free on OpenRouter ranked by
        src.llm.rank_models_for_codegen. With use_only_free_models and no
        allow_paid_fallback, paid ids are filtered out."""
        with self._lock:
            if self._resolved_models is not None:
                return self._resolved_models
            ordered: list[str] = list(self.configured_models)
            if self.auto_discover:
                try:
                    free = self.list_free_models(max_models=self.max_free_models)
                    for m in rank_models_for_codegen(free):
                        mid = m.get("id")
                        if mid and mid not in ordered:
                            ordered.append(mid)
                except requests.RequestException as exc:
                    logger.warning("Auto-discovery of free models failed (%s); using configured models only", exc)

            use_only_free = bool(self.cfg.llm.get("use_only_free_models", True))
            allow_paid_fallback = bool(self.cfg.llm.get("allow_paid_fallback", False))
            if use_only_free and not allow_paid_fallback:
                ordered = [m for m in ordered if m.endswith(":free")] or ordered

            # Models known to burn the whole token budget on hidden reasoning
            # (observed: nemotron-lightning, 100-290s per call, 0-1 usable
            # strategies out of 10): drop them unless nothing else is left.
            avoid = [str(a).lower() for a in (self.cfg.llm.get("avoid_models", []) or [])]
            kept = [m for m in ordered if not any(a in m.lower() for a in avoid)]
            if kept and len(kept) < len(ordered):
                logger.info("Avoiding models (llm.avoid_models): %s", [m for m in ordered if m not in kept])
                ordered = kept

            self._resolved_models = ordered
            logger.info("Resolved model fallback order: %s", ordered)
            return ordered

    # ---- failover state --------------------------------------------------

    def _next_model(self, tried: set[str]) -> tuple[str | None, float]:
        """Best usable model for this call, or (None, seconds_until_one_frees_up)
        if every remaining candidate is cooling down (None, inf) if there is
        nothing left to wait for."""
        order = {m: i for i, m in enumerate(self.resolve_models())}
        now = time.monotonic()
        soonest = float("inf")
        best: tuple[float, int, str] | None = None
        with self._lock:
            for m, idx in order.items():
                if m in self._dead_models or m in tried:
                    continue
                wait = self._cooldown_until.get(m, 0.0) - now
                if wait > 0:
                    soonest = min(soonest, wait)
                    continue
                key = (self._penalty.get(m, 0.0), idx, m)
                if best is None or key < best:
                    best = key
        return (best[2], 0.0) if best else (None, soonest)

    def _on_success(self, model: str) -> None:
        with self._lock:
            self._consecutive_fail[model] = 0
            self._penalty[model] = max(0.0, self._penalty.get(model, 0.0) - 0.5)

    def _on_failure(self, model: str, fatal: bool = False) -> None:
        with self._lock:
            self._penalty[model] = self._penalty.get(model, 0.0) + 1.0
            self._consecutive_fail[model] = self._consecutive_fail.get(model, 0) + 1
            if fatal or self._consecutive_fail[model] >= 3:
                self._dead_models.add(model)
                logger.warning("Dropping model %s for the rest of this run", model)

    def report_yield(self, model: str, usable: int, expected: int) -> None:
        """Feedback from the caller on how many of the `expected` blocks were
        actually usable. A model that answers (and bills tokens) but returns
        mostly garbage/truncation is demoted so later calls prefer models that
        deliver; a reliable one is promoted."""
        if expected <= 0:
            return
        ratio = usable / expected
        with self._lock:
            if ratio < 0.5:
                self._penalty[model] = self._penalty.get(model, 0.0) + 1.5 * (1.0 - ratio)
                if ratio == 0.0:
                    self._zero_yield[model] = self._zero_yield.get(model, 0) + 1
                    if self._zero_yield[model] >= 2:
                        self._dead_models.add(model)
                        logger.warning("Dropping model %s: returned no usable strategies twice", model)
            else:
                self._penalty[model] = max(0.0, self._penalty.get(model, 0.0) - 0.7)
                self._zero_yield[model] = 0

    def _on_rate_limit(self, model: str) -> None:
        with self._lock:
            self._cooldown_until[model] = time.monotonic() + self.cooldown_seconds

    # ---- the actual request ---------------------------------------------

    def _call_model_once(self, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
        url = f"{self.base_url}/chat/completions"
        reasoning: dict = {"exclude": True}
        if self.reasoning_effort:
            reasoning["effort"] = str(self.reasoning_effort)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Keep reasoning/"thinking" models from burning the token budget on
            # hidden chain-of-thought (the cause of most "returned empty
            # content" failures and 100-160s calls in the logs): low effort,
            # and never echo the trace back into `content`. Ignored by models
            # that don't support it.
            "reasoning": reasoning,
            "usage": {"include": True},
        }
        started = time.monotonic()
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        elapsed = time.monotonic() - started
        if resp.status_code == 429:
            if "free-models-per-day" in resp.text or "openrouter_free_tier_daily" in resp.text:
                raise DailyQuotaExhaustedError(
                    f"OpenRouter's account-wide free-model daily quota is exhausted (hit on model {model}): {resp.text[:300]}"
                )
            raise RateLimitError(f"Model {model} rate-limited (HTTP 429): {resp.text[:300]}")
        if resp.status_code in (401, 403):
            raise AuthError(f"HTTP {resp.status_code} from OpenRouter (check OPENROUTER_API_KEY): {resp.text[:200]}")
        if resp.status_code >= 400:
            raise LLMError(f"Model {model} returned HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        # Record before validating content: an empty/invalid reply is still billed.
        self._record_usage(model, data, elapsed)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"Model {model} returned no choices: {str(data)[:200]}")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise LLMError(f"Model {model} returned empty content (finish_reason={choices[0].get('finish_reason')})")
        if choices[0].get("finish_reason") == "length":
            logger.warning("Model %s hit max_tokens=%d; returning the truncated text (complete blocks are still usable)",
                           model, max_tokens)
        return content

    def complete_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> LLMTextResponse:
        """Try models in adaptive order until one returns non-empty text.

        Raises DailyQuotaExhaustedError / AuthError immediately (nothing to
        fail over to) and LLMError once every model has been exhausted."""
        max_tokens = int(max_tokens or self.max_tokens)
        temperature = self.temperature if temperature is None else temperature
        tried: set[str] = set()
        last_error: Exception | None = None
        waited = 0.0

        while True:
            model, wait = self._next_model(tried)
            if model is None:
                if wait != float("inf") and waited + wait <= self.max_cooldown_wait:
                    logger.info("All models cooling down; waiting %.0fs for the next one", wait + 0.5)
                    time.sleep(wait + 0.5)
                    waited += wait + 0.5
                    continue
                raise LLMError(f"All models failed or are rate-limited. Last error: {last_error}")
            tried.add(model)

            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self._call_model_once(model, prompt, max_tokens, temperature)
                    self._on_success(model)
                    return LLMTextResponse(text=text, model_used=model)
                except (DailyQuotaExhaustedError, AuthError):
                    raise
                except RateLimitError as exc:
                    last_error = exc
                    self._on_rate_limit(model)
                    logger.warning("Model %s rate-limited; cooling it down %.0fs and moving on", model, self.cooldown_seconds)
                    break
                except (requests.RequestException, LLMError) as exc:
                    last_error = exc
                    fatal = "HTTP 404" in str(exc)
                    logger.warning("Model %s failed (attempt %d/%d): %s", model, attempt, self.max_retries, str(exc)[:200])
                    if fatal or attempt >= self.max_retries:
                        self._on_failure(model, fatal=fatal)
                        break
                    time.sleep(self.retry_wait * attempt)
