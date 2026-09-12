"""
Shared API-call machinery used by both the VLM describer (stage 1) and the
LLM summarizer (stage 2).

Centralizes the three fixes that came out of the free-tier quota post-mortem:

* P2: a shared `pause_event` that, the instant ANY worker hits a quota /
  auth error, is set so every other in-flight worker on every task stops
  hammering the same API key. Cleared when the crisis passes.
* P1: retry backoff that honors the provider's suggested wait (retryDelay /
  Retry-After) instead of blindly sleeping a fixed 1s.
* 401 handling: an UNAUTHENTICATED response is almost never transient, so we
  surface it immediately to the caller instead of burning retries.

Nothing here talks to a specific provider; it only inspects an exception and
answers "should I pause / how long should I wait / should I give up".
"""
import re
import threading
import time
import logging
import requests

logger = logging.getLogger(__name__)

# Set by whichever worker first sees a quota/auth crisis; read by every other
# worker before each new attempt and at the top of each retry sleep.
SHARED_PAUSE = threading.Event()

# Minimum pause (seconds) enforced before we let workers resume after a
# quota/auth event. Google's free tier resets on a rolling window, so we
# back off generously.
PAUSE_SECONDS = 30.0


class TokenUsage:
    """Accumulates billed token counts across API calls.

    Values are fetched from each provider's success response:
      - OpenAI-compatible (DeepSeek / OpenRouter / custom):
          usage.prompt_tokens, usage.completion_tokens,
          usage.prompt_tokens_details.cached_tokens
      - Gemini:
          usage_metadata.prompt_token_count, candidates_token_count,
          cached_content_token_count

    Only responses that actually succeed contribute; failed / retried
    attempts charge input tokens on the provider side but never return
    a usage object, so they cannot be measured from the client.
    """

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0

    def add(self, usage) -> None:
        """Normalize a provider usage dict / object and accumulate it."""
        if not usage:
            return
        if hasattr(usage, "get"):
            usage = usage  # dict-like
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = details.get("cached_tokens")
        else:
            # google.genai usage_metadata-style object
            prompt_tokens = getattr(usage, "prompt_token_count", None)
            completion_tokens = getattr(usage, "candidates_token_count", None)
            cached_tokens = getattr(usage, "cached_content_token_count", None)
        self.calls += 1
        if prompt_tokens:
            self.prompt_tokens += int(prompt_tokens)
        if completion_tokens:
            self.completion_tokens += int(completion_tokens)
        if cached_tokens:
            self.cached_tokens += int(cached_tokens)

    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        cached = f", cache_hit={self.cached_tokens}" if self.cached_tokens else ""
        return (f"calls={self.calls} input={self.prompt_tokens} "
                f"output={self.completion_tokens}{cached} total={self.total()}")


def is_paused() -> bool:
    return SHARED_PAUSE.is_set()


def wait_if_paused() -> bool:
    """Block while another worker is paused for a quota/auth crisis.

    Returns True once it is safe to continue, False if it never becomes safe
    (should not normally happen)."""
    while SHARED_PAUSE.is_set():
        time.sleep(0.25)
    return True


def pause_all() -> None:
    """Global freeze: every other worker on every task checks this and stops."""
    SHARED_PAUSE.set()
    logger.warning("API quota/auth crisis detected - pausing all workers")


def resume_all() -> None:
    SHARED_PAUSE.clear()


def pause_and_wait(seconds: float = PAUSE_SECONDS) -> None:
    """Set the global pause, sleep locally so we DO NOT retry immediately,
    then clear it. During the sleep, other workers also see the pause and
    hold off, so the whole pipeline backs off together instead of a stampede."""
    SHARED_PAUSE.set()
    deadline = time.time() + max(seconds, 1.0)
    try:
        while time.time() < deadline:
            if not SHARED_PAUSE.is_set():  # something else cleared it early
                return
            time.sleep(0.25)
    finally:
        SHARED_PAUSE.clear()


def extract_status(exc) -> int:
    """Best-effort pull of an HTTP status code out of any exception shape.

    Handles requests.Response (raise_for_status), google.genai errors
    (.code / .status_code / .status), and raw ints."""
    if exc is None:
        return 0
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    resp = getattr(exc, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if isinstance(code, int):
            return code
    return 0


def extract_retry_delay(exc) -> float:
    """Pull the provider's suggested wait from a 429.

    Strategies, in order:
      1. google.genai error detail: 429 body has `{'retryDelay': '16s'}`.
      2. requests.Response `Retry-After` header (seconds or HTTP date).
      3. We rely on the caller's own backoff otherwise."""
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(resp, requests.Response):
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                # Seconds form
                return float(retry_after)
            except ValueError:
                pass
            # HTTP-date form - fall through and use our own backoff
    try:
        message = str(getattr(exc, "message", "") or exc)
        import re as _re
        m = _re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)", message, _re.I)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 0.0


def should_give_up(status: int) -> bool:
    """401 (and 403 for free-tier unmatched regions) are not worth retrying.

    401 UNAUTHENTICATED -> the key is dead; retrying the same key in a loop
    only makes the quota worse. 404/400 are permanent config errors."""
    if status in (400, 401, 403, 404, 422):
        return True
    return False


def classify_crisis(status: int) -> bool:
    """True if this status means 'stop all workers, we are hammering a shared
    quota / a shared key'."""
    return status == 429 or status == 401 or status == 403


def compute_sleep(status: int, exc, attempt, base_delay=1.0, cap=20.0) -> float:
    """Pick how long to sleep before the next attempt.

    * 429: honor the provider's retryDelay when present, otherwise exponential
      backoff scaled by attempt. Capped so we don't hang forever.
    * Others: exponential backoff on our own base_delay.
    """
    if status == 429:
        provider_delay = extract_retry_delay(exc)
        if provider_delay > 0:
            return min(provider_delay, cap)
        return min(base_delay * (2 ** attempt), cap)
    return min(base_delay * (2 ** attempt), cap)


# Cosmetic descriptions of the most common provider status codes, used to turn
# the raw per-unit exception into a user-facing diagnosis.
_STATUS_TEXT = {
    400: "bad request",
    401: "authentication failed - the API key is invalid, revoked, or not valid for this provider",
    403: "authorization failed - the API key is not allowed for the selected model/endpoint",
    404: "endpoint/model not found - check the API URL and model name",
    422: "unprocessable request - check the model name and payload",
    429: "quota or rate limit exceeded",
    500: "provider server error",
    502: "provider gateway error",
    503: "provider service unavailable",
}

# Precedence among categories when several statuses show up across units:
# a permanent auth/config/network failure should be reported instead of the
# quota message even if a quota hiccup happened first.
_CATEGORY_RANK = {
    "network": 0,
    "other": 1,
    "quota": 2,
    "config": 3,
    "auth": 4,
}


def extract_status_from_error(exc) -> int:
    """Like extract_status, but also scans the exception MESSAGE text for a
    provider-embedded status like ``HTTPError 401``, ``(status 401)``,
    ``401 Client Error`` or ``429 RESOURCE_EXHAUSTED``. Handles the
    RuntimeError wrappers that vlm_describer / llm_summarizer raise (which
    only carry the code inside the string, since HTTPError.__str__ includes
    it)."""
    status = extract_status(exc)
    if status:
        return status
    text = str(exc or "")
    patterns = (
        r"\b(?:HTTP(?:Error)?|status|status_code|code)\s*[:=]?\s*(\d{3})\b",
        r"\b(\d{3})\s+(?:Client|Server)\s+Error\b",
        r"\b(429|400|401|403|404|422)\s+(?:RESOURCE_EXHAUSTED|INVALID_ARGUMENT|UNAUTHENTICATED|PERMISSION_DENIED|NOT_FOUND|UNKNOWN)\b",
        r"\([^)]*?(\d{3})\)",
    )
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            code = int(m.group(1))
            if 400 <= code <= 599:
                return code
    return 0


def categorize_error(exc) -> str:
    """Classify an exception into a bucket the UI / abort logic can act on:
    auth / config / quota / network / other."""
    status = extract_status_from_error(exc)
    if status in (401, 403):
        return "auth"
    if status in (400, 404, 422):
        return "config"
    if status == 429:
        return "quota"
    if status in (500, 502, 503, 504):
        return "other"
    if status:
        return "other"
    text = str(exc or "").lower()
    if isinstance(exc, requests.exceptions.RequestException) or any(k in text for k in (
        "connectionerror", "timeout", "timed out", "no such host", "connection refused",
        "printer in a fire", "connection aborted", "failed to establish", "unreachable",
        "temporary failure in name resolution",
    )):
        return "network"
    return "other"


def dominant_category(counter: dict) -> str:
    """Pick which category best explains a batch of failures. When the counts
    tie, prefer the more serious category (auth > config > quota > other > net).

    An empty counter means every unit returned 200 OK with no HTTP errors, so
    the most likely cause is the model silently returning empty content (e.g.
    the model does not support image input or the model name is wrong)."""
    if not counter:
        return "empty"
    best_cat = "quota"
    best_score = -1
    for cat, count in counter.items():
        score = count * 10 + _CATEGORY_RANK.get(cat, 1)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


def quote_error_detail(exc) -> str:
    """Return a short, single-line error detail safe for user-facing messages."""
    text = str(exc or "").replace("\n", " ").replace("\r", " ").strip()
    return text[:200]


def response_body(exc, limit=500) -> str:
    """Extract the provider's response body (if any) from a requests
    exception OR a raw requests.Response, flattened and truncated for
    logging.

    This is how we surface the real provider-side reason for a 400/401/429
    (e.g. "model does not support image inputs") that the exception's own
    ``__str__`` hides."""
    if exc is None:
        return ""
    resp = exc if isinstance(exc, requests.Response) else getattr(exc, "response", None)
    if resp is None:
        return ""
    try:
        text = getattr(resp, "text", "") or ""
    except Exception:
        return ""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if not text:
        return ""
    return text[:limit]


def build_stage1_blocked_message(category: str, success_count: int, total: int, error_detail: str = "") -> str:
    """User-facing explanation for why stage 2 was skipped. Replaces the old
    unconditional 'quota limit reached' message with a diagnosis derived from
    the actual API errors observed during stage 1."""
    detail = f" Last error: {error_detail}" if error_detail else ""
    common = (
        f"only {success_count}/{total} shots could be described "
        f"(full audio descriptions need ~{2 * total} API calls). "
        "Stage 2 (AD summarization) was skipped, so final.csv and _AD.csv were NOT "
        "generated. The partial shot descriptions are saved in _DetailsDescription.csv."
    )
    if category == "auth":
        return (
            "API authentication failed: the API key is invalid, revoked, or not valid for the "
            f"selected provider/model (check the API key and the endpoint URL/model). "
            f"During the run {common}{detail}"
        )
    if category == "config":
        return (
            "API configuration error: the endpoint URL or model name appears to be wrong for the "
            f"selected backend. During the run {common}{detail}"
        )
    if category == "network":
        return (
            "Network/connection error occurred while calling the API provider. "
            f"During the run {common}{detail}"
        )
    if category == "empty":
        return (
            f"The API returned empty descriptions for all {total} shots — the model "
            "likely does not support image input, or the model name / endpoint URL "
            f"is incorrect.{detail} During the run {common}"
        )
    # quota (includes genuinely exhausted free-tier daily caps and 429s)
    return (
        f"API quota limit reached: {common} "
        "Consider using a higher-quota model or a paid tier."
    )


def build_stage2_failed_message(category: str, error_detail: str = "") -> str:
    """User-facing explanation for a stage-2 (summarization) failure, keeping
    the stage-1 results as the recoverable artifact."""
    detail = f" Details: {error_detail}" if error_detail else ""
    if category == "auth":
        return (
            f"Stage 2 (AD summarization) failed: API authentication error - the API key is "
            f"invalid, revoked, or not valid for the selected provider.{detail} "
            "The stage-1 shot descriptions were saved and can be downloaded above."
        )
    if category == "config":
        return (
            f"Stage 2 (AD summarization) failed: API configuration error - the endpoint URL or "
            f"model name appears to be wrong for the selected backend.{detail} "
            "The stage-1 shot descriptions were saved and can be downloaded above."
        )
    if category == "network":
        return (
            f"Stage 2 (AD summarization) failed: network/connection error while calling the API "
            f"provider.{detail} The stage-1 shot descriptions were saved and can be downloaded above."
        )
    return (
        f"Stage 2 (AD summarization) failed: API quota or provider error likely exhausted the "
        f"quota.{detail} "
        "The stage-1 shot descriptions were saved and can be downloaded above."
    )
