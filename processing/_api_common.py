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
