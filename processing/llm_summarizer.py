"""
LLM Summarizer - supports Gemini, Qwen, DeepSeek, and OpenAI-compatible backends
"""
import os
import re
import json
import random
import logging
import pandas as pd
import requests
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Import original repo prompt function
try:
    from stage2.promptloader import get_user_prompt
except ImportError:
    def get_user_prompt(mode, prompt_idx, verb_list, text_pred, word_limit, examples, lang="en"):
        return f"Summarize: {text_pred} in {word_limit} words."

# API URLs
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.8-flash"

# Shared API-call helpers (pause-on-quota, provider backoff, 401 give-up).
# Imported lazily via _get_api_common() at call time so both frozen builds
# (desktop/web) keep a single source of truth.
def _get_api_common():
    import processing._api_common as m
    return m

# Dataset-specific verb lists (from original repo)
VERB_LISTS = {
    "movie": ['look', 'turn', 'take', 'hold', 'pull', 'walk', 'run', 'watch', 'stare', 'grab', 'fall', 'get', 'go', 'open', 'smile'],
    "tv_series": ['look', 'walk', 'turn', 'stare', 'take', 'hold', 'smile', 'leave', 'pull', 'watch', 'open', 'go', 'step', 'get', 'enter'],
    "stage_performance": ['sing', 'dance', 'perform', 'move', 'gesture', 'walk', 'turn', 'look', 'smile', 'wave', 'jump', 'spin', 'play', 'hold', 'raise']
}

# Same lists (same order = same priority) in Traditional Chinese, used when the
# UI language is Chinese so the prompt stays in one language end to end.
VERB_LISTS_ZH = {
    "movie": ['看', '轉身', '拿', '握住', '拉', '走', '跑', '注視', '凝視', '抓', '倒下', '取得', '前往', '打開', '微笑'],
    "tv_series": ['看', '走', '轉身', '凝視', '拿', '握住', '微笑', '離開', '拉', '注視', '打開', '前往', '跨步', '取得', '進入'],
    "stage_performance": ['唱', '跳舞', '表演', '移動', '比劃', '走', '轉身', '看', '微笑', '揮手', '跳', '旋轉', '演奏', '握住', '抬起']
}

# AD pacing used to size the narration budget: SECONDS PER UNIT, not units per
# second. The limit is duration / pace, so 0.275 s/word means ~3.64 words per
# second - the rate measured on the cmdad/tvad training data.
#
# NOTE: despite the historical name, this same table also picks the few-shot
# examples (those are English sentences from those datasets), so it must stay on
# the English rate even when the prompt language is Chinese.
AD_SPEED = {
    "movie": 0.275,        # cmdad dataset
    "tv_series": 0.2695,   # tvad dataset
    "stage_performance": 0.2695  # use tvad speed
}

# Chinese narration pace for the same job: 0.2 s/character == 5 characters per
# second. Chinese narration is paced per character, so the English word rate
# would under-fill the window. Applies to the word limit only - the few-shot
# examples stay English (see the note above).
AD_SPEED_ZH = {
    "movie": 0.2,
    "tv_series": 0.2,
    "stage_performance": 0.2,
}

# Floor applied to user-picked ranges ("describe only / in addition"): they can
# be far shorter than any detected shot or dialogue gap.
CUSTOM_MIN_WORD_LIMIT = 20

# Load few-shot training examples
GT_EXAMPLES = {}
TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'stage2', 'gt_ad_train')

def _load_training_data():
    """Load ground truth AD training examples for few-shot learning."""
    global GT_EXAMPLES
    try:
        cmdad_df = pd.read_csv(os.path.join(TRAIN_DIR, 'cmdad_train.csv'))
        GT_EXAMPLES['movie'] = cmdad_df['text_gt_wo_char'].tolist()
        
        tvad_df = pd.read_csv(os.path.join(TRAIN_DIR, 'tvad_train.csv'))
        GT_EXAMPLES['tv_series'] = tvad_df['text_gt_wo_char'].tolist()
        GT_EXAMPLES['stage_performance'] = tvad_df['text_gt_wo_char'].tolist()
    except Exception as e:
        print(f"Warning: Could not load training data: {e}")
        GT_EXAMPLES = {'movie': [], 'tv_series': [], 'stage_performance': []}

_load_training_data()


def estimate_word_limit(duration_seconds: float, video_type: str = "movie", lang: str = None) -> int:
    """How long an AD may be for a window of this duration.

    Both pace tables are SECONDS PER UNIT, so the limit is duration / pace:
    ~3.64 words/s in English, ~5 characters/s when lang is "zh" (a shorter
    pace gives a longer budget). Anything other than "zh", including None,
    keeps the original English budget.
    """
    pace = AD_SPEED_ZH if str(lang or "").lower().startswith("zh") else AD_SPEED
    return max(1, round(duration_seconds / pace.get(video_type, pace["movie"])))


def sample_few_shot_examples(video_type: str, duration_seconds: float, num_examples: int = 10) -> List[str]:
    """Sample few-shot examples with similar word count from training data."""
    examples = GT_EXAMPLES.get(video_type, [])
    if not examples:
        return []
    
    # Calculate target word count. Deliberately the English pace: the examples
    # are English sentences from the cmdad/tvad datasets, so they have to be
    # matched in words even when the prompt language is Chinese.
    speed = AD_SPEED.get(video_type, 0.275)
    target_words = round(duration_seconds / speed)
    
    # Calculate word count for each example
    example_words = [len(str(e).strip().split(" ")) for e in examples]
    
    # Find examples with similar word count (±1 word)
    candid_indices = [i for i, w in enumerate(example_words) 
                      if target_words - 1 <= w <= target_words + 1]
    
    # If not enough candidates, use all examples
    if len(candid_indices) < num_examples:
        candid_indices = list(range(len(examples)))
    
    # Sample random examples
    sampled_indices = random.choices(candid_indices, k=min(num_examples, len(candid_indices)))
    return [examples[i] for i in sampled_indices]


def _call_gemini(prompt, api_key, model="gemini-3.5-flash", max_retries=3, retry_delay=1.0, usage_acc=None):
    """Call Gemini API using google-genai library.

    Uses the shared API helpers: on a 429/401/403 it pauses every worker on
    every task (via the shared pause event), honors the provider's retryDelay
    during backoff, and gives up immediately on permanent auth errors
    instead of burning retries against a dead key."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    common = _get_api_common()

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    for attempt in range(max_retries):
        try:
            # Wait out any global quota crisis triggered by another worker
            # before we touch the API again.
            common.wait_if_paused()
            response = client.models.generate_content(model=model, contents=prompt)
            result = response.text.strip() if response.text else ""
            if usage_acc is not None:
                usage_acc.add(getattr(response, "usage_metadata", None))
            return result
        except Exception as e:
            status = common.extract_status(e)
            logger.warning(f"Gemini API attempt {attempt+1} failed (status={status or 'n/a'}): {e}")

            if common.classify_crisis(status):
                # Pause ALL workers so the whole pipeline backs off together
                # instead of a stampede on the same shared quota / key.
                common.pause_all()
                delay = common.compute_sleep(status, e, attempt, base_delay=retry_delay)
                common.pause_and_wait(delay)

            if common.should_give_up(status):
                # Permanent auth/config error - retrying a dead key is pointless
                # and only worsens the quota picture.
                raise RuntimeError(f"Gemini API failed (permanent error, status {status}): {e}")

            if attempt < max_retries - 1:
                delay = common.compute_sleep(status, e, attempt, base_delay=retry_delay)
                logger.info(f"Gemini API retry in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini API failed: {e}")


def _apply_no_thinking(payload, base_url):
    """Disable chain-of-thought for the reasoning-capable providers we route to.

    Qwen/DashScope and DeepSeek run thinking by default and burn the output
    budget on hidden reasoning; OpenRouter relays any reasoning-capable model.
    Unknown OpenAI-compatible hosts are left untouched so we never inject a
    field a strict provider rejects."""
    if not base_url:
        return
    if base_url.startswith(QWEN_API_URL) or "dashscope" in base_url:
        payload["enable_thinking"] = False
    elif base_url.startswith(DEEPSEEK_API_URL) or "deepseek" in base_url:
        payload["thinking"] = {"type": "disabled"}
    elif "openrouter" in base_url:
        payload["reasoning"] = {"enabled": False}


def _call_openai(prompt, api_key, model="gpt-latest", max_retries=3, retry_delay=1.0, base_url=None, usage_acc=None):
    """Call OpenAI-compatible API."""
    common = _get_api_common()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6
    }
    # No max_tokens: reasoning-mode models (qwen3.8-flash, DeepSeek thinking,
    # etc.) share the output budget between chain-of-thought and the final
    # answer; a small cap gets burned on reasoning and content comes back
    # empty (HTTP 200). Let the provider use its own default instead.
    _apply_no_thinking(payload, base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url}/chat/completions" if base_url else OPENAI_API_URL
    logger.info(f"OpenAI API: url={url}, model={model}")
    for attempt in range(max_retries):
        try:
            common.wait_if_paused()
            logger.info(f"OpenAI API attempt {attempt+1}/{max_retries} -> POST {url} (model={model})")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            if usage_acc is not None:
                usage_acc.add(data.get("usage"))
            return (data["choices"][0]["message"]["content"] or "").strip()
        except requests.exceptions.RequestException as e:
            status = common.extract_status(e)
            logger.warning(f"OpenAI API attempt {attempt+1} failed (status={status or 'n/a'}) url={url} model={model}: {e}")
            body = common.response_body(e)
            if body:
                logger.warning(f"OpenAI API response body: {body}")
            if common.classify_crisis(status):
                common.pause_all()
                delay = common.compute_sleep(status, e, attempt, base_delay=retry_delay)
                common.pause_and_wait(delay)
            if common.should_give_up(status):
                raise RuntimeError(f"OpenAI API failed (permanent error, status {status}): {e}")
            if attempt < max_retries - 1:
                delay = common.compute_sleep(status, e, attempt, base_delay=retry_delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"OpenAI API failed: {e}")


def _extract_ad_sentence(raw: str) -> str:
    """Pull the actual AD sentence out of a model response.

    Models like to wrap the JSON payload in markdown code fences, emit
    trailing prose, or return the value as a list - the old naive
    `'{"summarized_AD":' in text` + json.loads approach leaked all of
    those straight into the output CSVs.
    """
    text = (raw or "").strip()
    # 1) Strip markdown code fences
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    # 2) Strict JSON on the first {...} block
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            val = json.loads(m.group()).get("summarized_AD")
            if isinstance(val, list):
                val = " ".join(str(x) for x in val).strip()
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            pass
    # 3) Regex fallback: grab the string value even from broken JSON
    m2 = re.search(r'"summarized_AD"\s*:\s*"([^"]*)"?', text)
    if m2:
        return m2.group(1).strip()
    # 4) Plain text response - drop any leftover fences/braces
    return text.replace("```", "").strip()


def _http_error_hint(status_code: int) -> str:
    hints = {
        400: "Bad request - check model name / request format",
        401: "Unauthorized - invalid or missing API key",
        403: "Forbidden - this key has no access to the model",
        404: "Not found - model name does not exist on this endpoint",
        422: "Unprocessable request - check parameters",
        429: "Rate limit / quota exceeded - wait or top up billing",
    }
    if status_code in hints:
        return hints[status_code]
    if 500 <= status_code <= 599:
        return "Provider server error - try again later"
    return "Request failed"


def test_connection(api_key, backend="gemini-3.7-flash", openai_url=None, openai_model=None):
    """Send a tiny prompt to verify the credentials work end-to-end.

    Returns (ok, message). The message is UI-ready: either a confirmation
    or a mapped HTTP error (401/403/404/429/5xx), timeout, network failure,
    or empty-response report."""
    prompt = "Reply with the single word OK."
    try:
        if not api_key or not str(api_key).strip():
            return False, "No API key provided"

        if backend.startswith("gemini"):
            from google import genai
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(model=backend, contents=prompt)
            text = ""
            try:
                text = (resp.text or "").strip()
            except Exception:
                pass
            if not text:
                return False, "Empty response (possible safety block or quota issue)"
            return True, f"Connection OK - {backend} replied: {text[:50]}"

        if backend == "qwen":
            url = QWEN_API_URL + "/chat/completions"
            model = openai_model or QWEN_MODEL
        elif backend == "deepseek":
            url = DEEPSEEK_API_URL + "/chat/completions"
            model = openai_model or DEEPSEEK_MODEL
        else:  # openai-compatible
            base = (openai_url or "https://api.openai.com/v1").rstrip("/")
            url = base + "/chat/completions"
            model = openai_model or "gpt-latest"

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            # Reasoning-style models (DeepSeek V4 thinking, QwQ, etc.) spend
            # part of max_tokens on reasoning before emitting content; a tiny
            # budget gets burned on `reasoning_content` and `content` comes back
            # empty (HTTP 200). Give it enough room.
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 800},
            timeout=25,
        )
        logger.info(f"Test connection -> {backend} POST {url} (model={model})")
        if resp.status_code != 200:
            body = _get_api_common().response_body(resp, limit=500)
            body_suffix = f" | {body}" if body else ""
            return False, f"HTTP {resp.status_code} - {_http_error_hint(resp.status_code)}{body_suffix}"
        try:
            data = resp.json()
        except ValueError:
            return False, "HTTP 200 but the response body was not valid JSON"
        if not isinstance(data, dict):
            return False, "HTTP 200 but the response did not contain a JSON object"
        choices = data.get("choices") or []
        if not choices:
            error = data.get("error")
            if error:
                return False, f"Endpoint returned an error: {str(error)[:200]}"
            return False, "HTTP 200 but the response contained no choices"
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "").strip()
        if content:
            return True, f"Connection OK - {model} replied: {content[:50]}"
        # Reasoning models may answer entirely inside `reasoning_content` and
        # leave the final `content` empty for a tiny prompt; the connection,
        # key and model are all fine in that case.
        reasoning = str(
            message.get("reasoning_content") or message.get("reasoning") or ""
        ).strip()
        if reasoning:
            return True, (
                f"Connection OK - {model} responded (received reasoning text, "
                "no final content for this tiny prompt)"
            )
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "length":
            return False, "HTTP 200 but the response was cut off (finish_reason=length) - try a model with a longer output budget"
        return False, "HTTP 200 but the model returned empty content"

    except requests.exceptions.Timeout:
        return False, "Timeout - server did not respond within 25s"
    except requests.exceptions.ConnectionError as e:
        return False, f"Network error - cannot reach endpoint ({e.__class__.__name__})"
    except Exception as e:
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if isinstance(code, int):
            return False, f"HTTP {code} - {_http_error_hint(code)} | {str(e)[:150]}"
        return False, f"{e.__class__.__name__}: {str(e)[:200]}"


def summarize_to_ad(
    stage1_description: str,
    api_key: str,
    backend: str = "gemini",
    model: str = None,
    duration_seconds: float = None,
    word_limit: int = None,
    video_type: str = "movie",
    examples: List[str] = None,
    mode: str = "single",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    openai_url: str = None,
    openai_model: str = None,
    usage_acc=None,
    lang: str = None
) -> str:
    """Summarize Stage 1 description into concise AD sentence."""
    if not stage1_description:
        return ""
    
    # Calculate word limit from duration if provided
    if duration_seconds is not None and word_limit is None:
        word_limit = estimate_word_limit(duration_seconds, video_type, lang)
    elif word_limit is None:
        word_limit = 15
    
    # Sample few-shot examples if not provided
    if examples is None and duration_seconds is not None:
        examples = sample_few_shot_examples(video_type, duration_seconds)
    
    verb_lists = VERB_LISTS_ZH if str(lang or "").lower().startswith("zh") else VERB_LISTS
    verb_list = verb_lists.get(video_type, verb_lists["movie"])
    prompt = get_user_prompt(mode=mode, prompt_idx=0, verb_list=verb_list, 
                            text_pred=stage1_description, word_limit=word_limit, examples=examples or [],
                            lang=lang)
    
    if backend.startswith("gemini"):
        model = backend  # Use the full model name from dropdown
        ad_text = _call_gemini(prompt, api_key, model, max_retries, retry_delay, usage_acc=usage_acc)
    elif backend == "qwen":
        model = QWEN_MODEL
        ad_text = _call_openai(prompt, api_key, model, max_retries, retry_delay, base_url=QWEN_API_URL, usage_acc=usage_acc)
    elif backend == "deepseek":
        model = DEEPSEEK_MODEL
        ad_text = _call_openai(prompt, api_key, model, max_retries, retry_delay, base_url=DEEPSEEK_API_URL, usage_acc=usage_acc)
    elif backend in ("openai", "openai-compatible"):
        model = openai_model or model or "gpt-latest"
        ad_text = _call_openai(prompt, api_key, model, max_retries, retry_delay, base_url=openai_url, usage_acc=usage_acc)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    ad_text = _extract_ad_sentence(ad_text)
    if not ad_text.endswith("."):
        ad_text += "."
    return ad_text


def batch_summarize(stage1_descriptions: List[dict], api_key: str, backend: str = "gemini", 
                    video_type: str = "movie", examples: List[str] = None,
                    openai_url: str = None, openai_model: str = None,
                    usage_acc=None, lang: str = None) -> List[dict]:
    """Batch summarize multiple Stage 1 descriptions."""
    results = []
    for item in stage1_descriptions:
        duration = item["end"] - item["start"]
        # A range the user picked can be arbitrarily short, and duration/speed
        # would then ask for a one- or two-word AD sentence, which is useless.
        word_limit = estimate_word_limit(duration, video_type, lang)
        if item.get("mode") == "custom":
            word_limit = max(CUSTOM_MIN_WORD_LIMIT, word_limit)
        try:
            ad_sentence = summarize_to_ad(item["description"], api_key, backend, 
                                         duration_seconds=duration, video_type=video_type, examples=examples,
                                         word_limit=word_limit,
                                         openai_url=openai_url, openai_model=openai_model, usage_acc=usage_acc,
                                         lang=lang)
        except Exception as e:
            logger.warning(f"summarize_to_ad failed for shot {item['shot_id']}: {e}", exc_info=True)
            ad_sentence = ""
        results.append({"shot_id": item["shot_id"], "start": item["start"], 
                       "end": item["end"], "ad_sentence": ad_sentence})
    if usage_acc is not None:
        logger.info(f"Stage 2 token usage: {usage_acc.summary()}")
    return results
