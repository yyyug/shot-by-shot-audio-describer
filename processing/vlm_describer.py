"""
VLM Describer - supports Gemini, Qwen, DeepSeek, and OpenAI-compatible backends
"""
import base64
import time
import logging
import requests
from typing import List, Optional, Dict
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

# Import original repo prompt class
from stage1.promptloader import PromptLoader

# API URLs
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3.8-flash"
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"


def _get_api_common():
    import processing._api_common as m
    return m


def build_film_grammar_prompt(
    video_type: str = "movie",
    label_type: str = "circles",
    char_text: str = "",
    current_shots: List[int] = None,
    threads: List[List[int]] = None,
    shot_scales: List[int] = None,
    prompt_variant: int = None,
    custom_opening: str = None,
    lang: str = None
) -> str:
    """
    Build prompt with film grammar information.
    Uses original repo's PromptLoader class.

    ``lang`` ("zh") renders the prompt in Traditional Chinese; anything else
    (including None) keeps the original English wording.
    """
    prompt_loader = PromptLoader(
        prompt_idx=prompt_variant or 0,
        video_type=video_type,
        label_type=label_type,
        custom_opening=custom_opening,
        lang=lang
    )
    
    prompt = prompt_loader.apply(
        char_text=char_text,
        current_shots=current_shots or [],
        threads=threads or [],
        shot_scales=shot_scales or [2]
    )
    
    return prompt


def _build_prompt(frames_base64, prompt, film_grammar):
    """Build prompt from film grammar if not provided."""
    if prompt is None and film_grammar:
        prompt = build_film_grammar_prompt(**film_grammar)
    elif prompt is None:
        prompt = (
            "Please describe what happened in this video clip in the following steps:\n"
            "1. Identify main characters\n"
            "2. Describe the actions of characters\n"
            "3. Describe the interactions between characters\n"
            "4. Describe the environment\n"
            "Note: Focus on movements and interactions. Do not hallucinate information.\n"
            "Provide the result in Traditional Chinese."
        )
    return prompt


def _call_gemini(frames_base64, api_key, prompt, model=GEMINI_DEFAULT_MODEL, max_retries=3, retry_delay=1.0, usage_acc=None):
    """Call Gemini API using google-genai library.

    Shares the pause-on-quota / provider-backoff / 401-give-up behaviour with
    the stage-2 summarizer so all workers across all tasks back off together
    on a shared free-tier quota."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    common = _get_api_common()

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    # Build content with images in correct format
    content = [prompt]
    for frame_b64 in frames_base64:
        img_bytes = base64.b64decode(frame_b64)
        content.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    logger.info(f"Gemini API: model={model}, frames={len(frames_base64)}, prompt_len={len(prompt)}")

    # Gemini 3.5 Flash-Lite supports thinking_level (minimal/low/medium/high).
    # Stage-1 shot descriptions benefit from deeper visual reasoning, so force
    # 'high'. Other models keep their default thinking behaviour.
    config = None
    if model and "flash-lite" in model:
        try:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high")
            )
            logger.info(f"Gemini API: thinking_level=high for model {model}")
        except Exception as e:
            logger.warning(f"Gemini API: could not set thinking_level=high ({e}); using model default")

    crisis_slept = False
    for attempt in range(max_retries):
        try:
            common.wait_if_paused()
            logger.info(f"Gemini API attempt {attempt+1}/{max_retries}...")
            response = client.models.generate_content(model=model, contents=content, config=config)
            result = response.text.strip() if response.text else ""
            if usage_acc is not None:
                usage_acc.add(getattr(response, "usage_metadata", None))
            logger.info(f"Gemini API success: {len(result)} chars")
            return result
        except Exception as e:
            status = common.extract_status(e)
            logger.error(f"Gemini API attempt {attempt+1} failed (status={status or 'n/a'}): {e}")
            if common.classify_crisis(status):
                common.pause_all()
                delay = common.compute_sleep(status, e, attempt, base_delay=retry_delay)
                common.pause_and_wait(delay)
                crisis_slept = True
            if common.should_give_up(status):
                raise RuntimeError(f"Gemini API failed (permanent error, status {status}): {e}")
            if attempt < max_retries - 1:
                if crisis_slept:
                    logger.info("Gemini API retry after provider-required wait")
                    continue
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


def _call_openai(frames_base64, api_key, prompt, model="gpt-latest", max_retries=3, retry_delay=1.0, base_url=None, usage_acc=None):
    """Call OpenAI-compatible API."""
    common = _get_api_common()
    # Most providers cap images per request (DeepSeek allows up to 600). If a
    # unit produced more frames than the cap, keep even temporal coverage by
    # subsampling instead of sending every frame and getting a 400.
    MAX_IMAGES = 600
    if len(frames_base64) > MAX_IMAGES:
        import numpy as _np
        idxs = set(_np.linspace(0, len(frames_base64) - 1, MAX_IMAGES, dtype=int))
        frames_base64 = [f for i, f in enumerate(frames_base64) if i in idxs]
        logger.info(f"OpenAI API: capped frames to {len(frames_base64)} (provider limit {MAX_IMAGES})")
    content = [{"type": "text", "text": prompt}]
    for frame_b64 in frames_base64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "low"}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
    }
    # No max_tokens: reasoning-mode models (qwen3.8-flash, DeepSeek thinking,
    # etc.) share the output budget between chain-of-thought and the final
    # answer; a small cap gets burned on reasoning and content comes back
    # empty (HTTP 200). Let the provider use its own default instead.
    _apply_no_thinking(payload, base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url}/chat/completions" if base_url else OPENAI_API_URL
    logger.info(f"OpenAI API: url={url}, model={model}, frames={len(frames_base64)}, prompt_len={len(prompt)}")

    crisis_slept = False
    for attempt in range(max_retries):
        try:
            common.wait_if_paused()
            logger.info(f"OpenAI API attempt {attempt+1}/{max_retries} -> POST {url} (model={model})")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            if usage_acc is not None:
                usage_acc.add(data.get("usage"))
            return data["choices"][0]["message"]["content"] or ""
        except requests.exceptions.RequestException as e:
            status = common.extract_status(e)
            logger.error(f"OpenAI API attempt {attempt+1} failed (status={status or 'n/a'}) url={url} model={model}: {e}")
            body = common.response_body(e)
            if body:
                logger.error(f"OpenAI API response body: {body}")
            if common.classify_crisis(status):
                common.pause_all()
                delay = common.compute_sleep(status, e, attempt, base_delay=retry_delay)
                common.pause_and_wait(delay)
                crisis_slept = True
            if common.should_give_up(status):
                raise RuntimeError(f"OpenAI API failed (permanent error, status {status}): {e}")
            if attempt < max_retries - 1:
                if crisis_slept:
                    logger.info("OpenAI API retry after provider-required wait")
                    continue
                delay = common.compute_sleep(status, e, attempt, base_delay=retry_delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"OpenAI API failed: {e}")


def describe_frames(
    frames_base64: List[str],
    api_key: str,
    backend: str = "gemini",
    model: str = None,
    prompt: str = None,
    film_grammar: Dict = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    openai_url: str = None,
    openai_model: str = None,
    usage_acc=None
) -> str:
    """
    Describe video frames using specified backend.
    
    Args:
        frames_base64: List of base64-encoded JPEG images
        api_key: API key for the selected backend
        backend: "gemini", "qwen", "deepseek", or "openai-compatible"
        model: Model identifier (uses backend default if None)
        prompt: Custom prompt (overrides film_grammar)
        film_grammar: Dict with film grammar parameters
        max_retries: Number of retry attempts
        retry_delay: Delay between retries
        openai_url: Custom API base URL for OpenAI-compatible
        openai_model: Custom model name for OpenAI-compatible
        
    Returns:
        Description string
    """
    if not frames_base64:
        return ""
    
    prompt = _build_prompt(frames_base64, prompt, film_grammar)
    
    if backend == "gemini":
        model = model or GEMINI_DEFAULT_MODEL
        return _call_gemini(frames_base64, api_key, prompt, model, max_retries, retry_delay, usage_acc=usage_acc)
    elif backend == "qwen":
        model = model or QWEN_MODEL
        return _call_openai(frames_base64, api_key, prompt, model, max_retries, retry_delay, base_url=QWEN_API_URL, usage_acc=usage_acc)
    elif backend == "deepseek":
        model = model or DEEPSEEK_MODEL
        return _call_openai(frames_base64, api_key, prompt, model, max_retries, retry_delay, base_url=DEEPSEEK_API_URL, usage_acc=usage_acc)
    elif backend == "openai-compatible":
        model = openai_model or "gpt-4o"
        return _call_openai(frames_base64, api_key, prompt, model, max_retries, retry_delay, base_url=openai_url, usage_acc=usage_acc)
    else:
        raise ValueError(f"Unknown backend: {backend}")
