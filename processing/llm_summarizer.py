"""
LLM Summarizer - supports Gemini, OpenRouter, and OpenAI backends
"""
import os
import random
import pandas as pd
import requests
import time
from typing import List, Optional

# Import original repo prompt function
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage2'))

try:
    from promptloader import get_user_prompt
except ImportError:
    def get_user_prompt(mode, prompt_idx, verb_list, text_pred, word_limit, examples):
        return f"Summarize: {text_pred} in {word_limit} words."

# API URLs
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Dataset-specific verb lists (from original repo)
VERB_LISTS = {
    "movie": ['look', 'turn', 'take', 'hold', 'pull', 'walk', 'run', 'watch', 'stare', 'grab', 'fall', 'get', 'go', 'open', 'smile'],
    "tv_series": ['look', 'walk', 'turn', 'stare', 'take', 'hold', 'smile', 'leave', 'pull', 'watch', 'open', 'go', 'step', 'get', 'enter'],
    "stage_performance": ['sing', 'dance', 'perform', 'move', 'gesture', 'walk', 'turn', 'look', 'smile', 'wave', 'jump', 'spin', 'play', 'hold', 'raise']
}

# Words per second (from original repo)
AD_SPEED = {
    "movie": 0.275,        # cmdad dataset
    "tv_series": 0.2695,   # tvad dataset
    "stage_performance": 0.2695  # use tvad speed
}

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


def estimate_word_limit(duration_seconds: float, video_type: str = "movie") -> int:
    """Estimate word limit based on AD interval duration (from original repo formula)."""
    speed = AD_SPEED.get(video_type, 0.275)
    return max(1, round(duration_seconds / speed))


def sample_few_shot_examples(video_type: str, duration_seconds: float, num_examples: int = 10) -> List[str]:
    """Sample few-shot examples with similar word count from training data."""
    examples = GT_EXAMPLES.get(video_type, [])
    if not examples:
        return []
    
    # Calculate target word count
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


def _call_gemini(prompt, api_key, model="gemini-3.5-flash", max_retries=3, retry_delay=1.0):
    """Call Gemini API using google-genai library."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")
    
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini API failed: {e}")


def _call_openrouter(prompt, api_key, model="qwen/qwen3.7-plus", max_retries=3, retry_delay=1.0):
    """Call OpenRouter API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.6
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter API failed: {e}")


def _call_openai(prompt, api_key, model="gpt-latest", max_retries=3, retry_delay=1.0):
    """Call OpenAI API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.6
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(f"OpenAI API failed: {e}")


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
    retry_delay: float = 1.0
) -> str:
    """Summarize Stage 1 description into concise AD sentence."""
    if not stage1_description:
        return ""
    
    # Calculate word limit from duration if provided
    if duration_seconds is not None and word_limit is None:
        word_limit = estimate_word_limit(duration_seconds, video_type)
    elif word_limit is None:
        word_limit = 15
    
    # Sample few-shot examples if not provided
    if examples is None and duration_seconds is not None:
        examples = sample_few_shot_examples(video_type, duration_seconds)
    
    verb_list = VERB_LISTS.get(video_type, VERB_LISTS["movie"])
    prompt = get_user_prompt(mode=mode, prompt_idx=0, verb_list=verb_list, 
                            text_pred=stage1_description, word_limit=word_limit, examples=examples or [])
    
    if backend.startswith("gemini"):
        model = backend  # Use the full model name from dropdown
        ad_text = _call_gemini(prompt, api_key, model, max_retries, retry_delay)
    elif backend == "openrouter":
        model = model or "qwen/qwen3.7-plus"
        ad_text = _call_openrouter(prompt, api_key, model, max_retries, retry_delay)
    elif backend == "openai":
        model = model or "gpt-latest"
        ad_text = _call_openai(prompt, api_key, model, max_retries, retry_delay)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Parse JSON output if present
    if '{"summarized_AD":' in ad_text:
        import json
        try:
            ad_text = json.loads(ad_text).get("summarized_AD", ad_text)
        except:
            pass
    
    ad_text = ad_text.strip('"').strip("'")
    if not ad_text.endswith("."):
        ad_text += "."
    return ad_text


def batch_summarize(stage1_descriptions: List[dict], api_key: str, backend: str = "gemini", 
                    video_type: str = "movie", examples: List[str] = None) -> List[dict]:
    """Batch summarize multiple Stage 1 descriptions."""
    results = []
    for item in stage1_descriptions:
        duration = item["end"] - item["start"]
        try:
            ad_sentence = summarize_to_ad(item["description"], api_key, backend, 
                                         duration_seconds=duration, video_type=video_type, examples=examples)
        except Exception:
            ad_sentence = ""
        results.append({"shot_id": item["shot_id"], "start": item["start"], 
                       "end": item["end"], "ad_sentence": ad_sentence})
    return results
