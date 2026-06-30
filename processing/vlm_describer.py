import base64
import requests
import time
from typing import List, Optional

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def describe_frames(
    frames_base64: List[str],
    api_key: str,
    model: str = "qwen/qwen-2.5-vl-7b-instruct:free",
    prompt: str = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Describe video frames using OpenRouter API with Qwen 2.5 VL.

    Args:
        frames_base64: List of base64-encoded JPEG images
        api_key: OpenRouter API key
        model: Model identifier (default: qwen/qwen-2.5-vl-7b-instruct:free)
        prompt: Custom prompt (default uses project template)
        max_retries: Number of retry attempts on failure
        retry_delay: Delay between retries in seconds

    Returns:
        Description string
    """
    if not frames_base64:
        return ""

    if prompt is None:
        prompt = (
            "Please describe what happened in this video clip in the following steps:\n"
            "1. Identify main characters\n"
            "2. Describe the actions of characters\n"
            "3. Describe the interactions between characters\n"
            "4. Describe the environment\n"
            "Note: Focus on movements and interactions. Do not hallucinate information."
        )

    # Build content with images
    content = [{"type": "text", "text": prompt}]

    for frame_b64 in frames_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}"
            }
        })

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 512
    }

    # Retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter API failed after {max_retries} attempts: {e}")
