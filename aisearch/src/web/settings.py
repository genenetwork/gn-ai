import requests


def call_claude(api_key):
    """Validate an Anthropic API key with a tiny test request."""
    if not api_key:
        raise requests.exceptions.RequestException("API key is missing")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello world!"}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raises an exception for HTTP errors

def configure(lm, api_key, model):
    lm.kwargs["api_key"] = api_key
    lm.model = model
    return lm
