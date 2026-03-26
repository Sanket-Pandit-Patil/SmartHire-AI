import os
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def _generate_with_ollama(prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "llama3")
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response returned by Ollama.")
    except Exception as e:
        return f"Ollama Error: {str(e)}"


def _generate_with_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        return "GROQ_API_KEY is missing. Add it to your environment or Streamlit secrets."

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI career assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content or "No response returned by Groq."
    except Exception as e:
        return f"Groq Error: {str(e)}"


def generate_llm_response(prompt: str) -> str:
    provider = _get_provider()

    if provider == "ollama":
        return _generate_with_ollama(prompt)
    elif provider == "groq":
        return _generate_with_groq(prompt)
    else:
        return (
            f"Unsupported LLM_PROVIDER='{provider}'. "
            "Use 'ollama' for local or 'groq' for hosted."
        )