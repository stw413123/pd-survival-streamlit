import os
import streamlit as st


def _get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        if value is None:
            return default
        return str(value).strip()
    except Exception:
        return os.getenv(name, default).strip()


DEEPSEEK_API_KEY = _get_secret("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _get_secret("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = _get_secret("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TIMEOUT = int(_get_secret("LLM_TIMEOUT", "45") or "45")


def deepseek_ready() -> bool:
    return bool(DEEPSEEK_API_KEY)
