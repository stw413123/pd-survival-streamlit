import requests

from cloud_config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TIMEOUT,
)
from prompts import PD_QA_SYSTEM_PROMPT


def ask_pd_education_question(question: str, history: list | None = None) -> dict:
    if not DEEPSEEK_API_KEY:
        return {"ok": False, "error": "未检测到 DEEPSEEK_API_KEY 或 Streamlit secrets 配置。", "answer": None}

    history = history or []
    trimmed_history = history[-6:] if len(history) > 6 else history

    messages = [{"role": "system", "content": PD_QA_SYSTEM_PROMPT}]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question.strip()})

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return {"ok": True, "error": None, "answer": content}
    except requests.HTTPError as e:
        return {"ok": False, "error": f"DeepSeek 接口报错：{str(e)}；响应内容：{getattr(e.response, 'text', '')[:500]}", "answer": None}
    except Exception as e:
        return {"ok": False, "error": f"科普问答失败：{str(e)}", "answer": None}
