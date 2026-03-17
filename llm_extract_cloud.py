import json
import re
from typing import Any, Dict

import requests

from cloud_config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TIMEOUT,
)
from prompts import SYSTEM_PROMPT

REQUIRED_FIELDS = [
    "Age_at_onset",
    "GBA1_mutation",
    "T2D",
    "DBS",
    "UPDRS_Part_III",
    "HY_Stage",
    "Falls",
    "Depression",
    "Cognitive_dysfunction",
]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    return json.loads(_strip_code_fence(text))


def _normalize_output(data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "Age_at_onset": data.get("Age_at_onset"),
        "GBA1_mutation": data.get("GBA1_mutation"),
        "T2D": data.get("T2D"),
        "DBS": data.get("DBS"),
        "UPDRS_Part_III": data.get("UPDRS_Part_III"),
        "HY_Stage": data.get("HY_Stage"),
        "Falls": data.get("Falls"),
        "Depression": data.get("Depression"),
        "Cognitive_dysfunction": data.get("Cognitive_dysfunction"),
        "missing_fields": data.get("missing_fields", []),
        "uncertainties": data.get("uncertainties", []),
        "can_predict": bool(data.get("can_predict", False)),
    }
    missing = [k for k in REQUIRED_FIELDS if result.get(k) in [None, "", []]]
    result["missing_fields"] = sorted(list(set(result.get("missing_fields", []) + missing)))
    result["can_predict"] = len(result["missing_fields"]) == 0
    return result


def extract_variables_from_text(case_text: str) -> Dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        return {"ok": False, "error": "未检测到 DEEPSEEK_API_KEY 或 Streamlit secrets 配置。", "data": None}

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请从下面病例描述中抽取 PD 生存风险模型变量，不要计算风险：\n\n{case_text.strip()}"},
        ],
        "temperature": 0,
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
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = _safe_json_loads(content)
        return {"ok": True, "error": None, "data": _normalize_output(parsed)}
    except requests.HTTPError as e:
        return {"ok": False, "error": f"DeepSeek 接口报错：{str(e)}；响应内容：{getattr(e.response, 'text', '')[:500]}", "data": None}
    except Exception as e:
        return {"ok": False, "error": f"AI 提取失败：{str(e)}", "data": None}
