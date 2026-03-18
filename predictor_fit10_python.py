import json
import math
from pathlib import Path
from typing import Dict, Any

MODEL_PATH = Path(__file__).with_name("fit10_export_for_python.json")
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    MODEL_EXPORT = json.load(f)

COEFS = MODEL_EXPORT["coefficients"]
BASELINE_SURVIVAL = MODEL_EXPORT["baseline_survival_at_lp0"]
FACTOR_LEVELS = MODEL_EXPORT["factor_levels"]

# 说明：
# 1) 新模型已加入 disease_duration_baseline（起病至基线病程，单位：年）。
# 2) 这里仍然读取导出的 centered linear predictor 常数。
# 3) 你需要使用“包含 disease_duration_baseline 系数”的最新 JSON 文件替换同名文件。
LP_CENTERING_CONSTANT = MODEL_EXPORT.get("lp_centering_constant", MODEL_EXPORT.get("LP_CENTERING_CONSTANT", 0.0))

REQUIRED_FIELDS = [
    "Age_at_onset",
    "disease_duration_baseline",
    "GBA1_mutation",
    "T2D",
    "DBS",
    "UPDRS_Part_III",
    "HY_Stage",
    "Falls",
    "Depression",
    "Cognitive_dysfunction",
]


def validate_payload(payload: Dict[str, Any]) -> list[str]:
    missing = [k for k in REQUIRED_FIELDS if payload.get(k) in [None, "", []]]
    return missing


def _assert_model_ready() -> None:
    if "disease_duration_baseline" not in COEFS:
        raise ValueError(
            "当前 fit10_export_for_python.json 仍是旧版本，缺少 disease_duration_baseline 系数。"
            "请替换为包含新模型系数的 JSON 文件后再部署。"
        )


def _assert_allowed(payload: Dict[str, Any]) -> None:
    _assert_model_ready()
    for field, levels in FACTOR_LEVELS.items():
        if payload[field] not in levels:
            raise ValueError(f"{field} 取值非法：{payload[field]}，允许值为 {levels}")
    float(payload["disease_duration_baseline"])
    float(payload["UPDRS_Part_III"])


def compute_raw_score(payload: Dict[str, Any]) -> float:
    _assert_allowed(payload)
    s = 0.0
    if payload["Age_at_onset"] == ">50":
        s += COEFS["Age_at_onset=>50"]

    s += float(payload["disease_duration_baseline"]) * COEFS["disease_duration_baseline"]

    if payload["GBA1_mutation"] == "Yes":
        s += COEFS["GBA1_mutation=Yes"]
    if payload["T2D"] == "Yes":
        s += COEFS["T2D=Yes"]
    if payload["DBS"] == "Yes":
        s += COEFS["DBS=Yes"]

    s += float(payload["UPDRS_Part_III"]) * COEFS["UPDRS_Part_III"]

    hy_stage = payload["HY_Stage"]
    if hy_stage != "1":
        s += COEFS[f"HY_Stage={hy_stage}"]

    if payload["Falls"] == "Yes":
        s += COEFS["Falls=Yes"]
    if payload["Depression"] == "Yes":
        s += COEFS["Depression=Yes"]
    if payload["Cognitive_dysfunction"] == "Yes":
        s += COEFS["Cognitive_dysfunction=Yes"]
    return s


def compute_linear_predictor(payload: Dict[str, Any]) -> float:
    return compute_raw_score(payload) - float(LP_CENTERING_CONSTANT)


def predict_fit10_python(payload: Dict[str, Any]) -> Dict[str, Any]:
    missing = validate_payload(payload)
    if missing:
        raise ValueError(f"以下字段缺失，无法计算风险：{', '.join(missing)}")

    lp = compute_linear_predictor(payload)
    hr = math.exp(lp)

    surv_3y = BASELINE_SURVIVAL["3y"] ** hr
    surv_5y = BASELINE_SURVIVAL["5y"] ** hr
    surv_7y = BASELINE_SURVIVAL["7y"] ** hr

    return {
        "input": payload,
        "predictions": {
            "linear_predictor": round(lp, 4),
            "survival_3y": round(surv_3y, 4),
            "survival_5y": round(surv_5y, 4),
            "survival_7y": round(surv_7y, 4),
            "risk_3y": round(1 - surv_3y, 4),
            "risk_5y": round(1 - surv_5y, 4),
            "risk_7y": round(1 - surv_7y, 4),
        },
    }
