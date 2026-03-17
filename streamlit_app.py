import pandas as pd
import streamlit as st

from cloud_config import deepseek_ready
from llm_extract_cloud import extract_variables_from_text
from predictor_fit10_python import predict_fit10_python, validate_payload

st.set_page_config(page_title="PD生存风险预测平台", layout="wide")

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

DEFAULT_PAYLOAD = {
    "Age_at_onset": ">50",
    "GBA1_mutation": "No",
    "T2D": "No",
    "DBS": "No",
    "UPDRS_Part_III": 0.0,
    "HY_Stage": "3",
    "Falls": "No",
    "Depression": "No",
    "Cognitive_dysfunction": "No",
}


def init_session_state():
    for k, v in DEFAULT_PAYLOAD.items():
        if k not in st.session_state:
            st.session_state[k] = v

    defaults = {
        "ai_result": None,
        "ai_raw_text": "",
        "ai_message": None,
        "pending_fill": None,
        "pending_predict": False,
        "latest_prediction": None,
        "latest_payload": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_pending_fill_if_any():
    pending = st.session_state.get("pending_fill")
    if not pending:
        return []

    changed = []
    for field, value in pending.items():
        if value is None:
            continue
        if field == "UPDRS_Part_III":
            try:
                value = float(value)
            except Exception:
                continue
        st.session_state[field] = value
        changed.append(field)

    st.session_state["pending_fill"] = None
    return changed


def show_flash_message():
    msg_obj = st.session_state.get("ai_message")
    if not msg_obj:
        return

    level, msg = msg_obj
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    elif level == "info":
        st.info(msg)
    else:
        st.info(msg)

    st.session_state["ai_message"] = None


def classify_risk(preds: dict) -> str:
    risk_7y = preds.get("risk_7y")
    if risk_7y is None:
        return "未知"
    if risk_7y < 0.20:
        return "低风险"
    if risk_7y < 0.50:
        return "中风险"
    return "高风险"


def get_current_payload_from_session():
    return {
        "Age_at_onset": st.session_state["Age_at_onset"],
        "GBA1_mutation": st.session_state["GBA1_mutation"],
        "T2D": st.session_state["T2D"],
        "DBS": st.session_state["DBS"],
        "UPDRS_Part_III": float(st.session_state["UPDRS_Part_III"]),
        "HY_Stage": st.session_state["HY_Stage"],
        "Falls": st.session_state["Falls"],
        "Depression": st.session_state["Depression"],
        "Cognitive_dysfunction": st.session_state["Cognitive_dysfunction"],
    }


def show_prediction_result(result: dict):
    preds = result.get("predictions", {})

    st.markdown("## 二、预测结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("3年风险", f"{preds.get('risk_3y', 0):.4f}")
    c2.metric("5年风险", f"{preds.get('risk_5y', 0):.4f}")
    c3.metric("7年风险", f"{preds.get('risk_7y', 0):.4f}")

    risk_label = classify_risk(preds)
    st.info(f"当前总体风险等级：**{risk_label}**")

    rows = [
        ("线性预测值（LP）", preds.get("linear_predictor")),
        ("3年生存概率", preds.get("survival_3y")),
        ("5年生存概率", preds.get("survival_5y")),
        ("7年生存概率", preds.get("survival_7y")),
        ("3年风险", preds.get("risk_3y")),
        ("5年风险", preds.get("risk_5y")),
        ("7年风险", preds.get("risk_7y")),
    ]
    df = pd.DataFrame(rows, columns=["指标", "数值"])
    st.dataframe(df, use_container_width=True)

    st.markdown("## 三、基础解释")
    st.write(f"总体风险等级：{risk_label}")
    st.write("解释说明：")
    st.write("该结果由 fit10 Cox 主模型的 Python 复现版计算生成。")
    st.write("当前平台仅用于科研与辅助评估，不替代临床诊断和治疗决策。")


def build_pending_fill_from_ai(data: dict):
    pending_fill = {}
    for field in REQUIRED_FIELDS:
        value = data.get(field, None)
        if value is not None:
            pending_fill[field] = value
    return pending_fill


# -----------------------------
# 初始化
# -----------------------------
init_session_state()
applied_now = apply_pending_fill_if_any()

# 如果上一轮要求“回填后立即预测”，则在控件渲染前执行
if st.session_state["pending_predict"]:
    try:
        auto_payload = get_current_payload_from_session()
        missing_after_fill = validate_payload(auto_payload)
        if missing_after_fill:
            st.session_state["latest_prediction"] = None
            st.session_state["latest_payload"] = auto_payload
            st.session_state["ai_message"] = (
                "warning",
                f"AI 已完成回填，但仍缺失字段：{', '.join(missing_after_fill)}，因此不能自动预测。",
            )
        else:
            auto_result = predict_fit10_python(auto_payload)
            st.session_state["latest_prediction"] = auto_result
            st.session_state["latest_payload"] = auto_payload
            st.session_state["ai_message"] = (
                "success",
                "AI 已完成提取、回填并自动预测。请核对结果。",
            )
    except Exception as e:
        st.session_state["latest_prediction"] = None
        st.session_state["ai_message"] = ("error", f"自动预测失败：{str(e)}")
    finally:
        st.session_state["pending_predict"] = False


# -----------------------------
# 页面头部
# -----------------------------
st.title("PD生存风险预测平台")
st.caption("单个 Streamlit 应用版本：已完成云端部署测试")

with st.sidebar:
    st.write("预测内核：Python 版 fit10")
    if deepseek_ready():
        st.success("DeepSeek API Key 已检测到")
    else:
        st.warning("未检测到 DeepSeek API Key，AI 提取功能不可用")
    st.info("请仅输入脱敏病例摘要。")

show_flash_message()

if applied_now:
    st.info("本轮已自动回填字段：" + ", ".join(applied_now))

st.markdown("---")

# -----------------------------
# 一、结构化变量输入
# -----------------------------
st.markdown("## 一、结构化变量输入")

col1, col2 = st.columns(2)
with col1:
    age = st.selectbox("Age at onset", ["≤50", ">50"], key="Age_at_onset")
    gba1 = st.selectbox("GBA1 mutation", ["No", "Yes"], key="GBA1_mutation")
    t2d = st.selectbox("T2D", ["No", "Yes"], key="T2D")
    dbs = st.selectbox("DBS", ["No", "Yes"], key="DBS")
    updrs = st.number_input("UPDRS Part III", min_value=0.0, step=1.0, key="UPDRS_Part_III")
with col2:
    hy = st.selectbox("H&Y Stage", ["1", "2", "2.5", "3", "4", "5"], key="HY_Stage")
    falls = st.selectbox("Falls", ["No", "Yes"], key="Falls")
    depression = st.selectbox("Depression", ["No", "Yes"], key="Depression")
    cog = st.selectbox("Cognitive dysfunction", ["No", "Yes"], key="Cognitive_dysfunction")

payload = {
    "Age_at_onset": age,
    "GBA1_mutation": gba1,
    "T2D": t2d,
    "DBS": dbs,
    "UPDRS_Part_III": float(updrs),
    "HY_Stage": hy,
    "Falls": falls,
    "Depression": depression,
    "Cognitive_dysfunction": cog,
}

if st.button("开始预测", type="primary"):
    missing_fields = validate_payload(payload)
    if missing_fields:
        st.error(f"以下字段缺失，无法计算风险：{', '.join(missing_fields)}")
        st.session_state["latest_prediction"] = None
        st.session_state["latest_payload"] = payload
    else:
        try:
            result = predict_fit10_python(payload)
            st.session_state["latest_prediction"] = result
            st.session_state["latest_payload"] = payload
        except Exception as e:
            st.error(f"预测失败：{str(e)}")
            st.session_state["latest_prediction"] = None
            st.session_state["latest_payload"] = payload

if st.session_state["latest_prediction"] is not None:
    show_prediction_result(st.session_state["latest_prediction"])

st.markdown("---")

# -----------------------------
# 四、AI 智能录入
# -----------------------------
st.markdown("## 四、AI 智能录入")
st.warning("请仅输入脱敏后的病例摘要，不要输入姓名、住院号、身份证号、联系方式、详细住址等身份信息。")

case_text = st.text_area(
    "请输入脱敏病例描述",
    value=st.session_state["ai_raw_text"],
    height=180,
    placeholder="例如：患者发病年龄大于50岁，GBA1阳性，T2D阳性，未行DBS，UPDRS III 45分，H&Y 3期，有跌倒史，无抑郁，无认知障碍。",
)
st.session_state["ai_raw_text"] = case_text

col_ai1, col_ai2 = st.columns(2)
with col_ai1:
    btn_fill_only = st.button("AI提取并自动回填")
with col_ai2:
    btn_fill_and_predict = st.button("AI提取、回填并预测")

if btn_fill_only or btn_fill_and_predict:
    if not case_text.strip():
        st.session_state["ai_message"] = ("error", "请先输入病例描述。")
        st.rerun()

    if not deepseek_ready():
        st.session_state["ai_message"] = ("error", "未检测到 DeepSeek API Key，当前无法使用 AI 提取功能。")
        st.rerun()

    with st.spinner("正在进行 AI 结构化提取..."):
        ai_result = extract_variables_from_text(case_text)

    if not ai_result["ok"]:
        st.session_state["ai_result"] = None
        st.session_state["ai_message"] = ("error", ai_result["error"])
        st.rerun()

    data = ai_result["data"]
    st.session_state["ai_result"] = data
    st.session_state["pending_fill"] = build_pending_fill_from_ai(data)

    if btn_fill_and_predict:
        st.session_state["pending_predict"] = True
        if data["can_predict"]:
            st.session_state["ai_message"] = ("success", "AI 提取完成，页面刷新后将自动回填并尝试预测。")
        else:
            msg = "AI 提取完成，页面刷新后将自动回填已识别字段。"
            if data.get("missing_fields"):
                msg += f" 当前仍缺失字段：{', '.join(data['missing_fields'])}，因此不能自动预测。"
            st.session_state["ai_message"] = ("warning", msg)
    else:
        st.session_state["pending_predict"] = False
        if data["can_predict"]:
            st.session_state["ai_message"] = ("success", "AI 提取完成，页面刷新后将自动回填字段。字段已完整，请核对后点击“开始预测”。")
        else:
            msg = "AI 提取完成，页面刷新后将自动回填已识别字段。"
            if data.get("missing_fields"):
                msg += f" 当前仍缺失字段：{', '.join(data['missing_fields'])}，因此不能计算风险。"
            st.session_state["ai_message"] = ("warning", msg)

    st.rerun()

if st.session_state["ai_result"] is not None:
    data = st.session_state["ai_result"]
    st.markdown("### 最近一次 AI 提取结果")
    st.json(data)

    if data["can_predict"]:
        st.success("字段已完整。你可以手动点击“开始预测”，或使用“一键提取、回填并预测”。")
    else:
        st.warning("信息不足，当前不能计算风险。系统不会补全，也不会编造。")

    if data.get("missing_fields"):
        st.error("缺失字段：" + ", ".join(data["missing_fields"]))
    if data.get("uncertainties"):
        st.info("存在歧义字段：" + ", ".join(data["uncertainties"]))

st.markdown("---")
st.caption("说明：AI 仅用于结构化提取与辅助录入，风险值由 Python 复现版 fit10 模型计算。")