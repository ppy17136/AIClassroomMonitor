import os
import json
import time
import requests
import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image

# -------------------- 基础配置 --------------------
st.set_page_config(page_title="AI课堂状态监测与智能反馈", layout="wide")

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问兼容 OpenAI 的接口
MODEL_NAME = "qwen-max"  # 你也可以换成你账号有权限的模型，比如 qwen-turbo / qwen-plus

def state_to_color(attention: str) -> str:
    if attention == "专注":
        return "#2ecc71"   # green
    elif attention == "需要关注":
        return "#f1c40f"   # yellow
    else:
        return "#e74c3c"   # red

def get_qwen_api_key() -> str:
    # 优先读取 Streamlit Cloud Secrets，其次读取环境变量
    return st.secrets.get("QWEN_API_KEY", os.environ.get("QWEN_API_KEY", "")).strip()

QWEN_API_KEY = get_qwen_api_key()
if not QWEN_API_KEY:
    st.error("未配置 QWEN_API_KEY。请在 Streamlit Cloud 的 Secrets 中设置：\n\nQWEN_API_KEY=\"sk-你的真实Key\"")
    st.stop()

# -------------------- LLM 调用（做缓存，避免重复烧 API）--------------------
@st.cache_data(ttl=300, show_spinner=False)
def llm_feedback_cached(prompt: str) -> str:
    """对同一个 prompt 做 5 分钟缓存，避免页面重绘时重复请求。"""
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个智慧课堂助理，回答要具体、可执行、适合教师课堂使用。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 256
    }
    resp = requests.post(BASE_URL + "/chat/completions", headers=headers, json=data, timeout=20)

    # 尽量给出可读错误信息
    if resp.status_code != 200:
        raise RuntimeError(f"LLM接口返回非200：{resp.status_code}，body={resp.text[:300]}")

    j = resp.json()
    return j["choices"][0]["message"]["content"]

def build_prompt(context, student_name, emotion, attention, history, custom_question=None) -> str:
    prompt = (
        "教师在课堂上关注学生状态。请结合下述信息生成自然语言反馈和改进建议：\n"
        f"- 学生姓名：{student_name}\n"
        f"- 当前情绪：{emotion}\n"
        f"- 当前专注度状态：{attention}\n"
        f"- 学生历史状态：{history}\n"
        f"- 当前课堂内容：{context}\n"
    )
    if custom_question:
        prompt += f"\n教师提问：{custom_question}\n"
    prompt += "\n输出要求：\n1) 用简短要点列出判断依据；2) 给出3条可执行建议；3) 语气友好专业。"
    return prompt

def safe_llm_feedback(context, student_name, emotion, attention, history, custom_question=None) -> str:
    prompt = build_prompt(context, student_name, emotion, attention, history, custom_question)
    try:
        return llm_feedback_cached(prompt)
    except Exception:
        return "（大语言模型反馈调用失败：请检查 API Key、模型权限或网络；也可能是调用频率过高。）"

# -------------------- UI --------------------
st.title("AI课堂状态实时监测与智能反馈系统")
st.markdown("本系统支持上传班级照片进行人脸检测与状态标注，并生成个性化教学反馈建议。")

# 教师端配置
classroom_context = st.text_input("当前课堂内容描述", value="微积分第3章：链式法则讲解")

st.sidebar.header("图片采集")
st.sidebar.info("提示：Streamlit Cloud 上无法使用你的本地摄像头，建议使用“上传照片”模式。")

use_camera = st.sidebar.checkbox("启用本地摄像头（仅本地运行有效）", value=False)
uploaded_image = st.sidebar.file_uploader("上传班级照片", type=["jpg", "jpeg", "png"])

frame_rgb = None

if use_camera:
    cap = cv2.VideoCapture(0)
    ret, frame_bgr = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame_bgr, caption="摄像头采集画面", channels="BGR")
    else:
        st.error("摄像头采集失败（云端环境通常不可用，请改用上传照片）。")

elif uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    frame_rgb = np.array(img)
    st.image(frame_rgb, caption="上传图片（RGB）")

# -------------------- 人脸检测（MediaPipe）--------------------
student_status = []
draw_frame_bgr = None

if frame_rgb is not None:
    h, w = frame_rgb.shape[:2]
    draw_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()

    mp_fd = mp.solutions.face_detection
    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        results = fd.process(frame_rgb)
        dets = results.detections or []

        for i, det in enumerate(dets):
            score = float(det.score[0]) if det.score else 0.0
            box = det.location_data.relative_bounding_box

            left = max(0, int(box.xmin * w))
            top = max(0, int(box.ymin * h))
            right = min(w, int((box.xmin + box.width) * w))
            bottom = min(h, int((box.ymin + box.height) * h))

            # 防止异常框
            if right <= left or bottom <= top:
                continue

            match_name = f"Stu{i+1}"

            # --- 情绪：先做 Demo（可部署、可演示） ---
            # 你后续如果要接入真实情绪模型，可以在这里替换
            emotion = "neutral"

            # --- 注意力：用规则（可解释 + 稳定）---
            # 这里用检测置信度和人脸框大小做简单启发式
            face_area = (right - left) * (bottom - top)
            area_ratio = face_area / max(1, (w * h))

            if score >= 0.85 and area_ratio >= 0.01:
                attention = "专注"
            elif score >= 0.65:
                attention = "需要关注"
            else:
                attention = "状态不佳"

            color = state_to_color(attention)

            student_status.append({
                "name": match_name,
                "face_box": (top, right, bottom, left),
                "emotion": emotion,
                "attention": attention,
                "color": color,
                "history": []
            })

            # 画框与编号
            cv2.rectangle(draw_frame_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(draw_frame_bgr, match_name, (left, max(0, top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    st.image(draw_frame_bgr, caption="检测结果（人脸已编号）", channels="BGR")

# -------------------- 状态表 --------------------
st.header("班级学生状态监控（实时）")

if not student_status:
    st.info("未检测到人脸：请上传更清晰、人物正面较多的班级照片。")
    st.stop()

cols = st.columns(len(student_status))
for i, s in enumerate(student_status):
    with cols[i]:
        st.markdown(
            f"""
            <div style="background:{s['color']}; padding:10px; border-radius:12px; text-align:center;">
                <b>{s['name']}</b><br>
                情绪：{s['emotion']}<br>
                专注度：{s['attention']}
            </div>
            """,
            unsafe_allow_html=True
        )
        if s["attention"] == "状态不佳":
            st.warning(f"{s['name']} 状态不佳，已自动提醒！")

# -------------------- AI 个性化反馈（只生成一次并复用，避免烧 API）--------------------
st.header("AI个性化反馈/智能分析建议")

feedback_cache = {}
with st.spinner("正在生成AI分析（已启用缓存，避免重复调用）..."):
    for s in student_status:
        feedback_cache[s["name"]] = safe_llm_feedback(
            classroom_context, s["name"], s["emotion"], s["attention"], s["history"]
        )

for s in student_status:
    with st.expander(f"{s['name']} - 详细AI分析", expanded=False):
        st.markdown(feedback_cache.get(s["name"], "（暂无反馈）"))

# -------------------- 教师与AI交互 --------------------
st.header("教师与AI交互")

stu_names = [s["name"] for s in student_status]
selected_student = st.selectbox("选择学生", options=stu_names)
custom_q = st.text_input("向AI系统提问", value="该生当前学习状态及改进建议？")

if st.button("获取AI分析和建议"):
    s = next((x for x in student_status if x["name"] == selected_student), None)
    if s:
        resp = safe_llm_feedback(
            classroom_context, s["name"], s["emotion"], s["attention"], s["history"], custom_q
        )
        st.success(resp)

# -------------------- 教师纠正/优化（演示用） --------------------
st.header("教师纠正/优化AI模型（演示）")
err_stu = st.selectbox("如需纠正，请选择学生", options=stu_names, key="correct_stu")
err_correct = st.text_area("请描述纠正意见/实际状态（将记录用于后续改进）", "")

if st.button("提交反馈优化AI"):
    st.info("您的反馈已记录（演示版本不做持久化存储）。")

st.caption("注：本版本为云端可部署演示版（MediaPipe 人脸检测 + 千问生成建议）。")
