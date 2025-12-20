import os
import json
import time
import requests
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps
import base64
import re

# -------------------- 基础配置 --------------------
st.set_page_config(page_title="AI课堂状态监测与智能反馈", layout="wide")

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问兼容 OpenAI 的接口
TEXT_MODEL = "qwen-max"        # 文本模型：用于生成教学建议
VISION_MODEL = "qwen-vl-plus"  # 视觉模型：用于判断专注度/情绪（可改 qwen-vl-max）

# -------------------- Key 读取 --------------------
def get_qwen_api_key() -> str:
    return st.secrets.get("QWEN_API_KEY", os.environ.get("QWEN_API_KEY", "")).strip()

QWEN_API_KEY = get_qwen_api_key()
if not QWEN_API_KEY:
    st.error('未配置 QWEN_API_KEY。请在 Streamlit Cloud 的 Secrets 中设置：\n\nQWEN_API_KEY="sk-你的真实Key"')
    st.stop()

# -------------------- 人脸检测：OpenCV DNN（Res10 SSD） --------------------
@st.cache_resource
def load_dnn_face_net():
    os.makedirs("models", exist_ok=True)
    proto_path = os.path.join("models", "deploy.prototxt")
    model_path = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")

    PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    def download(url, path):
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            return
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

    download(PROTO_URL, proto_path)
    download(MODEL_URL, model_path)

    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net

def detect_faces_dnn(frame_rgb: np.ndarray, conf_thr=0.65):
    h, w = frame_rgb.shape[:2]
    net = load_dnn_face_net()

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()

    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thr:
            continue

        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        fw, fh = x2 - x1, y2 - y1
        area_ratio = (fw * fh) / max(1, w * h)

        if fw < 40 or fh < 40:
            continue
        if area_ratio > 0.25:
            continue

        boxes.append((x1, y1, x2, y2, conf))

    # 让大脸优先，避免编号跳来跳去
    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes

# -------------------- 视觉AI：脸部截图→专注度/情绪 --------------------
def _img_to_data_url(face_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    if not ok:
        raise RuntimeError("encode jpg failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

@st.cache_data(ttl=600, show_spinner=False)
def qwen_vl_attention_cached(face_jpg_dataurl: str, classroom_context: str) -> dict:
    """
    返回：{"attention": "专注/需要关注/状态不佳", "emotion": "...", "reason": "..."}
    """
    api_key = get_qwen_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    prompt = f"""
你是课堂观察助手。请根据学生脸部截图，结合课堂内容“{classroom_context}”，估计其课堂状态。
只允许输出严格JSON（不要Markdown，不要多余文字）：
{{
  "attention": "专注/需要关注/状态不佳",
  "emotion": "平静/困倦/紧张/好奇/烦躁/未知",
  "reason": "不超过20字的依据"
}}
注意：这是概率估计，不要涉及身份识别。
""".strip()

    data = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": "你是严谨的课堂观察助手。必须输出严格JSON。"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": face_jpg_dataurl}}
            ]}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    resp = requests.post(BASE_URL + "/chat/completions", headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        return {"attention": "未知", "emotion": "未知", "reason": "解析失败"}

def state_to_color(attention: str) -> str:
    if attention == "专注":
        return "#2ecc71"
    elif attention == "需要关注":
        return "#f1c40f"
    elif attention == "状态不佳":
        return "#e74c3c"
    else:
        return "#95a5a6"  # 未知：灰色

# -------------------- 文本LLM：生成教学建议（缓存） --------------------
@st.cache_data(ttl=300, show_spinner=False)
def llm_feedback_cached(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {get_qwen_api_key()}", "Content-Type": "application/json"}
    data = {
        "model": TEXT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个智慧课堂助理，回答要具体、可执行、适合教师课堂使用。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 256
    }
    resp = requests.post(BASE_URL + "/chat/completions", headers=headers, json=data, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"LLM接口非200：{resp.status_code} {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]

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
    prompt += "\n输出要求：1) 简短要点列出依据；2) 给出3条可执行建议；3) 语气友好专业。"
    return prompt

def safe_llm_feedback(context, student_name, emotion, attention, history, custom_question=None) -> str:
    prompt = build_prompt(context, student_name, emotion, attention, history, custom_question)
    try:
        return llm_feedback_cached(prompt)
    except Exception:
        return "（教学建议生成失败：请检查 API Key、模型权限或网络；也可能是调用频率过高。）"

# -------------------- UI --------------------
st.title("AI课堂状态实时监测与智能反馈系统")
st.markdown("支持上传班级照片进行人脸检测与编号，并可用视觉大模型判断专注度/情绪，再生成教学建议。")

classroom_context = st.text_input("当前课堂内容描述", value="微积分第3章：链式法则讲解")

st.sidebar.header("图片采集")
st.sidebar.info("Streamlit Cloud 上无法使用本地摄像头，建议上传照片。")

use_camera = st.sidebar.checkbox("启用本地摄像头（仅本地有效）", value=False)
uploaded_image = st.sidebar.file_uploader("上传班级照片", type=["jpg", "jpeg", "png"])

st.sidebar.header("AI判断开关")
use_ai = st.sidebar.checkbox("使用AI判断专注度/情绪（较慢/耗API）", value=True)
max_ai_faces = st.sidebar.slider("最多AI分析人数", 1, 12, 6)
conf_thr = st.sidebar.slider("人脸检测置信度阈值", 0.40, 0.90, 0.65, 0.01)
pad = st.sidebar.slider("人脸框扩展比例(PAD)", 0.00, 0.20, 0.05, 0.01)

frame_rgb = None
if use_camera:
    cap = cv2.VideoCapture(0)
    ret, frame_bgr = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame_bgr, caption="摄像头采集画面", channels="BGR")
    else:
        st.error("摄像头采集失败（云端通常不可用，请改用上传照片）。")
elif uploaded_image is not None:
    img = ImageOps.exif_transpose(Image.open(uploaded_image)).convert("RGB")
    frame_rgb = np.array(img)
    st.image(frame_rgb, caption="上传图片（用于检测的原始RGB）", channels="RGB")

# -------------------- 人脸检测 + AI状态 --------------------
student_status = []
draw_frame_rgb = None

if frame_rgb is not None:
    h, w = frame_rgb.shape[:2]
    draw_frame_rgb = frame_rgb.copy()

    boxes = detect_faces_dnn(frame_rgb, conf_thr=conf_thr)  # (x1,y1,x2,y2,conf)
    st.caption(f"检测到人脸数量：{len(boxes)}（按面积从大到小编号）")

    for i, (left, top, right, bottom, conf) in enumerate(boxes):
        bw, bh = right - left, bottom - top
        px, py = int(bw * pad), int(bh * pad)
        left = max(0, left - px)
        top = max(0, top - py)
        right = min(w, right + px)
        bottom = min(h, bottom + py)

        face_roi = frame_rgb[top:bottom, left:right]
        if face_roi.size == 0:
            continue

        match_name = f"Stu{i+1}"

        # 默认（不开AI时）
        attention, emotion, reason = "未知", "未知", f"det_conf={conf:.2f}"

        # 用AI判断（仅前N个，避免太慢）
        if use_ai and i < max_ai_faces:
            try:
                dataurl = _img_to_data_url(face_roi)
                r = qwen_vl_attention_cached(dataurl, classroom_context)
                attention = r.get("attention", "未知")
                emotion = r.get("emotion", "未知")
                reason = r.get("reason", "")
            except Exception:
                attention, emotion, reason = "未知", "未知", "AI调用失败"

        color = state_to_color(attention)

        student_status.append({
            "name": match_name,
            "emotion": emotion,
            "attention": attention,
            "reason": reason,
            "color": color,
            "history": []
        })

        # 画框与标注
        cv2.rectangle(draw_frame_rgb, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            draw_frame_rgb,
            f"{match_name} {attention}",
            (left, max(0, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    st.image(draw_frame_rgb, caption="检测结果（RGB画框）", channels="RGB")

# -------------------- 状态表 --------------------
st.header("班级学生状态监控（实时）")
if not student_status:
    st.info("未检测到人脸：请上传更清晰、人物正面较多的班级照片。")
    st.stop()

cols = st.columns(len(student_status) if len(student_status) <= 6 else 6)
for idx, s in enumerate(student_status[:len(cols)]):
    with cols[idx]:
        html = (
            f'<div style="background:{s["color"]}; padding:14px; border-radius:14px; text-align:center;">'
            f'<b>{s["name"]}</b><br>'
            f'情绪：{s["emotion"]}<br>'
            f'专注度：{s["attention"]}<br>'
            f'<span style="font-size:12px; opacity:0.85;">{s["reason"]}</span>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        if s["attention"] == "状态不佳":
            st.warning(f'{s["name"]} 状态不佳，已自动提醒！')

# -------------------- 教学建议（文本LLM） --------------------
st.header("AI个性化反馈/智能分析建议")

feedback_cache = {}
with st.spinner("正在生成教学建议（已启用缓存）..."):
    for s in student_status:
        feedback_cache[s["name"]] = safe_llm_feedback(
            classroom_context, s["name"], s["emotion"], s["attention"], s["history"]
        )

for s in student_status:
    with st.expander(f'{s["name"]} - 教学建议', expanded=False):
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

# -------------------- 教师纠正/优化（演示） --------------------
st.header("教师纠正/优化AI模型（演示）")
err_stu = st.selectbox("如需纠正，请选择学生", options=stu_names, key="correct_stu")
err_correct = st.text_area("请描述纠正意见/实际状态（将记录用于后续改进）", "")

if st.button("提交反馈优化AI"):
    st.info("您的反馈已记录（演示版本不做持久化存储）。")

st.caption("注：本系统对学生状态的判断为AI估计结果，仅供教学参考。")
