import os
import json
import time
import requests
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

# -------------------- 基础配置 --------------------
st.set_page_config(page_title="AI课堂状态监测与智能反馈", layout="wide")

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问兼容 OpenAI 的接口
MODEL_NAME = "qwen-max"  # 你也可以换成你账号有权限的模型，比如 qwen-turbo / qwen-plus

@st.cache_resource
def get_face_cascades():
    base = cv2.data.haarcascades
    c1 = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
    c2 = cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml")
    return [c1, c2]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms(boxes, iou_thr=0.35):
    # boxes: [(x1,y1,x2,y2), ...]
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for b in boxes:
        if all(iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep

def detect_faces_opencv(frame_rgb: np.ndarray):
    """
    输入: RGB ndarray
    输出: list of (left, top, right, bottom)
    """
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)  # 提升对比度，减少漏检

    cascades = get_face_cascades()

    def run(scaleFactor, minNeighbors, minSize):
        boxes = []
        for cas in cascades:
            faces = cas.detectMultiScale(
                gray,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=(minSize, minSize)
            )
            for (x, y, fw, fh) in faces:
                x1, y1, x2, y2 = x, y, x + fw, y + fh
                # 过滤：比例异常、太小、太大（减少误检“大框”）
                ar = fw / max(1, fh)
                area_ratio = (fw * fh) / max(1, w * h)
                if ar < 0.65 or ar > 1.6:
                    continue
                if area_ratio < 0.002:   # 太小（很多是噪声）
                    continue
                if area_ratio > 0.20:    # 太大（容易把一排人当脸）
                    continue
                boxes.append((x1, y1, x2, y2))
        return nms(boxes, iou_thr=0.35)

    # 第一轮：偏“稳”，减少误检
    boxes = run(scaleFactor=1.1, minNeighbors=7, minSize=60)

    # 回退：如果一个都没检出，放宽条件以减少漏检
    if not boxes:
        boxes = run(scaleFactor=1.08, minNeighbors=5, minSize=40)

    # 再回退：仍然没有，就再放宽一点
    if not boxes:
        boxes = run(scaleFactor=1.05, minNeighbors=4, minSize=30)

    return boxes


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
    img = ImageOps.exif_transpose(Image.open(uploaded_image)).convert("RGB")
    frame_rgb = np.array(img)
    st.image(frame_rgb, caption="上传图片（用于检测的原始RGB）", channels="RGB")    
    

# -------------------- 人脸检测（OpenCV Haar Cascade）--------------------
student_status = []
draw_frame_rgb = None

if frame_rgb is not None:
    h, w = frame_rgb.shape[:2]
    draw_frame_rgb = frame_rgb.copy()

    boxes = detect_faces_opencv(frame_rgb)  # (left, top, right, bottom)

    PAD = 0.05  # 12% padding，让框更贴合/更好看
    for i, (left, top, right, bottom) in enumerate(boxes):
        bw, bh = right - left, bottom - top
        px, py = int(bw * PAD), int(bh * PAD)
        left  = max(0, left - px)
        top   = max(0, top - py)
        right = min(w, right + px)
        bottom= min(h, bottom + py)

        match_name = f"Stu{i+1}"
        emotion = "neutral"
        attention = "专注"
        color = state_to_color(attention)

        student_status.append({
            "name": match_name, "emotion": emotion,
            "attention": attention, "color": color, "history": []
        })

        # 在RGB上画红框
        cv2.rectangle(draw_frame_rgb, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(draw_frame_rgb, match_name, (left, max(0, top-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    st.image(draw_frame_rgb, caption="检测结果（RGB画框）", channels="RGB")
        

# -------------------- 状态表 --------------------
st.header("班级学生状态监控（实时）")
if not student_status:
    st.info("未检测到人脸：请上传更清晰、人物正面较多的班级照片。")
    st.stop()

cols = st.columns(len(student_status) or 1)
for i, s in enumerate(student_status):
    with cols[i]:
        html = (
            f'<div style="background:{s["color"]}; padding:10px; border-radius:12px; text-align:center;">'
            f'<b>{s["name"]}</b><br>'
            f'情绪：{s["emotion"]}<br>'
            f'专注度：{s["attention"]}'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
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
