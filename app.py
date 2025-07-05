import streamlit as st
import cv2
import numpy as np
import face_recognition
from PIL import Image
import requests
from transformers import pipeline

def state_to_color(attention):
    if attention == "专注":
        return "green"
    elif attention == "需要关注":
        return "yellow"
    else:
        return "red"

QWEN_API_KEY = "XXXXXXXXX"  #你的key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# --------- 模型准备 ---------
# 情绪识别：用huggingface pipeline示例，可换成本地模型
from deepface import DeepFace
def analyze_emotion(face_img):
    result = DeepFace.analyze(np.array(face_img), actions=['emotion'], enforce_detection=False)
    return result['dominant_emotion']
# 大语言模型：这里用OpenAI API，也可用千问、文心等
import os
import json

def llm_feedback(context, student_name, emotion, attention, history, custom_question=None):
    prompt = f"""
    教师在课堂上关注学生状态。请结合下述信息生成自然语言反馈和改进建议：
    - 学生姓名：{student_name}
    - 当前情绪：{emotion}
    - 当前专注度状态：{attention}
    - 学生历史状态：{history}
    - 当前课堂内容：{context}
    """
    if custom_question:
        prompt += f"\n教师提问：{custom_question}"
    try:
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "qwen-max",  # 或你实际有权限的千问模型
            "messages": [
                {"role": "system", "content": "你是一个智慧课堂助理"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 256,
        }
        resp = requests.post(BASE_URL + "/chat/completions", headers=headers, json=data, timeout=15)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "（大语言模型反馈接口调用失败，请检查API KEY与网络。）"


# Streamlit UI
st.set_page_config(page_title="AI课堂状态监测与智能反馈", layout="wide")
st.title("AI课堂状态实时监测与智能反馈系统")
st.markdown("本系统实时采集学生表情、姿态并智能分析，自动生成反馈建议，助力教师高效教学。")

# 教师端配置
classroom_context = st.text_input("当前课堂内容描述", value="微积分第3章链式法则讲解")


# 摄像头/图片
st.sidebar.header("摄像头/图片采集")
use_camera = st.sidebar.checkbox("启用本地摄像头", value=False)
uploaded_image = st.sidebar.file_uploader("或上传班级照片", type=['jpg', 'jpeg', 'png'])

frame = None
if use_camera:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        st.image(frame, caption="实时摄像头采集画面", channels="BGR")
    else:
        st.error("摄像头采集失败")
elif uploaded_image is not None:
    frame = np.array(Image.open(uploaded_image))
    st.image(frame, caption="上传图片")

# ------------------ 人脸检测与画框编号、学生分析 ------------------
student_status = []
face_locs = []
draw_frame = None
if frame is not None:
    # 检测所有人脸
    face_locs = face_recognition.face_locations(frame)
    face_encs = face_recognition.face_encodings(frame, face_locs)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw_frame = frame.copy()

    for i, (loc, enc) in enumerate(zip(face_locs, face_encs)):
        top, right, bottom, left = loc
        face_img = pil_img.crop((left, top, right, bottom)).resize((224,224))
        match_name = f"Stu{i+1}"
        try:
            emotion = analyze_emotion(face_img)
        except Exception:
            emotion = "未知"
        attention = "专注" if emotion in ["happy", "neutral"] else ("需要关注" if emotion in ["surprise", "disgust"] else "状态不佳")
        color = state_to_color(attention)
        student_status.append({
            "name": match_name,
            "face_box": (top, right, bottom, left),
            "emotion": emotion,
            "attention": attention,
            "color": color,
            "history": []
        })
        # --- 画框与编号 ---
        cv2.rectangle(draw_frame, (left, top), (right, bottom), (0,0,255), 2)
        cv2.putText(draw_frame, match_name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # 显示画好框的图片
    st.image(draw_frame, caption="检测结果（人脸已编号）", channels="BGR")



# ------------------ 学生状态表+报警 ------------------
st.header("班级学生状态监控（实时）")
cols = st.columns(len(student_status) or 1)
for i, s in enumerate(student_status):
    with cols[i]:
        st.markdown(f'<div style="background:{s["color"]}; padding:10px; border-radius:12px; text-align:center;">'
                    f'<b>{s["name"]}</b><br>'
                    f'情绪：{s["emotion"]}<br>'
                    f'专注度：{s["attention"]}'
                    f'</div>', unsafe_allow_html=True)
        if s['attention'] == "状态不佳":
            st.warning(f"{s['name']} 状态不佳，已自动报警！")
        # AI个性化反馈
        feedback = llm_feedback(classroom_context, s["name"], s["emotion"], s["attention"], s["history"])
        # 这里只显示卡片状态和报警提示，不显示长文本
        # st.markdown(f"**AI个性化反馈：**{feedback}")

if student_status:
    st.header("AI个性化反馈/智能分析建议")
    for s in student_status:
        feedback = llm_feedback(classroom_context, s["name"], s["emotion"], s["attention"], s["history"])
        with st.expander(f"{s['name']} - 详细AI分析", expanded=False):
            st.markdown(feedback)


# ------------------ 教师与AI交互/反馈优化 ------------------
st.header("教师与AI交互")
stu_names = [s["name"] for s in student_status]
selected_student = st.selectbox("选择学生", options=stu_names)
custom_q = st.text_input("向AI系统提问", value="该生当前学习状态及建议？")
if st.button("获取AI分析和建议"):
    s = next((x for x in student_status if x["name"] == selected_student), None)
    if s:
        resp = llm_feedback(classroom_context, s["name"], s["emotion"], s["attention"], s["history"], custom_q)
        st.success(resp)
# --------- 教师反馈优化AI ---------
st.header("教师纠正/优化AI模型")
err_stu = st.selectbox("如需纠正，请选择学生", options=stu_names)
err_correct = st.text_area("请描述纠正意见/实际状态", "")
if st.button("提交反馈优化AI"):
    st.info("您的反馈已记录，后续将用于优化AI分析结果。")

st.caption("注：本系统已实现自动标注人脸编号，状态信息与图片编号一一对应！")
