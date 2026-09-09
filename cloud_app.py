# app_api.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"  # ① 換網址
HEADERS = {"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"}             # ② 加金鑰
MODEL = "gemini-flash-latest"                                                     # ③ 指定型號

st.title("💬 小聊天機器人（雲端版）")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是友善的中文助理，回答簡潔。"}
    ]

for m in st.session_state.messages[1:]:
    st.chat_message(m["role"]).write(m["content"])

if user := st.chat_input("說點什麼…"):
    st.session_state.messages.append({"role": "user", "content": user})
    st.chat_message("user").write(user)

    with st.spinner("思考中…"):
        resp = requests.post(
            URL,
            headers=HEADERS,                                                      # ← 多這個
            json={
                "model": MODEL,                                                   # ← 多這個
                "messages": st.session_state.messages,
                "temperature": 0.7,
            },
            timeout=120,
        )
    data = resp.json()
    if resp.status_code != 200:
        st.error(f"API 錯誤（{resp.status_code}）：{data.get('error', data)}")
        st.stop()

    reply = data["choices"][0]["message"]["content"]

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})