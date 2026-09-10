# cloud_app.py（RAG 版介面：只管畫面，問答交給 cloud_chat_bot）
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scr"))   # 模組都在 scr/

import streamlit as st
from google.genai import errors
import cloud_chat_bot as bot                    # ← 換掉直接打 API 的 requests

st.title("💬 小聊天機器人（雲端版）")

if "history" not in st.session_state:
    st.session_state.history = []               # 跟 CLI 同一份格式：role / parts

for m in st.session_state.history:              # model → assistant 才畫得出來
    role = "user" if m["role"] == "user" else "assistant"
    st.chat_message(role).write(m["parts"][0]["text"])

if user := st.chat_input("說點什麼…"):
    st.chat_message("user").write(user)

    err = None
    with st.spinner("思考中…"):
        try:
            reply, _ = bot.ask(user, st.session_state.history)   # ← 檢索 + 生成
        except Exception as e:
            err = e                     # 先接住，離開 spinner 再顯示

    if err:                             # 在 spinner 裡 st.stop() 的話，轉圈會停不下來
        if isinstance(err, errors.APIError) and err.code == 503:
            st.error("gemini暫時過載，請稍後再試。")
        else:
            st.error(f"✗ {err}")
    else:
        st.chat_message("assistant").write(reply)
