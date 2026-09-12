# cloud_app.py（雲端版進入點：streamlit run cloud_app.py）
import os

import streamlit as st

from shared.app import run

# Streamlit Cloud 沒有 .env，金鑰放在 secrets。要在第一次讀 settings() 之前
# 塞進環境變數——settings() 有快取，晚了就讀不到。
try:                                     # 本機通常沒有 secrets.toml，不要炸
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except Exception:
    pass

run("cloud", "💬 小聊天機器人（雲端版）")
