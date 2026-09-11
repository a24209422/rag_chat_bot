# cloud_app.py（RAG 版介面：只管畫面，問答交給 cloud_chat_bot）
import os
import sys
from pathlib import Path

import streamlit as st
from google.genai import errors

sys.path.insert(0, str(Path(__file__).parent / "scr"))   # 模組都在 scr/

try:                                     # 雲端有 secrets，本機通常沒有
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except Exception:                        # 本機沒有 secrets.toml 時不要炸
    pass

import cloud_chat_bot as bot             # noqa: E402 ← 必須在設好 key 之後

st.title("💬 小聊天機器人（雲端版）")

st.session_state.setdefault("history", [])      # 跟 CLI 同一份格式：role / parts
st.session_state.setdefault("sources", [])      # ← 與 history 等長；user 那格放 None
st.session_state.setdefault("usage", {"in": 0, "out": 0})   # 本次 session 的 token 累計

for m, src in zip(st.session_state.history, st.session_state.sources):
    role = "user" if m["role"] == "user" else "assistant"   # model → assistant 才畫得出來
    with st.chat_message(role):                 # 一格要放兩樣東西，所以用 with
        st.write(m["parts"][0]["text"])
        for d, s in (src or []):                # user 那格是 None，不能直接迭代
            st.caption(f"`{s:.3f}` {d}")

if user := st.chat_input("說點什麼…"):
    st.chat_message("user").write(user)

    err = None
    with st.spinner("思考中…"):
        try:
            reply, hits, u = bot.ask(user, st.session_state.history)   # ← 檢索 + 生成
        except Exception as e:
            err = e                     # 先接住，離開 spinner 再顯示

    if err:                             # 在 spinner 裡 st.stop() 的話，轉圈會停不下來
        if isinstance(err, errors.APIError) and err.code == 503:
            st.error("gemini暫時過載，請稍後再試。")
        elif isinstance(err, errors.APIError) and err.code == 429:
            # 免費方案觀測到的上限：這個模型每天 20 次生成。429 也可能是每分鐘超速
            st.error("已達 Gemini 免費方案的用量上限，請稍後或明天再試。")
        else:
            st.error(f"✗ {err}")
    else:
        st.session_state.sources += [None, hits]     # ← 只在「成功」這條路加
        st.session_state.usage["in"] += u["in"]
        st.session_state.usage["out"] += u["out"]
        with st.chat_message("assistant"):
            st.write(reply)
            for d, s in hits:           # 這次的來源要自己畫，頂端迴圈還看不到它
                st.caption(f"`{s:.3f}` {d}")

# 擺在最後才畫，這樣數字包含剛才那一輪（sidebar 位置跟程式順序無關）
st.sidebar.metric("輸入 token", st.session_state.usage["in"])
st.sidebar.metric("輸出 token", st.session_state.usage["out"])
st.sidebar.caption("輸出含模型的思考 token——看不到但會計費")
