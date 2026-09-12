# cloud_app.py（RAG 版介面：只管畫面，問答交給 cloud/chat_bot.py）
import os

import streamlit as st
from google.genai import errors

import providers

try:                                     # 雲端有 secrets，本機通常沒有
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except Exception:                        # 本機沒有 secrets.toml 時不要炸
    pass


@st.cache_resource                       # ← 換成 "onperm" 就是地端版
def get_bot():
    """建一次就好。

    Streamlit 每次互動都會從頭重跑整支腳本，沒有這個 decorator 的話，每問一句
    就會重建一次 Retriever——索引跟著重算（雲端是 140 個 API request）。
    以前靠 rag.py 的模組層 _DOC_VECS 全域「順便」達到這個效果，現在狀態在
    實例上，就得明確講出快取的範圍。
    """
    return providers.chat_for("cloud")


bot = get_bot()

st.title("💬 小聊天機器人（雲端版）")

st.session_state.setdefault("history", [])      # 跟 CLI 同一份格式：role / parts
st.session_state.setdefault("sources", [])      # ← 與 history 等長；user 那格放 None
st.session_state.setdefault("usage", {"in": 0, "out": 0})   # 本次 session 的 token 累計

# strict=True：history 與 sources 必須等長。不加的話 zip() 會沉默截斷，
# 症狀是來源被標到別人的回答底下——寧可當場報錯也不要默默錯位。
for m, src in zip(st.session_state.history, st.session_state.sources, strict=True):
    role = "user" if m["role"] == "user" else "assistant"   # model → assistant 才畫得出來
    with st.chat_message(role):                 # 一格要放兩樣東西，所以用 with
        st.write(m["parts"][0]["text"])
        for d, s in (src or []):                # user 那格是 None，不能直接迭代
            st.caption(f"`{s:.3f}` {d.label}")

if user := st.chat_input("說點什麼…"):
    st.chat_message("user").write(user)

    err = None
    with st.spinner("思考中…"):
        try:
            reply, hits, u = bot.ask(user, st.session_state.history)  # ← 檢索 + 生成
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
                st.caption(f"`{s:.3f}` {d.label}")

# 擺在最後才畫，這樣數字包含剛才那一輪（sidebar 位置跟程式順序無關）
st.sidebar.metric("輸入 token", st.session_state.usage["in"])
st.sidebar.metric("輸出 token", st.session_state.usage["out"])
st.sidebar.caption("輸出含模型的思考 token——看不到但會計費")
