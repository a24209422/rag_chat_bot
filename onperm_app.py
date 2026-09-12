# onperm_app.py（RAG 版介面：只管畫面，問答交給 onperm/chat_bot.py）
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "onperm"))   # ← 換成 "cloud" 就是雲端版

import chat_bot as bot                   # noqa: E402

st.title("💬 小聊天機器人（地端版）")

st.session_state.setdefault("history", [])      # 跟 CLI 同一份格式：role / content
st.session_state.setdefault("sources", [])      # ← 與 history 等長；user 那格放 None

for m, src in zip(st.session_state.history, st.session_state.sources):
    with st.chat_message(m["role"]):            # 地端是 OpenAI 格式，role 直接就能畫
        st.write(m["content"])                  # （雲端要把 model 轉成 assistant）
        for d, s in (src or []):                # user 那格是 None，不能直接迭代
            st.caption(f"`{s:.3f}` {d.label}")

if user := st.chat_input("說點什麼…"):
    st.chat_message("user").write(user)

    err = None
    with st.spinner("思考中…"):
        try:
            reply, hits = bot.ask(user, st.session_state.history)   # ← 地端只回兩個值
        except Exception as e:
            err = e                     # 先接住，離開 spinner 再顯示

    if err:                             # 在 spinner 裡 st.stop() 的話，轉圈會停不下來
        if isinstance(err, requests.exceptions.ConnectionError):
            st.error(f"連不上 {bot.URL}——llama.cpp server 沒開。")
        elif isinstance(err, requests.exceptions.Timeout):
            st.error("等超過 120 秒，模型可能還在載入，稍等再問。")
        else:
            st.error(f"✗ {err}")
    else:
        st.session_state.sources += [None, hits]     # ← 只在「成功」這條路加
        with st.chat_message("assistant"):
            st.write(reply)
            for d, s in hits:           # 這次的來源要自己畫，頂端迴圈還看不到它
                st.caption(f"`{s:.3f}` {d.label}")

    # ask() 會就地截短 history（onperm/chat_bot.py 的 len > 11），雲端版沒這回事。
    # sources 不跟著切的話，zip() 會從頭配對 → 來源標到別人的回答底下。
    # 兩者都是一次 append 兩則、history 只從前面砍，所以取相同長度的尾段就對齊了。
    n = len(st.session_state.history)
    st.session_state.sources = st.session_state.sources[-n:] if n else []
