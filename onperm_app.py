# onperm_app.py（RAG 版介面：只管畫面，問答交給 onperm/chat_bot.py）
import requests
import streamlit as st

import providers


@st.cache_resource                       # ← 換成 "cloud" 就是雲端版
def get_bot():
    """建一次就好。

    Streamlit 每次互動都會從頭重跑整支腳本，沒有這個 decorator 的話，每問一句
    就會重載一次 e5 模型並重建索引。以前靠 rag.py 的模組層 _EMBEDDER /
    _DOC_VECS 全域「順便」達到這個效果，現在狀態在實例上，就得明確講出來。
    """
    return providers.chat_for("onperm")


bot = get_bot()

st.title("💬 小聊天機器人（地端版）")

st.session_state.setdefault("history", [])      # 跟 CLI 同一份格式：role / content
st.session_state.setdefault("sources", [])      # ← 與 history 等長；user 那格放 None

# strict=True：history 與 sources 必須等長。不加的話 zip() 會沉默截斷，
# 症狀是來源被標到別人的回答底下——寧可當場報錯也不要默默錯位。
for m, src in zip(st.session_state.history, st.session_state.sources, strict=True):
    with st.chat_message(m["role"]):            # 地端是 OpenAI 格式，role 直接就能畫
        st.write(m["content"])                  # （雲端要把 model 轉成 assistant）
        for d, s in (src or []):                # user 那格是 None，不能直接迭代
            st.caption(f"`{s:.3f}` {d.label}")

if user := st.chat_input("說點什麼…"):
    st.chat_message("user").write(user)

    err = None
    with st.spinner("思考中…"):
        try:
            reply, hits = bot.ask(user, st.session_state.history)  # ← 地端只回兩個值
        except Exception as e:
            err = e                     # 先接住，離開 spinner 再顯示

    if err:                             # 在 spinner 裡 st.stop() 的話，轉圈會停不下來
        if isinstance(err, requests.exceptions.ConnectionError):
            st.error(f"連不上 {bot.url}——llama.cpp server 沒開。")
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

    # ask() 會就地截短 history（onperm/chat_bot.py 的 HISTORY_LIMIT），雲端版沒這回事。
    # sources 不跟著切的話，zip() 會從頭配對 → 來源標到別人的回答底下。
    # 兩者都是一次 append 兩則、history 只從前面砍，所以取相同長度的尾段就對齊了。
    n = len(st.session_state.history)
    st.session_state.sources = st.session_state.sources[-n:] if n else []
