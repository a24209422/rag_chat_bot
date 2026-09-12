# app.py（Streamlit 介面：兩邊共用一份）
#
#   這一層只管畫面。問答走 HTTP 打後端，不再直接 import ChatBot——
#   所以後端可以換機器、可以重啟、可以同時被別的客戶端用，UI 不用知道。
#   history 由這裡保管（st.session_state），後端完全無狀態：
#   併發使用者永遠不會看到彼此的對話。
import streamlit as st

from shared.api_client import ApiClient, ApiError
from shared.llm import Usage


@st.cache_resource
def get_client():
    """建一次就好——Streamlit 每次互動都會從頭重跑整支腳本。

    （以前這裡快取的是整個 ChatBot，因為索引跟著它。現在索引在後端，
    這裡只剩一個很輕的 HTTP 客戶端，快取只是順手。）
    """
    return ApiClient()


def run(side, title):
    client = get_client()

    st.title(title)

    st.session_state.setdefault("history", [])   # 中性格式：role / content
    st.session_state.setdefault("sources", [])   # ← 與 history 等長；user 那格放 None
    st.session_state.setdefault("usage", Usage())

    # strict=True：history 與 sources 必須等長。不加的話 zip() 會沉默截斷，
    # 症狀是來源被標到別人的回答底下——寧可當場報錯也不要默默錯位。
    for m, src in zip(st.session_state.history, st.session_state.sources, strict=True):
        with st.chat_message(m["role"]):         # 中性格式的 role 直接就能畫
            st.write(m["content"])
            for s in (src or []):                # user 那格是 None，不能直接迭代
                st.caption(f"`{s['score']:.3f}` {s['label']}")

    if user := st.chat_input("說點什麼…"):
        st.chat_message("user").write(user)

        err = None
        with st.spinner("思考中…"):
            try:
                ans = client.ask(side, user, st.session_state.history)
            except ApiError as e:
                err = e                 # 先接住，離開 spinner 再顯示

        if err:                         # 在 spinner 裡 st.stop() 的話，轉圈會停不下來
            st.error(f"✗ {err}")        # 訊息在後端就翻成人話了，直接顯示
        else:
            # 後端回的 history 才是權威版本——它可能就地截短過（地端有上限）。
            st.session_state.history = ans.history
            st.session_state.sources += [None, ans.sources]
            st.session_state.usage += ans.usage
            with st.chat_message("assistant"):
                st.write(ans.reply)
                for s in ans.sources:   # 這次的來源要自己畫，頂端迴圈還看不到它
                    st.caption(f"`{s['score']:.3f}` {s['label']}")

            # history 被截短時 sources 要跟著切，不然下一輪的 zip(strict=True) 會炸。
            # 兩者都是一次 append 兩則、history 只從前面砍，取相同長度的尾段就對齊了。
            n = len(st.session_state.history)
            st.session_state.sources = st.session_state.sources[-n:] if n else []

    # 擺在最後才畫，這樣數字包含剛才那一輪（sidebar 位置跟程式順序無關）
    st.sidebar.metric("輸入 token", st.session_state.usage.prompt)
    st.sidebar.metric("輸出 token", st.session_state.usage.output)
    st.sidebar.caption("雲端的輸出含模型的思考 token——看不到但會計費。"
                       "地端不計費，但看得出離 context 上限還有多遠。")
