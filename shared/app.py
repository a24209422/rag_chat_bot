# app.py（Streamlit 介面：兩邊共用一份）
#
#   在 history 統一成中性格式、ask() 統一回傳 (reply, hits, usage) 之前，
#   兩支 app 各有一份幾乎一樣的畫面邏輯，差在 role 怎麼轉、要不要畫 token。
#   現在差異只剩「標題」和「哪一邊」，所以 cloud_app.py / onperm_app.py
#   各自只剩幾行——它們仍然是兩個獨立的進入點（streamlit run 要指定檔案）。
import streamlit as st

import providers
from shared.llm import Usage


def run(side, title):
    @st.cache_resource(show_spinner="準備中…")
    def get_bot(side):
        """建一次就好。

        Streamlit 每次互動都會從頭重跑整支腳本，沒有這個 decorator 的話，
        每問一句就會重建 Retriever——索引跟著重算（雲端是 140 個 API request，
        地端是重載 e5）。以前靠模組層全域「順便」達到這個效果，現在狀態在
        實例上，就得明確講出快取的範圍。
        """
        return providers.chat_for(side)

    bot = get_bot(side)

    st.title(title)

    st.session_state.setdefault("history", [])   # 中性格式：role / content
    st.session_state.setdefault("sources", [])   # ← 與 history 等長；user 那格放 None
    st.session_state.setdefault("usage", Usage())

    # strict=True：history 與 sources 必須等長。不加的話 zip() 會沉默截斷，
    # 症狀是來源被標到別人的回答底下——寧可當場報錯也不要默默錯位。
    for m, src in zip(st.session_state.history, st.session_state.sources, strict=True):
        with st.chat_message(m["role"]):         # 中性格式的 role 直接就能畫
            st.write(m["content"])
            for d, s in (src or []):             # user 那格是 None，不能直接迭代
                st.caption(f"`{s:.3f}` {d.label}")

    if user := st.chat_input("說點什麼…"):
        st.chat_message("user").write(user)

        err = None
        with st.spinner("思考中…"):
            try:
                reply, hits, used = bot.ask(user, st.session_state.history)
            except Exception as e:
                err = e                 # 先接住，離開 spinner 再顯示

        if err:                         # 在 spinner 裡 st.stop() 的話，轉圈會停不下來
            st.error(bot.llm.explain(err) or f"✗ {err}")
        else:
            st.session_state.sources += [None, hits]     # ← 只在「成功」這條路加
            st.session_state.usage += used
            with st.chat_message("assistant"):
                st.write(reply)
                for d, s in hits:       # 這次的來源要自己畫，頂端迴圈還看不到它
                    st.caption(f"`{s:.3f}` {d.label}")

        # ask() 可能就地截短 history（地端有上限，雲端沒有）。sources 不跟著切的話，
        # 下一輪的 zip(strict=True) 會直接炸。兩者都是一次 append 兩則、history 只從
        # 前面砍，所以取相同長度的尾段就對齊了。
        n = len(st.session_state.history)
        st.session_state.sources = st.session_state.sources[-n:] if n else []

    # 擺在最後才畫，這樣數字包含剛才那一輪（sidebar 位置跟程式順序無關）
    st.sidebar.metric("輸入 token", st.session_state.usage.prompt)
    st.sidebar.metric("輸出 token", st.session_state.usage.output)
    st.sidebar.caption("雲端的輸出含模型的思考 token——看不到但會計費。"
                       "地端不計費，但看得出離 context 上限還有多遠。")
