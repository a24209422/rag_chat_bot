# providers.py（依名稱取得雲端或地端的實作）
#
#   取代原本的 sys.path.insert(ROOT / side) 切換。那個做法有個藏得很深的問題：
#   cloud/rag.py 與 onperm/rag.py 都叫 rag，兩邊都不是 package，所以 import rag
#   的結果由 sys.path 順序決定，而且會被 sys.modules 快取——同一個 process 裡
#   「只能」載入其中一邊，第二次 import 會靜默拿到第一邊的模組，不報錯。
#   Streamlit 一次只跑一支所以碰不到，但測試、後端服務、任何要並排比較兩邊的
#   工具都會撞上。改成正式 package + 這個工廠之後，兩邊可以同時存在。
SIDES = ("cloud", "onperm")


def _check(side):
    if side not in SIDES:
        raise ValueError("side 要是 %s 之一，收到 %r" % (" 或 ".join(SIDES), side))


def retriever_for(side, **kw):
    """建一個 Retriever。kw 直接轉給建構子（docs、min_score…）。

    import 刻意放在函式裡：載入地端會連帶拉進 sentence_transformers（好幾秒），
    只想用雲端的人不該付這個成本。
    """
    _check(side)
    if side == "cloud":
        from cloud.rag import CloudRetriever
        return CloudRetriever(**kw)
    from onperm.rag import OnpremRetriever
    return OnpremRetriever(**kw)


def chat_for(side, **kw):
    """建一個 ChatBot。不傳 retriever 的話，建構子會自己建一個對應的。

    ⚠ 兩邊的 ask() 形狀不同，這是刻意的，不要包一層統一介面把它藏起來：
        cloud  → (reply, hits, usage)   history 格式 parts / model
        onperm → (reply, hits)          history 格式 content / assistant
    呼叫端本來就得知道自己在用哪一邊（UI 要畫 token 用量、要轉 role），
    假裝一樣只會讓差異在更遠的地方以更難懂的形式冒出來。
    """
    _check(side)
    if side == "cloud":
        from cloud.chat_bot import CloudChatBot
        return CloudChatBot(**kw)
    from onperm.chat_bot import OnpremChatBot
    return OnpremChatBot(**kw)
