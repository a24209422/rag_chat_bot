# providers.py（依名稱組出某一邊的零件）
#
#   取代原本的 sys.path.insert(ROOT / side) 切換。那個做法有個藏得很深的問題：
#   cloud/rag.py 與 onperm/rag.py 都叫 rag，兩邊都不是 package，所以 import rag
#   的結果由 sys.path 順序決定，而且會被 sys.modules 快取——同一個 process 裡
#   「只能」載入其中一邊，第二次 import 會靜默拿到第一邊的模組，不報錯。
#
#   現在兩邊的差異只剩三個零件：
#       Retriever  怎麼算向量、min_score 是多少
#       LLM        怎麼跟模型講話、usage 怎麼讀
#       history 上限   地端要砍（context 小），雲端不砍
#   ChatBot 本身只有一份（shared/chat_bot.py）。加第三家就是多一個 LLM 子類。
from shared.chat_bot import ChatBot
from shared.settings import settings

SIDES = ("cloud", "onperm")


def _check(side):
    if side not in SIDES:
        raise ValueError("side 要是 %s 之一，收到 %r" % (" 或 ".join(SIDES), side))


def retriever_for(side, config=None, **kw):
    """建一個 Retriever。kw 直接轉給建構子（docs、min_score、store_path…）。

    沒有明確傳 docs 的話會接上共用的 registry——索引才會跟上傳的文件同步。
    傳了 docs 就是「測試那條路」：不碰磁碟、不碰 registry。

    import 刻意放在函式裡：載入地端會連帶拉進 sentence_transformers，
    只想用雲端的人不該付這個成本。
    """
    _check(side)
    cfg = config or settings()
    kw.setdefault("config", cfg)
    if kw.get("docs") is None and "registry" not in kw:
        from shared.registry import registry_at
        kw["registry"] = registry_at(cfg.registry_path)
    if side == "cloud":
        from cloud.rag import CloudRetriever
        return CloudRetriever(**kw)
    from onperm.rag import OnpremRetriever
    return OnpremRetriever(**kw)


def llm_for(side, **kw):
    """建一個 LLM client。"""
    _check(side)
    if side == "cloud":
        from cloud.llm import GeminiLLM
        return GeminiLLM(**kw)
    from onperm.llm import LlamaCppLLM
    return LlamaCppLLM(**kw)


def chat_for(side, config=None, retriever=None, llm=None, **kw):
    """組出一個 ChatBot。兩邊回傳的形狀完全一樣：ask() → (reply, hits, usage)。"""
    _check(side)
    cfg = config or settings()
    limit = cfg.cloud_history_limit if side == "cloud" else cfg.onperm_history_limit
    return ChatBot(
        retriever=retriever if retriever is not None else retriever_for(side, config=cfg),
        llm=llm if llm is not None else llm_for(side, config=cfg),
        history_limit=kw.pop("history_limit", limit),
        config=cfg,
        **kw,
    )
