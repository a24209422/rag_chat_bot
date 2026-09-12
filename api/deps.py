# api/deps.py（依賴注入）
import threading
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends

import providers
from shared.registry import Registry, registry_at
from shared.settings import settings

# side → ChatBot。用自己的 dict 而不是 lru_cache，是因為 lru_cache 沒有
# 「只看不建」的 API——想知道哪幾邊已經載入，就只能去呼叫它，而那正好會
# 把它建起來。/health 因此會把每一邊都建一遍，是個安靜的 bug。
_bots: dict[str, object] = {}
_lock = threading.Lock()


def bot_for(side):
    """每一邊建一次就好，之後重用。

    第一次呼叫才建——建索引很貴（雲端要讀快取或打 140 個 request，地端要載
    e5 再算 140 塊），只用雲端的人不該在啟動時付地端的成本。要預熱就設
    API_WARM（見 shared/settings.py）。

    上鎖是因為 FastAPI 的同步端點跑在 threadpool 裡：兩個並發的「第一次」
    請求會同時看到快取是空的，然後各建一次索引（雲端等於打 280 個 request）。
    進鎖之後再檢查一次，才不會白白重建。
    """
    bot = _bots.get(side)
    if bot is None:
        with _lock:
            bot = _bots.get(side)          # ← 進鎖後再檢查一次
            if bot is None:
                bot = providers.chat_for(side)
                _bots[side] = bot
    return bot


def loaded():
    """索引已經算好的那幾邊。給 /health 用，順便看得出預熱有沒有生效。

    回報的是「索引算好了」而不是「物件建好了」——後者幾乎不花時間，
    講出來沒有意義。只是查表，不會把沒載入的建起來。
    """
    return [s for s in providers.SIDES
            if s in _bots and _bots[s].retriever.indexed]


def get_bot_factory():
    """回傳「side → ChatBot」的函式。

    包一層是為了測試能整個換掉（app.dependency_overrides），
    不必真的去建 retriever 或打模型。
    """
    return bot_for


def get_registry():
    return registry_at(settings().registry_path)


def refresh_loaded():
    """上傳或刪除之後，把「已經建好索引」的那幾邊拉回同步。

    還沒載入的不用管——它們第一次被建起來時就會自己跟 registry 對齊
    （見 BaseRetriever._sync）。先建好再同步是浪費，尤其地端要載 e5。
    """
    out = {}
    for side, bot in _bots.items():
        if bot.retriever.indexed:
            added, removed = bot.retriever.refresh()
            out[side] = {"added": added, "removed": removed}
    return out


def chunk_counts():
    return {side: len(bot.retriever.store)
            for side, bot in _bots.items() if bot.retriever.indexed}


# 型別別名，端點簽名才不會被依賴注入的雜訊塞滿。
# 用 Annotated 而不是 factory=Depends(...) 是 FastAPI 現在推薦的寫法：
# 預設值裡呼叫函式本來就是可疑的（ruff 的 B008 會抓），Annotated 把它
# 放進型別註記，兩邊都乾淨。
BotFactoryDep = Annotated[Callable, Depends(get_bot_factory)]
RegistryDep = Annotated[Registry, Depends(get_registry)]
