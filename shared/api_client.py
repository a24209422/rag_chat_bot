# api_client.py（打後端 API 的客戶端）
#
#   Streamlit 與 CLI 都走這裡，不再直接 import ChatBot。這就是前後端分離的
#   實質好處：後端可以獨立測試、可以被別的東西呼叫（curl、React、別的服務）、
#   可以部署在另一台機器；而且 UI 掛掉不會拖垮索引。
#
#   用 requests 而不是 httpx：requests 本來就是地端那側的依賴，不必多一個。
import json

import requests

from shared.llm import Usage
from shared.settings import settings


class ApiError(RuntimeError):
    """後端回的錯誤。

    訊息已經在後端被 llm.explain() 翻成人話了，UI 直接顯示就好——
    不需要在這裡再翻一次，也不該在這裡認得 Gemini 的 429 是什麼意思。
    """

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Answer:
    """一次問答的結果。"""

    def __init__(self, payload):
        self.reply = payload["reply"]
        self.sources = payload["sources"]          # [{code, label, score}]
        self.history = payload["history"]          # 後端回的才是權威版本
        self.usage = Usage(payload["usage"]["prompt"], payload["usage"]["output"])
        # 模型說「資料裡沒有」但 sources 不是空的。後端已經為此重問過一次，
        # 這個旗標代表「重問完還是矛盾」——UI 該把使用者的視線導向來源列。
        self.contradiction = payload.get("contradiction", False)


class StreamedAnswer:
    """串流版的一次問答。

    建好的時候 sources 就有了（後端第一個送的就是它），所以 UI 可以在模型
    還沒講完之前先把來源列出來。文字要迭代 tokens() 才會一段一段出來；
    跑完之後 reply / usage / history 才有值。
    """

    def __init__(self, events):
        self.sources = []
        self.reply = ""
        self.usage = Usage()
        self.history = []
        self.contradiction = False        # done 事件才知道，見 tokens()
        self._events = events
        for event, data in self._events:      # 先讀到 sources 為止
            if event == "sources":
                self.sources = data["sources"]
                return
            if event == "error":
                raise ApiError(data["detail"])
        raise ApiError("後端沒有送出 sources 事件就結束了")

    def tokens(self):
        for event, data in self._events:
            if event == "token":
                self.reply += data["text"]
                yield data["text"]
            elif event == "done":
                self.usage = Usage(data["usage"]["prompt"], data["usage"]["output"])
                self.history = data["history"]
                self.contradiction = data.get("contradiction", False)
            elif event == "error":
                # 生成中途壞掉。HTTP 狀態早就送出去了（200），所以錯誤只能
                # 以事件的形式回來——不處理的話畫面會停在半截答案上。
                raise ApiError(data["detail"])


class ApiClient:
    def __init__(self, base_url=None, timeout=None, config=None):
        cfg = config or settings()
        self.base_url = (base_url or cfg.api_url).rstrip("/")
        self.timeout = timeout if timeout is not None else cfg.api_timeout

    def health(self):
        return self._request("GET", "/health")

    def ask(self, side, question, history, k=None):
        return Answer(self._request("POST", "/chat", json=self._body(
            side, question, history, k)))

    def stream(self, side, question, history, k=None):
        """開一條 SSE 連線。回傳時 sources 已經拿到了。"""
        # stream=True 一定要傳下去，不然 requests 會把整個回應緩衝起來——
        # 對串流端點來說那等於「等它全部講完」，逐段吐字完全失效。
        resp = self._open("POST", "/chat/stream", stream=True,
                          json=self._body(side, question, history, k))
        return StreamedAnswer(_sse_events(resp))

    @staticmethod
    def _body(side, question, history, k):
        body = {"side": side, "question": question, "history": history}
        if k is not None:
            body["k"] = k
        return body

    def _request(self, method, path, **kw):
        return self._open(method, path, **kw).json()

    def _open(self, method, path, stream=False, **kw):
        try:
            resp = requests.request(method, self.base_url + path,
                                    timeout=self.timeout, stream=stream, **kw)
        except requests.exceptions.ConnectionError as e:
            raise ApiError(f"連不上後端 {self.base_url}——"
                           f"先跑 uvicorn api.main:app --reload") from e
        except requests.exceptions.Timeout as e:
            raise ApiError(f"後端超過 {self.timeout} 秒沒回應。") from e

        if resp.status_code >= 400:
            raise ApiError(self._detail(resp), status=resp.status_code)
        return resp

    @staticmethod
    def _detail(resp):
        """FastAPI 的錯誤是 {"detail": ...}，但 422 的 detail 是一個清單
        （每個欄位一則），不是字串——直接丟給 UI 會很難讀。"""
        try:
            detail = resp.json().get("detail")
        except ValueError:
            return f"後端回了 {resp.status_code}：{resp.text[:200]}"
        if isinstance(detail, list):
            return "；".join(str(d.get("msg", d)) for d in detail)
        return str(detail or f"後端回了 {resp.status_code}")


def _sse_events(resp):
    """把 SSE 的位元組流拆成 (事件名, 資料) 的序列。"""
    # ⚠ 後端有宣告 charset=utf-8，但這裡再設一次當保險。HTTP 對 text/* 的規定
    #   是沒宣告就退回 ISO-8859-1，requests 照做——中文會變成 latin-1 亂碼，
    #   而且完全不報錯。踩過一次就知道要兩邊都防。
    resp.encoding = "utf-8"
    event = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue                       # 空行是事件分隔
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            yield event, json.loads(line[6:])
            event = None
