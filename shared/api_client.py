# api_client.py（打後端 API 的客戶端）
#
#   Streamlit 與 CLI 都走這裡，不再直接 import ChatBot。這就是前後端分離的
#   實質好處：後端可以獨立測試、可以被別的東西呼叫（curl、React、別的服務）、
#   可以部署在另一台機器；而且 UI 掛掉不會拖垮索引。
#
#   用 requests 而不是 httpx：requests 本來就是地端那側的依賴，不必多一個。
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


class ApiClient:
    def __init__(self, base_url=None, timeout=None, config=None):
        cfg = config or settings()
        self.base_url = (base_url or cfg.api_url).rstrip("/")
        self.timeout = timeout if timeout is not None else cfg.api_timeout

    def health(self):
        return self._request("GET", "/health")

    def ask(self, side, question, history, k=None):
        body = {"side": side, "question": question, "history": history}
        if k is not None:
            body["k"] = k
        return Answer(self._request("POST", "/chat", json=body))

    def _request(self, method, path, **kw):
        try:
            resp = requests.request(method, self.base_url + path,
                                    timeout=self.timeout, **kw)
        except requests.exceptions.ConnectionError as e:
            raise ApiError(f"連不上後端 {self.base_url}——"
                           f"先跑 uvicorn api.main:app --reload") from e
        except requests.exceptions.Timeout as e:
            raise ApiError(f"後端超過 {self.timeout} 秒沒回應。") from e

        if resp.status_code >= 400:
            raise ApiError(self._detail(resp), status=resp.status_code)
        return resp.json()

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
