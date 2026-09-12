"""shared/api_client.py：打後端的客戶端。

換掉 requests.request，所以不需要真的起一個後端。要驗的是「後端回什麼、
UI 看到什麼」——尤其是錯誤訊息，因為那是使用者唯一看得到的東西。
"""
import pytest
import requests

from shared.api_client import ApiClient, ApiError
from shared.llm import Usage

OK_PAYLOAD = {
    "reply": "模型的回答",
    "sources": [{"code": "MAT-04", "label": "MAT-04 · 公司 · 職缺", "score": 0.91}],
    "usage": {"prompt": 2764, "output": 73},
    "history": [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "模型的回答"}],
}


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("不是 JSON")
        return self._payload


@pytest.fixture
def sent(monkeypatch):
    box = {"response": FakeResponse(payload=OK_PAYLOAD)}

    def fake_request(method, url, timeout=None, **kw):
        box.update(method=method, url=url, timeout=timeout, **kw)
        resp = box["response"]
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(requests, "request", fake_request)
    return box


# ── 正常路徑 ──────────────────────────────────────────────────────────
def test_送出的請求長什麼樣(sent):
    ApiClient(base_url="http://x:8000").ask("onperm", "問題", [{"role": "user",
                                                                "content": "舊"}])

    assert sent["method"] == "POST"
    assert sent["url"] == "http://x:8000/chat"
    assert sent["json"] == {"side": "onperm", "question": "問題",
                            "history": [{"role": "user", "content": "舊"}]}


def test_沒給k就不要送這個欄位(sent):
    """讓後端用它自己的預設，而不是在客戶端硬塞一個數字。"""
    ApiClient().ask("cloud", "問題", [])
    assert "k" not in sent["json"]

    ApiClient().ask("cloud", "問題", [], k=20)
    assert sent["json"]["k"] == 20


def test_回應被拆成好用的形狀(sent):
    a = ApiClient().ask("cloud", "q", [])

    assert a.reply == "模型的回答"
    assert a.sources[0]["code"] == "MAT-04"
    assert a.usage == Usage(2764, 73)
    assert len(a.history) == 2


def test_網址結尾的斜線不會變成雙斜線(sent):
    ApiClient(base_url="http://x:8000/").health()
    assert sent["url"] == "http://x:8000/health"


# ── 錯誤 ──────────────────────────────────────────────────────────────
def test_後端的錯誤訊息原樣往上傳(sent):
    """訊息在後端就被 llm.explain() 翻成人話了，客戶端不該再翻一次，
    也不該在這裡認得 Gemini 的 429 是什麼意思。"""
    sent["response"] = FakeResponse(503, {"detail": "Gemini 暫時過載，請稍後再試。"})

    with pytest.raises(ApiError, match="暫時過載") as e:
        ApiClient().ask("cloud", "q", [])
    assert e.value.status == 503


def test_422的detail是清單要拼成一句話(sent):
    """FastAPI 的驗證錯誤每個欄位一則，直接丟給 UI 會是一坨 JSON。"""
    sent["response"] = FakeResponse(422, {"detail": [
        {"msg": "String should have at least 1 character"},
        {"msg": "Input should be 'cloud' or 'onperm'"},
    ]})

    with pytest.raises(ApiError, match="at least 1 character；.*cloud"):
        ApiClient().ask("cloud", "", [])


def test_回的不是JSON也要給得出訊息(sent):
    """例如反向代理回了一頁 HTML。"""
    sent["response"] = FakeResponse(502, None, text="<html>Bad Gateway</html>")

    with pytest.raises(ApiError, match="502"):
        ApiClient().ask("cloud", "q", [])


def test_連不上後端時要講怎麼辦(sent):
    """最常見的狀況就是忘了開後端。訊息要直接告訴使用者下一步。"""
    sent["response"] = requests.exceptions.ConnectionError()

    with pytest.raises(ApiError, match="uvicorn api.main:app"):
        ApiClient(base_url="http://x:8000").ask("cloud", "q", [])


def test_逾時的訊息要帶秒數(sent):
    sent["response"] = requests.exceptions.Timeout()

    with pytest.raises(ApiError, match="180 秒"):
        ApiClient(timeout=180).ask("cloud", "q", [])
