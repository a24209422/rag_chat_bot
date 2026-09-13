"""shared/api_client.py：打後端的客戶端。

換掉 requests.request，所以不需要真的起一個後端。要驗的是「後端回什麼、
UI 看到什麼」——尤其是錯誤訊息，因為那是使用者唯一看得到的東西。
"""
import json

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


# ══ 串流 ══════════════════════════════════════════════════════════════
class FakeSSEResponse:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status
        self.encoding = "ISO-8859-1"     # requests 對 text/* 的預設猜法
        self.text = ""

    def json(self):
        raise ValueError("串流回應沒有整份 JSON")

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


def sse(event, data):
    return [f"event: {event}", f"data: {json.dumps(data, ensure_ascii=False)}", ""]


OK_LINES = (sse("sources", {"sources": [{"code": "MAT-04", "label": "L", "score": 0.9}]})
            + sse("token", {"text": "資料"})
            + sse("token", {"text": "裡有"})
            + sse("done", {"usage": {"prompt": 2764, "output": 78},
                           "history": [{"role": "user", "content": "q"},
                                       {"role": "assistant", "content": "資料裡有"}]}))


@pytest.fixture
def streamed(monkeypatch):
    box = {"response": FakeSSEResponse(OK_LINES)}

    def fake_request(method, url, timeout=None, stream=False, **kw):
        box.update(method=method, url=url, stream=stream, **kw)
        resp = box["response"]
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(requests, "request", fake_request)
    return box


def test_串流要告訴requests別緩衝(streamed):
    """不傳 stream=True 的話 requests 會把整個回應讀完才回來——
    對串流端點而言那等於「等它全部講完」，逐段吐字完全失效。"""
    ApiClient().stream("onperm", "q", [])

    assert streamed["stream"] is True
    assert streamed["url"].endswith("/chat/stream")


def test_sources在開始吐字之前就拿得到(streamed):
    s = ApiClient().stream("onperm", "q", [])
    assert s.sources == [{"code": "MAT-04", "label": "L", "score": 0.9}]
    assert s.reply == ""                    # 還沒迭代


def test_迭代tokens之後才有reply與usage(streamed):
    s = ApiClient().stream("onperm", "q", [])

    assert list(s.tokens()) == ["資料", "裡有"]
    assert s.reply == "資料裡有"
    assert s.usage == Usage(2764, 78)
    assert len(s.history) == 2


def test_客戶端也要把編碼設成utf8(streamed):
    resp = FakeSSEResponse(OK_LINES)
    streamed["response"] = resp

    list(ApiClient().stream("onperm", "q", []).tokens())

    assert resp.encoding == "utf-8"


def test_中途的error事件要變成例外(streamed):
    """不丟例外的話，UI 會以為答案正常結束，停在半截文字上。"""
    streamed["response"] = FakeSSEResponse(
        sse("sources", {"sources": []})
        + sse("token", {"text": "一"})
        + sse("error", {"detail": "llama.cpp server 沒開。"}))

    s = ApiClient().stream("onperm", "q", [])
    got = []
    with pytest.raises(ApiError, match="沒開"):
        for piece in s.tokens():
            got.append(piece)
    assert got == ["一"]


def test_一開始就是error事件(streamed):
    streamed["response"] = FakeSSEResponse(sse("error", {"detail": "壞了"}))
    with pytest.raises(ApiError, match="壞了"):
        ApiClient().stream("onperm", "q", [])


def test_沒送sources就結束也要丟例外(streamed):
    streamed["response"] = FakeSSEResponse([])
    with pytest.raises(ApiError, match="sources"):
        ApiClient().stream("onperm", "q", [])


def test_串流端點的HTTP錯誤仍然照常處理(streamed):
    streamed["response"] = FakeSSEResponse([], status=429)
    streamed["response"].json = lambda: {"detail": "已達用量上限。"}
    with pytest.raises(ApiError, match="用量上限") as e:
        ApiClient().stream("onperm", "q", [])
    assert e.value.status == 429


# ── contradiction：模型說沒有但檢索有 ────────────────────────────────
def test_contradiction_旗標要傳到UI(sent):
    """後端已經為此重問過一次，這個旗標代表「重問完還是矛盾」。
    UI 要據此把使用者的視線導向來源列——那才是完整精確的清單。"""
    sent["response"] = FakeResponse(payload={**OK_PAYLOAD, "contradiction": True})

    assert ApiClient().ask("onperm", "q", []).contradiction is True


def test_後端沒給contradiction就當作沒有(sent):
    """欄位是後來才加的。舊版後端或別的實作沒給時不該炸。"""
    assert ApiClient().ask("onperm", "q", []).contradiction is False


def test_串流的contradiction要到done事件才有(streamed):
    streamed["response"] = FakeSSEResponse(
        sse("sources", {"sources": [{"code": "INT-01", "label": "L", "score": 0.8}]})
        + sse("token", {"text": "資料裡沒有符合南部地點要求的職缺。"})
        + sse("done", {"usage": {"prompt": 1, "output": 1}, "history": [],
                       "contradiction": True}))

    s = ApiClient().stream("onperm", "南部呢？", [])
    assert s.contradiction is False        # 還沒迭代，done 還沒到

    list(s.tokens())
    assert s.contradiction is True
