"""兩個 ChatBot 的對話邏輯。

這裡不碰網路：地端把 requests.post 換掉，雲端把 client 換掉。
測的是「檢索結果怎麼變成 prompt」「出錯時 history 有沒有壞掉」這類
純邏輯——它們才是會默默壞掉、又不會被使用者立刻發現的部分。
"""
import pytest
import requests

from cloud.chat_bot import CloudChatBot
from onperm.chat_bot import HISTORY_LIMIT, OnpremChatBot
from shared.knowledge import Doc


class StubRetriever:
    """回傳指定 hits 的假檢索器。"""

    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, k=5):
        return self.hits[:k] if k else self.hits


def hit(code, n=1):
    d = Doc(id=f"{code}#1", text="塊", label=f"{code} · 公司 · 職缺",
            full=f"代號：{code}\n完整職缺內容", group=code)
    return (d, 0.9)


# ══ 地端 ══════════════════════════════════════════════════════════════
class FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._ok = status_ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.exceptions.HTTPError("500")

    def json(self):
        return self._payload


def reply_payload(text="模型的回答"):
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def sent(monkeypatch):
    """攔下送出去的 payload，並回一個固定答案。"""
    box = {}

    def fake_post(url, json=None, timeout=None):
        box["url"], box["json"] = url, json
        return FakeResponse(reply_payload())

    monkeypatch.setattr(requests, "post", fake_post)
    return box


def test_第一句就離題會短路不打模型(monkeypatch):
    """檢索不到又沒有上下文時，直接回「資料裡沒有。」，省掉一次推論。"""
    called = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: called.append(1))
    bot = OnpremChatBot(retriever=StubRetriever([]))
    history = []

    reply, hits = bot.ask("推薦一家餐廳", history)

    assert reply == "資料裡沒有。"
    assert hits == []
    assert called == []                                   # 沒打模型
    assert len(history) == 2                              # 但對話還是有記錄


def test_有上下文時就算檢索不到也要交給模型(sent):
    """「那薪水呢？」這種跟隨問句檢索分數本來就低，短路會誤殺。"""
    bot = OnpremChatBot(retriever=StubRetriever([]))
    history = [{"role": "user", "content": "前一題"},
               {"role": "assistant", "content": "前一答"}]

    reply, _ = bot.ask("那薪水呢？", history)

    assert reply == "模型的回答"
    assert sent["json"] is not None                       # 有打模型


def test_送出去的訊息帶資料但history存乾淨的問題(sent):
    bot = OnpremChatBot(retriever=StubRetriever([hit("A-01")]))
    history = []

    bot.ask("A-01 是什麼職缺？", history)

    sent_last = sent["json"]["messages"][-1]["content"]
    assert "【資料】" in sent_last and "代號：A-01" in sent_last
    assert history[0] == {"role": "user", "content": "A-01 是什麼職缺？"}   # 沒有【資料】


def test_system不進history但每次都要送(sent):
    bot = OnpremChatBot(retriever=StubRetriever([hit("A-01")]))
    history = []

    bot.ask("問題", history)

    assert sent["json"]["messages"][0]["role"] == "system"
    assert all(m["role"] != "system" for m in history)


def test_超過五筆改送label清單(sent):
    """篩選型問句可能撈回 20 個職缺，塞完整內容會爆 context。
    完整清單本來就由 UI 的來源列負責，模型只要當摘要。"""
    bot = OnpremChatBot(retriever=StubRetriever([hit(f"J{i}") for i in range(8)]))

    bot.ask("有哪些台北的職缺？", [], k=8)

    sent_last = sent["json"]["messages"][-1]["content"]
    assert "J0 · 公司 · 職缺" in sent_last                 # label
    assert "完整職缺內容" not in sent_last                  # 沒塞 full


def test_五筆以內送完整職缺(sent):
    """少數幾筆就給完整內容，模型才答得出細節。"""
    bot = OnpremChatBot(retriever=StubRetriever([hit("A-01"), hit("B-01")]))

    bot.ask("問題", [])

    assert "完整職缺內容" in sent["json"]["messages"][-1]["content"]


def test_連線失敗時history要維持呼叫前的樣子(monkeypatch):
    """不收回剛 append 的問題，history 會變成「問題、問題、答案」錯位。"""
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("server 沒開")

    monkeypatch.setattr(requests, "post", boom)
    bot = OnpremChatBot(retriever=StubRetriever([hit("A-01")]))
    history = [{"role": "user", "content": "舊問題"},
               {"role": "assistant", "content": "舊答案"}]
    before = list(history)

    with pytest.raises(requests.exceptions.ConnectionError):
        bot.ask("新問題", history)

    assert history == before


def test_回傳格式不對也要收回問題(monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: FakeResponse({"error": "什麼鬼"}))
    bot = OnpremChatBot(retriever=StubRetriever([hit("A-01")]))
    history = []

    with pytest.raises(RuntimeError, match="非預期"):
        bot.ask("問題", history)

    assert history == []


def test_history超過上限會從前面砍(sent):
    """上限是奇數，切完第一則仍然是 user，問答配對不會歪掉。"""
    bot = OnpremChatBot(retriever=StubRetriever([hit("A-01")]))
    history = []
    for i in range(10):
        bot.ask(f"問題{i}", history)

    assert len(history) <= HISTORY_LIMIT + 1        # ask() 結尾又 append 了答案
    assert history[0]["role"] == "user"


# ══ 雲端 ══════════════════════════════════════════════════════════════
class FakeUsage:
    def __init__(self, prompt, candidates, thoughts):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts


class FakeGeminiClient:
    def __init__(self, text="雲端的回答", usage=None):
        self._text = text
        self._usage = usage or FakeUsage(100, 50, 30)
        self.models = self
        self.sent = None

    def generate_content(self, model=None, contents=None, config=None):
        self.sent = contents
        return type("R", (), {"text": self._text, "usage_metadata": self._usage})()


def test_雲端的輸出token要含思考token():
    """thoughts 是使用者看不到的內部草稿，但按輸出計費——不加會嚴重低估。"""
    client = FakeGeminiClient(usage=FakeUsage(100, 50, 30))
    bot = CloudChatBot(retriever=StubRetriever([hit("A-01")]), client=client)

    _, _, usage = bot.ask("問題", [])

    assert usage == {"in": 100, "out": 80}          # 50 + 30，不是 50


def test_雲端的usage欄位可能是None():
    """google-genai 的每個 token 欄位都是 Optional[int]，None 不能讓它炸。"""
    client = FakeGeminiClient(usage=FakeUsage(None, None, None))
    bot = CloudChatBot(retriever=StubRetriever([hit("A-01")]), client=client)

    _, _, usage = bot.ask("問題", [])

    assert usage == {"in": 0, "out": 0}


def test_雲端短路時usage是零():
    """沒打生成 API，就不該記任何 token。"""
    bot = CloudChatBot(retriever=StubRetriever([]), client=FakeGeminiClient())

    reply, hits, usage = bot.ask("推薦一家餐廳", [])

    assert reply == "資料裡沒有。"
    assert usage == {"in": 0, "out": 0}


def test_雲端history用的是gemini格式():
    """parts / model，不是 OpenAI 的 content / assistant。
    兩邊刻意不同，UI 要分別處理。"""
    bot = CloudChatBot(retriever=StubRetriever([hit("A-01")]), client=FakeGeminiClient())
    history = []

    bot.ask("問題", history)

    assert history[0]["role"] == "user" and "parts" in history[0]
    assert history[1]["role"] == "model"


def test_雲端沒拿到內容要收回問題():
    bot = CloudChatBot(retriever=StubRetriever([hit("A-01")]),
                       client=FakeGeminiClient(text=""))
    history = []

    with pytest.raises(RuntimeError, match="沒拿到內容"):
        bot.ask("問題", history)

    assert history == []
