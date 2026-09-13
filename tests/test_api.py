"""api/ 的測試。

端點不碰真的模型：用 dependency_overrides 把「side → ChatBot」整個換掉。
這一層要驗的是 HTTP 的事——參數驗證、Doc → Source 的轉換、例外翻成哪個
狀態碼、以及「伺服器不記 history」這件事。
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.deps import get_bot_factory, get_registry
from api.main import app
from shared.chat_bot import Turn
from shared.knowledge import Doc
from shared.llm import Stream, Usage
from shared.registry import Registry


class FakeBot:
    def __init__(self, hits=(), reply="模型的回答", usage=Usage(10, 5),
                 error=None, explain=None):
        self.hits, self.reply, self.usage = list(hits), reply, usage
        self.error, self._explain = error, explain
        self.calls = []
        bot = self

        class _LLM:
            def explain(self, exc):
                return bot._explain

        self.llm = _LLM()

    def ask(self, question, history, k=None):
        self.calls.append({"question": question, "history": list(history), "k": k})
        if self.error:
            raise self.error
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": self.reply})
        return self.reply, self.hits, self.usage


def hit(code, score=0.9):
    return (Doc(id=f"{code}#1", text="塊", label=f"{code} · 公司 · 職缺",
                full="完整", group=code), score)


@pytest.fixture
def client(tmp_path):
    """預設每一邊都給同一個 FakeBot。要換就重設 dependency_overrides。

    registry 一定要換成暫存的——不然測試會在專案的 data/ 裡留下 registry.db。
    """
    app.dependency_overrides[get_registry] = lambda: Registry(tmp_path / "reg.db")
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def use(bot_or_factory):
    factory = (bot_or_factory if callable(bot_or_factory)
               else lambda side: bot_or_factory)
    app.dependency_overrides[get_bot_factory] = lambda: factory


# ── /health ───────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["sides"] == ["cloud", "onperm"]


# ── /chat 正常路徑 ────────────────────────────────────────────────────
def test_chat_回傳完整的四樣東西(client):
    use(FakeBot(hits=[hit("MAT-04", 0.91)], usage=Usage(2764, 73)))

    r = client.post("/chat", json={"question": "有哪些無人機職缺？", "side": "onperm"})

    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "模型的回答"
    assert body["sources"] == [{"code": "MAT-04", "label": "MAT-04 · 公司 · 職缺",
                                "score": 0.91}]
    assert body["usage"] == {"prompt": 2764, "output": 73}
    assert body["history"] == [{"role": "user", "content": "有哪些無人機職缺？"},
                               {"role": "assistant", "content": "模型的回答"}]


def test_代號是獨立欄位不是散文裡的字(client):
    """實測 3B 模型被要求列 20 筆只列得出 16~17 筆，而且常常不附代號。
    所以完整清單一定要由程式給，前端不該去 reply 裡撈。"""
    use(FakeBot(hits=[hit("A-01"), hit("B-02")], reply="我懶得列代號"))

    body = client.post("/chat", json={"question": "q"}).json()

    assert [s["code"] for s in body["sources"]] == ["A-01", "B-02"]


def test_伺服器不記history(client):
    """兩次獨立請求之間不該有任何殘留——這是無狀態設計的重點。
    對方專案就是把 ChatHistory 做成模組層單例，導致併發使用者互看對話。"""
    bot = FakeBot()
    use(bot)

    client.post("/chat", json={"question": "第一題"})
    body = client.post("/chat", json={"question": "第二題"}).json()

    assert bot.calls[1]["history"] == []          # 第二次進來時是空的
    assert len(body["history"]) == 2              # 只有這一輪


def test_history要原樣帶進去(client):
    bot = FakeBot()
    use(bot)
    prior = [{"role": "user", "content": "舊問題"},
             {"role": "assistant", "content": "舊答案"}]

    body = client.post("/chat", json={"question": "新問題", "history": prior}).json()

    assert bot.calls[0]["history"] == prior
    assert len(body["history"]) == 4               # 舊的兩則 + 新的兩則


def test_side會挑到對應的那一邊(client):
    seen = []
    use(lambda side: seen.append(side) or FakeBot())

    client.post("/chat", json={"question": "q", "side": "onperm"})
    client.post("/chat", json={"question": "q", "side": "cloud"})

    assert seen == ["onperm", "cloud"]


def test_side預設是雲端(client):
    seen = []
    use(lambda side: seen.append(side) or FakeBot())
    client.post("/chat", json={"question": "q"})
    assert seen == ["cloud"]


def test_k會傳下去(client):
    bot = FakeBot()
    use(bot)
    client.post("/chat", json={"question": "q", "k": 20})
    assert bot.calls[0]["k"] == 20


# ── 參數驗證 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("body", [
    {"question": ""},                       # 空問題
    {"question": "q", "side": "gcp"},       # 不存在的 side
    {"question": "q", "k": 0},              # k 太小
    {"question": "q", "k": 999},            # k 太大
    {"history": [{"role": "user", "content": "x"}]},              # 少了 question
    {"question": "q", "history": [{"role": "system", "content": "x"}]},  # 角色不合法
])
def test_參數不合法回422(client, body):
    use(FakeBot())
    assert client.post("/chat", json=body).status_code == 422


# ── 例外翻成狀態碼 ────────────────────────────────────────────────────
def test_上游過載回503(client):
    """explain() 認得的都是「上游的可預期狀況」，不是我們自己壞了。"""
    use(FakeBot(error=RuntimeError("boom"), explain="Gemini 暫時過載，請稍後再試。"))

    r = client.post("/chat", json={"question": "q"})

    assert r.status_code == 503
    assert r.json()["detail"] == "Gemini 暫時過載，請稍後再試。"


def test_配額用完回429(client):
    """429 跟 503 要分開：一個是「等一下再試」，一個是「今天沒了」，
    呼叫端的處理不一樣。"""
    err = RuntimeError("quota")
    err.code = 429
    use(FakeBot(error=err, explain="已達免費方案的用量上限。"))

    assert client.post("/chat", json={"question": "q"}).status_code == 429


def test_不認得的錯誤回500(client):
    """500 要留給「我們自己壞了」，不然呼叫端沒辦法判斷該不該重試。"""
    use(FakeBot(error=ValueError("誰知道"), explain=None))

    r = client.post("/chat", json={"question": "q"})

    assert r.status_code == 500
    assert "誰知道" in r.json()["detail"]


def test_那一邊起不來回503(client):
    """例如雲端沒設金鑰。"""
    def boom(side):
        raise RuntimeError("找不到 GEMINI_API_KEY")

    use(boom)
    r = client.post("/chat", json={"question": "q"})

    assert r.status_code == 503
    assert "GEMINI_API_KEY" in r.json()["detail"]


# ── /health 不該有副作用 ──────────────────────────────────────────────
def test_health不會把還沒用到的那一邊建起來(monkeypatch):
    """一開始這裡用 lru_cache，而 lru_cache 沒有「只看不建」的 API——
    為了知道哪幾邊已載入就得去呼叫它，結果 /health 把每一邊都建了一遍。
    地端那邊會連帶載 e5，健康檢查不該做這種事。"""
    from api import deps

    built = []

    def fake_chat_for(side):
        built.append(side)
        return FakeBotWithIndex(indexed=True)

    monkeypatch.setattr(deps.providers, "chat_for", fake_chat_for)
    monkeypatch.setattr(deps, "_bots", {})

    assert deps.loaded() == []
    deps.loaded()
    deps.loaded()
    assert built == []                      # 查了三次也沒建任何東西

    deps.bot_for("cloud")                   # 真的要用才建
    assert built == ["cloud"]
    assert deps.loaded() == ["cloud"]


def test_loaded回報的是索引算好了不是物件建好了(monkeypatch):
    """建 ChatBot 幾乎不花時間，算索引才是貴的（地端實測 47 秒）。
    /health 講「已載入」如果只代表前者，那句話沒有意義——
    而且會讓人以為 API_WARM 生效了，其實沒有。"""
    from api import deps

    monkeypatch.setattr(deps, "_bots", {"cloud": FakeBotWithIndex(indexed=False)})
    assert deps.loaded() == []              # 物件在了，但索引還沒算

    deps._bots["cloud"].retriever.indexed = True
    assert deps.loaded() == ["cloud"]


class FakeBotWithIndex:
    def __init__(self, indexed):
        self.retriever = type("R", (), {"indexed": indexed})()


# ══ /chat/stream ══════════════════════════════════════════════════════
def sse_events(text):
    """把回應本文拆成 [(事件名, 資料)]。"""
    out, event = [], None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            out.append((event, json.loads(line[6:])))
            event = None
    return out


class FakeStreamBot(FakeBot):
    """ask_stream 用真的 Turn，這樣測到的是實際會跑的那份程式。"""

    def __init__(self, pieces=("好", "的"), error_at=None, **kw):
        super().__init__(**kw)
        self.pieces, self.error_at = list(pieces), error_at

    def ask_stream(self, question, history, k=None):
        self.calls.append({"question": question, "history": list(history), "k": k})
        if self.error:
            raise self.error
        history.append({"role": "user", "content": question})
        pieces, error_at = self.pieces, self.error_at

        def gen():
            for i, p in enumerate(pieces):
                if error_at is not None and i == error_at:
                    raise RuntimeError("串到一半壞了")
                yield p, None
            yield None, self.usage

        return Turn(self.hits, stream=Stream(gen()), history=history)


def test_串流的事件順序(client):
    use(FakeStreamBot(hits=[hit("MAT-04", 0.91)], pieces=("資料", "裡有"),
                      usage=Usage(2764, 78)))

    r = client.post("/chat/stream", json={"question": "q", "side": "onperm"})

    assert r.status_code == 200
    assert r.headers["content-type"] == "text/event-stream; charset=utf-8"
    events = sse_events(r.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "done"]

    assert events[0][1]["sources"] == [{"code": "MAT-04",
                                        "label": "MAT-04 · 公司 · 職缺", "score": 0.91}]
    assert [d["text"] for _, d in events[1:3]] == ["資料", "裡有"]
    assert events[-1][1]["usage"] == {"prompt": 2764, "output": 78}
    assert events[-1][1]["history"][-1] == {"role": "assistant", "content": "資料裡有"}


def test_sources是第一個事件(client):
    """檢索比生成快得多。前端可以先把來源列出來，不必等模型講完。"""
    use(FakeStreamBot(hits=[hit("A-01")], pieces=("一", "二", "三")))

    events = sse_events(client.post("/chat/stream", json={"question": "q"}).text)

    assert events[0][0] == "sources"


def test_中文不會變成latin1亂碼(client):
    """SSE 的 Content-Type 若沒宣告 charset，HTTP 規定 text/* 退回 ISO-8859-1。
    所以伺服器一定要宣告——實測 requests 就是照規定猜的。"""
    use(FakeStreamBot(pieces=("耐能智慧", "在徵人")))

    r = client.post("/chat/stream", json={"question": "q"})

    assert "charset=utf-8" in r.headers["content-type"]
    assert [d["text"] for e, d in sse_events(r.text) if e == "token"] == ["耐能智慧",
                                                                          "在徵人"]


def test_生成中途壞掉要用error事件回報(client):
    """這時 HTTP 狀態早就送出去了（200），改不了。呼叫端一定要處理這個事件，
    不然畫面會停在半截答案上。"""
    use(FakeStreamBot(pieces=("一", "二", "三"), error_at=2,
                      explain="llama.cpp server 沒開。"))

    r = client.post("/chat/stream", json={"question": "q"})

    assert r.status_code == 200             # 標頭早就送出去了
    events = sse_events(r.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "error"]
    assert events[-1][1]["detail"] == "llama.cpp server 沒開。"


def test_連線階段的錯誤仍然是正確的狀態碼(client):
    """先預抽第一段，抽得出來才開始串流——所以 503 不會偽裝成 200。"""
    err = RuntimeError("quota")
    err.code = 429
    use(FakeStreamBot(error=err, explain="已達用量上限。"))

    r = client.post("/chat/stream", json={"question": "q"})

    assert r.status_code == 429
    assert r.json()["detail"] == "已達用量上限。"


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "q", "side": "gcp"}])
def test_串流端點的參數驗證也是422(client, body):
    use(FakeStreamBot())
    assert client.post("/chat/stream", json=body).status_code == 422


# ══ /documents ════════════════════════════════════════════════════════
@pytest.fixture
def parsed(monkeypatch):
    """換掉 PDF 解析——這一層要測的是 HTTP 與 registry，不是解析。
    解析本身有 tests/test_ingest.py。"""
    from api import main

    box = {"chunks": [{"id": "A-01#基本", "text": "塊", "group": "A-01"}], "jobs": 1}

    def fake_parse(data, source=""):
        if box.get("error"):
            raise box["error"]
        return [dict(c, source=source) for c in box["chunks"]], box["jobs"]

    monkeypatch.setattr(main, "docs_from_pdf", fake_parse)
    return box


def upload(client, name="甲公司.pdf", content=b"%PDF-1.4 fake"):
    return client.post("/documents",
                       files={"file": (name, content, "application/pdf")})


def test_上傳成功(client, parsed):
    r = upload(client)

    assert r.status_code == 201
    doc = r.json()["document"]
    assert doc["filename"] == "甲公司.pdf"
    assert doc["jobs"] == 1
    assert doc["size"] == len(b"%PDF-1.4 fake")
    assert len(doc["version"]) == 16          # 內容的指紋
    assert client.get("/documents").json()[0]["doc_id"] == doc["doc_id"]


def test_只收pdf(client, parsed):
    r = upload(client, name="筆記.md", content=b"# hello")
    assert r.status_code == 400
    assert ".pdf" in r.json()["detail"]


def test_檔案太大(client, parsed, monkeypatch):
    from shared.settings import Settings, settings

    small = Settings(_env_file=None, max_upload_bytes=8)
    monkeypatch.setattr("api.main.settings", lambda: small)
    assert settings                            # 原本那個沒被動到

    assert upload(client, content=b"x" * 100).status_code == 413


def test_同名同內容回409(client, parsed):
    """沒有事情要做——重算索引是白費力氣。"""
    upload(client)
    r = upload(client)

    assert r.status_code == 409
    assert "已經上傳過" in r.json()["detail"]


def test_同名不同內容是更新(client, parsed):
    """這就是增量更新要處理的情況：文件改了，只重算那一份的塊。"""
    first = upload(client).json()["document"]
    parsed["chunks"] = [{"id": "A-01#基本", "text": "新的", "group": "A-01"},
                        {"id": "A-02#基本", "text": "又一個", "group": "A-02"}]
    parsed["jobs"] = 2
    second = upload(client, content=b"%PDF-1.4 changed").json()["document"]

    assert second["doc_id"] == first["doc_id"]      # 同一份文件
    assert second["version"] != first["version"]    # 不同版本
    assert second["jobs"] == 2
    assert len(client.get("/documents").json()) == 1


def test_解析不了回400(client, parsed):
    parsed["error"] = ValueError("這不是 PDF")
    r = upload(client)
    assert r.status_code == 400
    assert "解析不了" in r.json()["detail"]


def test_解析不出職缺回400(client, parsed):
    """檔案是合法 PDF，但不是媒合會職缺表的格式。"""
    parsed["chunks"], parsed["jobs"] = [], 0
    r = upload(client)
    assert r.status_code == 400
    assert "職缺" in r.json()["detail"]


def test_刪除(client, parsed):
    doc_id = upload(client).json()["document"]["doc_id"]

    r = client.delete(f"/documents/{doc_id}")

    assert r.status_code == 200
    assert r.json()["doc_id"] == doc_id
    assert client.get("/documents").json() == []


def test_刪不存在的回404(client, parsed):
    assert client.delete("/documents/不存在").status_code == 404


def test_上傳會更新已載入那幾邊的索引(client, parsed, monkeypatch):
    """還沒載入的不用管——它們第一次被建起來時就會自己跟 registry 對齊。
    先建好再同步是浪費，尤其地端要載 e5。"""
    from api import deps

    class FakeRetrieverStub:
        indexed = True

        def __init__(self):
            self.refreshed = 0

        def refresh(self):
            self.refreshed += 1
            return 3, 0

    stub = FakeRetrieverStub()
    monkeypatch.setattr(deps, "_bots",
                        {"cloud": type("B", (), {"retriever": stub})()})

    body = upload(client).json()

    assert body["index"] == {"cloud": {"added": 3, "removed": 0}}
    assert stub.refreshed == 1


def test_health會報告塊數與文件數(client, parsed):
    upload(client)
    body = client.get("/health").json()
    assert body["documents"] == 1
    assert body["chunks"] == {}               # 這個測試沒有任何一邊被載入


def test_cors允許前端的來源(client, parsed):
    """React 是從瀏覽器打 API 的，沒有這個會被擋下來——而且錯誤只出現在
    devtools 的 console，後端日誌什麼都看不到。"""
    r = client.options("/chat", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    })
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors放行本機的其他埠(client, parsed):
    """5173 被佔走時 vite 會自己跳 5174，`vite preview` 是 4173。埠一換就整個
    前端打不進來太脆，所以本機任意埠都放行（cors_origin_regex）。"""
    for origin in ["http://localhost:5174", "http://127.0.0.1:4173"]:
        r = client.options("/chat/stream", headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        assert r.status_code == 200, origin
        assert r.headers["access-control-allow-origin"] == origin


def test_cors擋下本機以外的來源並在日誌說明(client, parsed, caplog):
    """放行本機不等於放行全部。而且被擋時要在後端日誌講出是哪個來源——
    Starlette 的說明寫在預檢的 body 裡，那是瀏覽器自己發的請求，沒人看得到。"""
    r = client.options("/chat/stream", headers={
        "Origin": "https://evil.example.com",
        "Access-Control-Request-Method": "POST",
    })
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers
    assert "evil.example.com" in caplog.text
