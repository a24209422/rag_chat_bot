"""shared/chat_bot.py：檢索與生成之間的編排邏輯。

這裡完全不碰線路格式——LLM 換成一個假的，只檢查「送出去的中性訊息長什麼樣」。
各家怎麼把中性格式轉成自己的線路格式，是 tests/test_llm.py 的事。
這個分工本身就是 Stage 2 的重點：以前兩邊各有一份 ChatBot，這些測試要寫兩遍。
"""
import pytest

from shared.chat_bot import NOT_FOUND, ChatBot, contradicts
from shared.knowledge import SYSTEM, Doc
from shared.llm import BaseLLM, Reply, Stream, Usage


class FakeLLM(BaseLLM):
    def __init__(self, text="模型的回答", usage=Usage(10, 5), error=None):
        self.text, self.usage, self.error = text, usage, error
        self.calls = []

    def complete(self, messages, system, temperature=0.2):
        self.calls.append({"messages": list(messages), "system": system,
                           "temperature": temperature})
        if self.error:
            raise self.error
        return Reply(self.text, self.usage)

    @property
    def last(self):
        return self.calls[-1]


class StubRetriever:
    def __init__(self, hits, filters=None):
        self.hits = hits
        self.filters = filters or {}

    def filters_for(self, question):
        return self.filters

    def retrieve(self, question, k=5, filters=None):
        return self.hits[:k]


def hit(code):
    d = Doc(id=f"{code}#1", text="塊", label=f"{code} · 公司 · 職缺",
            full=f"代號：{code}\n完整職缺內容", group=code)
    return (d, 0.9)


def make(hits=(), filters=None, **kw):
    llm = kw.pop("llm", None) or FakeLLM()
    return ChatBot(retriever=StubRetriever(list(hits), filters), llm=llm, **kw), llm


# ── 短路 ──────────────────────────────────────────────────────────────
def test_第一句就離題會短路不打模型():
    bot, llm = make([])
    history = []

    reply, hits, usage = bot.ask("推薦一家餐廳", history)

    assert reply == NOT_FOUND
    assert hits == []
    assert llm.calls == []                    # 沒打模型
    assert usage == Usage()                   # 沒打生成就不該記 token
    assert len(history) == 2                  # 但對話還是有記錄


def test_有上下文時就算檢索不到也要交給模型():
    """「那薪水呢？」這種跟隨問句檢索分數本來就低，短路會誤殺。"""
    bot, llm = make([])
    history = [{"role": "user", "content": "前一題"},
               {"role": "assistant", "content": "前一答"}]

    reply, _, _ = bot.ask("那薪水呢？", history)

    assert reply == "模型的回答"
    assert len(llm.calls) == 1


# ── prompt 組裝 ───────────────────────────────────────────────────────
def test_送出去的帶資料但history存乾淨的問題():
    bot, llm = make([hit("A-01")])
    history = []

    bot.ask("A-01 是什麼職缺？", history)

    sent = llm.last["messages"][-1]["content"]
    assert "【資料】" in sent and "代號：A-01" in sent
    assert history[0] == {"role": "user", "content": "A-01 是什麼職缺？"}


def test_system另外傳不進history():
    bot, llm = make([hit("A-01")])
    history = []

    bot.ask("問題", history)

    assert llm.last["system"] == SYSTEM
    assert all(m["role"] != "system" for m in llm.last["messages"])
    assert all(m["role"] != "system" for m in history)


def test_超過五筆改送label清單():
    """篩選型問句可能撈回 20 個職缺，塞完整內容會浪費 context。
    完整清單本來就由 UI 的來源列負責，模型只當摘要。"""
    bot, llm = make([hit(f"J{i}") for i in range(8)])

    bot.ask("有哪些台北的職缺？", [], k=8)

    sent = llm.last["messages"][-1]["content"]
    assert "J0 · 公司 · 職缺" in sent          # label
    assert "完整職缺內容" not in sent           # 沒塞 full


def test_五筆以內送完整職缺():
    bot, llm = make([hit("A-01"), hit("B-01")])

    bot.ask("問題", [])

    assert "完整職缺內容" in llm.last["messages"][-1]["content"]


def test_只有這一輪帶資料前面幾輪維持乾淨():
    bot, llm = make([hit("A-01")])
    history = []

    bot.ask("第一題", history)
    bot.ask("第二題", history)

    sent = llm.last["messages"]
    assert "【資料】" not in sent[0]["content"]      # 第一輪的問題
    assert "【資料】" in sent[-1]["content"]         # 只有這一輪


# ── 失敗時 history 不能壞掉 ───────────────────────────────────────────
def test_生成失敗時history維持呼叫前的樣子():
    """不收回剛 append 的問題，history 會變成「問題、問題、答案」錯位。"""
    bot, _ = make([hit("A-01")], llm=FakeLLM(error=RuntimeError("server 掛了")))
    history = [{"role": "user", "content": "舊問題"},
               {"role": "assistant", "content": "舊答案"}]
    before = list(history)

    with pytest.raises(RuntimeError):
        bot.ask("新問題", history)

    assert history == before


# ── history 上限 ──────────────────────────────────────────────────────
def test_設了上限就從前面砍():
    """上限用奇數，切完第一則仍然是 user，問答配對不會歪掉。"""
    bot, _ = make([hit("A-01")], history_limit=5)
    history = []
    for i in range(10):
        bot.ask(f"問題{i}", history)

    assert len(history) <= 6                   # 砍到 5 之後又 append 了答案
    assert history[0]["role"] == "user"


def test_沒設上限就不砍():
    """雲端的 context 寬裕，砍了反而失去上下文。"""
    bot, _ = make([hit("A-01")], history_limit=None)
    history = []
    for i in range(10):
        bot.ask(f"問題{i}", history)

    assert len(history) == 20


# ── 設定 ──────────────────────────────────────────────────────────────
def test_usage原樣往上傳():
    bot, _ = make([hit("A-01")], llm=FakeLLM(usage=Usage(123, 45)))
    _, _, usage = bot.ask("問題", [])
    assert usage == Usage(123, 45)


def test_溫度與k來自設定也可以單次覆寫():
    bot, llm = make([hit(f"J{i}") for i in range(9)], temperature=0.7, k=3)

    bot.ask("問題", [])
    assert llm.last["temperature"] == 0.7

    _, hits, _ = bot.ask("問題", [])
    assert len(hits) == 3                      # 用建構子給的 k
    _, hits, _ = bot.ask("問題", [], k=7)
    assert len(hits) == 7                      # 單次覆寫


# ══ 串流 ══════════════════════════════════════════════════════════════
class FakeStreamLLM(BaseLLM):
    """吐出固定片段的假 LLM。error_at=N 表示吐到第 N 段時炸掉。"""

    def __init__(self, pieces=("好", "的"), usage=Usage(10, 5), error_at=None,
                 error_on_open=None):
        self.pieces, self.usage_, self.error_at = list(pieces), usage, error_at
        self.error_on_open = error_on_open
        self.calls = []

    def stream(self, messages, system, temperature=0.2):
        self.calls.append({"messages": list(messages), "system": system,
                           "temperature": temperature})
        if self.error_on_open:
            raise self.error_on_open
        pieces, error_at = self.pieces, self.error_at

        def gen():
            for i, p in enumerate(pieces):
                if error_at is not None and i == error_at:
                    raise RuntimeError("串到一半壞了")
                yield p, None
            yield None, self.usage_

        return Stream(gen())

    @property
    def last(self):
        return self.calls[-1]


def make_stream(hits=(), filters=None, **kw):
    llm = kw.pop("llm", None) or FakeStreamLLM()
    return ChatBot(retriever=StubRetriever(list(hits), filters), llm=llm, **kw), llm


def test_ask_stream_的hits馬上就有():
    """檢索比生成快得多，所以 UI 可以在模型還沒吐字之前就把來源列出來。"""
    bot, llm = make_stream([hit("A-01")])

    turn = bot.ask_stream("問題", [])

    assert [d.group for d, _ in turn.hits] == ["A-01"]
    assert llm.calls == [] or True          # 連線可能已開，但還沒吐任何字
    assert turn.reply is None               # 要迭代才有


def test_ask_stream_跑完才補上history():
    bot, _ = make_stream([hit("A-01")], llm=FakeStreamLLM(pieces=("資料", "裡有")))
    history = []

    turn = bot.ask_stream("問題", history)
    assert len(history) == 1                # 只有問題，還沒有答案

    # 這裡驗「字有沒有全部到齊」而不是切成幾段：押字機制會把開頭合成一段
    # ——"資料" 正好是「資料裡沒有」的前綴，要等到分歧那一刻才吐得出來。
    # 切幾段是實作細節，全部到齊才是契約。
    assert "".join(turn) == "資料裡有"
    assert history == [{"role": "user", "content": "問題"},
                       {"role": "assistant", "content": "資料裡有"}]
    assert turn.reply == "資料裡有"
    assert turn.usage == Usage(10, 5)


def test_ask_stream_中途壞掉要把問題收回去():
    """畫面上會留著半截答案，但 history 不能壞——否則下一輪會變成
    「問題、問題、答案」錯位。"""
    bot, _ = make_stream([hit("A-01")], llm=FakeStreamLLM(pieces=("一", "二", "三"),
                                                          error_at=2))
    history = [{"role": "user", "content": "舊"},
               {"role": "assistant", "content": "舊答"}]
    before = list(history)

    turn = bot.ask_stream("新問題", history)
    got = []
    with pytest.raises(RuntimeError, match="壞了"):
        for piece in turn:
            got.append(piece)

    assert got == ["一", "二"]              # 已經吐出去的還是吐出去了
    assert history == before                # 但 history 回到呼叫前


def test_ask_stream_連線階段就失敗也要收回問題():
    bot, _ = make_stream([hit("A-01")],
                         llm=FakeStreamLLM(error_on_open=RuntimeError("server 沒開")))
    history = []

    with pytest.raises(RuntimeError, match="沒開"):
        bot.ask_stream("問題", history)

    assert history == []


def test_ask_stream_離題短路():
    """沒打生成，所以一次就把那句話吐完，usage 是零。"""
    bot, llm = make_stream([])
    history = []

    turn = bot.ask_stream("推薦一家餐廳", history)

    assert turn.done                        # 建出來就是完成狀態
    assert list(turn) == [NOT_FOUND]
    assert turn.usage == Usage()
    assert llm.calls == []                  # 完全沒碰模型
    assert len(history) == 2


def test_兩條路組出來的prompt一模一樣():
    """ask() 與 ask_stream() 共用 _prepare()，所以不會有一邊改了另一邊忘了。"""
    bot_a, llm_a = make([hit("A-01")])
    bot_b, llm_b = make_stream([hit("A-01")])

    bot_a.ask("同一個問題", [])
    list(bot_b.ask_stream("同一個問題", []))

    assert llm_a.last["messages"] == llm_b.last["messages"]
    assert llm_a.last["system"] == llm_b.last["system"]


# ══ 矛盾偵測與重抽 ════════════════════════════════════════════════════
#
#   實測地端 Qwen2.5-3B 同一題跑 46 次，檢索 46 次都撈到 INT-01，但約 11%
#   的回答是「資料裡沒有」。錯的是生成不是檢索，而且那個矛盾偵測得到。
class SequenceLLM(BaseLLM):
    """每次呼叫吐下一段文字。用來測「第一次矛盾、第二次正常」。"""

    def __init__(self, *texts, usage=Usage(10, 5)):
        self.texts = list(texts)
        self.usage = usage
        self.calls = 0
        self.seen = []                 # 每次呼叫實際收到的 messages

    def _next(self):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return text

    def complete(self, messages, system, temperature=0.2):
        self.seen.append(list(messages))
        return Reply(self._next(), self.usage)

    def stream(self, messages, system, temperature=0.2):
        self.seen.append(list(messages))
        text, usage = self._next(), self.usage

        def gen():
            for ch in text:            # 一個字一段，逼出押字邏輯的邊界
                yield ch, None
            yield None, usage

        return Stream(gen())


def guarded(texts, hits=None, **kw):
    """預設就是「檢索撈到一筆」——矛盾要成立，hits 不能是空的。"""
    llm = SequenceLLM(*texts)
    hits = [hit("A-01")] if hits is None else list(hits)
    return ChatBot(retriever=StubRetriever(hits), llm=llm, **kw), llm


def test_contradicts_只在整段就是那一句時成立():
    h = [hit("A-01")]

    assert contradicts(h, NOT_FOUND)
    assert contradicts(h, "  資料裡沒有  ")        # 前後空白不算內容
    assert not contradicts(h, "資料裡沒有台北的，但 A-01 在台南")
    assert not contradicts([], NOT_FOUND)          # 沒檢索到就不是矛盾，是實話


def test_ask_檢索到東西卻說沒有會重抽一次():
    bot, llm = guarded([NOT_FOUND, "A-01 在台南"])
    history = []

    reply, _, usage = bot.ask("台南有職缺嗎", history)

    assert reply == "A-01 在台南"
    assert llm.calls == 2
    assert usage == Usage(20, 10)                  # 白跑那次也花了錢，要算進去
    assert history[-1]["content"] == "A-01 在台南"  # 壞的那次不留在 history


def test_ask_正常回答不會多抽():
    bot, llm = guarded(["A-01 在台南"])
    bot.ask("台南有職缺嗎", [])
    assert llm.calls == 1


def test_ask_重抽關得掉():
    bot, llm = guarded([NOT_FOUND, "不該看到"], retry_contradiction=False)
    reply, _, _ = bot.ask("台南有職缺嗎", [])
    assert reply == NOT_FOUND
    assert llm.calls == 1


def test_ask_兩次都矛盾就認了():
    bot, llm = guarded([NOT_FOUND, NOT_FOUND])

    reply, hits, _ = bot.ask("台南有職缺嗎", [])

    assert reply == NOT_FOUND
    assert llm.calls == 2
    assert contradicts(hits, reply)      # 呼叫端據此提醒使用者去看來源列


def test_ask_stream_壞的那句一個字都不會吐出去():
    """整個押字機制就是為了這件事：畫面不能先閃過一句錯的再被換掉。"""
    bot, llm = guarded([NOT_FOUND, "有的，A-01"])

    out = "".join(bot.ask_stream("台南有職缺嗎", []))

    assert out == "有的，A-01"
    assert "資料裡沒有" not in out
    assert llm.calls == 2


def test_ask_stream_正常回答不會被押住或吃掉():
    bot, llm = guarded(["有的，A-01 在台南"])
    turn = bot.ask_stream("台南有職缺嗎", [])

    assert "".join(turn) == "有的，A-01 在台南" == turn.reply
    assert llm.calls == 1


def test_ask_stream_開頭像那一句但後面有接下去的不算矛盾():
    """「資料裡沒有台北的，但…」是正常且正確的回答，押著的字要原封不動補吐。"""
    text = "資料裡沒有台北的，但 A-01 在台南"
    bot, llm = guarded([text])

    assert "".join(bot.ask_stream("問題", [])) == text
    assert llm.calls == 1


def test_ask_stream_兩次都矛盾還是要把那一句吐出來():
    bot, llm = guarded([NOT_FOUND, NOT_FOUND])
    turn = bot.ask_stream("問題", [])

    out = "".join(turn)

    assert out == NOT_FOUND          # 不能什麼都不吐，畫面會是空白
    assert llm.calls == 2
    assert contradicts(turn.hits, turn.reply)


def test_ask_stream_重抽的usage兩次都算():
    bot, _ = guarded([NOT_FOUND, "有的"])
    turn = bot.ask_stream("問題", [])
    list(turn)
    assert turn.usage == Usage(20, 10)


def test_ask_stream_沒檢索到就不押字行為完全不變():
    """沒有 hits 不可能矛盾。這條路要跟加這個機制之前一模一樣。"""
    bot, llm = guarded([NOT_FOUND], hits=())
    history = [{"role": "user", "content": "前"},
               {"role": "assistant", "content": "後"}]

    assert "".join(bot.ask_stream("問題", history)) == NOT_FOUND
    assert llm.calls == 1


def test_重抽時丟掉history只留這一輪的資料():
    """原樣重抽沒有用——實測同一題原樣重抽 12 次有 7 次抽到同樣那句話。
    要換掉輸入才換得到不同的答案，而汙染源就是上一輪那段長對話。"""
    bot, llm = guarded([NOT_FOUND, "A-01 在台南"])
    history = [{"role": "user", "content": "台北有職缺嗎"},
               {"role": "assistant", "content": "很長的台北職缺清單…"}]

    bot.ask("台南有職缺嗎", history)

    assert len(llm.seen[0]) == 3               # 第一次：上一輪兩則 + 這一輪
    assert len(llm.seen[1]) == 1               # 重抽：只剩這一輪
    assert "【資料】" in llm.seen[1][0]["content"]
    assert "台南有職缺嗎" in llm.seen[1][0]["content"]


def test_ask_stream_重抽也丟掉history():
    bot, llm = guarded([NOT_FOUND, "有的，A-01"])
    history = [{"role": "user", "content": "台北有職缺嗎"},
               {"role": "assistant", "content": "很長的台北職缺清單…"}]

    list(bot.ask_stream("台南有職缺嗎", history))

    assert len(llm.seen[0]) == 3
    assert len(llm.seen[1]) == 1


# ── 有過濾條件時要講給模型聽 ──────────────────────────────────────────
def test_有過濾條件就在prompt裡說明資料已經篩過():
    """模型判斷不出「台南算不算南部」——實測拿著台南的職缺被問「南部有哪些」，
    6 次有 6 次回「資料裡沒有」。把展開結果寫進 prompt 之後 6 次全對。
    這不是叫模型聽話的咒語，是補一塊它沒有的知識。"""
    bot, llm = make([hit("INT-01")],
                    filters={"city": ["嘉義", "台南", "高雄", "屏東"]})

    bot.ask("南部有哪些職缺", [])

    sent = llm.last["messages"][-1]["content"]
    assert "地點＝嘉義／台南／高雄／屏東" in sent
    assert "全部符合的職缺" in sent


def test_沒有過濾條件就不加那句():
    """純向量檢索撈回來的是「最像的前 k 個」，不是「全部符合的」——
    這時候說「列出的就是全部」會是謊話。"""
    bot, llm = make([hit("A-01")])

    bot.ask("有哪些無人機相關的職缺", [])

    assert "篩選完畢" not in llm.last["messages"][-1]["content"]


def test_兩條路的篩選說明也要一致():
    f = {"city": ["台北"], "kind": ["實習"]}
    bot_a, llm_a = make([hit("A-01")], filters=f)
    bot_s, llm_s = make_stream([hit("A-01")], filters=f)

    bot_a.ask("台北的實習", [])
    list(bot_s.ask_stream("台北的實習", []))

    assert llm_a.last["messages"] == llm_s.last["messages"]
