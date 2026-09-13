# chat_bot.py（檢索 + 生成的唯一一份實作）
#
#   雲端與地端共用這一個類別。差異全部收在建構時傳進來的兩個零件：
#       retriever  CloudRetriever / OnpremRetriever（差在 embed 與 min_score）
#       llm        GeminiLLM / LlamaCppLLM（差在線路格式與 usage 怎麼讀）
#   加第三家就是傳一個新的 llm 進來，這個檔案不用動。
from shared.facets import describe
from shared.knowledge import SYSTEM
from shared.llm import Usage
from shared.settings import settings

NOT_FOUND = "資料裡沒有。"     # 措辭跟 SYSTEM 一致，兩條路說法才不會打架

# 模型拒絕時的開頭。SYSTEM 要它講「資料裡沒有」，但小模型會自己加料——
# 實測看過「資料裡沒有符合南部地點要求的職缺。」這種冗長版本，而它跟乾淨的
# 那一句一樣是錯的（INT-01 就在台南）。只認字面相等會漏掉它。
REFUSALS = ("資料裡沒有", "資料中沒有", "資料沒有", "資料裡找不到", "資料中找不到")
FULL_TEXT_LIMIT = 5           # 超過這個數量就只給 label，不給完整職缺


def contradicts(hits, reply):
    """檢索撈到了東西，模型卻說「沒有」——這是矛盾，而且偵測得到。

    實測地端 Qwen2.5-3B 約每 10 次有 1 次這樣：【資料】裡明明就有那個職缺，
    它還是跳過去挑了 SYSTEM 裡那句話（那句話對小模型是很強的吸引子）。
    同一題跑 46 次，檢索 46 次都對，錯的一直是生成。

    判斷分兩個條件，缺一不可：
      1. 回答以「資料裡沒有」那一族的措辭開頭（見 REFUSALS）
      2. 回答從頭到尾沒提到任何一個撈回來的代號

    第 2 個條件才是關鍵，它把「真的在拒絕」跟「先否定再回答」分開：
      「資料裡沒有台北的，但 A-01 在台南」→ 提到 A-01，是正常且正確的回答
      「資料裡沒有符合南部地點要求的職缺。」→ 一個代號都沒有，是矛盾
    而它剛好搭在既有的契約上——SYSTEM 本來就要求提到職缺一定要附代號。

    ⚠ 第一版只認字面相等（整段就是「資料裡沒有。」），實際上線就被那個冗長
      版本繞過去了：偵測不到 → 不重問 → 使用者看到一句假話。**能被繞過的
      偵測器比沒有更危險**，因為它會讓人以為已經有防護。

    所以「來源列」跟「模型的散文」打架時該信誰，答案本來就寫在
    api/schemas.py 的 Source 上：來源列才是完整精確的清單，散文只當摘要。
    這個函式就是把那句話翻成程式看得懂的形式。

    ⚠ 偵測到之後，重抽「不能只是再抽一次」。實測原樣重抽 12 次有 7 次抽到
      同樣那句話——temperature 0.2 之下，只要 context 把模型推向那句話，
      重抽就是在同一個眾數上打轉。

      有效的是換掉輸入：把 history 拿掉、只留這一輪的【資料】+【問題】。
      實測不帶 history 時同一題從來沒失敗過，帶著 history 最高到 7/12 失敗
      ——汙染源就是上一輪那段長對話。所以下面的重抽是「丟掉上下文再問
      一次」，不是重抽樂透。

      代價：真正需要上下文的跟隨問句（「那薪水呢？」）重抽時會失去上下文。
      但那時候第一次已經答錯了，用【資料】答得出來的版本仍然比一句假的
      「資料裡沒有」好。
    """
    text = (reply or "").strip()
    if not hits or not text.startswith(REFUSALS):
        return False
    return not _cites(hits, text)


def _cites(hits, text):
    """回答裡有沒有提到任何一個撈回來的代號。"""
    return any(d.group in text for d, _ in hits)


def _could_still_be(text, hits=()):
    """串流途中：到目前為止的字，還有可能是一句「沒引用代號的拒絕」嗎。

    押著不吐是為了不讓畫面先閃過一句錯的再被換掉。兩種情況要繼續押：
      · 還在打那個開頭（「資」「資料」…）——最多幾個字
      · 已經是拒絕開頭，但還沒看到任何代號

    看到代號就立刻放行：那代表它在回答，不是在拒絕。所以
    「資料裡沒有台北的，但 A-01 在台南」只會被押到 A-01 出現為止。
    """
    s = text.strip()
    if not s:
        return True
    if s.startswith(REFUSALS):
        return not _cites(hits, s)
    return any(o.startswith(s) for o in REFUSALS)


class Turn:
    """串流的一輪對話。

    hits 馬上就有（檢索先跑完），所以 UI 可以在模型還沒吐字之前就把來源
    列出來。文字要迭代這個物件才會一段一段出來；跑完之後 reply 與 usage
    才有值——usage 只在最後一個 chunk 才知道。

    給了 retry 就多做一件事：開頭幾個字先押著不吐，直到確定它不是「資料裡
    沒有」那一句為止。真的是那一句而檢索又有東西，就呼叫 retry 再問一次
    （刻意不帶 history，理由見 contradicts）——押著的字還沒送出去，所以
    畫面上不會閃過那個錯的答案再被換掉。

    這裡有一個成立的不變式：contradicts 為真代表整段就是那一句，而那代表
    它從頭到尾都被押著、一個字都沒吐出去。所以「重抽」永遠不會發生在
    已經送出東西之後。
    """

    def __init__(self, hits, stream=None, history=None, reply=None, retry=None):
        self.hits = hits
        self.reply = reply
        self.usage = Usage()
        self._stream = stream
        self._history = history
        self._retry = retry               # () -> Stream，重抽一次用
        self.done = stream is None        # 短路那條路建出來就是完成狀態

    def __iter__(self):
        if self.done:
            if self.reply:
                yield self.reply          # 短路：一次吐完那一句
            return

        # 沒有 hits 就不可能矛盾，沒給 retry 就沒得重抽——兩種情況都不押字，
        # 走原本那條「收到什麼就吐什麼」的路，行為完全不變。
        guard = bool(self.hits) and self._retry is not None
        stream, text, hold, usage = self._stream, "", guard, Usage()

        for last in (False, True):        # 最多抽兩次
            text, hold = "", guard
            try:
                for piece in stream:
                    text += piece
                    if not hold:
                        yield piece
                    elif not _could_still_be(text, self.hits):
                        hold = False
                        yield text        # 分歧了，把押著的一次補上
            except Exception:
                if self._history is not None:
                    self._history.pop()   # 中途斷掉，把問題收回去
                raise
            usage = usage + stream.usage  # 白跑那次的 token 也真的花了錢

            if last or not contradicts(self.hits, text):
                break
            try:
                stream = self._retry()
            except Exception:
                if self._history is not None:
                    self._history.pop()
                raise

        if hold and text:
            yield text                    # 整段都被押著＝它真的就是那一句

        self.reply = text
        self.usage = usage
        if self._history is not None:
            self._history.append({"role": "assistant", "content": self.reply})
        self.done = True


class ChatBot:
    """history 是中性格式：[{"role": "user"|"assistant", "content": str}]。

    會被就地更新（呼叫端拿著同一個 list），失敗時維持呼叫前的樣子。
    """

    def __init__(self, retriever, llm, k=None, history_limit=None, temperature=None,
                 config=None, retry_contradiction=True):
        cfg = config or settings()
        self.retriever = retriever
        self.llm = llm
        self.k = k if k is not None else cfg.retrieve_k
        self.history_limit = history_limit        # None = 不砍
        self.temperature = temperature if temperature is not None else cfg.temperature
        # 偵測到矛盾就丟掉 history 再問一次（見 contradicts）。
        # 可以關掉是為了測試——要驗「不重抽會怎樣」就得關得掉。
        self.retry_contradiction = retry_contradiction

    def ask(self, user, history, k=None):
        """一次回完。回傳 (reply, hits, usage)。"""
        hits, to_send = self._prepare(user, history, k)
        if to_send is None:                       # 離題短路，沒打生成
            return NOT_FOUND, hits, Usage()

        try:
            reply = self._generate(to_send)
            usage = reply.usage
            if self.retry_contradiction and contradicts(hits, reply.text):
                # 只留最後一則（【資料】+【問題】）。原樣重抽沒有用，理由見
                # contradicts()——要換掉輸入才換得到不同的答案。
                again = self._generate(to_send[-1:])
                usage = usage + again.usage       # 白跑那次也計費，不能藏起來
                reply = again
        except Exception:
            history.pop()      # 把剛剛 append 的問題收回來，不然歷史會壞掉
            raise

        history.append({"role": "assistant", "content": reply.text})
        return reply.text, hits, usage

    def _generate(self, to_send):
        return self.llm.complete(to_send, system=SYSTEM,
                                 temperature=self.temperature)

    def ask_stream(self, user, history, k=None):
        """逐段回。回傳一個 Turn：先拿得到 hits，迭代它才吐文字。

        history 要等串流「跑完」才補上回答——中途斷掉的話會把問題收回去，
        跟 ask() 失敗時的行為一致。
        """
        hits, to_send = self._prepare(user, history, k)
        if to_send is None:
            return Turn(hits, reply=NOT_FOUND)    # 已完成，迭代會吐出那一句

        def open_stream(messages):
            return self.llm.stream(messages, system=SYSTEM,
                                   temperature=self.temperature)

        try:
            stream = open_stream(to_send)
        except Exception:
            history.pop()      # 連線階段就失敗（例如 server 沒開）
            raise
        # 重抽刻意不帶 history，只留【資料】+【問題】。理由見 contradicts()。
        return Turn(hits, stream=stream, history=history,
                    retry=(lambda: open_stream(to_send[-1:]))
                    if self.retry_contradiction else None)

    def _prepare(self, user, history, k):
        """兩條路共用的前半段：檢索、決定要不要短路、組 prompt、更新 history。

        回傳 (hits, to_send)。to_send 是 None 代表短路——history 已經補好
        一問一答，呼叫端不必再做事。
        """
        filters = self.retriever.filters_for(user)
        hits = self.retriever.retrieve(user, k=k if k is not None else self.k,
                                       filters=filters)

        if not hits and not history:
            # 第一句就離題才短路；有上下文的話（「那薪水呢？」這種跟隨問句
            # 檢索分數本來就低）交給模型判斷，短路會誤殺。
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": NOT_FOUND})
            return hits, None

        # 有過濾條件就講給模型聽。它不知道「台南算南部」，而過濾已經替它
        # 判斷過了——不說的話它會自己再判一次而且判錯（見 facets.describe）。
        note = ""
        if filters:
            note = ("（【資料】已依「%s」篩選完畢，列出的就是全部符合的職缺。）\n"
                    % describe(filters))
        prompt = "【資料】\n%s\n\n%s【問題】\n%s" % (
            self._context(hits), note, user)

        history.append({"role": "user", "content": user})     # 歷史存乾淨的
        if self.history_limit and len(history) > self.history_limit:
            del history[:-self.history_limit]

        to_send = list(history)                               # ← 分離送出的版本
        to_send[-1] = {"role": "user", "content": prompt}     # 只有這一輪帶【資料】
        return hits, to_send

    @staticmethod
    def _context(hits):
        """真正可靠的完整清單是 UI 的來源列（每一筆 hit 都會顯示，label 以代號
        開頭），模型的散文只當摘要看——實測 3B 模型不照指示抄代號，20 筆也只
        列得出 16~17 筆。所以筆數多的時候不必浪費 context 塞完整職缺。"""
        if len(hits) > FULL_TEXT_LIMIT:
            # 篩選型問句可能撈回 20 個職缺（例如「台北的職缺」）
            return "\n".join("[%d] %s" % (i + 1, d.label)     # label 本身以代號開頭
                             for i, (d, _) in enumerate(hits))
        # 少數幾筆就給完整職缺，模型才答得出細節
        return "\n\n".join("[%d] 代號 %s\n%s" % (i + 1, d.group, d.full)
                           for i, (d, _) in enumerate(hits))
