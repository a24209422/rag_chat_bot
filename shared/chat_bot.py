# chat_bot.py（檢索 + 生成的唯一一份實作）
#
#   雲端與地端共用這一個類別。差異全部收在建構時傳進來的兩個零件：
#       retriever  CloudRetriever / OnpremRetriever（差在 embed 與 min_score）
#       llm        GeminiLLM / LlamaCppLLM（差在線路格式與 usage 怎麼讀）
#   加第三家就是傳一個新的 llm 進來，這個檔案不用動。
from shared.knowledge import SYSTEM
from shared.llm import Usage
from shared.settings import settings

NOT_FOUND = "資料裡沒有。"     # 措辭跟 SYSTEM 一致，兩條路說法才不會打架
FULL_TEXT_LIMIT = 5           # 超過這個數量就只給 label，不給完整職缺


class Turn:
    """串流的一輪對話。

    hits 馬上就有（檢索先跑完），所以 UI 可以在模型還沒吐字之前就把來源
    列出來。文字要迭代這個物件才會一段一段出來；跑完之後 reply 與 usage
    才有值——usage 只在最後一個 chunk 才知道。
    """

    def __init__(self, hits, stream=None, history=None, reply=None):
        self.hits = hits
        self.reply = reply
        self.usage = Usage()
        self._stream = stream
        self._history = history
        self.done = stream is None        # 短路那條路建出來就是完成狀態

    def __iter__(self):
        if self.done:
            if self.reply:
                yield self.reply          # 短路：一次吐完那一句
            return

        try:
            yield from self._stream
        except Exception:
            if self._history is not None:
                self._history.pop()       # 中途斷掉，把問題收回去
            raise

        self.reply = self._stream.text
        self.usage = self._stream.usage
        if self._history is not None:
            self._history.append({"role": "assistant", "content": self.reply})
        self.done = True


class ChatBot:
    """history 是中性格式：[{"role": "user"|"assistant", "content": str}]。

    會被就地更新（呼叫端拿著同一個 list），失敗時維持呼叫前的樣子。
    """

    def __init__(self, retriever, llm, k=None, history_limit=None, temperature=None,
                 config=None):
        cfg = config or settings()
        self.retriever = retriever
        self.llm = llm
        self.k = k if k is not None else cfg.retrieve_k
        self.history_limit = history_limit        # None = 不砍
        self.temperature = temperature if temperature is not None else cfg.temperature

    def ask(self, user, history, k=None):
        """一次回完。回傳 (reply, hits, usage)。"""
        hits, to_send = self._prepare(user, history, k)
        if to_send is None:                       # 離題短路，沒打生成
            return NOT_FOUND, hits, Usage()

        try:
            reply = self.llm.complete(to_send, system=SYSTEM,
                                      temperature=self.temperature)
        except Exception:
            history.pop()      # 把剛剛 append 的問題收回來，不然歷史會壞掉
            raise

        history.append({"role": "assistant", "content": reply.text})
        return reply.text, hits, reply.usage

    def ask_stream(self, user, history, k=None):
        """逐段回。回傳一個 Turn：先拿得到 hits，迭代它才吐文字。

        history 要等串流「跑完」才補上回答——中途斷掉的話會把問題收回去，
        跟 ask() 失敗時的行為一致。
        """
        hits, to_send = self._prepare(user, history, k)
        if to_send is None:
            return Turn(hits, reply=NOT_FOUND)    # 已完成，迭代會吐出那一句

        try:
            stream = self.llm.stream(to_send, system=SYSTEM,
                                     temperature=self.temperature)
        except Exception:
            history.pop()      # 連線階段就失敗（例如 server 沒開）
            raise
        return Turn(hits, stream=stream, history=history)

    def _prepare(self, user, history, k):
        """兩條路共用的前半段：檢索、決定要不要短路、組 prompt、更新 history。

        回傳 (hits, to_send)。to_send 是 None 代表短路——history 已經補好
        一問一答，呼叫端不必再做事。
        """
        hits = self.retriever.retrieve(user, k=k if k is not None else self.k)

        if not hits and not history:
            # 第一句就離題才短路；有上下文的話（「那薪水呢？」這種跟隨問句
            # 檢索分數本來就低）交給模型判斷，短路會誤殺。
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": NOT_FOUND})
            return hits, None

        prompt = f"【資料】\n{self._context(hits)}\n\n【問題】\n{user}"

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
