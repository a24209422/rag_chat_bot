# retriever.py（兩邊共用的檢索骨架）
#
#   雲端與地端的 retrieve() 本來是兩份逐字相同的程式碼，靠「請照著改」維持一致。
#   收成 base class 之後，兩邊的差異縮到只剩三樣：
#       embed()             怎麼把文字變成向量（Gemini API / 本機 e5）
#       min_score           對各自的模型量出來的門檻（差 30 倍，見各子類的註解）
#       embed_model / dim   給索引檔做相容性檢查用
#   「同名同形狀」從此是繼承保證的，不是靠約定。
#
#   狀態（索引、區名詞彙表、知識庫）全部掛在實例上，不是模組層全域——
#   所以同一個 process 裡可以同時存在雲端與地端兩個 Retriever，
#   測試也能各建各的、互不污染。
import numpy as np

from shared.facets import known_districts, match, parse_query
from shared.knowledge import Doc, default_docs
from shared.store import NumpyStore


class BaseRetriever:
    """檢索器。子類要提供 embed()、embed_model、dim。"""

    # 子類覆寫。0.0 代表「不擋」，是刻意安全的預設：
    # 忘了量門檻的後果應該是「撈回太多」，不是「靜默地撈不到」。
    min_score = 0.0
    embed_model = ""       # 存進索引檔，重開時比對——換了模型就必須整個重算
    dim = 0

    def __init__(self, docs=None, min_score=None, store_path=None, registry=None):
        """docs 是「基礎語料」——索引檔不存在時拿來初始化的那批。
        不傳就用 shared/knowledge.py 的 default_docs()。

        store_path 不傳就不落地（測試預設如此：不碰磁碟）。
        registry 給了的話，索引會跟上傳的文件保持同步（見 refresh）。
        """
        self._seed = docs
        self._store = None
        self._districts = None
        self.store_path = store_path
        self.registry = registry
        if min_score is not None:
            self.min_score = min_score

    # ── 延遲載入的狀態：建構不做重活，第一次用到才算 ──────────────────
    @property
    def store(self):
        if self._store is None:
            self._store = self._build_store()
        return self._store

    @property
    def docs(self):
        return self.store.docs

    @property
    def indexed(self):
        """索引算好了沒。給 /health 與預熱用——建構出 Retriever 不代表
        索引就算好了，這兩件事的成本差好幾個數量級。"""
        return self._store is not None

    @property
    def districts(self):
        """區名詞彙表：問句裡的「內湖」沒有「區」字可當錨點，只能靠詞彙表比對。
        從 docs 長出來，所以上傳新文件之後會跟著更新（見 refresh）。"""
        if self._districts is None:
            self._districts = known_districts(self.docs)
        return self._districts

    def doc_vecs(self):
        """整個索引的向量。給 tools/probe_threshold.py 用。"""
        return self.store.vectors

    # ── 子類負責的部分 ──────────────────────────────────────────────
    def embed(self, texts, task_type):
        """task_type 是 "RETRIEVAL_QUERY" 或 "RETRIEVAL_DOCUMENT"。

        沿用 Gemini 的命名：地端 e5 是用 "query: " / "passage: " 前綴達到同樣效果，
        在子類裡把 task_type 映成前綴，兩邊的函式形狀就一致了。
        """
        raise NotImplementedError

    def _embed_docs(self, docs):
        return self.embed([d.text for d in docs], "RETRIEVAL_DOCUMENT")

    # ── 索引的建立與同步 ────────────────────────────────────────────
    def _build_store(self):
        store = NumpyStore(self.embed_model, self.dim, self.store_path)

        if not store.load():                      # 沒有索引檔，或模型/維度對不上
            seed = self._seed if self._seed is not None else default_docs()
            store.add(seed, self._embed_docs(seed))
        self._sync(store)
        store.save()
        return store

    def _sync(self, store):
        """讓索引跟 registry 對齊：補上新上傳的文件、拿掉被刪掉的。

        這就是「增量」的意思——只算差集，不重建整個索引。雲端重算一次是
        140 個 API request 加上跨配額視窗的兩分鐘等待，差很多。
        """
        if self.registry is None:
            return 0, 0

        want = self.registry.doc_ids()
        have = store.sources()

        removed = sum(store.remove_source(gone) for gone in have - want)

        added, missing = 0, want - have
        if missing:
            chunks = self.registry.chunks_of(missing)
            docs = [Doc(**c) for part in chunks.values() for c in part]
            if docs:
                added = store.add(docs, self._embed_docs(docs))
        return added, removed

    def refresh(self):
        """registry 被改過之後呼叫。回傳 (加了幾塊, 刪了幾塊)。"""
        added, removed = self._sync(self.store)
        if added or removed:
            self.store.save()
            self._districts = None        # 區名詞彙表要跟著重算
        return added, removed

    # ── 共用的檢索邏輯 ──────────────────────────────────────────────
    def retrieve(self, question, k=5, min_score=None, filters=None):
        """回傳 k 個「不同職缺」，不是 k 個塊。

        先看問句有沒有可以精確過濾的條件（地點、工作性質、學歷），有的話就
        先篩再排序。這是為了「篩選型」問題——問「有哪些台北的職缺」時，純語意
        相似度只會給你最像的前 k 個，沒辦法保證「全部」；但地點本來就是結構化
        欄位，直接篩就能保證完整。見 shared/facets.py。

        有過濾條件時會放掉 min_score 和 k：篩選已經保證相關，分數門檻反而會
        誤殺；k 也要放開，不然問「有哪些台北的職缺」只給 5 個（實際有 20 個）。

        沒有過濾條件時就是純向量檢索。

        同職缺去重：一個職缺切成好幾塊，實測 12 個問句有 9 個會同時撈到同職缺
        的多塊，不去重的話同一份 full 會被重複塞進 prompt。
        """
        if min_score is None:
            min_score = self.min_score
        if filters is None:
            filters = parse_query(question, self.districts)

        docs = self.docs
        qv = self.embed([question], "RETRIEVAL_QUERY")[0]
        scores = self.store.scores(qv)

        if filters:
            cand = [i for i in range(len(docs)) if match(docs[i].facets, filters)]
            min_score = 0.0                                    # 篩過了，不再用分數擋
            k = max(k, len({docs[i].group for i in cand}))      # 要給全部，不是前 k 個
            order = sorted(cand, key=lambda i: -scores[i])
        else:
            order = np.argsort(-scores)

        out, seen = [], set()
        for i in order:
            if scores[i] < min_score:
                break                              # 已排序，低於門檻後面不用看了
            d = docs[i]
            if d.group in seen:                    # 同職缺只留最高分那塊
                continue
            seen.add(d.group)
            out.append((d, float(scores[i])))
            if len(out) == k:
                break
        return out
