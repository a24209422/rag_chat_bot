# retriever.py（兩邊共用的檢索骨架）
#
#   雲端與地端的 retrieve() 本來是兩份逐字相同的程式碼，靠「請照著改」維持一致。
#   收成 base class 之後，兩邊的差異縮到只剩三樣：
#       embed()      怎麼把文字變成向量（Gemini API / 本機 e5）
#       min_score    對各自的模型量出來的門檻（差 30 倍，見各子類的註解）
#       doc_vecs 的建法  雲端要快取（燒配額），地端不用（重算只是慢十秒）
#   「同名同形狀」從此是繼承保證的，不是靠約定。
#
#   狀態（索引、區名詞彙表、知識庫）全部掛在實例上，不是模組層全域——
#   所以同一個 process 裡可以同時存在雲端與地端兩個 Retriever，
#   測試也能各建各的、互不污染。
import numpy as np

from shared.facets import known_districts, match, parse_query
from shared.knowledge import default_docs


class BaseRetriever:
    """檢索器。子類要提供 embed()，其餘共用。"""

    # 子類覆寫。0.0 代表「不擋」，是刻意安全的預設：
    # 忘了量門檻的後果應該是「撈回太多」，不是「靜默地撈不到」。
    min_score = 0.0

    def __init__(self, docs=None, min_score=None):
        """docs 不傳就用 shared.knowledge.default_docs()（延遲到第一次用才讀檔）。

        測試要塞自己的知識庫就傳 docs=[Doc(...), ...]，不必有 data/jobs.json。
        """
        self._docs = docs
        self._districts = None
        self._doc_vecs = None
        if min_score is not None:
            self.min_score = min_score

    # ── 延遲載入的狀態：建構不做重活，第一次用到才算 ──────────────────
    @property
    def docs(self):
        if self._docs is None:
            self._docs = default_docs()
        return self._docs

    @property
    def districts(self):
        """區名詞彙表：問句裡的「內湖」沒有「區」字可當錨點，只能靠詞彙表比對。
        從 docs 長出來，所以重建 jobs.json 之後會自動跟著更新。"""
        if self._districts is None:
            self._districts = known_districts(self.docs)
        return self._districts

    def doc_vecs(self):
        if self._doc_vecs is None:
            self._doc_vecs = self._build_doc_vecs()
        return self._doc_vecs

    # ── 子類負責的部分 ──────────────────────────────────────────────
    def embed(self, texts, task_type):
        """task_type 是 "RETRIEVAL_QUERY" 或 "RETRIEVAL_DOCUMENT"。

        沿用 Gemini 的命名：地端 e5 是用 "query: " / "passage: " 前綴達到同樣效果，
        在子類裡把 task_type 映成前綴，兩邊的函式形狀就一致了。
        """
        raise NotImplementedError

    def _build_doc_vecs(self):
        """建索引。雲端會覆寫這個加上磁碟快取（重算要燒 140 個 request 配額）。"""
        return self.embed([d.text for d in self.docs], "RETRIEVAL_DOCUMENT")

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
        scores = self.doc_vecs() @ qv

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
