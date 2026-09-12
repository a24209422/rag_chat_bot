# store.py（向量儲存層）
#
#   在這之前，索引是 BaseRetriever 裡一個 numpy 陣列，跟 docs 用「位置對齊」
#   隱性綁在一起，而且只能整批重算。知識庫一旦能在執行期增刪，那個做法就不夠了：
#   要能加一份文件而不重算全部、要能精準刪掉某份文件的塊、要能重啟後還在。
#
#   為什麼不直接上 Chroma：這批語料只有 140 塊，numpy 全掃是微秒級，
#   Chroma 的價值要到十萬塊以上才顯現。而它有一個對這個專案很痛的限制——
#   metadata 只吃 str/int/float/bool，不支援 list。這裡的 facets 全部是多值的
#   （city: ["台北","新竹"]），進 Chroma 就得攤平成 city_台北=True 這種布林欄，
#   derive() 與 match() 都要改寫。那是整個專案最有價值也最脆弱的一塊。
#
#   所以抽一層介面、先用 numpy 實作。語料真的長到需要 HNSW 時，
#   多一個子類就好，上層完全不用動——但屆時要記得處理 facets 攤平的問題。
import json

import numpy as np

from shared.knowledge import Doc


class VectorStore:
    """子類要實作下面五個。docs 與向量永遠等長、位置對齊。"""

    def add(self, docs, vecs):
        raise NotImplementedError

    def remove_source(self, source):
        """刪掉某份上傳文件的所有塊，回傳刪了幾塊。"""
        raise NotImplementedError

    def scores(self, qv):
        """對每一塊算相似度，回傳跟 docs 等長的陣列。"""
        raise NotImplementedError

    @property
    def docs(self):
        raise NotImplementedError

    def sources(self):
        """目前索引裡有哪些上傳文件（基礎語料的 source 是空字串，不算）。"""
        raise NotImplementedError


class NumpyStore(VectorStore):
    """全部放在記憶體、存成單一個 .npz。

    向量是正規化過的，所以 scores 就是矩陣乘法（餘弦相似度）。
    """

    def __init__(self, model, dim, path=None):
        self.model = model          # 哪個 embedding 模型算的
        self.dim = dim
        self.path = path
        self._docs = []
        self._vecs = np.zeros((0, dim), dtype="float32")

    @property
    def docs(self):
        return self._docs

    @property
    def vectors(self):
        return self._vecs

    def __len__(self):
        return len(self._docs)

    def add(self, docs, vecs):
        vecs = np.asarray(vecs, dtype="float32")
        assert len(docs) == len(vecs), "docs 與向量數量對不起來"
        if len(docs) == 0:
            return 0
        if vecs.shape[1] != self.dim:
            # 設定裡的維度跟模型實際吐出來的對不上。當場報錯，不要默默存進去
            # ——維度不合的向量算出來的相似度是沒有意義的數字。
            raise ValueError(
                "維度對不上：設定說 %d，%s 實際吐出 %d"
                % (self.dim, self.model, vecs.shape[1]))
        self._docs = self._docs + list(docs)
        self._vecs = np.vstack([self._vecs, vecs])
        return len(docs)

    def remove_source(self, source):
        if not source:
            raise ValueError("不能刪 source 為空的塊——那是建置階段的基礎語料")
        keep = [i for i, d in enumerate(self._docs) if d.source != source]
        removed = len(self._docs) - len(keep)
        if removed:
            self._docs = [self._docs[i] for i in keep]
            self._vecs = self._vecs[keep]
        return removed

    def scores(self, qv):
        if len(self._docs) == 0:
            return np.zeros(0, dtype="float32")
        return self._vecs @ qv

    def sources(self):
        return {d.source for d in self._docs if d.source}

    # ── 持久化 ────────────────────────────────────────────────────────
    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            vecs=self._vecs,
            docs=np.array(json.dumps([_as_dict(d) for d in self._docs],
                                     ensure_ascii=False)),
            meta=np.array(json.dumps({"model": self.model, "dim": self.dim})),
        )

    def load(self):
        """讀回上次存的。回傳有沒有讀成功。

        ⚠ 換了 embedding 模型或維度就「必須」整個重算——不同模型的向量空間
          不相通，混在一起算出來的相似度是沒有意義的數字，而且不會報錯。
          所以這裡拿 meta 比對，對不上就當作沒有快取。
        """
        if self.path is None or not self.path.exists():
            return False
        z = np.load(self.path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("model") != self.model or meta.get("dim") != self.dim:
            return False
        self._docs = [Doc(**d) for d in json.loads(str(z["docs"]))]
        self._vecs = z["vecs"]
        return True


def _as_dict(doc):
    return {"id": doc.id, "text": doc.text, "label": doc.label, "full": doc.full,
            "meta": doc.meta, "facets": doc.facets, "group": doc.group,
            "source": doc.source}
