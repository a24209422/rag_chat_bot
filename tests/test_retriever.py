"""shared/retriever.py 的測試。

這些測試在 Stage 0 之前寫不出來：那時 DOCS 是模組層全域、索引也是全域，
沒辦法塞一批自己的資料進去。現在 docs 是建構子參數、embed() 是子類職責，
所以可以用一個「分數由字典決定」的假檢索器，把檢索邏輯跟模型完全隔開——
不載 e5、不打 Gemini、整批跑不到一秒。
"""
import numpy as np
import pytest

from shared.knowledge import Doc
from shared.retriever import BaseRetriever


class FakeRetriever(BaseRetriever):
    """分數由字典指定的檢索器。

    手法：文件向量做成 (n, 1)、查詢向量做成 [1.0]，那麼相似度就剛好等於
    我們指定的分數。這樣就能精準佈置「剛好卡在門檻上」這類邊界。

    ⚠ embed() 只能用傳進來的 texts，不能碰 self.docs——docs 會觸發建索引，
      而建索引又要呼叫 embed()，直接無窮遞迴。（踩過一次。）
    """

    min_score = 0.5
    embed_model = "fake"
    dim = 1

    def __init__(self, docs, scores, **kw):
        super().__init__(docs=docs, **kw)
        self.scores = scores                  # {doc.text: 分數}；doc() 讓 text == id
        self.embed_calls = 0

    def embed(self, texts, task_type):
        self.embed_calls += 1
        if task_type == "RETRIEVAL_QUERY":
            return np.array([[1.0]], dtype="float32")
        return np.array([[self.scores[t]] for t in texts], dtype="float32")


def doc(id_, group=None, **facets):
    return Doc(id=id_, text=id_, group=group or id_, facets=facets)


# ── 同職缺去重 ────────────────────────────────────────────────────────
def test_同一個職缺的多塊只留最高分那塊():
    """一個職缺切成好幾塊，實測 12 個問句有 9 個會同時撈到同職缺的多塊。
    不去重的話同一份 full 會被重複塞進 prompt。"""
    docs = [doc("A#1", group="A"), doc("A#2", group="A"), doc("B#1", group="B")]
    r = FakeRetriever(docs, {"A#1": 0.9, "A#2": 0.95, "B#1": 0.8})
    hits = r.retrieve("q", filters={})

    assert [d.id for d, _ in hits] == ["A#2", "B#1"]     # A 只出現一次，且是高分那塊
    assert [d.group for d, _ in hits] == ["A", "B"]


def test_回傳的是k個不同職缺不是k個塊():
    docs = [doc("A#1", group="A"), doc("A#2", group="A"),
            doc("B#1", group="B"), doc("C#1", group="C")]
    r = FakeRetriever(docs, {"A#1": 0.9, "A#2": 0.88, "B#1": 0.8, "C#1": 0.7})
    assert len(r.retrieve("q", k=2, filters={})) == 2
    assert {d.group for d, _ in r.retrieve("q", k=2, filters={})} == {"A", "B"}


# ── 門檻 ──────────────────────────────────────────────────────────────
def test_低於門檻的不回傳():
    docs = [doc("A"), doc("B"), doc("C")]
    r = FakeRetriever(docs, {"A": 0.9, "B": 0.6, "C": 0.4}, min_score=0.5)
    assert [d.id for d, _ in r.retrieve("q", filters={})] == ["A", "B"]


def test_剛好等於門檻要留下():
    """判斷是 `scores[i] < min_score` 才 break，所以等於門檻的要留著。"""
    r = FakeRetriever([doc("A")], {"A": 0.5}, min_score=0.5)
    assert len(r.retrieve("q", filters={})) == 1


def test_門檻可以用類別預設或實例或單次呼叫覆寫():
    docs = [doc("A"), doc("B")]
    assert FakeRetriever(docs, {"A": 0.9, "B": 0.3}).min_score == 0.5      # 類別預設
    r = FakeRetriever(docs, {"A": 0.9, "B": 0.3}, min_score=0.2)           # 實例
    assert len(r.retrieve("q", filters={})) == 2
    assert len(r.retrieve("q", min_score=0.8, filters={})) == 1            # 單次


def test_全部低於門檻就回空():
    """雲端側靠這條短路，省掉一通生成 API。"""
    r = FakeRetriever([doc("A")], {"A": 0.1}, min_score=0.5)
    assert r.retrieve("q", filters={}) == []


# ── 過濾條件會放掉門檻與 k ────────────────────────────────────────────
def test_有過濾條件時放掉門檻():
    """篩選已經保證相關，分數門檻反而會誤殺。"""
    docs = [doc("A", city=["台北"]), doc("B", city=["台北"])]
    r = FakeRetriever(docs, {"A": 0.9, "B": 0.01}, min_score=0.5)
    assert len(r.retrieve("q", filters={"city": ["台北"]})) == 2      # B 低於門檻仍要給


def test_有過濾條件時放掉k():
    """不放開的話問「有哪些台北的職缺」只給 5 個，實際有 20 個。"""
    docs = [doc(f"J{i}", city=["台北"]) for i in range(12)]
    r = FakeRetriever(docs, {f"J{i}": 0.9 for i in range(12)})
    assert len(r.retrieve("q", k=5, filters={"city": ["台北"]})) == 12


def test_過濾後仍然按分數排序():
    docs = [doc("A", city=["台北"]), doc("B", city=["台北"]), doc("C", city=["台北"])]
    r = FakeRetriever(docs, {"A": 0.3, "B": 0.9, "C": 0.6})
    assert [d.id for d, _ in r.retrieve("q", filters={"city": ["台北"]})] == ["B", "C", "A"]


def test_過濾條件不符的不回傳():
    docs = [doc("A", city=["台北"]), doc("B", city=["高雄"])]
    r = FakeRetriever(docs, {"A": 0.9, "B": 0.95})
    assert [d.id for d, _ in r.retrieve("q", filters={"city": ["台北"]})] == ["A"]


def test_過濾時同職缺一樣會去重():
    """放開 k 用的是「不同 group 的數量」，不是候選塊的數量。"""
    docs = [doc("A#1", group="A", city=["台北"]), doc("A#2", group="A", city=["台北"]),
            doc("B#1", group="B", city=["台北"])]
    r = FakeRetriever(docs, {"A#1": 0.9, "A#2": 0.8, "B#1": 0.7})
    hits = r.retrieve("q", filters={"city": ["台北"]})
    assert [d.group for d, _ in hits] == ["A", "B"]


# ── 沒有過濾條件就是純向量檢索 ────────────────────────────────────────
def test_沒有過濾條件時按分數由高到低():
    docs = [doc("A"), doc("B"), doc("C")]
    r = FakeRetriever(docs, {"A": 0.6, "B": 0.9, "C": 0.7})
    assert [d.id for d, _ in r.retrieve("q", filters={})] == ["B", "C", "A"]


def test_不傳filters時會自己從問句抽():
    """filters=None（預設）才會呼叫 parse_query。上面的測試都明確傳 filters
    是為了隔離，這一個才是驗證兩者接得起來。"""
    docs = [doc("A", city=["台北"]), doc("B", city=["高雄"])]
    r = FakeRetriever(docs, {"A": 0.1, "B": 0.99}, min_score=0.5)
    hits = r.retrieve("有哪些台北的職缺？")           # 沒傳 filters
    assert [d.id for d, _ in hits] == ["A"]           # 低分的 A 靠過濾勝出


# ── 延遲建立與快取 ────────────────────────────────────────────────────
def test_索引只建一次():
    r = FakeRetriever([doc("A")], {"A": 0.9})
    assert r.embed_calls == 0                  # 建構不算任何東西
    r.doc_vecs()
    r.doc_vecs()
    assert r.embed_calls == 1                  # 第二次走快取


def test_可以傳自己的docs不必有jobs_json():
    """store 會自己持有一份（因為它之後還要被增刪），所以比內容不比身分。"""
    mine = [doc("X")]
    assert FakeRetriever(mine, {"X": 0.9}).docs == mine


def test_區名詞彙表從傳進來的docs長出來():
    docs = [doc("A", district=["內湖"]), doc("B", district=["三重"])]
    assert FakeRetriever(docs, {"A": 0.9, "B": 0.9}).districts == ["三重", "內湖"]


def test_base_class_沒實作embed():
    """子類一定要提供 embed()，忘了就當場炸，不要默默算出爛向量。"""
    with pytest.raises(NotImplementedError):
        BaseRetriever(docs=[doc("A")]).embed(["x"], "RETRIEVAL_QUERY")


# ── 跟上傳的文件保持同步（增量，不重建） ──────────────────────────────
def chunk(id_, source, score_key=None):
    """registry 裡存的塊：dict，形狀跟 Doc 一致。"""
    return {"id": id_, "text": score_key or id_, "group": id_.split("#")[0],
            "source": source}


def test_建索引時會把registry裡的文件一起收進來(tmp_path):
    from shared.registry import Registry

    reg = Registry(tmp_path / "reg.db")
    reg.put("甲.pdf", "application/pdf", 1, "v1",
            [chunk("A-01#基本", "doc1"), chunk("A-01#內容", "doc1")], jobs=1)
    doc_id = next(iter(reg.doc_ids()))
    for c in reg.get(doc_id)["chunks"]:
        c["source"] = doc_id                 # put 之後才知道 doc_id
    reg.put("甲.pdf", "application/pdf", 1, "v1",
            [chunk("A-01#基本", doc_id), chunk("A-01#內容", doc_id)], jobs=1)

    r = FakeRetriever([doc("SEED")], {"SEED": 0.9, "A-01#基本": 0.8, "A-01#內容": 0.7},
                      registry=reg)

    assert {d.id for d in r.docs} == {"SEED", "A-01#基本", "A-01#內容"}
    assert r.store.sources() == {doc_id}


def make_synced(tmp_path, extra_scores=None):
    from shared.registry import Registry

    reg = Registry(tmp_path / "reg.db")
    scores = {"SEED": 0.9}
    scores.update(extra_scores or {})
    r = FakeRetriever([doc("SEED")], scores, registry=reg)
    assert r.docs                             # 先把索引建起來
    return r, reg


def add_doc(reg, filename, doc_id_chunks):
    doc_id = reg.put(filename, "application/pdf", 1, "v1", [], jobs=1)
    reg.put(filename, "application/pdf", 1, "v1",
            [chunk(i, doc_id) for i in doc_id_chunks], jobs=1)
    return doc_id


def test_refresh_只算新的不重建整個索引(tmp_path):
    """雲端重算一次是 140 個 API request 加上跨配額視窗的兩分鐘等待，
    所以「只算差集」不是優化，是可用性。"""
    r, reg = make_synced(tmp_path, {"A-01#基本": 0.8})
    calls_before = r.embed_calls

    add_doc(reg, "甲.pdf", ["A-01#基本"])
    added, removed = r.refresh()

    assert (added, removed) == (1, 0)
    assert r.embed_calls == calls_before + 1     # 只算了新加的那一塊
    assert {d.id for d in r.docs} == {"SEED", "A-01#基本"}


def test_refresh_會拿掉被刪掉的文件(tmp_path):
    r, reg = make_synced(tmp_path, {"A-01#基本": 0.8, "A-01#內容": 0.7})
    doc_id = add_doc(reg, "甲.pdf", ["A-01#基本", "A-01#內容"])
    r.refresh()

    reg.delete(doc_id)
    added, removed = r.refresh()

    assert (added, removed) == (0, 2)
    assert {d.id for d in r.docs} == {"SEED"}
    assert r.store.sources() == set()


def test_refresh_沒變動就什麼都不做(tmp_path):
    r, reg = make_synced(tmp_path)
    calls_before = r.embed_calls

    assert r.refresh() == (0, 0)
    assert r.embed_calls == calls_before


def test_refresh之後區名詞彙表會重算(tmp_path):
    """新文件可能帶進新的區名。詞彙表不更新的話，問「板橋的職缺」永遠
    抽不出過濾條件——而且不會報錯，只會默默退回純向量檢索。"""
    from shared.registry import Registry

    reg = Registry(tmp_path / "reg.db")
    r = FakeRetriever([doc("SEED", district=["內湖"])], {"SEED": 0.9, "NEW": 0.8},
                      registry=reg)
    assert r.districts == ["內湖"]

    doc_id = reg.put("甲.pdf", "application/pdf", 1, "v1", [], 1)
    reg.put("甲.pdf", "application/pdf", 1, "v1",
            [{"id": "NEW", "text": "NEW", "group": "NEW", "source": doc_id,
              "facets": {"district": ["板橋"]}}], 1)
    r.refresh()

    assert r.districts == ["內湖", "板橋"]


def test_沒有registry就不同步(tmp_path):
    """測試那條路：只給 docs，不接 registry。"""
    r = FakeRetriever([doc("A")], {"A": 0.9})
    assert r.refresh() == (0, 0)
