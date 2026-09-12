"""shared/store.py：向量儲存層。

在知識庫能於執行期增刪之前，索引只是 BaseRetriever 裡一個 numpy 陣列，
靠「位置對齊」隱性綁著 docs，而且只能整批重算。這一層要保證的是：
增刪之後 docs 與向量仍然對齊、重開還在、換了模型不會拿舊向量硬算。
"""
import numpy as np
import pytest

from shared.knowledge import Doc
from shared.store import NumpyStore, VectorStore


def doc(id_, source=""):
    return Doc(id=id_, text=id_, group=id_, source=source)


def vecs(*values):
    """(n, 1) 的向量，值就是之後算出來的相似度（查詢向量固定是 [1.0]）。"""
    return np.array([[v] for v in values], dtype="float32")


def store(path=None):
    return NumpyStore("fake-model", 1, path)


def test_抽象介面沒實作():
    with pytest.raises(NotImplementedError):
        VectorStore().add([], [])


def test_加入後docs與向量對齊():
    s = store()
    s.add([doc("A"), doc("B")], vecs(0.9, 0.5))

    assert [d.id for d in s.docs] == ["A", "B"]
    assert list(s.scores(np.array([1.0], dtype="float32"))) == [0.9, 0.5]
    assert len(s) == 2


def test_數量對不上要當場炸():
    with pytest.raises(AssertionError):
        store().add([doc("A")], vecs(0.9, 0.5))


def test_維度對不上要當場炸():
    """設定裡的維度跟模型實際吐出來的不一樣——維度不合的向量算出來的相似度
    是沒有意義的數字，而且不會自己報錯，所以要在寫入的當下擋住。"""
    s = NumpyStore("fake-model", 384, None)
    with pytest.raises(ValueError, match="維度對不上"):
        s.add([doc("A")], vecs(0.9))


def test_按來源刪除():
    s = store()
    s.add([doc("A", "doc1"), doc("B", "doc2"), doc("C", "doc1")], vecs(0.9, 0.8, 0.7))

    assert s.remove_source("doc1") == 2
    assert [d.id for d in s.docs] == ["B"]
    assert list(s.scores(np.array([1.0], dtype="float32"))) == [0.8]   # 向量跟著切


def test_不能刪基礎語料():
    """source 是空字串的那些是建置階段的基礎語料，不是使用者管理的東西。
    允許用空字串刪會一次清掉全部，那是個很難查的意外。"""
    s = store()
    s.add([doc("A")], vecs(0.9))
    with pytest.raises(ValueError, match="基礎語料"):
        s.remove_source("")


def test_列出有哪些上傳來源():
    s = store()
    s.add([doc("A"), doc("B", "doc1"), doc("C", "doc1")], vecs(0.9, 0.8, 0.7))
    assert s.sources() == {"doc1"}          # 基礎語料（空 source）不算


def test_空的索引也要能查():
    assert len(store().scores(np.array([1.0], dtype="float32"))) == 0


def test_存了再讀回來(tmp_path):
    path = tmp_path / "s.npz"
    a = store(path)
    a.add([doc("A", "doc1"), doc("B")], vecs(0.9, 0.5))
    a.save()

    b = store(path)
    assert b.load()
    assert [d.id for d in b.docs] == ["A", "B"]
    assert [d.source for d in b.docs] == ["doc1", ""]
    assert list(b.scores(np.array([1.0], dtype="float32"))) == [0.9, 0.5]


def test_沒有檔案就讀不到(tmp_path):
    assert not store(tmp_path / "沒有這個.npz").load()


def test_沒給路徑就不落地():
    s = store(None)
    s.add([doc("A")], vecs(0.9))
    s.save()                                 # 不該炸，只是什麼都不做
    assert not store(None).load()


@pytest.mark.parametrize("model, dim", [("別的模型", 1), ("fake-model", 384)])
def test_換了模型或維度就當作沒有快取(tmp_path, model, dim):
    """不同模型的向量空間不相通。拿舊索引硬算出來的相似度是沒有意義的數字，
    而且不會報錯——所以寧可整個重算。"""
    path = tmp_path / "s.npz"
    a = store(path)
    a.add([doc("A")], vecs(0.9))
    a.save()

    b = NumpyStore(model, dim, path)
    assert not b.load()
    assert len(b) == 0
